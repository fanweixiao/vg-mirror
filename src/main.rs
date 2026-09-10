//! Local proxy for codex: exposes OpenAI `/v1/responses`, forwards to an
//! upstream Chat Completions endpoint, and logs finish_reason + usage.

mod convert;
mod report;
mod stream;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error, info, warn};
use tracing_subscriber::EnvFilter;

const DEFAULT_UPSTREAM: &str = "https://api.vivgrid.com/v1/chat/completions";
const DEFAULT_LISTEN: &str = "127.0.0.1:33333";

/// Incoming headers forwarded upstream under a new name: (incoming, upstream).
const HEADER_RENAMES: &[(&str, &str)] = &[("x-codex-turn-metadata", "x-viv-meta")];

#[derive(Clone)]
struct AppState {
    client: reqwest::Client,
    upstream: String,
    models_url: String,
    /// Used only when the incoming request carries no Authorization header.
    fallback_key: Option<String>,
    counter: Arc<AtomicU64>,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("codex_proxy=info")),
        )
        .with_target(false)
        .init();

    let listen = std::env::var("LISTEN").unwrap_or_else(|_| DEFAULT_LISTEN.into());
    let upstream = std::env::var("UPSTREAM_URL").unwrap_or_else(|_| DEFAULT_UPSTREAM.into());
    let models_url = match upstream.strip_suffix("/chat/completions") {
        Some(base) => format!("{base}/models"),
        None => upstream.clone(),
    };
    let fallback_key = std::env::var("UPSTREAM_API_KEY").ok().filter(|k| !k.is_empty());

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .build()
        .expect("build http client");

    let state = AppState {
        client,
        upstream: upstream.clone(),
        models_url,
        fallback_key,
        counter: Arc::new(AtomicU64::new(0)),
    };

    let app = Router::new()
        .route("/v1/responses", post(responses))
        .route("/responses", post(responses))
        .route("/v1/models", get(models))
        .route("/health", get(|| async { "ok" }))
        .fallback(|method: axum::http::Method, uri: axum::http::Uri, _headers: HeaderMap| async move {
            warn!("unhandled route: {method} {uri}");
            // warn!("  headers ({}):\n{}", _headers.len(), format_headers(&_headers));
            error_json(StatusCode::NOT_FOUND, &format!("no route for {method} {uri}"))
        })
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&listen).await.expect("bind listen address");
    info!("codex-proxy listening on http://{listen}  →  upstream {upstream}");
    axum::serve(listener, app).await.expect("server error");
}

fn error_json(status: StatusCode, msg: &str) -> Response {
    (status, Json(json!({ "error": { "message": msg, "type": "proxy_error" } }))).into_response()
}

fn auth_header(st: &AppState, headers: &HeaderMap) -> Option<HeaderValue> {
    if let Some(v) = headers.get(header::AUTHORIZATION) {
        return Some(v.clone());
    }
    let key = st.fallback_key.as_ref()?;
    HeaderValue::from_str(&format!("Bearer {key}")).ok()
}

/// One header per line; the Authorization token is masked to its first 6 / last 4 chars.
/// Currently unused: the header log lines are commented out.
#[allow(dead_code)]
fn format_headers(headers: &HeaderMap) -> String {
    let mut lines: Vec<String> = headers
        .iter()
        .map(|(name, value)| {
            let v = String::from_utf8_lossy(value.as_bytes()).to_string();
            let v = if name == header::AUTHORIZATION { mask_auth(&v) } else { v };
            format!("    {name}: {v}")
        })
        .collect();
    lines.sort();
    lines.join("\n")
}

#[allow(dead_code)]
fn mask_auth(v: &str) -> String {
    let (scheme, token) = v.split_once(' ').unwrap_or(("", v));
    let chars: Vec<char> = token.chars().collect();
    let masked = if chars.len() <= 12 {
        "*".repeat(chars.len())
    } else {
        let head: String = chars[..6].iter().collect();
        let tail: String = chars[chars.len() - 4..].iter().collect();
        format!("{head}…{tail} ({} chars)", chars.len())
    };
    if scheme.is_empty() { masked } else { format!("{scheme} {masked}") }
}

async fn responses(State(st): State<AppState>, headers: HeaderMap, body: Bytes) -> Response {
    let req_id = st.counter.fetch_add(1, Ordering::Relaxed) + 1;
    let started = Instant::now();

    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            error!(req = req_id, "invalid JSON body: {e}");
            return error_json(StatusCode::BAD_REQUEST, &format!("invalid JSON body: {e}"));
        }
    };
    let is_stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let model = req.get("model").and_then(Value::as_str).unwrap_or("").to_string();
    let chat = convert::to_chat_request(&req, is_stream);

    let n_messages = chat.body["messages"].as_array().map_or(0, Vec::len);
    let n_tools = chat.body.get("tools").and_then(Value::as_array).map_or(0, Vec::len);
    let n_input = req.get("input").and_then(Value::as_array).map_or(1, Vec::len);
    let effort = req.pointer("/reasoning/effort").and_then(Value::as_str).unwrap_or("-");
    let mode = if is_stream { "stream" } else { "non-stream" };
    let tools = report::tools_summary(&req, &chat.dropped_tools);
    let line = format!(
        "#{req_id} ▶ request  [{mode}, model={model}, input_items={n_input} → messages={n_messages}, tools={n_tools}, reasoning_effort={effort}]\n{tools}"
    );
    if tools.contains('⚠') {
        warn!("{line}");
    } else {
        info!("{line}");
    }
    // info!("#{req_id} headers ({}):\n{}", headers.len(), format_headers(&headers));
    debug!("#{req_id} upstream request body: {}", chat.body);

    let Some(auth) = auth_header(&st, &headers) else {
        warn!("#{req_id} no Authorization header (and UPSTREAM_API_KEY unset)");
        return error_json(StatusCode::UNAUTHORIZED, "missing Authorization: Bearer <token>");
    };

    let accept = if is_stream { "text/event-stream" } else { "application/json" };
    let mut rb = st
        .client
        .post(&st.upstream)
        .header(header::AUTHORIZATION, auth)
        .header(header::ACCEPT, accept);
    if let Some(ua) = headers.get(header::USER_AGENT) {
        rb = rb.header(header::USER_AGENT, ua.clone());
    }
    for (from, to) in HEADER_RENAMES {
        if let Some(v) = headers.get(*from) {
            debug!("#{req_id} forwarding header {from} → {to}");
            rb = rb.header(*to, v.clone());
        }
    }
    let upstream = match rb.json(&chat.body).send().await {
        Ok(r) => r,
        Err(e) => {
            error!("#{req_id} ✖ upstream request failed: {e}");
            return error_json(StatusCode::BAD_GATEWAY, &format!("upstream request failed: {e}"));
        }
    };

    let status = upstream.status();
    let content_type = upstream
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    if !status.is_success() || (is_stream && content_type.starts_with("application/json")) {
        let text = upstream.text().await.unwrap_or_default();
        error!(
            "#{req_id} ✖ upstream HTTP {status} ({:.2}s)\n    stop   ▸ n/a (no completion)\n    body   ▸ {text}",
            started.elapsed().as_secs_f64()
        );
        let code = if status.is_success() { StatusCode::BAD_GATEWAY } else { status };
        return (code, [(header::CONTENT_TYPE, "application/json")], text).into_response();
    }

    if is_stream {
        let (tx, rx) = mpsc::channel(64);
        let translator = stream::Translator::new(tx, req_id, started, model, chat.custom_tools);
        tokio::spawn(translator.run(upstream));
        return Sse::new(ReceiverStream::new(rx)).into_response();
    }

    let chat_resp: Value = match upstream.json().await {
        Ok(v) => v,
        Err(e) => {
            error!("#{req_id} ✖ failed to read upstream JSON: {e}");
            return error_json(StatusCode::BAD_GATEWAY, &format!("invalid upstream JSON: {e}"));
        }
    };
    debug!("#{req_id} upstream response body: {chat_resp}");

    let resp = convert::chat_to_response(&chat_resp, &convert::new_id("resp"), &model, &chat.custom_tools);
    let choice = chat_resp.pointer("/choices/0").cloned().unwrap_or(Value::Null);
    let mut notes = Vec::new();
    if let Some(n) = chat_resp.get("choices").and_then(Value::as_array).map(Vec::len)
        && n != 1
    {
        notes.push(format!("upstream returned {n} choices (only choice 0 used)"));
    }
    report::log(&report::Report {
        req_id,
        stream: false,
        elapsed: started.elapsed(),
        model: resp["model"].as_str().unwrap_or(&model),
        finish_reason: choice.get("finish_reason").and_then(Value::as_str),
        extra_stop: &report::extra_stop_fields(&choice),
        status: resp["status"].as_str().unwrap_or("?"),
        usage: chat_resp.get("usage"),
        notes: &notes,
    });

    Json(resp).into_response()
}

async fn models(State(st): State<AppState>, headers: HeaderMap) -> Response {
    // info!("GET /v1/models\n  headers ({}):\n{}", headers.len(), format_headers(&headers));
    let mut rb = st.client.get(&st.models_url);
    if let Some(auth) = auth_header(&st, &headers) {
        rb = rb.header(header::AUTHORIZATION, auth);
    }
    if let Some(ua) = headers.get(header::USER_AGENT) {
        rb = rb.header(header::USER_AGENT, ua.clone());
    }
    match rb.send().await {
        Ok(r) => {
            let status = r.status();
            let text = r.text().await.unwrap_or_default();
            (status, [(header::CONTENT_TYPE, "application/json")], text).into_response()
        }
        Err(e) => error_json(StatusCode::BAD_GATEWAY, &format!("upstream request failed: {e}")),
    }
}
