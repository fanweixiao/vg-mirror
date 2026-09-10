//! Responses API <-> Chat Completions API conversion (non-streaming parts).

use std::collections::HashSet;

use serde_json::{Map, Value, json};
use tracing::{debug, warn};

pub struct ChatRequest {
    pub body: Value,
    /// Names of Responses `custom` (freeform) tools, e.g. codex's `apply_patch`.
    /// They are exposed upstream as functions taking `{"input": string}`.
    pub custom_tools: HashSet<String>,
    pub dropped_tools: Vec<String>,
}

pub fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

pub fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Request: Responses -> Chat Completions
// ---------------------------------------------------------------------------

pub fn to_chat_request(req: &Value, stream: bool) -> ChatRequest {
    let mut messages: Vec<Value> = Vec::new();

    if let Some(ins) = req.get("instructions").and_then(Value::as_str)
        && !ins.is_empty()
    {
        messages.push(json!({ "role": "system", "content": ins }));
    }

    match req.get("input") {
        Some(Value::String(s)) => messages.push(json!({ "role": "user", "content": s })),
        Some(Value::Array(items)) => {
            for item in items {
                convert_input_item(item, &mut messages);
            }
        }
        _ => {}
    }

    let mut body = Map::new();
    if let Some(m) = req.get("model") {
        body.insert("model".into(), m.clone());
    }
    body.insert("messages".into(), Value::Array(messages));
    body.insert("stream".into(), Value::Bool(stream));
    if stream {
        body.insert("stream_options".into(), json!({ "include_usage": true }));
    }

    for key in ["temperature", "top_p", "user"] {
        if let Some(v) = req.get(key).filter(|v| !v.is_null()) {
            body.insert(key.into(), v.clone());
        }
    }
    if let Some(v) = req.get("max_output_tokens").filter(|v| !v.is_null()) {
        body.insert("max_tokens".into(), v.clone());
    }
    if let Some(effort) = req.pointer("/reasoning/effort").filter(|v| !v.is_null()) {
        body.insert("reasoning_effort".into(), effort.clone());
    }
    if let Some(fmt) = req.pointer("/text/format")
        && fmt.get("type").and_then(Value::as_str) == Some("json_schema")
    {
        body.insert(
            "response_format".into(),
            json!({
                "type": "json_schema",
                "json_schema": {
                    "name": fmt.get("name").cloned().unwrap_or(json!("output")),
                    "schema": fmt.get("schema").cloned().unwrap_or(json!({})),
                    "strict": fmt.get("strict").cloned().unwrap_or(json!(false)),
                }
            }),
        );
    }

    let mut custom_tools = HashSet::new();
    let mut dropped_tools = Vec::new();
    let mut tools = Vec::new();
    for tool in req.get("tools").and_then(Value::as_array).into_iter().flatten() {
        let ty = tool.get("type").and_then(Value::as_str).unwrap_or("");
        let name = tool.get("name").and_then(Value::as_str).unwrap_or("");
        match ty {
            "function" => {
                let mut f = Map::new();
                f.insert("name".into(), json!(name));
                if let Some(d) = tool.get("description") {
                    f.insert("description".into(), d.clone());
                }
                f.insert(
                    "parameters".into(),
                    tool.get("parameters")
                        .cloned()
                        .unwrap_or(json!({ "type": "object", "properties": {} })),
                );
                if let Some(s) = tool.get("strict").filter(|v| !v.is_null()) {
                    f.insert("strict".into(), s.clone());
                }
                tools.push(json!({ "type": "function", "function": f }));
            }
            "custom" => {
                let mut desc = tool
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if let Some(def) = tool.pointer("/format/definition").and_then(Value::as_str) {
                    let syntax = tool
                        .pointer("/format/syntax")
                        .and_then(Value::as_str)
                        .unwrap_or("grammar");
                    desc.push_str(&format!(
                        "\n\nThe `input` argument must be raw text following this {syntax} grammar:\n{def}"
                    ));
                }
                custom_tools.insert(name.to_string());
                tools.push(json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": {
                            "type": "object",
                            "properties": { "input": { "type": "string", "description": "Raw tool input" } },
                            "required": ["input"],
                        }
                    }
                }));
            }
            other => dropped_tools.push(if name.is_empty() {
                other.to_string()
            } else {
                format!("{other}:{name}")
            }),
        }
    }
    if !tools.is_empty() {
        body.insert("tools".into(), Value::Array(tools));
        if let Some(tc) = req.get("tool_choice").filter(|v| !v.is_null()) {
            body.insert("tool_choice".into(), convert_tool_choice(tc));
        }
        if let Some(p) = req.get("parallel_tool_calls").filter(|v| !v.is_null()) {
            body.insert("parallel_tool_calls".into(), p.clone());
        }
    }

    ChatRequest {
        body: Value::Object(body),
        custom_tools,
        dropped_tools,
    }
}

fn convert_tool_choice(tc: &Value) -> Value {
    match tc {
        Value::Object(o) if o.get("type").and_then(Value::as_str) == Some("function") => {
            json!({ "type": "function", "function": { "name": o.get("name").cloned().unwrap_or(Value::Null) } })
        }
        Value::Object(_) => json!("auto"),
        other => other.clone(),
    }
}

fn convert_input_item(item: &Value, messages: &mut Vec<Value>) {
    let ty = item
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or(if item.get("role").is_some() { "message" } else { "" });
    let str_field = |k: &str| item.get(k).and_then(Value::as_str).unwrap_or("").to_string();

    match ty {
        "message" => {
            let role = match item.get("role").and_then(Value::as_str).unwrap_or("user") {
                "developer" | "system" => "system",
                "assistant" => "assistant",
                _ => "user",
            };
            let content = convert_message_content(item.get("content"), role);
            messages.push(json!({ "role": role, "content": content }));
        }
        "function_call" => {
            let mut args = str_field("arguments");
            if args.trim().is_empty() {
                args = "{}".into();
            }
            push_tool_call(messages, &str_field("call_id"), &str_field("name"), args);
        }
        "custom_tool_call" => {
            let args = json!({ "input": str_field("input") }).to_string();
            push_tool_call(messages, &str_field("call_id"), &str_field("name"), args);
        }
        "function_call_output" | "custom_tool_call_output" => {
            messages.push(json!({
                "role": "tool",
                "tool_call_id": str_field("call_id"),
                "content": tool_output_to_string(item.get("output")),
            }));
        }
        // Reasoning items (often encrypted) cannot be replayed to a chat backend.
        "reasoning" => {}
        other => debug!(item_type = other, "skipping unsupported input item"),
    }
}

/// Chat Completions requires tool_calls to live on an assistant message;
/// merge consecutive calls (and a preceding assistant text) into one message.
fn push_tool_call(messages: &mut Vec<Value>, call_id: &str, name: &str, args: String) {
    let tc = json!({
        "id": call_id,
        "type": "function",
        "function": { "name": name, "arguments": args },
    });
    if let Some(last) = messages.last_mut()
        && last.get("role").and_then(Value::as_str) == Some("assistant")
        && let Some(obj) = last.as_object_mut()
    {
        if let Some(arr) = obj
            .entry("tool_calls")
            .or_insert_with(|| json!([]))
            .as_array_mut()
        {
            arr.push(tc);
        }
        return;
    }
    messages.push(json!({ "role": "assistant", "content": null, "tool_calls": [tc] }));
}

fn convert_message_content(content: Option<&Value>, role: &str) -> Value {
    let parts = match content {
        Some(Value::String(s)) => return json!(s),
        Some(Value::Array(parts)) => parts,
        _ => return json!(""),
    };

    let mut out = Vec::new();
    let mut has_image = false;
    for p in parts {
        match p.get("type").and_then(Value::as_str).unwrap_or("") {
            "input_text" | "output_text" | "text" => {
                out.push(json!({ "type": "text", "text": p.get("text").and_then(Value::as_str).unwrap_or("") }))
            }
            "refusal" => {
                out.push(json!({ "type": "text", "text": p.get("refusal").and_then(Value::as_str).unwrap_or("") }))
            }
            "input_image" => {
                let url = p
                    .get("image_url")
                    .and_then(|u| u.as_str().map(str::to_string).or_else(|| u.get("url")?.as_str().map(str::to_string)))
                    .unwrap_or_default();
                if url.is_empty() {
                    warn!("input_image without image_url (file_id is not supported), dropped");
                    continue;
                }
                has_image = true;
                let mut image = json!({ "url": url });
                if let Some(d) = p.get("detail").filter(|v| !v.is_null()) {
                    image["detail"] = d.clone();
                }
                out.push(json!({ "type": "image_url", "image_url": image }));
            }
            other => debug!(part_type = other, "skipping unsupported content part"),
        }
    }

    if has_image && role == "user" {
        return Value::Array(out);
    }
    let text: Vec<&str> = out
        .iter()
        .filter_map(|p| p.get("text").and_then(Value::as_str))
        .collect();
    json!(text.join("\n"))
}

fn tool_output_to_string(output: Option<&Value>) -> String {
    match output {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Response: Chat Completions -> Responses
// ---------------------------------------------------------------------------

/// finish_reason -> (Responses status, incomplete_details.reason)
pub fn map_finish_reason(fr: Option<&str>) -> (&'static str, Option<&'static str>) {
    match fr {
        Some("length") => ("incomplete", Some("max_output_tokens")),
        Some("content_filter") => ("incomplete", Some("content_filter")),
        _ => ("completed", None),
    }
}

pub fn convert_usage(u: Option<&Value>) -> Value {
    let Some(u) = u.filter(|u| u.is_object()) else {
        return Value::Null;
    };
    let n = |p: &str| u.pointer(p).and_then(Value::as_u64).unwrap_or(0);
    let input = n("/prompt_tokens");
    let output = n("/completion_tokens");
    let total = u.get("total_tokens").and_then(Value::as_u64).unwrap_or(input + output);
    json!({
        "input_tokens": input,
        "input_tokens_details": { "cached_tokens": n("/prompt_tokens_details/cached_tokens") },
        "output_tokens": output,
        "output_tokens_details": { "reasoning_tokens": n("/completion_tokens_details/reasoning_tokens") },
        "total_tokens": total,
    })
}

pub fn response_object(
    id: &str,
    created_at: i64,
    model: &str,
    status: &str,
    incomplete_reason: Option<&str>,
    output: Vec<Value>,
    usage: Value,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "output": output,
        "usage": usage,
        "incomplete_details": incomplete_reason.map(|r| json!({ "reason": r })),
        "error": null,
    })
}

pub fn message_item(id: &str, text: &str) -> Value {
    json!({
        "id": id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{ "type": "output_text", "text": text, "annotations": [] }],
    })
}

pub fn reasoning_item(id: &str, text: &str) -> Value {
    json!({
        "id": id,
        "type": "reasoning",
        "summary": [{ "type": "summary_text", "text": text }],
    })
}

pub fn tool_call_item(id: &str, call_id: &str, name: &str, args: &str, custom: bool) -> Value {
    if custom {
        // Unwrap {"input": "..."}; fall back to the raw string if the model didn't comply.
        let input = serde_json::from_str::<Value>(args)
            .ok()
            .and_then(|v| v.get("input")?.as_str().map(str::to_string))
            .unwrap_or_else(|| args.to_string());
        json!({
            "id": id,
            "type": "custom_tool_call",
            "status": "completed",
            "call_id": call_id,
            "name": name,
            "input": input,
        })
    } else {
        json!({
            "id": id,
            "type": "function_call",
            "status": "completed",
            "call_id": call_id,
            "name": name,
            "arguments": args,
        })
    }
}

pub fn reasoning_text(v: &Value) -> Option<&str> {
    v.get("reasoning_content")
        .or_else(|| v.get("reasoning"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
}

/// Convert a non-streaming chat completion into a Responses object.
pub fn chat_to_response(chat: &Value, resp_id: &str, fallback_model: &str, custom: &HashSet<String>) -> Value {
    let choice = chat.pointer("/choices/0").cloned().unwrap_or(Value::Null);
    let msg = choice.get("message").cloned().unwrap_or(Value::Null);
    let mut output = Vec::new();

    if let Some(r) = reasoning_text(&msg) {
        output.push(reasoning_item(&new_id("rs"), r));
    }
    if let Some(text) = msg.get("content").and_then(Value::as_str).filter(|s| !s.is_empty()) {
        output.push(message_item(&new_id("msg"), text));
    }
    for tc in msg.get("tool_calls").and_then(Value::as_array).into_iter().flatten() {
        let name = tc.pointer("/function/name").and_then(Value::as_str).unwrap_or("");
        let args = tc.pointer("/function/arguments").and_then(Value::as_str).unwrap_or("");
        let call_id = tc
            .get("id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| new_id("call"));
        output.push(tool_call_item(&new_id("fc"), &call_id, name, args, custom.contains(name)));
    }

    let (status, reason) = map_finish_reason(choice.get("finish_reason").and_then(Value::as_str));
    let model = chat.get("model").and_then(Value::as_str).unwrap_or(fallback_model);
    let created = chat.get("created").and_then(Value::as_i64).unwrap_or_else(now_secs);
    response_object(resp_id, created, model, status, reason, output, convert_usage(chat.get("usage")))
}
