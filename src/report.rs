//! Per-response log block: stop value + usage.

use std::fmt::Write;
use std::time::Duration;

use serde_json::Value;
use tracing::{info, warn};

pub struct Report<'a> {
    pub req_id: u64,
    pub stream: bool,
    pub elapsed: Duration,
    pub model: &'a str,
    /// Raw `finish_reason` from upstream (None = never received).
    pub finish_reason: Option<&'a str>,
    /// Other stop-ish fields upstream sent on the choice (stop_reason, native_finish_reason, ...).
    pub extra_stop: &'a [(String, Value)],
    /// Status we reported to the client (completed / incomplete / failed).
    pub status: &'a str,
    /// Raw upstream Chat Completions usage object.
    pub usage: Option<&'a Value>,
    pub notes: &'a [String],
}

pub fn log(r: &Report) {
    let mut s = String::new();
    let mode = if r.stream { "stream" } else { "non-stream" };
    let _ = writeln!(
        s,
        "#{} ◀ response  [{mode}, {:.2}s, model={}]",
        r.req_id,
        r.elapsed.as_secs_f64(),
        r.model
    );

    // --- stop ---
    let fr = match r.finish_reason {
        Some(v) => format!("{v:?}"),
        None => "<MISSING> (upstream never sent finish_reason)".to_string(),
    };
    let _ = write!(s, "    stop   ▸ finish_reason = {fr}  →  status = {}", r.status);
    for (k, v) in r.extra_stop {
        let _ = write!(s, "  | {k} = {v}");
    }
    s.push('\n');

    // --- usage ---
    match r.usage.filter(|u| u.is_object()) {
        None => s.push_str("    usage  ▸ <none> (upstream returned no usage)"),
        Some(u) => {
            let n = |p: &str| u.pointer(p).and_then(Value::as_u64);
            let input = n("/prompt_tokens").unwrap_or(0);
            let output = n("/completion_tokens").unwrap_or(0);
            let cached = match n("/prompt_tokens_details/cached_tokens") {
                Some(c) => {
                    let pct = if input > 0 { c as f64 * 100.0 / input as f64 } else { 0.0 };
                    format!("{} ({pct:.1}%)", fmt_num(c))
                }
                None => "n/a".into(),
            };
            let reasoning = n("/completion_tokens_details/reasoning_tokens").map_or("n/a".into(), fmt_num);
            let _ = write!(
                s,
                "    usage  ▸ inp: {}, cd-inp: {cached}, opt: {} (reasoning: {reasoning})",
                fmt_num(input),
                fmt_num(output)
            );
        }
    }

    for note in r.notes {
        let _ = write!(s, "\n    note   ▸ {note}");
    }

    let normal = matches!(r.finish_reason, Some("stop" | "tool_calls")) && r.notes.is_empty();
    if normal {
        info!("{s}");
    } else {
        warn!("{s}");
    }
}

pub fn fmt_num(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Multi-line summary of the tool-related parts of a Responses request.
pub fn tools_summary(req: &Value, dropped: &[String]) -> String {
    let mut s = String::new();

    let tools = req.get("tools").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]);
    let declared: Vec<String> = tools
        .iter()
        .filter_map(|t| {
            let ty = t.get("type").and_then(Value::as_str).unwrap_or("?");
            let name = t.get("name").and_then(Value::as_str).unwrap_or("?");
            match ty {
                "function" => {
                    let params: Vec<&str> = t
                        .pointer("/parameters/properties")
                        .and_then(Value::as_object)
                        .map(|p| p.keys().map(String::as_str).collect())
                        .unwrap_or_default();
                    Some(format!("{name}({}) [function]", params.join(", ")))
                }
                "custom" => Some(format!("{name}(input) [custom→function]")),
                _ => None,
            }
        })
        .collect();
    let _ = write!(s, "    tools  ▸ declared ({}): ", declared.len());
    s.push_str(if declared.is_empty() { "-" } else { "" });
    s.push_str(&declared.join(", "));
    if !dropped.is_empty() {
        let _ = write!(s, "\n             dropped ({}): {}", dropped.len(), dropped.join(", "));
    }
    let parallel = req
        .get("parallel_tool_calls")
        .filter(|v| !v.is_null())
        .map_or("-".to_string(), Value::to_string);
    let _ = write!(s, "\n             parallel_tool_calls={parallel}");

    // Tool calls / outputs replayed in the input history.
    let items = req.get("input").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]);
    let mut calls: Vec<(String, String)> = Vec::new(); // (call_id, name)
    let mut outputs: Vec<String> = Vec::new();
    for it in items {
        let call_id = it.get("call_id").and_then(Value::as_str).unwrap_or("").to_string();
        match it.get("type").and_then(Value::as_str).unwrap_or("") {
            "function_call" | "custom_tool_call" | "local_shell_call" => {
                let name = it.get("name").and_then(Value::as_str).unwrap_or("local_shell");
                calls.push((call_id, name.to_string()));
            }
            "function_call_output" | "custom_tool_call_output" | "local_shell_call_output" => outputs.push(call_id),
            _ => {}
        }
    }
    let mut counts: Vec<(String, usize)> = Vec::new();
    for (_, name) in &calls {
        match counts.iter_mut().find(|(n, _)| n == name) {
            Some(c) => c.1 += 1,
            None => counts.push((name.clone(), 1)),
        }
    }
    let by_name: Vec<String> = counts.iter().map(|(n, c)| format!("{n} ×{c}")).collect();
    let _ = write!(s, "\n             in input: {} calls", calls.len());
    if !by_name.is_empty() {
        let _ = write!(s, " ({})", by_name.join(", "));
    }
    let _ = write!(s, ", {} outputs", outputs.len());

    let no_output: Vec<&str> = calls
        .iter()
        .filter(|(id, _)| !outputs.contains(id))
        .map(|(id, _)| id.as_str())
        .collect();
    let no_call: Vec<&str> = outputs
        .iter()
        .filter(|id| !calls.iter().any(|(c, _)| c == *id))
        .map(String::as_str)
        .collect();
    if !no_output.is_empty() {
        let _ = write!(s, "\n             ⚠ calls without output: {}", no_output.join(", "));
    }
    if !no_call.is_empty() {
        let _ = write!(s, "\n             ⚠ outputs without call: {}", no_call.join(", "));
    }
    s
}

/// Extra stop-related fields some backends put on a choice.
pub fn extra_stop_fields(choice: &Value) -> Vec<(String, Value)> {
    ["stop_reason", "native_finish_reason", "matched_stop"]
        .iter()
        .filter_map(|k| {
            choice
                .get(*k)
                .filter(|v| !v.is_null())
                .map(|v| (k.to_string(), v.clone()))
        })
        .collect()
}
