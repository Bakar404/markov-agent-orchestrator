//! verified-kernel: the proved kernel behind a JSON-lines CLI, for the
//! differential test in backend/tests/test_verified_kernel.py.
//!
//! One JSON object per line in, one per line out. Floats travel as the u64 bit
//! pattern of the IEEE double (`"x": 4607182418800017408` is 1.0), so NaN, the
//! infinities and every last bit cross the boundary exactly. This file is not
//! verified: it is the adapter, TB3 in verified/README.md. It does one thing a
//! proof can't do for it: Verus `requires` clauses are not checked at run time,
//! so every precondition of a kernel call is checked here first, and a request
//! that would break one gets `{"error": ...}` instead of a verdict.

use markov_verified::*;
use serde_json::{json, Map, Value};
use std::io::{self, BufRead, Write};

const ACTION_NAMES: [&str; 10] = [
    "invoke_generalist",
    "invoke_planner",
    "invoke_researcher",
    "invoke_critic",
    "invoke_verifier",
    "invoke_executor",
    "invoke_memory",
    "run_parallel",
    "escalate",
    "terminate",
];

const TENSORS: [&str; 3] = ["q_weights", "synergy", "terminate_weights"];

type Res<T> = Result<T, String>;

fn field<'a>(o: &'a Map<String, Value>, k: &str) -> Res<&'a Value> {
    o.get(k).ok_or_else(|| format!("missing field {k}"))
}

fn float(o: &Map<String, Value>, k: &str) -> Res<f64> {
    let bits = field(o, k)?.as_u64().ok_or_else(|| format!("{k} must be u64 float bits"))?;
    Ok(f64::from_bits(bits))
}

/// A float, quantized, which a budget or gate call needs to be finite.
fn finite_micro(o: &Map<String, Value>, k: &str) -> Res<i64> {
    let q = quantize(float(o, k)?);
    if q == NONFINITE {
        return Err(format!("{k} is not finite"));
    }
    Ok(q)
}

fn int(o: &Map<String, Value>, k: &str) -> Res<i64> {
    field(o, k)?.as_i64().ok_or_else(|| format!("{k} must be an integer"))
}

fn boolean(o: &Map<String, Value>, k: &str) -> Res<bool> {
    field(o, k)?.as_bool().ok_or_else(|| format!("{k} must be a boolean"))
}

fn micro_out(q: i64) -> Value {
    if q == NONFINITE { Value::Null } else { json!(q) }
}

fn gate(o: &Map<String, Value>) -> Res<Value> {
    let g = GateInput {
        has_escalated: boolean(o, "has_escalated")?,
        allow_escalation: boolean(o, "allow_escalation")?,
        solo_steps: int(o, "solo_steps")?,
        min_solo_steps: int(o, "min_solo_steps")?,
        step: int(o, "step")?,
        min_steps_before_terminate: int(o, "min_steps_before_terminate")?,
        unresolved_subtasks: int(o, "unresolved_subtasks")?,
        budget_total: finite_micro(o, "budget_total")?,
        budget_spent: finite_micro(o, "budget_spent")?,
        latency_budget: finite_micro(o, "latency_budget")?,
        latency_consumed: finite_micro(o, "latency_consumed")?,
    };
    let mask = legal_actions(&g);
    let legal: Vec<&str> = (0..ACTION_COUNT).filter(|&a| mask[a]).map(|a| ACTION_NAMES[a]).collect();
    Ok(json!({ "legal": legal }))
}

fn budget(o: &Map<String, Value>) -> Res<Value> {
    let total = finite_micro(o, "total")?;
    let spent = finite_micro(o, "spent")?;
    Ok(json!({ "remaining": remaining(total, spent), "normalized": normalized(total, spent) }))
}

fn spend_op(o: &Map<String, Value>) -> Res<Value> {
    let spent = finite_micro(o, "spent")?;
    let cost = finite_micro(o, "cost")?;
    Ok(json!({ "spent": spend(spent, cost) }))
}

/// A dimension list from `shapes`, of the rank the policy gives that tensor.
fn dims_of(shapes: &Map<String, Value>, name: &str, rank: usize) -> Res<Vec<u64>> {
    let arr = field(shapes, name)?.as_array().ok_or_else(|| format!("shape of {name} must be a list"))?;
    if arr.len() != rank {
        return Err(format!("{name} must have rank {rank}"));
    }
    let mut out = Vec::new();
    for d in arr {
        let d = d.as_u64().ok_or_else(|| format!("dimension of {name} must be a non-negative integer"))?;
        if d > MAX_DIM {
            return Err(format!("dimension of {name} exceeds {MAX_DIM}"));
        }
        out.push(d);
    }
    Ok(out)
}

fn index_coord(v: &Value) -> i64 {
    // A coordinate that isn't an integer can't be in bounds; -1 says so. An
    // integer past i64 is past every bound too, and saturating keeps it so.
    match v {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i
            } else if n.as_u64().is_some() {
                i64::MAX
            } else {
                -1
            }
        }
        _ => -1,
    }
}

fn guard(o: &Map<String, Value>) -> Res<Value> {
    let shapes = field(o, "shapes")?.as_object().ok_or("shapes must be an object")?;
    let q = dims_of(shapes, "q_weights", 2)?;
    let sy = dims_of(shapes, "synergy", 2)?;
    let t = dims_of(shapes, "terminate_weights", 1)?;
    if sy[0] != sy[1] {
        return Err("synergy must be square".into());
    }
    let s = Shapes { q_rows: q[0], q_cols: q[1], s_n: sy[0], t_len: t[0] };
    let lens = [q[0] * q[1], sy[0] * sy[1], t[0]];

    let current = field(o, "current")?.as_object().ok_or("current must be an object")?;
    let mut cur: Vec<i64> = Vec::new();
    for (name, len) in TENSORS.iter().zip(lens) {
        let flat = field(current, name)?.as_array().ok_or_else(|| format!("current {name} must be a list"))?;
        if flat.len() as u64 != len {
            return Err(format!("current {name} does not match its shape"));
        }
        for b in flat {
            let bits = b.as_u64().ok_or("current values must be u64 float bits")?;
            cur.push(quantize(f64::from_bits(bits)));
        }
    }

    let lim_o = field(o, "limits")?.as_object().ok_or("limits must be an object")?;
    let lim = Limits {
        max_abs: finite_micro(lim_o, "max_abs_value")?,
        max_delta: finite_micro(lim_o, "max_abs_delta")?,
        max_fraction: finite_micro(lim_o, "max_changed_fraction")?,
    };
    if lim.max_abs < 0 || lim.max_delta < 0 || !(0..=MICRO).contains(&lim.max_fraction) {
        return Err("limits out of range".into());
    }

    let mut edits: Vec<Edit> = Vec::new();
    for e in field(o, "edits")?.as_array().ok_or("edits must be a list")? {
        let e = e.as_object().ok_or("an edit must be an object")?;
        let tensor = match e.get("tensor").and_then(Value::as_str) {
            Some("q_weights") => Q_WEIGHTS,
            Some("synergy") => SYNERGY,
            Some("terminate_weights") => TERMINATE_WEIGHTS,
            _ => 3,
        };
        let index = match e.get("index") {
            Some(Value::Array(a)) => a.iter().map(index_coord).collect(),
            _ => Vec::new(),
        };
        let value = quantize(f64::from_bits(field(e, "value")?.as_u64().ok_or("edit value must be u64 float bits")?));
        edits.push(Edit { tensor, index, value });
    }

    let v = check_patch(&s, &cur, &lim, &edits);
    if v.rule != 0 {
        return Ok(json!({ "rule": format!("R{}", v.rule), "edit": v.edit, "applied": Value::Null }));
    }
    let out = apply_patch(&s, &cur, &lim, &edits);
    let mut applied = Map::new();
    let mut at = 0usize;
    for (name, len) in TENSORS.iter().zip(lens) {
        let part: Vec<Value> = out[at..at + len as usize].iter().map(|&x| micro_out(x)).collect();
        applied.insert((*name).to_string(), Value::Array(part));
        at += len as usize;
    }
    Ok(json!({ "rule": Value::Null, "edit": Value::Null, "applied": applied }))
}

fn handle(line: &str) -> Value {
    let req: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => return json!({ "error": format!("bad json: {e}") }),
    };
    let Some(o) = req.as_object() else {
        return json!({ "error": "request must be an object" });
    };
    let res = match o.get("op").and_then(Value::as_str) {
        Some("gate") => gate(o),
        Some("budget") => budget(o),
        Some("spend") => spend_op(o),
        Some("guard") => guard(o),
        Some("quantize") => float(o, "x").map(|x| json!({ "q": micro_out(quantize(x)) })),
        Some("version") => Ok(json!({ "kernel": env!("CARGO_PKG_VERSION") })),
        _ => Err("unknown op".into()),
    };
    res.unwrap_or_else(|e| json!({ "error": e }))
}

fn main() {
    let stdin = io::stdin();
    let mut out = io::stdout().lock();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let reply = handle(&line);
        if writeln!(out, "{reply}").and_then(|_| out.flush()).is_err() {
            break;
        }
    }
}
