//! Command parsing and execution.
//!
//! Two command types:
//! - Memory commands (natural): remember, recall, forget, relate
//! - Meta commands (backslash): \stats, \conflicts, \think, \dt, \q, \h

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use crate::client::Client;
use crate::format;

pub fn execute(client: &Client, line: &str) -> Result<()> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(());
    }

    if let Some(meta) = line.strip_prefix('\\') {
        return execute_meta(client, meta);
    }

    execute_natural(client, line)
}

// ── Meta commands ──────────────────────────────────────────

fn execute_meta(client: &Client, line: &str) -> Result<()> {
    let mut parts = line.split_whitespace();
    let cmd = parts.next().unwrap_or("");
    let rest: Vec<&str> = parts.collect();

    match cmd {
        "q" | "quit" | "exit" => std::process::exit(0),
        "h" | "?" | "help" => {
            print_help();
            Ok(())
        }
        "stats" | "s" => {
            let stats = client.get("/v1/stats")?;
            format::print_stats(&stats);
            Ok(())
        }
        "health" => {
            let health = client.get("/v1/health")?;
            format::print_json(&health);
            Ok(())
        }
        "dt" | "databases" | "l" => {
            let dbs = client.get("/v1/databases")?;
            format::print_databases(&dbs);
            Ok(())
        }
        "conflicts" | "c" => {
            let conflicts = client.get("/v1/conflicts")?;
            format::print_conflicts(&conflicts);
            Ok(())
        }
        "personality" | "p" => {
            let personality = client.get("/v1/personality")?;
            format::print_personality(&personality);
            Ok(())
        }
        "think" | "t" => {
            let result = client.post("/v1/think", &json!({}))?;
            format::print_think_result(&result);
            Ok(())
        }
        "json" => {
            // \json <path> — raw GET
            if let Some(path) = rest.first() {
                let resp = client.get(path)?;
                format::print_json(&resp);
            } else {
                format::print_error("usage: \\json <path>");
            }
            Ok(())
        }
        other => {
            format::print_error(&format!("unknown meta command: \\{}", other));
            println!("type {} for help", "\\h".cyan());
            Ok(())
        }
    }
}

// ── Natural commands ──────────────────────────────────────

fn execute_natural(client: &Client, line: &str) -> Result<()> {
    let tokens = shell_words::split(line).map_err(|e| anyhow!("parse error: {}", e))?;
    if tokens.is_empty() {
        return Ok(());
    }

    let cmd = tokens[0].to_lowercase();
    let args = &tokens[1..];

    match cmd.as_str() {
        "remember" | "r" => cmd_remember(client, args),
        "recall" | "?" => cmd_recall(client, args),
        "forget" | "f" => cmd_forget(client, args),
        "relate" => cmd_relate(client, args),
        _ => {
            format::print_error(&format!("unknown command: {}", cmd));
            println!("type {} for help", "\\h".cyan());
            Ok(())
        }
    }
}

fn parse_kv_args(args: &[String]) -> (Vec<&String>, std::collections::HashMap<String, String>) {
    let mut positional = Vec::new();
    let mut kv = std::collections::HashMap::new();
    for arg in args {
        if let Some((k, v)) = arg.split_once('=') {
            kv.insert(k.to_string(), v.to_string());
        } else {
            positional.push(arg);
        }
    }
    (positional, kv)
}

fn cmd_remember(client: &Client, args: &[String]) -> Result<()> {
    if args.is_empty() {
        format::print_error("usage: remember \"text\" [importance=0.9] [domain=work]");
        return Ok(());
    }

    let (positional, kv) = parse_kv_args(args);
    let text = positional.first().ok_or_else(|| anyhow!("text required"))?;

    let mut body = json!({
        "text": text,
    });

    for (key, val) in &kv {
        match key.as_str() {
            "importance" | "valence" | "half_life" | "certainty" => {
                if let Ok(n) = val.parse::<f64>() {
                    body[key] = json!(n);
                }
            }
            _ => {
                body[key] = json!(val);
            }
        }
    }

    let resp = client.post("/v1/remember", &body)?;
    let rid = resp.get("rid").and_then(|v| v.as_str()).unwrap_or("?");
    format::print_success(&format!("stored: {}", rid));
    Ok(())
}

fn cmd_recall(client: &Client, args: &[String]) -> Result<()> {
    if args.is_empty() {
        format::print_error("usage: recall <query> [top=10] [domain=work]");
        return Ok(());
    }

    let (positional, kv) = parse_kv_args(args);
    let query = positional
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    let mut body = json!({ "query": query });

    if let Some(top) = kv.get("top").or_else(|| kv.get("top_k")) {
        if let Ok(n) = top.parse::<u64>() {
            body["top_k"] = json!(n);
        }
    }
    for key in ["domain", "source", "namespace", "memory_type"] {
        if let Some(v) = kv.get(key) {
            body[key] = json!(v);
        }
    }

    let resp = client.post("/v1/recall", &body)?;
    format::print_recall_results(&resp);
    Ok(())
}

fn cmd_forget(client: &Client, args: &[String]) -> Result<()> {
    let rid = args.first().ok_or_else(|| anyhow!("usage: forget <rid>"))?;
    let resp = client.post("/v1/forget", &json!({ "rid": rid }))?;
    let found = resp.get("found").and_then(|v| v.as_bool()).unwrap_or(false);
    if found {
        format::print_success(&format!("forgot: {}", rid));
    } else {
        format::print_error(&format!("not found: {}", rid));
    }
    Ok(())
}

fn cmd_relate(client: &Client, args: &[String]) -> Result<()> {
    // Syntax: relate <entity> -> <target> as <relationship> [weight=1.0]
    // Or: relate entity=Alice target=Acme rel=works_at
    let (positional, kv) = parse_kv_args(args);

    let (entity, target, relationship) = if !kv.is_empty() {
        let entity = kv
            .get("entity")
            .ok_or_else(|| anyhow!("entity= required"))?;
        let target = kv
            .get("target")
            .ok_or_else(|| anyhow!("target= required"))?;
        let rel = kv
            .get("rel")
            .or_else(|| kv.get("relationship"))
            .ok_or_else(|| anyhow!("rel= required"))?;
        (entity.clone(), target.clone(), rel.clone())
    } else {
        // Try arrow syntax: entity -> target as rel_type
        let joined = positional
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let (left, right) = joined
            .split_once("->")
            .ok_or_else(|| anyhow!("usage: relate <entity> -> <target> as <relationship>"))?;
        let (target_part, rel_part) = right
            .split_once(" as ")
            .ok_or_else(|| anyhow!("missing 'as <relationship>'"))?;
        (
            left.trim().to_string(),
            target_part.trim().to_string(),
            rel_part.trim().to_string(),
        )
    };

    let mut body = json!({
        "entity": entity,
        "target": target,
        "relationship": relationship,
    });
    if let Some(w) = kv.get("weight").and_then(|v| v.parse::<f64>().ok()) {
        body["weight"] = json!(w);
    }

    let resp = client.post("/v1/relate", &body)?;
    let edge_id = resp.get("edge_id").and_then(|v| v.as_str()).unwrap_or("?");
    format::print_success(&format!(
        "edge: {} ({} -[{}]-> {})",
        edge_id, entity, relationship, target
    ));
    Ok(())
}

// ── Help ──────────────────────────────────────────────────

fn print_help() {
    use colored::Colorize;
    println!();
    println!("{}", "Memory commands:".bold());
    println!("  remember \"<text>\" [importance=0.9] [domain=work]   store a memory");
    println!("  recall <query> [top=10] [domain=work]              semantic search");
    println!("  forget <rid>                                       tombstone a memory");
    println!("  relate <entity> -> <target> as <relationship>      create graph edge");
    println!();
    println!("{}", "Meta commands:".bold());
    println!("  \\stats     \\s    engine statistics");
    println!("  \\dt        \\l    list databases");
    println!("  \\conflicts \\c    list open conflicts");
    println!("  \\personality \\p  derived personality traits");
    println!("  \\think     \\t    run consolidation + conflict scan");
    println!("  \\health          server health");
    println!("  \\json <path>     raw GET request");
    println!("  \\help      \\h    this help");
    println!("  \\quit      \\q    exit");
    println!();
}

// Re-export for main.rs use
use colored::Colorize;
