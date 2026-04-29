//! Output formatting — tables, colored text, truncation.

use colored::Colorize;
use comfy_table::{Attribute, Cell, Color, ContentArrangement, Table};
use serde_json::Value;

const MAX_TEXT_WIDTH: usize = 60;

pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max - 3).collect();
        format!("{}...", truncated)
    }
}

pub fn print_recall_results(value: &Value) {
    let results = match value.get("results").and_then(|v| v.as_array()) {
        Some(r) => r,
        None => {
            print_json(value);
            return;
        }
    };

    if results.is_empty() {
        println!("{}", "(no results)".dimmed());
        return;
    }

    let mut table = Table::new();
    table
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["#", "score", "text", "domain", "why"]);

    for (i, r) in results.iter().enumerate() {
        let score = r.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let text = r.get("text").and_then(|v| v.as_str()).unwrap_or("");
        let domain = r.get("domain").and_then(|v| v.as_str()).unwrap_or("");
        let why = r
            .get("why_retrieved")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();

        table.add_row(vec![
            Cell::new(i + 1),
            Cell::new(format!("{:.2}", score)),
            Cell::new(truncate(text, MAX_TEXT_WIDTH)),
            Cell::new(domain),
            Cell::new(truncate(&why, 30)),
        ]);
    }

    println!("{table}");
    let total = value
        .get("total")
        .and_then(|v| v.as_u64())
        .unwrap_or(results.len() as u64);
    println!("{}", format!("({} rows)", total).dimmed());
}

pub fn print_stats(value: &Value) {
    let mut table = Table::new();
    table
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["metric", "value"]);

    let fields = [
        ("active_memories", "Active memories"),
        ("consolidated_memories", "Consolidated"),
        ("tombstoned_memories", "Tombstoned"),
        ("edges", "Graph edges"),
        ("entities", "Entities"),
        ("operations", "Operations"),
        ("open_conflicts", "Open conflicts"),
        ("pending_triggers", "Pending triggers"),
    ];

    for (key, label) in fields {
        let v = value
            .get(key)
            .map(|v| v.to_string())
            .unwrap_or_else(|| "-".to_string());
        table.add_row(vec![Cell::new(label), Cell::new(v)]);
    }

    println!("{table}");
}

pub fn print_databases(value: &Value) {
    let dbs = match value.get("databases").and_then(|v| v.as_array()) {
        Some(d) => d,
        None => {
            print_json(value);
            return;
        }
    };

    if dbs.is_empty() {
        println!("{}", "(no databases)".dimmed());
        return;
    }

    let mut table = Table::new();
    table
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["id", "name", "created"]);

    for db in dbs {
        let id = db.get("id").map(|v| v.to_string()).unwrap_or_default();
        let name = db.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let created = db.get("created_at").and_then(|v| v.as_str()).unwrap_or("");
        table.add_row(vec![Cell::new(id), Cell::new(name), Cell::new(created)]);
    }

    println!("{table}");
}

pub fn print_conflicts(value: &Value) {
    let conflicts = match value.get("conflicts").and_then(|v| v.as_array()) {
        Some(c) => c,
        None => {
            print_json(value);
            return;
        }
    };

    if conflicts.is_empty() {
        println!("{}", "(no conflicts)".dimmed());
        return;
    }

    let mut table = Table::new();
    table
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["id", "type", "priority", "entity", "reason"]);

    for c in conflicts {
        let id = c.get("conflict_id").and_then(|v| v.as_str()).unwrap_or("");
        let ctype = c
            .get("conflict_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let prio = c.get("priority").and_then(|v| v.as_str()).unwrap_or("");
        let entity = c.get("entity").and_then(|v| v.as_str()).unwrap_or("-");
        let reason = c
            .get("detection_reason")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        table.add_row(vec![
            Cell::new(truncate(id, 20)),
            Cell::new(ctype),
            Cell::new(prio),
            Cell::new(entity),
            Cell::new(truncate(reason, 40)),
        ]);
    }

    println!("{table}");
}

pub fn print_personality(value: &Value) {
    let traits = match value.get("traits").and_then(|v| v.as_array()) {
        Some(t) => t,
        None => {
            print_json(value);
            return;
        }
    };

    if traits.is_empty() {
        println!("{}", "(no traits derived yet — try \\think first)".dimmed());
        return;
    }

    let mut table = Table::new();
    table
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec!["trait", "score"]);

    for t in traits {
        let name = t.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let score = t.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        table.add_row(vec![Cell::new(name), Cell::new(format!("{:.3}", score))]);
    }

    println!("{table}");
}

pub fn print_think_result(value: &Value) {
    let consolidation = value
        .get("consolidation_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let conflicts = value
        .get("conflicts_found")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let duration = value
        .get("duration_ms")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let triggers = value
        .get("triggers")
        .and_then(|v| v.as_array())
        .map(|a| a.len())
        .unwrap_or(0);

    println!(
        "{} consolidated={} conflicts={} triggers={} ({}ms)",
        "thought:".green().bold(),
        consolidation,
        conflicts,
        triggers,
        duration
    );
}

pub fn print_cluster(value: &Value) {
    if value.get("clustered").and_then(|v| v.as_bool()) != Some(true) {
        println!("{}", "single-node mode (no replication)".dimmed());
        return;
    }

    let role = value.get("role").and_then(|v| v.as_str()).unwrap_or("?");
    let term = value
        .get("current_term")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let leader = value
        .get("leader_id")
        .and_then(|v| v.as_u64())
        .map(|n| n.to_string())
        .unwrap_or_else(|| "(none)".into());
    let node_id = value.get("node_id").and_then(|v| v.as_u64()).unwrap_or(0);
    let healthy = value
        .get("healthy")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let writes = value
        .get("accepts_writes")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let quorum = value
        .get("quorum_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let role_colored = match role {
        "Leader" => role.green().bold().to_string(),
        "Follower" => role.cyan().to_string(),
        "Candidate" => role.yellow().to_string(),
        "ReadOnly" => role.blue().to_string(),
        _ => role.dimmed().to_string(),
    };

    println!();
    println!("  {} #{} — {}", "node".bold(), node_id, role_colored);
    println!("  {}: {}", "term".bold(), term);
    println!("  {}: {}", "leader".bold(), leader);
    println!(
        "  {}: {} | {}: {}",
        "healthy".bold(),
        if healthy {
            "yes".green().to_string()
        } else {
            "no".red().to_string()
        },
        "writable".bold(),
        if writes {
            "yes".green().to_string()
        } else {
            "no".red().to_string()
        },
    );
    println!("  {}: {}", "quorum".bold(), quorum);
    println!();

    if let Some(peers) = value.get("peers").and_then(|v| v.as_array()) {
        if peers.is_empty() {
            return;
        }
        let mut table = Table::new();
        table
            .set_content_arrangement(ContentArrangement::Dynamic)
            .set_header(vec![
                "node_id",
                "addr",
                "role",
                "reachable",
                "term",
                "last_seen",
            ]);

        for p in peers {
            let nid = p
                .get("node_id")
                .and_then(|v| v.as_u64())
                .map(|n| n.to_string())
                .unwrap_or_else(|| "?".into());
            let addr = p.get("addr").and_then(|v| v.as_str()).unwrap_or("");
            let prole = p.get("role").and_then(|v| v.as_str()).unwrap_or("");
            let reach = p
                .get("reachable")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let pterm = p.get("current_term").and_then(|v| v.as_u64()).unwrap_or(0);
            let last_seen = p
                .get("last_seen_secs_ago")
                .and_then(|v| v.as_f64())
                .map(|s| format!("{:.1}s ago", s))
                .unwrap_or_else(|| "never".into());
            table.add_row(vec![
                Cell::new(nid),
                Cell::new(addr),
                Cell::new(prole),
                Cell::new(if reach { "✓" } else { "✗" }),
                Cell::new(pterm),
                Cell::new(last_seen),
            ]);
        }
        println!("{table}");
    }
}

/// Render the RFC 009 admission section of `/v1/health/deep` as a table.
/// Health responses without an `admission` block (older servers) print
/// a friendly note instead of a noisy error.
pub fn print_admission(value: &Value) {
    let admission = match value.get("admission") {
        Some(a) => a,
        None => {
            println!(
                "{} server response has no `admission` block — likely a pre-RFC-009 build",
                "!".yellow()
            );
            return;
        }
    };

    let mut table = Table::new();
    table.set_header(vec![
        Cell::new("Field").add_attribute(Attribute::Bold),
        Cell::new("Value").add_attribute(Attribute::Bold),
    ]);
    table.add_row(vec![
        Cell::new("hard top_k cap"),
        Cell::new(admission["hard_top_k_cap"].as_u64().unwrap_or(0)),
    ]);
    table.add_row(vec![
        Cell::new("max request body bytes"),
        Cell::new(admission["max_request_body_bytes"].as_u64().unwrap_or(0)),
    ]);

    let in_flight_max = admission["in_flight_recall"]["max"].as_u64().unwrap_or(0);
    let in_flight_used = admission["in_flight_recall"]["in_use"]
        .as_u64()
        .unwrap_or(0);
    let in_flight_pct = if in_flight_max > 0 {
        100 * in_flight_used / in_flight_max
    } else {
        0
    };
    let in_flight_cell = format!("{}/{} ({}%)", in_flight_used, in_flight_max, in_flight_pct);
    table.add_row(vec![
        Cell::new("in-flight recalls"),
        cell_with_pressure(&in_flight_cell, in_flight_pct),
    ]);

    let expanded_max = admission["expanded_recall"]["max"].as_u64().unwrap_or(0);
    let expanded_used = admission["expanded_recall"]["in_use"].as_u64().unwrap_or(0);
    let expanded_pct = if expanded_max > 0 {
        100 * expanded_used / expanded_max
    } else {
        0
    };
    let expanded_cell = format!("{}/{} ({}%)", expanded_used, expanded_max, expanded_pct);
    table.add_row(vec![
        Cell::new("expanded concurrent"),
        cell_with_pressure(&expanded_cell, expanded_pct),
    ]);

    if let Some(rt) = value.get("runtime") {
        let isolated = rt["control_runtime_isolated"].as_bool().unwrap_or(false);
        table.add_row(vec![
            Cell::new("control runtime isolated"),
            if isolated {
                Cell::new("yes").fg(Color::Green)
            } else {
                Cell::new("no").fg(Color::Yellow)
            },
        ]);
    }

    println!("{table}");
    println!();
    println!(
        "{} term changes, scheduling latency p99, and rejection counts are at /metrics",
        "tip:".dimmed()
    );
}

/// Render the RFC 017-A version section of `/v1/health/deep` as a table.
/// Surfaces local + cluster wire version + per-table schema versions
/// for rolling-upgrade visibility.
pub fn print_version(value: &Value) {
    let version = match value.get("version") {
        Some(v) => v,
        None => {
            println!(
                "{} server response has no `version` block — likely a pre-RFC-017 build",
                "!".yellow()
            );
            return;
        }
    };

    let mut table = Table::new();
    table.set_header(vec![
        Cell::new("Field").add_attribute(Attribute::Bold),
        Cell::new("Value").add_attribute(Attribute::Bold),
    ]);
    table.add_row(vec![
        Cell::new("binary build id"),
        Cell::new(version["build_id"].as_str().unwrap_or("?")),
    ]);
    table.add_row(vec![
        Cell::new("local wire version"),
        Cell::new(format!(
            "{}.{}",
            version["wire"]["major"].as_u64().unwrap_or(0),
            version["wire"]["minor"].as_u64().unwrap_or(0),
        )),
    ]);
    table.add_row(vec![
        Cell::new("min supported wire"),
        Cell::new(format!(
            "{}.{}",
            version["min_supported_wire"]["major"].as_u64().unwrap_or(0),
            version["min_supported_wire"]["minor"].as_u64().unwrap_or(0),
        )),
    ]);
    if let Some(cluster) = version.get("cluster") {
        let min_str = format!(
            "{}.{}",
            cluster["min"]["major"].as_u64().unwrap_or(0),
            cluster["min"]["minor"].as_u64().unwrap_or(0),
        );
        let max_str = format!(
            "{}.{}",
            cluster["max"]["major"].as_u64().unwrap_or(0),
            cluster["max"]["minor"].as_u64().unwrap_or(0),
        );
        table.add_row(vec![Cell::new("cluster min wire"), Cell::new(min_str)]);
        table.add_row(vec![Cell::new("cluster max wire"), Cell::new(max_str)]);
        table.add_row(vec![
            Cell::new("observed peers"),
            Cell::new(cluster["peer_count"].as_u64().unwrap_or(0)),
        ]);
    }
    println!("{table}");

    if let Some(tables) = version
        .get("table_schema_versions")
        .and_then(|t| t.as_array())
    {
        println!();
        let mut t2 = Table::new();
        t2.set_header(vec![
            Cell::new("Table").add_attribute(Attribute::Bold),
            Cell::new("Schema").add_attribute(Attribute::Bold),
        ]);
        for entry in tables {
            if let Some(arr) = entry.as_array() {
                if arr.len() == 2 {
                    let name = arr[0].as_str().unwrap_or("?");
                    let ver = arr[1].as_u64().unwrap_or(0);
                    t2.add_row(vec![Cell::new(name), Cell::new(format!("v{}", ver))]);
                }
            }
        }
        println!("{t2}");
    }
}

/// Render the RFC 010 PR-5 fault list. Empty list = "no faults active"
/// in green; populated = colored table.
pub fn print_faults(value: &Value) {
    let arr = match value.as_array() {
        Some(a) => a,
        None => {
            println!("{} unexpected response shape", "!".yellow());
            return;
        }
    };
    if arr.is_empty() {
        println!("{} no fault injections active", "✓".green());
        return;
    }
    let mut table = Table::new();
    table.set_header(vec![
        Cell::new("ID").add_attribute(Attribute::Bold),
        Cell::new("Kind").add_attribute(Attribute::Bold),
        Cell::new("TTL").add_attribute(Attribute::Bold),
    ]);
    for f in arr {
        let id = f["id"].as_u64().unwrap_or(0);
        let kind = f["kind"]["kind"].as_str().unwrap_or("?");
        let ttl_cell = match f["ttl_secs"].as_u64() {
            Some(n) => Cell::new(format!("{}s", n)).fg(Color::Yellow),
            None => Cell::new("persistent").fg(Color::Red),
        };
        table.add_row(vec![
            Cell::new(format!("fault_{}", id)),
            Cell::new(kind),
            ttl_cell,
        ]);
    }
    println!("{table}");
}

/// Render the RFC 019 jobs list as a colored table. Empty list = "no
/// jobs" in green.
pub fn print_jobs(value: &Value) {
    let arr = match value.as_array() {
        Some(a) => a,
        None => {
            println!("{} unexpected response shape", "!".yellow());
            return;
        }
    };
    if arr.is_empty() {
        println!("{} no jobs in queue", "✓".green());
        return;
    }
    let mut table = Table::new();
    table.set_header(vec![
        Cell::new("ID").add_attribute(Attribute::Bold),
        Cell::new("Tenant").add_attribute(Attribute::Bold),
        Cell::new("State").add_attribute(Attribute::Bold),
        Cell::new("Pri").add_attribute(Attribute::Bold),
        Cell::new("Kind").add_attribute(Attribute::Bold),
    ]);
    for j in arr {
        let id = j["id"].as_str().unwrap_or("?");
        let id_short: String = id.chars().take(8).collect();
        let tenant = j["tenant_id"].as_i64().unwrap_or(0);
        let state = j["state"].as_str().unwrap_or("?");
        let pri = j["priority"].as_u64().unwrap_or(0);
        let kind = j["kind"].as_str().unwrap_or("?");
        let state_cell = match state {
            "Pending" => Cell::new(state).fg(Color::Yellow),
            "Leased" => Cell::new(state).fg(Color::Cyan),
            "Succeeded" => Cell::new(state).fg(Color::Green),
            "Failed" => Cell::new(state).fg(Color::Red),
            "Cancelled" => Cell::new(state).fg(Color::DarkGrey),
            other => Cell::new(other),
        };
        table.add_row(vec![
            Cell::new(id_short),
            Cell::new(tenant),
            state_cell,
            Cell::new(pri),
            Cell::new(kind),
        ]);
    }
    println!("{table}");
}

/// Render the RFC 017-B migrations list as a colored table per-DB.
pub fn print_migrations(value: &Value) {
    let obj = match value.as_object() {
        Some(o) => o,
        None => {
            println!("{} unexpected response shape", "!".yellow());
            return;
        }
    };
    for (db, applied) in obj {
        println!("\n[{}]", db.bold());
        if let Some(err) = applied["error"].as_str() {
            println!("  {} {}", "✗".red(), err);
            continue;
        }
        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("ID").add_attribute(Attribute::Bold),
            Cell::new("Migration").add_attribute(Attribute::Bold),
        ]);
        if let Some(arr) = applied.as_array() {
            for m in arr {
                let id = m["id"].as_u64().unwrap_or(0);
                let name = m["name"].as_str().unwrap_or("?");
                table.add_row(vec![
                    Cell::new(format!("m{:03}", id)),
                    Cell::new(name).fg(Color::Green),
                ]);
            }
        }
        println!("{table}");
    }
}

/// RFC 010 PR-4 — pretty-print `/v1/cluster/raft` JSON.
pub fn print_raft_status(value: &Value) {
    let obj = match value.as_object() {
        Some(o) => o,
        None => {
            println!("{} unexpected response shape", "!".yellow());
            return;
        }
    };
    let node_id = obj.get("node_id").and_then(|v| v.as_u64()).unwrap_or(0);
    let state = obj
        .get("state")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let term = obj
        .get("current_term")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let leader = obj.get("current_leader").and_then(|v| v.as_u64());
    let last_log = obj.get("last_log_index").and_then(|v| v.as_u64());
    let last_applied = obj.get("last_applied_index").and_then(|v| v.as_u64());
    let snapshot = obj.get("snapshot_index").and_then(|v| v.as_u64());
    let healthy = obj
        .get("healthy")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let lag = obj.get("millis_since_quorum_ack").and_then(|v| v.as_u64());

    println!("\n{}", "openraft cluster".bold());
    let state_colored = match state {
        "leader" => state.green(),
        "follower" => state.cyan(),
        "candidate" => state.yellow(),
        _ => state.normal(),
    };
    println!("  this node    : node-{}  ({})", node_id, state_colored);
    match leader {
        Some(id) if id == node_id => println!("  leader       : {} (node-{})", "SELF".green(), id),
        Some(id) => println!("  leader       : node-{}", id),
        None => println!(
            "  leader       : {}",
            "(none — election in progress)".yellow()
        ),
    }
    println!("  current term : {}", term);
    println!(
        "  last log     : {}",
        last_log
            .map(|n| n.to_string())
            .unwrap_or_else(|| "(none)".into())
    );
    println!(
        "  last applied : {}",
        last_applied
            .map(|n| n.to_string())
            .unwrap_or_else(|| "(none)".into())
    );
    println!(
        "  snapshot @   : {}",
        snapshot
            .map(|n| n.to_string())
            .unwrap_or_else(|| "(no snapshot)".into())
    );
    if let Some(l) = lag {
        let lag_colored = if l > 5_000 {
            format!("{} ms ago", l).red()
        } else if l > 1_000 {
            format!("{} ms ago", l).yellow()
        } else {
            format!("{} ms ago", l).green()
        };
        println!("  quorum ack   : {}", lag_colored);
    }
    let health_str = if healthy { "OK".green() } else { "FATAL".red() };
    println!("  health       : {}", health_str);

    // Members table.
    if let Some(members) = obj.get("members").and_then(|v| v.as_array()) {
        let mut table = Table::new();
        table.set_header(vec![
            Cell::new("Node").add_attribute(Attribute::Bold),
            Cell::new("Role").add_attribute(Attribute::Bold),
            Cell::new("Address").add_attribute(Attribute::Bold),
        ]);
        for m in members {
            let id = m.get("node_id").and_then(|v| v.as_u64()).unwrap_or(0);
            let is_voter = m.get("is_voter").and_then(|v| v.as_bool()).unwrap_or(false);
            let addr = m.get("addr").and_then(|v| v.as_str()).unwrap_or("?");
            let role = if is_voter {
                Cell::new("voter").fg(Color::Green)
            } else {
                Cell::new("learner").fg(Color::Cyan)
            };
            table.add_row(vec![
                Cell::new(format!("node-{}", id)),
                role,
                Cell::new(addr),
            ]);
        }
        println!("{table}");
    }
}

/// Color-code a usage cell green/yellow/red by saturation percentage.
fn cell_with_pressure(text: &str, pct: u64) -> Cell {
    let color = if pct >= 90 {
        Color::Red
    } else if pct >= 70 {
        Color::Yellow
    } else {
        Color::Green
    };
    Cell::new(text).fg(color)
}

pub fn print_success(msg: &str) {
    println!("{} {}", "✓".green(), msg);
}

pub fn print_error(msg: &str) {
    eprintln!("{} {}", "✗".red(), msg);
}

pub fn print_json(value: &Value) {
    if let Ok(s) = serde_json::to_string_pretty(value) {
        println!("{}", s);
    }
}
