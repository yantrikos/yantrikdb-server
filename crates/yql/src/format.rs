//! Output formatting — tables, colored text, truncation.

use colored::Colorize;
use comfy_table::{Cell, ContentArrangement, Table};
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
