//! Profiling binary for recall() at 100K / 1M scale.
//!
//! Run: cargo run --example profile_recall --release --features profiling
//!
//! Environment variables:
//!   PROFILE_N=100000       Number of memories (default: 100000)
//!   PROFILE_DIM=384        Embedding dimension (default: 384)
//!   PROFILE_ITERS=10       Recall iterations (default: 10)
//!   PROFILE_TOP_K=10       top_k for recall (default: 10)
//!   PROFILE_GRAPH=1        Enable graph expansion (default: 0)

#[cfg(feature = "profiling")]
fn main() {
    use std::time::Instant;
    use aidb_core::bench_utils::{seed_db_scaled, query_embedding};
    use aidb_core::{AIDB, RecallTimings};

    let n: usize = std::env::var("PROFILE_N")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let dim: usize = std::env::var("PROFILE_DIM")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(384);
    let iters: usize = std::env::var("PROFILE_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let top_k: usize = std::env::var("PROFILE_TOP_K")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let with_graph: bool = std::env::var("PROFILE_GRAPH")
        .ok().and_then(|s| s.parse::<u8>().ok()).unwrap_or(0) != 0;

    println!("=== AIDB Recall Profiling ===");
    println!("N={n}, dim={dim}, iters={iters}, top_k={top_k}, graph={with_graph}");
    println!();

    // Estimate memory usage
    let emb_bytes = n * dim * 4;
    println!("Estimated embedding storage: {:.1} MB", emb_bytes as f64 / 1_048_576.0);
    println!();

    // Seed database
    println!("Seeding database...");
    let t_seed = Instant::now();
    let db = AIDB::new(":memory:", dim).unwrap();
    seed_db_scaled(&db, n, dim, with_graph);
    let seed_secs = t_seed.elapsed().as_secs_f64();
    println!("  Seeded {n} memories in {seed_secs:.1}s ({:.0} records/sec)",
        n as f64 / seed_secs);
    println!();

    // Run profiled recall iterations
    let query = query_embedding(dim);
    let query_text = if with_graph {
        Some("Memory about Entity_5 involving Entity_10")
    } else {
        None
    };

    let mut all_timings: Vec<RecallTimings> = Vec::with_capacity(iters);

    for i in 0..iters {
        let result = db.recall_profiled(
            &query, top_k, None, None,
            false, with_graph, query_text, false,
        ).unwrap();
        println!(
            "  iter {:>2}: total={:>8.2}ms  vec={:>8.2}ms  cache_score={:>6.2}ms  fetch={:>6.2}ms  graph={:>6.2}ms  reinforce={:>6.2}ms  candidates={}  results={}",
            i,
            result.timings.total_ms,
            result.timings.vec_search_ms,
            result.timings.cache_score_ms,
            result.timings.fetch_ms,
            result.timings.graph_ms,
            result.timings.reinforce_ms,
            result.timings.candidate_count,
            result.results.len(),
        );
        all_timings.push(result.timings);
    }

    // Aggregate statistics
    println!();
    println!("=== Timing Summary (ms) ===");
    println!("{:<20} {:>8} {:>8} {:>8} {:>8}",
        "Phase", "Mean", "Median", "Min", "Max");
    println!("{}", "-".repeat(60));

    let phases: Vec<(&str, Box<dyn Fn(&RecallTimings) -> f64>)> = vec![
        ("vec_search", Box::new(|t: &RecallTimings| t.vec_search_ms)),
        ("cache_score", Box::new(|t: &RecallTimings| t.cache_score_ms)),
        ("fetch_topk", Box::new(|t: &RecallTimings| t.fetch_ms)),
        ("graph", Box::new(|t: &RecallTimings| t.graph_ms)),
        ("reinforce", Box::new(|t: &RecallTimings| t.reinforce_ms)),
        ("sort_truncate", Box::new(|t: &RecallTimings| t.sort_truncate_ms)),
        ("TOTAL", Box::new(|t: &RecallTimings| t.total_ms)),
    ];

    for (name, extractor) in &phases {
        let mut vals: Vec<f64> = all_timings.iter().map(|t| extractor(t)).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let median = vals[vals.len() / 2];
        let min = vals[0];
        let max = vals[vals.len() - 1];
        println!("{:<20} {:>8.2} {:>8.2} {:>8.2} {:>8.2}", name, mean, median, min, max);
    }

    // Percentage breakdown
    println!();
    println!("=== Phase Breakdown (% of total) ===");
    let mean_total = all_timings.iter().map(|t| t.total_ms).sum::<f64>() / iters as f64;

    let breakdown_phases: Vec<(&str, Box<dyn Fn(&RecallTimings) -> f64>)> = vec![
        ("vec_search", Box::new(|t: &RecallTimings| t.vec_search_ms)),
        ("cache_score", Box::new(|t: &RecallTimings| t.cache_score_ms)),
        ("fetch_topk", Box::new(|t: &RecallTimings| t.fetch_ms)),
        ("graph", Box::new(|t: &RecallTimings| t.graph_ms)),
        ("reinforce", Box::new(|t: &RecallTimings| t.reinforce_ms)),
        ("sort_truncate", Box::new(|t: &RecallTimings| t.sort_truncate_ms)),
    ];

    for (name, extractor) in &breakdown_phases {
        let mean_phase = all_timings.iter().map(|t| extractor(t)).sum::<f64>() / iters as f64;
        let pct = if mean_total > 0.0 { mean_phase / mean_total * 100.0 } else { 0.0 };
        let bar = "#".repeat((pct / 2.0) as usize);
        println!("  {:<16} {:>5.1}%  {}", name, pct, bar);
    }

    // Stats summary
    println!();
    let stats = db.stats().unwrap();
    println!("=== Database Stats ===");
    println!("  Active memories: {}", stats.active_memories);
    println!("  Entities: {}", stats.entities);
    println!("  Edges: {}", stats.edges);
    println!("  Operations: {}", stats.operations);
    println!("  Scoring cache entries: {}", stats.scoring_cache_entries);
    println!("  Vec index entries: {}", stats.vec_index_entries);
}

#[cfg(not(feature = "profiling"))]
fn main() {
    eprintln!("ERROR: This binary requires the 'profiling' feature.");
    eprintln!("Run with: cargo run --example profile_recall --release --features profiling");
    std::process::exit(1);
}
