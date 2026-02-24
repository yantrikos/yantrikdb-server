use std::time::Duration;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use aidb_core::{AIDB, RecordInput};
use aidb_core::bench_utils::{vec_seed_dim, seed_db_scaled, query_embedding};

fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..dim).map(|i| (seed + i as f32) * 0.1).collect();
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    raw.iter().map(|x| x / norm).collect()
}

fn seed_db(db: &AIDB, n: usize, dim: usize) {
    let meta = serde_json::json!({});
    for i in 0..n {
        let emb = vec_seed(i as f32 * 0.37, dim);
        db.record(
            &format!("Memory number {} about topic {}", i, i % 10),
            if i % 2 == 0 { "episodic" } else { "semantic" },
            0.3 + (i % 7) as f64 * 0.1,
            (i % 5) as f64 * 0.2 - 0.4,
            604800.0,
            &meta,
            &emb,
            "default",
        )
        .unwrap();
    }
}

fn bench_record(c: &mut Criterion) {
    let dim = 64;
    let db = AIDB::new(":memory:", dim).unwrap();
    let meta = serde_json::json!({});
    let mut i = 0u64;

    c.bench_function("record", |b| {
        b.iter(|| {
            let emb = vec_seed(i as f32 * 0.37 + 10000.0, dim);
            db.record(
                black_box(&format!("bench record {}", i)),
                "episodic",
                0.5,
                0.0,
                604800.0,
                &meta,
                &emb,
                "default",
            )
            .unwrap();
            i += 1;
        })
    });
}

fn bench_recall(c: &mut Criterion) {
    let dim = 64;
    let mut group = c.benchmark_group("recall");

    for &n in &[100, 500, 1000] {
        let db = AIDB::new(":memory:", dim).unwrap();
        seed_db(&db, n, dim);
        let query = vec_seed(999.0, dim);

        group.bench_with_input(BenchmarkId::new("top10", n), &n, |b, _| {
            b.iter(|| db.recall(black_box(&query), 10, None, None, false, false, None, false, None).unwrap())
        });
    }
    group.finish();
}

fn bench_get(c: &mut Criterion) {
    let dim = 64;
    let db = AIDB::new(":memory:", dim).unwrap();
    let meta = serde_json::json!({});
    let rid = db
        .record("lookup target", "episodic", 0.5, 0.0, 604800.0, &meta, &vec_seed(1.0, dim), "default")
        .unwrap();

    c.bench_function("get", |b| {
        b.iter(|| db.get(black_box(&rid)).unwrap())
    });
}

fn bench_stats(c: &mut Criterion) {
    let dim = 64;
    let db = AIDB::new(":memory:", dim).unwrap();
    seed_db(&db, 100, dim);

    c.bench_function("stats_100", |b| {
        b.iter(|| db.stats(None).unwrap())
    });
}

fn bench_relate(c: &mut Criterion) {
    let dim = 64;
    let db = AIDB::new(":memory:", dim).unwrap();
    seed_db(&db, 100, dim);
    let mut i = 0u64;

    c.bench_function("relate", |b| {
        b.iter(|| {
            db.relate(
                &format!("entity_{}", i),
                &format!("entity_{}", i + 1),
                "related_to",
                1.0,
            )
            .unwrap();
            i += 1;
        })
    });
}

fn bench_decay(c: &mut Criterion) {
    let dim = 64;
    let db = AIDB::new(":memory:", dim).unwrap();
    seed_db(&db, 100, dim);

    c.bench_function("decay_100", |b| {
        b.iter(|| db.decay(black_box(0.01)).unwrap())
    });
}

fn bench_bulk_insert(c: &mut Criterion) {
    let dim = 64;
    let meta = serde_json::json!({});

    c.bench_function("bulk_insert_500", |b| {
        b.iter(|| {
            let db = AIDB::new(":memory:", dim).unwrap();
            for i in 0..500 {
                let emb = vec_seed(i as f32 * 0.37, dim);
                db.record(
                    &format!("bulk {}", i),
                    "episodic",
                    0.5,
                    0.0,
                    604800.0,
                    &meta,
                    &emb,
                    "default",
                )
                .unwrap();
            }
        })
    });
}

fn bench_record_batch(c: &mut Criterion) {
    let dim = 64;

    c.bench_function("record_batch_500", |b| {
        b.iter(|| {
            let db = AIDB::new(":memory:", dim).unwrap();
            let inputs: Vec<RecordInput> = (0..500).map(|i| RecordInput {
                text: format!("batch memory {i}"),
                memory_type: "episodic".to_string(),
                importance: 0.5,
                valence: 0.0,
                half_life: 604800.0,
                metadata: serde_json::json!({}),
                embedding: vec_seed(i as f32 * 0.37, dim),
                namespace: "default".to_string(),
            }).collect();
            db.record_batch(black_box(&inputs)).unwrap();
        })
    });
}

fn bench_archive(c: &mut Criterion) {
    let dim = 64;

    c.bench_function("archive", |b| {
        b.iter_batched(
            || {
                // Setup: create a DB with one hot memory
                let db = AIDB::new(":memory:", dim).unwrap();
                let emb = vec_seed(42.0, dim);
                let rid = db.record("archive target", "episodic", 0.5, 0.0, 604800.0,
                    &serde_json::json!({}), &emb, "default").unwrap();
                (db, rid)
            },
            |(db, rid)| {
                db.archive(black_box(&rid)).unwrap();
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_hydrate(c: &mut Criterion) {
    let dim = 64;

    c.bench_function("hydrate", |b| {
        b.iter_batched(
            || {
                // Setup: create a DB with one archived memory
                let db = AIDB::new(":memory:", dim).unwrap();
                let emb = vec_seed(42.0, dim);
                let rid = db.record("hydrate target", "episodic", 0.5, 0.0, 604800.0,
                    &serde_json::json!({}), &emb, "default").unwrap();
                db.archive(&rid).unwrap();
                (db, rid)
            },
            |(db, rid)| {
                db.hydrate(black_box(&rid)).unwrap();
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

fn bench_evict(c: &mut Criterion) {
    let dim = 64;

    c.bench_function("evict_500_to_200", |b| {
        b.iter(|| {
            let db = AIDB::new(":memory:", dim).unwrap();
            seed_db(&db, 500, dim);
            db.evict(black_box(200)).unwrap();
        })
    });
}

fn bench_recall_scaled(c: &mut Criterion) {
    let dim = 64;
    let mut group = c.benchmark_group("recall_scaled");

    for &n in &[100, 1000, 5000] {
        let db = AIDB::new(":memory:", dim).unwrap();
        seed_db(&db, n, dim);
        let query = vec_seed(999.0, dim);

        group.bench_with_input(BenchmarkId::new("top10", n), &n, |b, _| {
            b.iter(|| db.recall(black_box(&query), 10, None, None, false, false, None, true, None).unwrap())
        });
    }
    group.finish();
}

// ── Scaled benchmarks (dim=384, larger DB sizes) ──

fn bench_recall_dim_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_dim_comparison");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    for &dim in &[64, 384] {
        for &n in &[1000, 10_000] {
            let db = AIDB::new(":memory:", dim).unwrap();
            seed_db_scaled(&db, n, dim, false);
            let query = query_embedding(dim);

            group.bench_with_input(
                BenchmarkId::new(format!("dim{}_top10", dim), n),
                &n,
                |b, _| {
                    b.iter(|| {
                        db.recall(black_box(&query), 10, None, None, false, false, None, true, None)
                            .unwrap()
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_recall_100k(c: &mut Criterion) {
    let dim = 384;
    let n = 100_000;
    let mut group = c.benchmark_group("recall_100k");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(60));
    group.warm_up_time(Duration::from_secs(5));

    eprintln!("Seeding 100K memories at dim=384 (this takes a while)...");
    let db = AIDB::new(":memory:", dim).unwrap();
    seed_db_scaled(&db, n, dim, false);
    let query = query_embedding(dim);
    eprintln!("Seeding complete.");

    for &top_k in &[10, 50] {
        group.bench_with_input(
            BenchmarkId::new("no_graph", top_k),
            &top_k,
            |b, &k| {
                b.iter(|| {
                    db.recall(black_box(&query), k, None, None, false, false, None, true, None)
                        .unwrap()
                })
            },
        );
    }
    group.finish();
}

fn bench_recall_with_graph(c: &mut Criterion) {
    let dim = 384;
    let mut group = c.benchmark_group("recall_with_graph");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for &n in &[1000, 10_000] {
        let db = AIDB::new(":memory:", dim).unwrap();
        seed_db_scaled(&db, n, dim, true);
        let query = query_embedding(dim);

        group.bench_with_input(
            BenchmarkId::new("graph_expand", n),
            &n,
            |b, _| {
                b.iter(|| {
                    db.recall(
                        black_box(&query), 10, None, None,
                        false, true,
                        Some("Memory about Entity_5 involving Entity_10"),
                        true,
                        None,
                    ).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("no_graph", n),
            &n,
            |b, _| {
                b.iter(|| {
                    db.recall(black_box(&query), 10, None, None, false, false, None, true, None)
                        .unwrap()
                })
            },
        );
    }
    group.finish();
}

fn bench_reinforce_overhead(c: &mut Criterion) {
    let dim = 384;
    let n = 10_000;
    let mut group = c.benchmark_group("reinforce_overhead");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    let db_with = AIDB::new(":memory:", dim).unwrap();
    let db_without = AIDB::new(":memory:", dim).unwrap();
    seed_db_scaled(&db_with, n, dim, false);
    seed_db_scaled(&db_without, n, dim, false);
    let query = query_embedding(dim);

    group.bench_function("with_reinforce", |b| {
        b.iter(|| {
            db_with.recall(black_box(&query), 10, None, None, false, false, None, false, None)
                .unwrap()
        })
    });

    group.bench_function("without_reinforce", |b| {
        b.iter(|| {
            db_without.recall(black_box(&query), 10, None, None, false, false, None, true, None)
                .unwrap()
        })
    });

    group.finish();
}

fn bench_record_scaled(c: &mut Criterion) {
    let dim = 384;
    let mut group = c.benchmark_group("record_scaled");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &existing in &[0, 1000, 10_000] {
        let db = AIDB::new(":memory:", dim).unwrap();
        if existing > 0 {
            seed_db_scaled(&db, existing, dim, false);
        }
        let meta = serde_json::json!({});
        let mut i = 0u64;

        group.bench_with_input(
            BenchmarkId::new("single_dim384", existing),
            &existing,
            |b, _| {
                b.iter(|| {
                    let emb = vec_seed_dim(i as f32 * 0.37 + 100000.0, dim);
                    db.record(
                        black_box(&format!("bench scaled record {}", i)),
                        "episodic", 0.5, 0.0, 604800.0, &meta, &emb, "default",
                    ).unwrap();
                    i += 1;
                })
            },
        );
    }
    group.finish();
}

fn bench_record_batch_scaled(c: &mut Criterion) {
    let dim = 384;
    let mut group = c.benchmark_group("record_batch_scaled");
    group.sample_size(10);

    for &batch_size in &[100, 500, 1000] {
        group.bench_with_input(
            BenchmarkId::new("batch_dim384", batch_size),
            &batch_size,
            |b, &bs| {
                b.iter(|| {
                    let db = AIDB::new(":memory:", dim).unwrap();
                    let inputs: Vec<RecordInput> = (0..bs).map(|i| RecordInput {
                        text: format!("batch memory {i}"),
                        memory_type: "episodic".to_string(),
                        importance: 0.5,
                        valence: 0.0,
                        half_life: 604800.0,
                        metadata: serde_json::json!({}),
                        embedding: vec_seed_dim(i as f32 * 0.37, dim),
                        namespace: "default".to_string(),
                    }).collect();
                    db.record_batch(black_box(&inputs)).unwrap();
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_record,
    bench_get,
    bench_stats,
    bench_relate,
    bench_decay,
    bench_recall,
    bench_bulk_insert,
    bench_record_batch,
    bench_archive,
    bench_hydrate,
    bench_evict,
    bench_recall_scaled,
);

criterion_group! {
    name = scaled_benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(30));
    targets =
        bench_recall_dim_comparison,
        bench_recall_100k,
        bench_recall_with_graph,
        bench_reinforce_overhead,
        bench_record_scaled,
        bench_record_batch_scaled
}

criterion_main!(benches, scaled_benches);
