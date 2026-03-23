use std::collections::HashMap;

use rusqlite::params;

use crate::error::Result;
use crate::scoring;
use crate::types::*;

use super::{now, TextMetadataRow, YantrikDB};

/// Simple English suffix stripping for FTS5 query expansion.
///
/// Returns a stem suitable for FTS5 prefix matching (e.g., "reading" → "read",
/// used as `read*` in FTS5 MATCH). This is data-agnostic — it works for any
/// English text without hardcoded domain knowledge.
fn simple_stem(word: &str) -> Option<String> {
    if word.len() <= 4 {
        return None;
    }
    // Ordered longest-first so we strip the most specific suffix.
    let suffixes: &[(&str, usize)] = &[
        ("ations", 3), ("ation", 3), ("tions", 3), ("ments", 3),
        ("tion", 3), ("sion", 3), ("ment", 3), ("ence", 3),
        ("ance", 3), ("ness", 3), ("ible", 3), ("able", 3),
        ("ful", 3), ("ous", 3), ("ive", 3), ("ary", 3),
        ("ery", 3), ("ory", 3), ("ing", 3), ("ble", 3),
        ("ity", 3), ("ish", 3),
        ("ed", 3), ("er", 3), ("ly", 3), ("al", 3),
        ("es", 3), ("s", 3),
    ];
    for &(suffix, min_stem) in suffixes {
        if word.ends_with(suffix) && word.len() - suffix.len() >= min_stem {
            return Some(word[..word.len() - suffix.len()].to_string());
        }
    }
    None
}

/// Map of irregular verb forms → base form, used to expand FTS queries.
/// E.g., query contains "grow" → also search for "grew" and "grown".
/// Each entry: (form, &[all_forms]) — we map any form to all alternate forms.
const IRREGULAR_VERBS: &[(&str, &[&str])] = &[
    ("grow", &["grew", "grown", "growing"]),
    ("grew", &["grow", "grown", "growing"]),
    ("go", &["went", "gone", "going"]),
    ("went", &["go", "gone", "going"]),
    ("come", &["came", "coming"]),
    ("came", &["come", "coming"]),
    ("run", &["ran", "running"]),
    ("ran", &["run", "running"]),
    ("eat", &["ate", "eaten", "eating"]),
    ("ate", &["eat", "eaten", "eating"]),
    ("drink", &["drank", "drunk", "drinking"]),
    ("drank", &["drink", "drunk", "drinking"]),
    ("write", &["wrote", "written", "writing"]),
    ("wrote", &["write", "written", "writing"]),
    ("read", &["reading"]),
    ("speak", &["spoke", "spoken", "speaking"]),
    ("spoke", &["speak", "spoken", "speaking"]),
    ("think", &["thought", "thinking"]),
    ("thought", &["think", "thinking"]),
    ("buy", &["bought", "buying"]),
    ("bought", &["buy", "buying"]),
    ("teach", &["taught", "teaching"]),
    ("taught", &["teach", "teaching"]),
    ("feel", &["felt", "feeling"]),
    ("felt", &["feel", "feeling"]),
    ("keep", &["kept", "keeping"]),
    ("kept", &["keep", "keeping"]),
    ("leave", &["left", "leaving"]),
    ("left", &["leave", "leaving"]),
    ("meet", &["met", "meeting"]),
    ("met", &["meet", "meeting"]),
    ("take", &["took", "taken", "taking"]),
    ("took", &["take", "taken", "taking"]),
    ("give", &["gave", "given", "giving"]),
    ("gave", &["give", "given", "giving"]),
    ("know", &["knew", "known", "knowing"]),
    ("knew", &["know", "known", "knowing"]),
    ("see", &["saw", "seen", "seeing"]),
    ("saw", &["see", "seen", "seeing"]),
    ("begin", &["began", "begun", "beginning"]),
    ("began", &["begin", "begun", "beginning"]),
    ("break", &["broke", "broken", "breaking"]),
    ("broke", &["break", "broken", "breaking"]),
    ("drive", &["drove", "driven", "driving"]),
    ("drove", &["drive", "driven", "driving"]),
    ("sing", &["sang", "sung", "singing"]),
    ("sang", &["sing", "sung", "singing"]),
    ("swim", &["swam", "swum", "swimming"]),
    ("swam", &["swim", "swum", "swimming"]),
    ("choose", &["chose", "chosen", "choosing"]),
    ("chose", &["choose", "chosen", "choosing"]),
    ("lose", &["lost", "losing"]),
    ("lost", &["lose", "losing"]),
    ("win", &["won", "winning"]),
    ("won", &["win", "winning"]),
    ("sleep", &["slept", "sleeping"]),
    ("slept", &["sleep", "sleeping"]),
    ("build", &["built", "building"]),
    ("built", &["build", "building"]),
    ("send", &["sent", "sending"]),
    ("sent", &["send", "sending"]),
    ("spend", &["spent", "spending"]),
    ("spent", &["spend", "spending"]),
    ("fall", &["fell", "fallen", "falling"]),
    ("fell", &["fall", "fallen", "falling"]),
];

/// Get irregular verb alternate forms for a word (if any).
fn irregular_verb_forms(word: &str) -> Option<&'static [&'static str]> {
    for &(form, alts) in IRREGULAR_VERBS {
        if form == word {
            return Some(alts);
        }
    }
    None
}

impl YantrikDB {
    /// Retrieve memories using multi-signal fusion scoring.
    /// When `expand_entities` is true, graph edges are followed to pull in
    /// entity-connected memories that pure vector search would miss.
    #[tracing::instrument(skip(self, query_embedding), fields(top_k, expand_entities, namespace))]
    pub fn recall(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
        expand_entities: bool,
        query_text: Option<&str>,
        skip_reinforce: bool,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
    ) -> Result<Vec<RecallResult>> {
        let ts = now();

        // Load per-database learned weights (falls back to defaults if none learned yet)
        let learned_weights = self.load_learned_weights()?;

        // Detect query sentiment once for directional valence boosting
        let query_sentiment = query_text
            .map(scoring::detect_query_sentiment)
            .unwrap_or(0.0);

        // Step 1: Vector candidate generation via HNSW
        // Fetch a large pool to ensure diverse, high-quality candidates survive MMR filtering.
        let fetch_k = (top_k * 20).min(500);
        let vec_results = {
            let _span = tracing::debug_span!("hnsw_search", fetch_k).entered();
            self.vec_index.read().unwrap().search(query_embedding, fetch_k)?
        };

        if vec_results.is_empty() {
            return Ok(vec![]);
        }

        // Step 2: Score from in-memory cache (replaces fetch_memories_by_rids)
        let mut scored: Vec<RecallResult> = Vec::new();
        {
            let cache = self.scoring_cache.read().unwrap();
            for (rid, distance) in &vec_results {
                let Some(row) = cache.get(rid) else { continue };

                // Filter: consolidation_status
                let status_ok = if include_consolidated {
                    row.consolidation_status == "active" || row.consolidation_status == "consolidated"
                } else {
                    row.consolidation_status == "active"
                };
                if !status_ok { continue; }

                // Filter: memory_type
                if let Some(mt) = memory_type {
                    if row.memory_type != mt { continue; }
                }

                // Filter: time_window
                if let Some((start, end)) = time_window {
                    if row.created_at < start || row.created_at > end { continue; }
                }

                // Filter: namespace
                if let Some(ns) = namespace {
                    if row.namespace != ns { continue; }
                }

                // Filter: domain (V10)
                if let Some(d) = domain {
                    if row.domain != d { continue; }
                }

                // Filter: source (V10)
                if let Some(s) = source {
                    if row.source != s { continue; }
                }

                let sim_score = (1.0 - distance).max(0.0);
                let elapsed = ts - row.last_access;
                let decay = scoring::decay_score(row.importance, row.half_life, elapsed);
                let age = ts - row.created_at;
                let recency = scoring::recency_score(age);
                let composite = scoring::adaptive_composite_score(sim_score, decay, recency, row.importance, row.valence, query_sentiment, &learned_weights);
                let why = scoring::build_why(sim_score, recency, decay, row.valence);
                let contributions = scoring::adaptive_contributions(sim_score, decay, recency, row.importance, &learned_weights);
                let valence_multiplier = scoring::query_valence_boost(row.valence, query_sentiment);

                scored.push(RecallResult {
                    rid: rid.clone(),
                    memory_type: row.memory_type.clone(),
                    text: String::new(),  // hydrated after top_k selection
                    created_at: row.created_at,
                    importance: row.importance,
                    valence: row.valence,
                    score: composite,
                    scores: ScoreBreakdown {
                        similarity: sim_score,
                        decay,
                        recency,
                        importance: row.importance,
                        graph_proximity: 0.0,
                        contributions,
                        valence_multiplier,
                    },
                    why_retrieved: why,
                    metadata: serde_json::Value::Null,  // hydrated after top_k selection
                    namespace: row.namespace.clone(),
                    certainty: row.certainty,
                    domain: row.domain.clone(),
                    source: row.source.clone(),
                    emotional_state: row.emotional_state.clone(),
                });
            }
        } // drop cache borrow

        // Step 2.5: High-importance memory fallback (similarity-gated)
        //
        // Anchor memories define the user's life story. HNSW approximate search
        // may miss them when many noise memories dominate the nearest-neighbor pool.
        // Include important memories if they have at least moderate similarity.
        //
        // Thresholds adapt to database size: large databases need more aggressive
        // fallback because HNSW approximate search degrades with more vectors.
        {
            let total_memories = self.scoring_cache.read().unwrap().len();
            let high_imp_threshold = if total_memories > 5000 { 0.5 } else { 0.7 };
            let min_sim_for_fallback = if total_memories > 5000 { 0.15 } else { 0.20 };
            let existing_rids: std::collections::HashSet<&str> =
                scored.iter().map(|r| r.rid.as_str()).collect();
            let important_rids: Vec<String> = {
                let cache = self.scoring_cache.read().unwrap();
                cache
                    .iter()
                    .filter(|(rid, row)| {
                        row.importance >= high_imp_threshold
                            && row.consolidation_status == "active"
                            && !existing_rids.contains(rid.as_str())
                            && memory_type.map_or(true, |mt| row.memory_type == mt)
                            && time_window.map_or(true, |(s, e)| {
                                row.created_at >= s && row.created_at <= e
                            })
                            && namespace.map_or(true, |ns| row.namespace == ns)
                            && domain.map_or(true, |d| row.domain == d)
                            && source.map_or(true, |s| row.source == s)
                    })
                    .map(|(rid, _)| rid.clone())
                    .collect()
            };

            if !important_rids.is_empty() {
                let rid_refs: Vec<&str> = important_rids.iter().map(|r| r.as_str()).collect();
                let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;
                let cache = self.scoring_cache.read().unwrap();
                for rid in &important_rids {
                    let Some(row) = cache.get(rid) else { continue };
                    let Some(emb_blob) = emb_map.get(rid.as_str()) else { continue };
                    let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
                    let sim_score =
                        crate::consolidate::cosine_similarity(query_embedding, &mem_emb) as f64;

                    if sim_score < min_sim_for_fallback { continue; }

                    let elapsed = ts - row.last_access;
                    let decay = scoring::decay_score(row.importance, row.half_life, elapsed);
                    let age = ts - row.created_at;
                    let recency = scoring::recency_score(age);
                    let composite = scoring::adaptive_composite_score(
                        sim_score, decay, recency, row.importance, row.valence, query_sentiment, &learned_weights,
                    );
                    let why = scoring::build_why(sim_score, recency, decay, row.valence);
                    let contributions =
                        scoring::adaptive_contributions(sim_score, decay, recency, row.importance, &learned_weights);
                    let valence_multiplier = scoring::query_valence_boost(row.valence, query_sentiment);

                    scored.push(RecallResult {
                        rid: rid.clone(),
                        memory_type: row.memory_type.clone(),
                        text: String::new(),
                        created_at: row.created_at,
                        importance: row.importance,
                        valence: row.valence,
                        score: composite,
                        scores: ScoreBreakdown {
                            similarity: sim_score,
                            decay,
                            recency,
                            importance: row.importance,
                            graph_proximity: 0.0,
                            contributions,
                            valence_multiplier,
                        },
                        why_retrieved: why,
                        metadata: serde_json::Value::Null,
                        namespace: row.namespace.clone(),
                        certainty: row.certainty,
                        domain: row.domain.clone(),
                        source: row.source.clone(),
                        emotional_state: row.emotional_state.clone(),
                    });
                }
            }
        }

        // Step 1.5: FTS5 keyword fallback
        //
        // Catches queries where keywords appear in memory text but HNSW
        // misses the match (e.g., "What books am I reading?" where "reading"
        // is in the text). Uses importance-weighted BM25 ranking so anchor
        // memories naturally surface above noise — no hardcoded thresholds.
        //
        // FTS limit scales dynamically with database size.
        // FTS5 only works when encryption is disabled (the default).
        if !self.is_encrypted() {
            const FTS_MIN_SIM: f64 = 0.05;

            const STOPWORDS: &[&str] = &[
                "a", "an", "the", "is", "are", "am", "was", "were", "be", "been",
                "what", "who", "how", "when", "where", "which", "why",
                "do", "did", "does", "have", "has", "had",
                "i", "me", "my", "mine", "we", "our", "you", "your",
                "to", "of", "in", "on", "at", "by", "for", "with", "from",
                "about", "tell", "and", "or", "but", "not", "no",
                "it", "its", "that", "this", "there", "s",
                "she", "her", "he", "his", "they", "them",
                "most", "each", "any", "all", "every", "been", "being",
                "up", "out", "so", "if", "than", "very", "just", "also",
            ];

            {
                if let Some(qt) = query_text {
                    let raw_keywords: Vec<String> = qt
                        .split(|c: char| !c.is_alphanumeric())
                        .filter(|s| !s.is_empty() && s.len() > 1)
                        .filter(|s| !STOPWORDS.contains(&s.to_lowercase().as_str()))
                        .map(|s| s.to_string())
                        .collect();

                    // Filter out person-type entity names from FTS keywords.
                    // Person names (e.g., "Priya", "Meera") appear in thousands
                    // of memories, flooding FTS results with entity-matching noise.
                    // Topic entities (e.g., "yoga", "reading") are kept — they're
                    // valuable FTS keywords that graph expansion may not cover.
                    let mut keywords: Vec<String> = {
                        let gi = self.graph_index.read().unwrap();
                        let query_tokens = crate::graph::tokenize(qt);
                        let matched = gi.entity_matches_query(&query_tokens);
                        // Only filter entities with type "person" — these are the
                        // high-frequency names that cause FTS noise.
                        let person_matches: Vec<_> = matched
                            .into_iter()
                            .filter(|(_, etype, _)| etype == "person")
                            .collect();
                        if person_matches.is_empty() {
                            raw_keywords
                        } else {
                            let person_tokens: std::collections::HashSet<String> = person_matches
                                .into_iter()
                                .flat_map(|(name, _, _)| {
                                    let mut tokens = vec![name.to_lowercase()];
                                    for token in name.split_whitespace() {
                                        tokens.push(token.to_lowercase());
                                    }
                                    tokens
                                })
                                .collect();
                            let filtered: Vec<String> = raw_keywords
                                .iter()
                                .filter(|kw| !person_tokens.contains(&kw.to_lowercase()))
                                .cloned()
                                .collect();
                            if filtered.is_empty() { raw_keywords } else { filtered }
                        }
                    };

                    // Entity-seeded FTS for group/aggregation queries.
                    //
                    // For "Tell me about Priya's family", normal FTS keyword
                    // "family" won't match individual member memories ("My husband
                    // Arjun is a product manager"). Inject graph-connected person
                    // entity names as additional FTS keywords so those memories
                    // enter the scoring pool with keyword_match boost.
                    {
                        const GROUP_FTS_WORDS: &[&str] = &[
                            "team", "group", "colleagues", "coworkers", "friends",
                            "family", "staff", "members", "people",
                        ];
                        let qt_lower = qt.to_lowercase();
                        if GROUP_FTS_WORDS.iter().any(|kw| qt_lower.contains(kw)) {
                            let gi = self.graph_index.read().unwrap();
                            let query_tokens = crate::graph::tokenize(qt);
                            let matched = gi.entity_matches_query(&query_tokens);
                            if !matched.is_empty() {
                                let seed_names: Vec<&str> =
                                    matched.iter().map(|(n, _, _)| n.as_str()).collect();
                                let expanded = gi.expand_bfs(&seed_names, 2, 30);
                                let mut injected = 0usize;
                                for (name, hops, _) in &expanded {
                                    if *hops == 0 || injected >= 15 {
                                        continue;
                                    }
                                    if gi.entity_type(name).map_or(false, |t| t == "person") {
                                        for part in name.split_whitespace() {
                                            if part.len() > 1
                                                && !keywords
                                                    .iter()
                                                    .any(|k| k.eq_ignore_ascii_case(part))
                                            {
                                                keywords.push(part.to_string());
                                            }
                                        }
                                        injected += 1;
                                    }
                                }
                            }
                        }
                    }

                    if !keywords.is_empty() {
                        // Build FTS5 query with stemmed prefix expansion,
                        // irregular verb forms, and AND conjunction for
                        // multi-keyword selectivity.
                        //
                        // Each keyword becomes a group:
                        //   "reading" → ("reading" OR read*)
                        //   "grow"    → ("grow" OR "grew" OR "grown" OR "growing")
                        //
                        // Multiple groups are AND'd for selectivity:
                        //   ("books" OR book*) AND ("reading" OR read*)
                        //
                        // Falls back to OR if AND returns too few results.
                        let mut keyword_groups: Vec<String> = Vec::new();
                        for kw in &keywords {
                            let kw_lower = kw.to_lowercase();
                            let mut parts: Vec<String> = Vec::new();
                            parts.push(format!("\"{}\"", kw.replace('"', "")));
                            if let Some(stem) = simple_stem(&kw_lower) {
                                parts.push(format!("{}*", stem));
                            }
                            if let Some(alts) = irregular_verb_forms(&kw_lower) {
                                for alt in alts {
                                    parts.push(format!("\"{}\"", alt));
                                }
                            }
                            keyword_groups.push(if parts.len() == 1 {
                                parts[0].clone()
                            } else {
                                format!("({})", parts.join(" OR "))
                            });
                        }

                        // Use AND for selectivity when 2+ keyword groups,
                        // single keywords always use their group directly.
                        let fts_query_and = if keyword_groups.len() >= 2 {
                            Some(keyword_groups.join(" AND "))
                        } else {
                            None
                        };
                        let fts_query_or = keyword_groups.join(" OR ");
                        // Primary query: AND if available, else OR
                        let fts_query = fts_query_and.as_deref()
                            .unwrap_or(&fts_query_or);

                        // Adaptive FTS limit: scales with database size.
                        // Small DBs (<3K): 30 is enough. Large DBs (15K+): need 150+
                        // to surface important memories above keyword noise.
                        let total_memories = self.scoring_cache.read().unwrap().len();
                        let fts_limit = (total_memories / 100).max(30).min(200);

                        // Dynamic importance threshold for Phase 2.
                        // Phase 2 ensures important keyword-matching memories surface
                        // even when noise exhausts Phase 1's BM25-ranked LIMIT.
                        // Use 70% of mean to catch relevant anchors with modest importance
                        // (e.g., yoga at imp=0.30 when mean≈0.35).
                        let mean_importance = {
                            let cache = self.scoring_cache.read().unwrap();
                            if cache.is_empty() {
                                0.5
                            } else {
                                let sum: f64 = cache.values().map(|r| r.importance).sum();
                                let mean = sum / cache.len() as f64;
                                (mean * 0.7).max(0.25)
                            }
                        };

                        // Importance-weighted BM25 ranking.
                        // rank is negative in FTS5 (more negative = better BM25).
                        // Multiplying by (0.5 + importance) makes important memories
                        // more negative → sort higher.
                        let fts_sql = if memory_type.is_some() {
                            format!(
                                "SELECT m.rid FROM memories m \
                                 JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                 WHERE memories_fts MATCH ?1 \
                                 AND m.consolidation_status = 'active' \
                                 AND m.type = ?2 \
                                 {} \
                                 ORDER BY rank * (0.5 + m.importance) \
                                 LIMIT {}",
                                if namespace.is_some() { "AND m.namespace = ?3" } else { "" },
                                fts_limit,
                            )
                        } else {
                            format!(
                                "SELECT m.rid FROM memories m \
                                 JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                 WHERE memories_fts MATCH ?1 \
                                 AND m.consolidation_status = 'active' \
                                 {} \
                                 ORDER BY rank * (0.5 + m.importance) \
                                 LIMIT {}",
                                if namespace.is_some() { "AND m.namespace = ?2" } else { "" },
                                fts_limit,
                            )
                        };

                        // Helper closure to run an FTS query and collect RIDs.
                        let run_fts_phase1 = |q: &str| -> Vec<String> {
                            let conn = self.conn.lock().unwrap();
                            let mut stmt = conn.prepare_cached(&fts_sql).ok();
                            if let Some(ref mut stmt) = stmt {
                                let result: std::result::Result<Vec<String>, _> = if let Some(mt) = memory_type {
                                    if let Some(ns) = namespace {
                                        stmt.query_map(
                                            params![q, mt, ns],
                                            |row| row.get::<_, String>(0),
                                        ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    } else {
                                        stmt.query_map(
                                            params![q, mt],
                                            |row| row.get::<_, String>(0),
                                        ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    }
                                } else if let Some(ns) = namespace {
                                    stmt.query_map(
                                        params![q, ns],
                                        |row| row.get::<_, String>(0),
                                    ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                } else {
                                    stmt.query_map(
                                        params![q],
                                        |row| row.get::<_, String>(0),
                                    ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                };
                                result.unwrap_or_default()
                            } else {
                                vec![]
                            }
                        };

                        // Run AND query first (more selective), fall back to OR
                        // if AND returns too few results.
                        let mut fts_rids = run_fts_phase1(fts_query);
                        if fts_rids.len() < 5 && fts_query_and.is_some() {
                            fts_rids = run_fts_phase1(&fts_query_or);
                        }

                        // Phase 2: Importance-filtered FTS.
                        // Phase 1's BM25-ranked LIMIT can be exhausted by noise
                        // (e.g., "yoga" matches thousands of generated memories).
                        // Phase 2 ensures important memories with keyword matches
                        // always enter the scoring pool by filtering on importance
                        // and using a separate LIMIT.
                        //
                        // Uses AND first for selectivity, then falls back to OR
                        // when AND is too strict (e.g., "books AND reading" misses
                        // memories that only contain "reading" like "I'm reading Sapiens").
                        {
                            let imp_fts_sql = if memory_type.is_some() {
                                format!(
                                    "SELECT m.rid FROM memories m \
                                     JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                     WHERE memories_fts MATCH ?1 \
                                     AND m.consolidation_status = 'active' \
                                     AND m.importance > ?2 \
                                     AND m.type = ?3 \
                                     {} \
                                     ORDER BY m.importance DESC \
                                     LIMIT 100",
                                    if namespace.is_some() { "AND m.namespace = ?4" } else { "" },
                                )
                            } else {
                                format!(
                                    "SELECT m.rid FROM memories m \
                                     JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                     WHERE memories_fts MATCH ?1 \
                                     AND m.consolidation_status = 'active' \
                                     AND m.importance > ?2 \
                                     {} \
                                     ORDER BY m.importance DESC \
                                     LIMIT 100",
                                    if namespace.is_some() { "AND m.namespace = ?3" } else { "" },
                                )
                            };

                            // Helper to run Phase 2 with a given FTS query string.
                            let run_fts_phase2 = |q: &str| -> Vec<String> {
                                let conn = self.conn.lock().unwrap();
                                let mut stmt = conn.prepare_cached(&imp_fts_sql).ok();
                                if let Some(ref mut stmt) = stmt {
                                    let result: std::result::Result<Vec<String>, _> = if let Some(mt) = memory_type {
                                        if let Some(ns) = namespace {
                                            stmt.query_map(
                                                params![q, mean_importance, mt, ns],
                                                |row| row.get::<_, String>(0),
                                            ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                        } else {
                                            stmt.query_map(
                                                params![q, mean_importance, mt],
                                                |row| row.get::<_, String>(0),
                                            ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                        }
                                    } else if let Some(ns) = namespace {
                                        stmt.query_map(
                                            params![q, mean_importance, ns],
                                            |row| row.get::<_, String>(0),
                                        ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    } else {
                                        stmt.query_map(
                                            params![q, mean_importance],
                                            |row| row.get::<_, String>(0),
                                        ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    };
                                    result.unwrap_or_default()
                                } else {
                                    vec![]
                                }
                            };

                            // Run AND first (selective), fall back to OR if too few results.
                            let mut imp_rids = run_fts_phase2(fts_query);
                            if imp_rids.len() < 10 && fts_query_and.is_some() {
                                let or_rids = run_fts_phase2(&fts_query_or);
                                let existing: std::collections::HashSet<String> =
                                    imp_rids.iter().cloned().collect();
                                imp_rids.extend(
                                    or_rids.into_iter().filter(|r| !existing.contains(r))
                                );
                            }

                            // Merge Phase 2 into Phase 1 results (dedup).
                            let existing_set: std::collections::HashSet<String> =
                                fts_rids.iter().cloned().collect();
                            let new_imp: Vec<String> = imp_rids
                                .into_iter()
                                .filter(|r| !existing_set.contains(r))
                                .collect();
                            fts_rids.extend(new_imp);
                        }

                        // Phase 2.5: Per-keyword anchor scan.
                        //
                        // When keywords match many memories (e.g., "reading"),
                        // Phase 1+2 may not surface the specific anchor memory
                        // among thousands of noise matches. Scan each individual
                        // keyword for the most important matching memories.
                        //
                        // Only targets anchor memories (importance > 0.5) and
                        // applies NO keyword_boost — they compete on pure scoring.
                        if keyword_groups.len() >= 2 {
                            let anchor_fts_sql = if memory_type.is_some() {
                                format!(
                                    "SELECT m.rid FROM memories m \
                                     JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                     WHERE memories_fts MATCH ?1 \
                                     AND m.consolidation_status = 'active' \
                                     AND m.importance > 0.5 \
                                     AND m.type = ?2 \
                                     {} \
                                     ORDER BY m.importance DESC \
                                     LIMIT 10",
                                    if namespace.is_some() { "AND m.namespace = ?3" } else { "" },
                                )
                            } else {
                                format!(
                                    "SELECT m.rid FROM memories m \
                                     JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                     WHERE memories_fts MATCH ?1 \
                                     AND m.consolidation_status = 'active' \
                                     AND m.importance > 0.5 \
                                     {} \
                                     ORDER BY m.importance DESC \
                                     LIMIT 10",
                                    if namespace.is_some() { "AND m.namespace = ?2" } else { "" },
                                )
                            };

                            let existing_fts: std::collections::HashSet<String> =
                                fts_rids.iter().cloned().collect();

                            for group in &keyword_groups {
                                let anchor_rids: Vec<String> = {
                                    let conn = self.conn.lock().unwrap();
                                    let mut stmt = conn.prepare_cached(&anchor_fts_sql).ok();
                                    if let Some(ref mut stmt) = stmt {
                                        let result: std::result::Result<Vec<String>, _> = if let Some(mt) = memory_type {
                                            if let Some(ns) = namespace {
                                                stmt.query_map(
                                                    params![group, mt, ns],
                                                    |row| row.get::<_, String>(0),
                                                ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                            } else {
                                                stmt.query_map(
                                                    params![group, mt],
                                                    |row| row.get::<_, String>(0),
                                                ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                            }
                                        } else if let Some(ns) = namespace {
                                            stmt.query_map(
                                                params![group, ns],
                                                |row| row.get::<_, String>(0),
                                            ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                        } else {
                                            stmt.query_map(
                                                params![group],
                                                |row| row.get::<_, String>(0),
                                            ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                        };
                                        result.unwrap_or_default()
                                    } else {
                                        vec![]
                                    }
                                };

                                for rid in anchor_rids {
                                    if !existing_fts.contains(&rid) {
                                        fts_rids.push(rid);
                                    }
                                }
                            }
                        }

                        // Boost existing candidates that matched FTS5 keywords.
                        // Scale boost inversely with similarity: memories where vector
                        // search failed but keywords matched get more boost.
                        {
                            let fts_rid_set: std::collections::HashSet<&str> =
                                fts_rids.iter().map(|r| r.as_str()).collect();
                            for result in &mut scored {
                                if fts_rid_set.contains(result.rid.as_str())
                                    && !result.why_retrieved.iter().any(|w| w == "keyword_match")
                                {
                                    let sim = result.scores.similarity;
                                    let boost = learned_weights.keyword_boost * (1.0 - sim).max(0.2);
                                    result.score += boost;
                                    result.why_retrieved.push("keyword_match".to_string());
                                }
                            }
                        }

                        // Add new FTS candidates not already in the pool
                        let existing_rids: std::collections::HashSet<String> =
                            scored.iter().map(|r| r.rid.clone()).collect();
                        let new_fts_rids: Vec<String> = fts_rids.into_iter()
                            .filter(|r| !existing_rids.contains(r))
                            .collect();

                        if !new_fts_rids.is_empty() {
                            let rid_refs: Vec<&str> = new_fts_rids.iter().map(|r| r.as_str()).collect();
                            let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;

                            let cache = self.scoring_cache.read().unwrap();
                            for rid in &new_fts_rids {
                                let Some(row) = cache.get(rid) else { continue };

                                let status_ok = if include_consolidated {
                                    row.consolidation_status == "active" || row.consolidation_status == "consolidated"
                                } else {
                                    row.consolidation_status == "active"
                                };
                                if !status_ok { continue; }
                                if let Some((start, end)) = time_window {
                                    if row.created_at < start || row.created_at > end { continue; }
                                }

                                let Some(emb_blob) = emb_map.get(rid.as_str()) else { continue };
                                let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
                                let sim_score = crate::consolidate::cosine_similarity(
                                    query_embedding, &mem_emb,
                                ) as f64;

                                if sim_score < FTS_MIN_SIM { continue; }

                                let elapsed = ts - row.last_access;
                                let decay = scoring::decay_score(row.importance, row.half_life, elapsed);
                                let age = ts - row.created_at;
                                let recency = scoring::recency_score(age);
                                let composite = scoring::adaptive_composite_score(
                                    sim_score, decay, recency, row.importance, row.valence, query_sentiment, &learned_weights,
                                );
                                let kw_boost = learned_weights.keyword_boost * (1.0 - sim_score).max(0.2);
                                let mut why = scoring::build_why(sim_score, recency, decay, row.valence);
                                why.push("keyword_match".to_string());
                                why.push("fts_sourced".to_string());
                                let contributions = scoring::adaptive_contributions(
                                    sim_score, decay, recency, row.importance, &learned_weights,
                                );
                                let valence_multiplier = scoring::query_valence_boost(row.valence, query_sentiment);

                                scored.push(RecallResult {
                                    rid: rid.clone(),
                                    memory_type: row.memory_type.clone(),
                                    text: String::new(),
                                    created_at: row.created_at,
                                    importance: row.importance,
                                    valence: row.valence,
                                    score: composite + kw_boost,
                                    scores: ScoreBreakdown {
                                        similarity: sim_score,
                                        decay,
                                        recency,
                                        importance: row.importance,
                                        graph_proximity: 0.0,
                                        contributions,
                                        valence_multiplier,
                                    },
                                    why_retrieved: why,
                                    metadata: serde_json::Value::Null,
                                    namespace: row.namespace.clone(),
                                    certainty: row.certainty,
                                    domain: row.domain.clone(),
                                    source: row.source.clone(),
                                    emotional_state: row.emotional_state.clone(),
                                });
                            }
                        }
                    }
                }
            }
        }

        // Step 2.7: Valence-based retrieval for emotional queries
        //
        // For queries with strong sentiment (e.g., "stressful moments", "happiest times"),
        // scan the scoring cache for strongly-valenced memories that HNSW may miss due to
        // low semantic overlap. These memories are important for emotional retrieval.
        if query_sentiment.abs() > 0.5 {
            const VALENCE_SCAN_THRESHOLD: f64 = 0.4; // min |valence| to consider
            const VALENCE_SCAN_MAX: usize = 30;      // max new candidates
            const VALENCE_MIN_SIM: f64 = 0.02;       // very low floor — valence is the signal

            let existing_rids: std::collections::HashSet<&str> =
                scored.iter().map(|r| r.rid.as_str()).collect();

            // Find strongly-valenced memories matching query sentiment direction
            let valence_rids: Vec<String> = {
                let cache = self.scoring_cache.read().unwrap();
                let mut candidates: Vec<(String, f64)> = cache
                    .iter()
                    .filter(|(rid, row)| {
                        row.consolidation_status == "active"
                            && !existing_rids.contains(rid.as_str())
                            && row.valence.abs() >= VALENCE_SCAN_THRESHOLD
                            // Match direction: negative query wants negative memories
                            && (query_sentiment * row.valence > 0.0
                                || (query_sentiment < 0.0 && row.valence < -0.2))
                            && row.importance >= 0.5 // only important memories
                            && memory_type.map_or(true, |mt| row.memory_type == mt)
                            && time_window.map_or(true, |(s, e)| {
                                row.created_at >= s && row.created_at <= e
                            })
                            && namespace.map_or(true, |ns| row.namespace == ns)
                    })
                    .map(|(rid, row)| {
                        // Rank by |valence| * importance
                        let rank = row.valence.abs() * row.importance;
                        (rid.clone(), rank)
                    })
                    .collect();
                candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                candidates.into_iter().take(VALENCE_SCAN_MAX).map(|(rid, _)| rid).collect()
            };

            if !valence_rids.is_empty() {
                let rid_refs: Vec<&str> = valence_rids.iter().map(|r| r.as_str()).collect();
                let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;
                let cache = self.scoring_cache.read().unwrap();
                for rid in &valence_rids {
                    let Some(row) = cache.get(rid) else { continue };
                    let Some(emb_blob) = emb_map.get(rid.as_str()) else { continue };
                    let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
                    let sim_score =
                        crate::consolidate::cosine_similarity(query_embedding, &mem_emb) as f64;

                    if sim_score < VALENCE_MIN_SIM { continue; }

                    let elapsed = ts - row.last_access;
                    let decay = scoring::decay_score(row.importance, row.half_life, elapsed);
                    let age = ts - row.created_at;
                    let recency = scoring::recency_score(age);
                    let composite = scoring::adaptive_composite_score(
                        sim_score, decay, recency, row.importance, row.valence, query_sentiment, &learned_weights,
                    );
                    // Additive valence boost: helps valence-matched memories compete
                    // when cosine similarity is low. Scaled by |valence| * importance
                    // so only strongly-valenced important memories get meaningful lift.
                    let valence_additive = 0.20 * row.valence.abs() * row.importance;
                    let mut why = scoring::build_why(sim_score, recency, decay, row.valence);
                    why.push("valence_match".to_string());
                    let contributions =
                        scoring::adaptive_contributions(sim_score, decay, recency, row.importance, &learned_weights);
                    let valence_multiplier = scoring::query_valence_boost(row.valence, query_sentiment);

                    scored.push(RecallResult {
                        rid: rid.clone(),
                        memory_type: row.memory_type.clone(),
                        text: String::new(),
                        created_at: row.created_at,
                        importance: row.importance,
                        valence: row.valence,
                        score: composite + valence_additive,
                        scores: ScoreBreakdown {
                            similarity: sim_score,
                            decay,
                            recency,
                            importance: row.importance,
                            graph_proximity: 0.0,
                            contributions,
                            valence_multiplier,
                        },
                        why_retrieved: why,
                        metadata: serde_json::Value::Null,
                        namespace: row.namespace.clone(),
                        certainty: row.certainty,
                        domain: row.domain.clone(),
                        source: row.source.clone(),
                        emotional_state: row.emotional_state.clone(),
                    });
                }
            }
        }

        // Step 2.9: Cold memory fallback
        //
        // Memories that have NEVER been retrieved (access_count == 0) may contain
        // unique facts that get buried under frequently-accessed noise. When the
        // best score so far is weak, search cold memories via FTS to surface
        // forgotten knowledge.
        if !self.is_encrypted() {
            if let Some(qt) = query_text {
                let best_score = scored.iter().map(|r| r.score).fold(0.0f64, f64::max);
                const COLD_ACTIVATION_THRESHOLD: f64 = 0.55;
                const COLD_MIN_SIM: f64 = 0.10;
                const COLD_MAX_CANDIDATES: usize = 30;

                if best_score < COLD_ACTIVATION_THRESHOLD {
                    // Extract keywords (reusing same logic as FTS5 step)
                    let cold_keywords: Vec<String> = qt
                        .split(|c: char| !c.is_alphanumeric())
                        .filter(|s| !s.is_empty() && s.len() > 1)
                        .filter(|s| {
                            const STOP: &[&str] = &[
                                "a","an","the","is","are","am","was","were","be","been",
                                "what","who","how","when","where","which","why",
                                "do","did","does","have","has","had",
                                "i","me","my","mine","we","our","you","your",
                                "to","of","in","on","at","by","for","with","from",
                                "about","tell","and","or","but","not","no",
                                "it","its","that","this","there","s",
                                "she","her","he","his","they","them",
                            ];
                            !STOP.contains(&s.to_lowercase().as_str())
                        })
                        .map(|s| s.to_string())
                        .collect();

                    if !cold_keywords.is_empty() {
                        // Build OR query (looser than AND for cold memories)
                        let mut fts_parts: Vec<String> = Vec::new();
                        for kw in &cold_keywords {
                            let kw_lower = kw.to_lowercase();
                            fts_parts.push(format!("\"{}\"", kw.replace('"', "")));
                            if let Some(stem) = simple_stem(&kw_lower) {
                                fts_parts.push(format!("{}*", stem));
                            }
                            if let Some(alts) = irregular_verb_forms(&kw_lower) {
                                for alt in alts {
                                    fts_parts.push(format!("\"{}\"", alt));
                                }
                            }
                        }
                        let cold_fts = fts_parts.join(" OR ");

                        // Query ONLY cold memories (access_count = 0)
                        let cold_sql = if memory_type.is_some() {
                            format!(
                                "SELECT m.rid FROM memories m \
                                 JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                 WHERE memories_fts MATCH ?1 \
                                 AND m.consolidation_status = 'active' \
                                 AND m.access_count = 0 \
                                 AND m.type = ?2 \
                                 {} \
                                 ORDER BY m.importance DESC \
                                 LIMIT {}",
                                if namespace.is_some() { "AND m.namespace = ?3" } else { "" },
                                COLD_MAX_CANDIDATES,
                            )
                        } else {
                            format!(
                                "SELECT m.rid FROM memories m \
                                 JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                 WHERE memories_fts MATCH ?1 \
                                 AND m.consolidation_status = 'active' \
                                 AND m.access_count = 0 \
                                 {} \
                                 ORDER BY m.importance DESC \
                                 LIMIT {}",
                                if namespace.is_some() { "AND m.namespace = ?2" } else { "" },
                                COLD_MAX_CANDIDATES,
                            )
                        };

                        let cold_rids: Vec<String> = {
                            let conn = self.conn.lock().unwrap();
                            let mut stmt = conn.prepare_cached(&cold_sql).ok();
                            if let Some(ref mut stmt) = stmt {
                                let result: std::result::Result<Vec<String>, _> = if let Some(mt) = memory_type {
                                    if let Some(ns) = namespace {
                                        stmt.query_map(
                                            params![cold_fts, mt, ns],
                                            |row| row.get::<_, String>(0),
                                        ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    } else {
                                        stmt.query_map(
                                            params![cold_fts, mt],
                                            |row| row.get::<_, String>(0),
                                        ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    }
                                } else if let Some(ns) = namespace {
                                    stmt.query_map(
                                        params![cold_fts, ns],
                                        |row| row.get::<_, String>(0),
                                    ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                } else {
                                    stmt.query_map(
                                        params![cold_fts],
                                        |row| row.get::<_, String>(0),
                                    ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                };
                                result.unwrap_or_default()
                            } else {
                                vec![]
                            }
                        };

                        // Score cold candidates — filter to new RIDs only
                        let existing_rids: std::collections::HashSet<String> =
                            scored.iter().map(|r| r.rid.clone()).collect();
                        let new_cold: Vec<String> = cold_rids.into_iter()
                            .filter(|r| !existing_rids.contains(r))
                            .collect();

                        if !new_cold.is_empty() {
                            let rid_refs: Vec<&str> = new_cold.iter().map(|r| r.as_str()).collect();
                            let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;
                            let cache = self.scoring_cache.read().unwrap();

                            for rid in &new_cold {
                                let Some(row) = cache.get(rid) else { continue };
                                let Some(emb_blob) = emb_map.get(rid.as_str()) else { continue };
                                let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
                                let sim_score = crate::consolidate::cosine_similarity(
                                    query_embedding, &mem_emb,
                                ) as f64;

                                if sim_score < COLD_MIN_SIM { continue; }

                                let elapsed = ts - row.last_access;
                                let decay = scoring::decay_score(row.importance, row.half_life, elapsed);
                                let age = ts - row.created_at;
                                let recency = scoring::recency_score(age);
                                let composite = scoring::adaptive_composite_score(
                                    sim_score, decay, recency, row.importance, row.valence, query_sentiment, &learned_weights,
                                );
                                // Cold memory bonus: these haven't been surfaced before
                                let cold_boost = 0.15 * row.importance;
                                let mut why = scoring::build_why(sim_score, recency, decay, row.valence);
                                why.push("cold_memory".to_string());
                                let contributions = scoring::adaptive_contributions(
                                    sim_score, decay, recency, row.importance, &learned_weights,
                                );
                                let valence_multiplier = scoring::query_valence_boost(row.valence, query_sentiment);

                                scored.push(RecallResult {
                                    rid: rid.clone(),
                                    memory_type: row.memory_type.clone(),
                                    text: String::new(),
                                    created_at: row.created_at,
                                    importance: row.importance,
                                    valence: row.valence,
                                    score: composite + cold_boost,
                                    scores: ScoreBreakdown {
                                        similarity: sim_score,
                                        decay,
                                        recency,
                                        importance: row.importance,
                                        graph_proximity: 0.0,
                                        contributions,
                                        valence_multiplier,
                                    },
                                    why_retrieved: why,
                                    metadata: serde_json::Value::Null,
                                    namespace: row.namespace.clone(),
                                    certainty: row.certainty,
                                    domain: row.domain.clone(),
                                    source: row.source.clone(),
                                    emotional_state: row.emotional_state.clone(),
                                });
                            }
                        }
                    }
                }
            }
        }

        // Step 3: Graph expansion (when enabled)
        if expand_entities {
            let _span = tracing::debug_span!("graph_expansion").entered();
            let gi = self.graph_index.read().unwrap();
            let query_entities: Vec<(String, String, u32)> = if let Some(qt) = query_text {
                let query_tokens = crate::graph::tokenize(qt);
                gi.entity_matches_query(&query_tokens)
            } else {
                vec![]
            };

            let (mut base_boost, mut seed_entities, entity_idfs): (f64, Vec<String>, std::collections::HashMap<String, f64>) = if !query_entities.is_empty() {
                let has_person = query_entities.iter().any(|(_, etype, _)| etype == "person");
                let factor = if has_person {
                    0.20
                } else if query_entities.len() >= 2 {
                    0.15
                } else {
                    0.12
                };
                let idfs: std::collections::HashMap<String, f64> = query_entities
                    .iter()
                    .map(|(name, _, mc)| {
                        let idf = 1.0 / (1.0 + (*mc as f64).max(1.0).ln());
                        (name.to_lowercase(), idf)
                    })
                    .collect();
                let names: Vec<String> = query_entities.into_iter().map(|(n, _, _)| n).collect();
                (factor, names, idfs)
            } else if query_text.is_none() {
                // Embedding-only search (no query text): seed from top results
                let mut seed_sorted = scored.clone();
                seed_sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                let seed_count = 3.min(seed_sorted.len());
                let seed_rids: Vec<&str> = seed_sorted[..seed_count].iter().map(|r| r.rid.as_str()).collect();
                let seeds = gi.entities_for_memories(&seed_rids);
                (0.05, seeds, std::collections::HashMap::new())
            } else {
                (0.0, vec![], std::collections::HashMap::new())
            };

            // Group query expansion: if query mentions "team", "family", etc.,
            // seed expansion with person entities CONNECTED to query entities.
            //
            // Previous approach: entities_by_type("person").take(15) grabbed
            // arbitrary person entities, often filler characters unrelated to
            // the query. New approach: BFS from query entities finds people
            // actually connected to the subject (e.g., Priya → Arjun, Meera,
            // Appa, Amma for "family"; Priya → Deepa, Neha for "team").
            const GROUP_KEYWORDS: &[&str] = &[
                "team", "group", "colleagues", "coworkers", "friends", "family",
                "staff", "members", "people",
            ];
            if let Some(qt) = query_text {
                let qt_lower = qt.to_lowercase();
                if GROUP_KEYWORDS.iter().any(|kw| qt_lower.contains(kw)) {
                    if !seed_entities.is_empty() {
                        // BFS from query entities to find connected person entities
                        let query_seeds: Vec<&str> = seed_entities.iter().map(|s| s.as_str()).collect();
                        let nearby = gi.expand_bfs(&query_seeds, 2, 50);
                        for (name, hops, _) in &nearby {
                            if *hops > 0
                                && gi.entity_type(name).map_or(false, |t| t == "person")
                                && !seed_entities.contains(&name.to_string())
                            {
                                seed_entities.push(name.clone());
                            }
                        }
                    } else {
                        // No query entities — fall back to type-based expansion
                        let person_entities = gi.entities_by_type("person");
                        for person in person_entities.into_iter().take(15) {
                            if !seed_entities.contains(&person) {
                                seed_entities.push(person);
                            }
                        }
                    }
                    base_boost = base_boost.max(0.20_f64);
                }
            }

            const MAX_BOOST_PER_MEMORY: f64 = 0.25;
            const MAX_GRAPH_FRACTION: f64 = 1.0;
            const MAX_SEED_ENTITIES: usize = 8;

            // Cap seed entities to prevent graph explosion with many entities
            if seed_entities.len() > MAX_SEED_ENTITIES {
                seed_entities.truncate(MAX_SEED_ENTITIES);
            }

            if !seed_entities.is_empty() && base_boost > 0.0 {
                let seed_refs: Vec<&str> = seed_entities.iter().map(|s| s.as_str()).collect();
                let expanded = gi.expand_bfs(&seed_refs, 2, 30);

                let expanded_map: std::collections::HashMap<String, (u8, f64)> = expanded
                    .iter()
                    .map(|(name, hops, weight)| (name.clone(), (*hops, *weight)))
                    .collect();

                // (a) IDF-weighted additive boost for existing results
                for result in &mut scored {
                    let prox = gi.graph_proximity(&result.rid, &expanded_map);
                    if prox > 0.0 {
                        let mem_entities: Vec<String> = gi.entities_for_memory(&result.rid).into_iter().map(|s| s.to_string()).collect();

                        let mut best_idf = 1.0f64;
                        let mut connecting_entity = String::new();
                        for entity in &mem_entities {
                            if expanded_map.contains_key(entity) {
                                let idf = entity_idfs.get(&entity.to_lowercase()).copied().unwrap_or(1.0);
                                if connecting_entity.is_empty() || idf > best_idf {
                                    best_idf = idf;
                                    connecting_entity = entity.clone();
                                }
                            }
                        }

                        // Consolidation penalty: use consolidation_status as proxy
                        let cache = self.scoring_cache.read().unwrap();
                        let consolidation_factor = cache.get(&result.rid)
                            .map(|r| if r.consolidation_status == "consolidated" { 0.5 } else { 1.0 })
                            .unwrap_or(1.0);
                        drop(cache);

                        let boost = (base_boost * prox * best_idf * consolidation_factor)
                            .min(MAX_BOOST_PER_MEMORY);
                        result.scores.graph_proximity = prox;
                        result.score += boost;
                        if !connecting_entity.is_empty() {
                            result.why_retrieved.push(format!("graph-connected via {connecting_entity}"));
                        }
                    }
                }

                // (b) Graph-only candidates: score from cache + batch embedding fetch
                let max_graph_only = ((MAX_GRAPH_FRACTION * top_k as f64).ceil() as usize).max(1);
                let all_entity_names: Vec<&str> = expanded.iter().map(|(n, _, _)| n.as_str()).collect();
                let graph_rids = gi.memories_for_entities(&all_entity_names);

                let existing_rids: std::collections::HashSet<&str> = scored.iter().map(|r| r.rid.as_str()).collect();
                let new_rids: Vec<String> = graph_rids
                    .into_iter()
                    .filter(|r| !existing_rids.contains(r.as_str()))
                    .collect();

                // Filter graph-only candidates, rank by importance * graph_proximity.
                // This ensures memories directly linked to seed entities (prox≈1.0)
                // outrank memories linked through distant neighbors (prox≈0.25).
                let preselect_pool = max_graph_only * 5; // fetch more, let full scoring pick best
                let filtered_rids: Vec<String> = {
                    let cache = self.scoring_cache.read().unwrap();
                    let mut candidates: Vec<(String, f64)> = new_rids.into_iter()
                        .filter_map(|rid| {
                            let row = cache.get(&rid)?;
                            let status_ok = if include_consolidated {
                                row.consolidation_status == "active" || row.consolidation_status == "consolidated"
                            } else {
                                row.consolidation_status == "active"
                            };
                            if !status_ok { return None; }
                            if let Some(mt) = memory_type {
                                if row.memory_type != mt { return None; }
                            }
                            if let Some((start, end)) = time_window {
                                if row.created_at < start || row.created_at > end { return None; }
                            }
                            if let Some(ns) = namespace {
                                if row.namespace != ns { return None; }
                            }
                            let prox = gi.graph_proximity(&rid, &expanded_map);
                            let rank = row.importance * (0.3 + 0.7 * prox);
                            Some((rid, rank))
                        })
                        .collect();
                    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    candidates.into_iter().take(preselect_pool).map(|(rid, _)| rid).collect()
                };

                if !filtered_rids.is_empty() {
                    // Batch fetch embeddings for cosine similarity
                    let rid_refs: Vec<&str> = filtered_rids.iter().map(|r| r.as_str()).collect();
                    let embeddings = self.fetch_embeddings_by_rids(&rid_refs)?;

                    let cache = self.scoring_cache.read().unwrap();
                    for rid in &filtered_rids {
                        let Some(row) = cache.get(rid) else { continue };
                        let Some(emb_blob_row) = embeddings.get(rid) else { continue };

                        let mem_embedding = crate::serde_helpers::deserialize_f32(emb_blob_row);
                        let sim_score = crate::consolidate::cosine_similarity(query_embedding, &mem_embedding) as f64;

                        let elapsed = ts - row.last_access;
                        let decay = scoring::decay_score(row.importance, row.half_life, elapsed);
                        let age = ts - row.created_at;
                        let recency = scoring::recency_score(age);

                        let prox = gi.graph_proximity(rid, &expanded_map);
                        let composite = scoring::adaptive_graph_composite_score(
                            sim_score, decay, recency, row.importance, row.valence, prox, query_sentiment, &learned_weights,
                        );
                        let contributions = scoring::adaptive_graph_contributions(sim_score, decay, recency, row.importance, prox, &learned_weights);
                        let valence_multiplier = scoring::query_valence_boost(row.valence, query_sentiment);

                        let mut why = scoring::build_why(sim_score, recency, decay, row.valence);
                        let mem_entities: Vec<String> = gi.entities_for_memory(rid).into_iter().map(|s| s.to_string()).collect();
                        for entity in &mem_entities {
                            if expanded_map.contains_key(entity) {
                                why.push(format!("graph-connected via {entity}"));
                                break;
                            }
                        }

                        scored.push(RecallResult {
                            rid: rid.clone(),
                            memory_type: row.memory_type.clone(),
                            text: String::new(),
                            created_at: row.created_at,
                            importance: row.importance,
                            valence: row.valence,
                            score: composite,
                            scores: ScoreBreakdown {
                                similarity: sim_score,
                                decay,
                                recency,
                                importance: row.importance,
                                graph_proximity: prox,
                                contributions,
                                valence_multiplier,
                            },
                            why_retrieved: why,
                            metadata: serde_json::Value::Null,
                            namespace: row.namespace.clone(),
                            certainty: row.certainty,
                            domain: row.domain.clone(),
                            source: row.source.clone(),
                            emotional_state: row.emotional_state.clone(),
                        });
                    }
                    drop(cache);
                }
            }
        }

        // Step 3.5: Keyword slot reservation
        //
        // Topic-relevant keyword matches (e.g., "yoga", "reading") often have
        // moderate importance (0.30-0.40) that can't compete with high-importance
        // memories (0.70-1.00) in the composite score formula. Reserve up to 3
        // top_k slots for the best keyword-matched candidates (by similarity).
        //
        // The boost is minimal (just above cutoff). The real benefit comes from
        // step 4 where keyword_reserved items are exempt from MMR diversity
        // penalty, guaranteeing they survive into the final results.
        {
            scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

            let cutoff_idx = top_k.min(scored.len()).saturating_sub(1);
            let cutoff_score = scored.get(cutoff_idx).map(|r| r.score).unwrap_or(0.0);

            const KEYWORD_RESERVE_SLOTS: usize = 3;
            const KEYWORD_RESERVE_MIN_SIM: f64 = 0.25;
            // Find keyword-matched candidates ranked below cutoff, sorted by similarity
            let mut kw_below: Vec<(usize, f64)> = scored.iter().enumerate()
                .filter(|(_, r)| {
                    r.why_retrieved.iter().any(|w| w == "keyword_match")
                        && r.scores.similarity >= KEYWORD_RESERVE_MIN_SIM
                        && r.score < cutoff_score
                })
                .map(|(i, r)| (i, r.scores.similarity))
                .collect();
            kw_below.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Boost best keyword candidates just above the cutoff
            for (idx, _) in kw_below.into_iter().take(KEYWORD_RESERVE_SLOTS) {
                scored[idx].score = cutoff_score + 0.001;
                scored[idx].why_retrieved.push("keyword_reserved".to_string());
            }
        }

        // Step 4: MMR diversity selection
        //
        // Without diversity filtering, near-duplicate generated memories (e.g.,
        // "Arjun cooked X today" x50) dominate all K result slots. MMR ensures
        // each result adds new information by penalizing candidates too similar
        // to already-selected results.
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        let min_pool_for_mmr = (top_k * 3).max(20);
        if scored.len() > top_k && scored.len() >= min_pool_for_mmr {
            // Fetch embeddings for top candidates to compute pairwise similarity
            let pool_size = scored.len().min(top_k * 10);
            scored.truncate(pool_size);

            let pool_rids: Vec<&str> = scored.iter().map(|r| r.rid.as_str()).collect();
            let emb_map = self.fetch_embeddings_by_rids(&pool_rids)?;

            // Parse embeddings for each candidate
            let pool_embeddings: Vec<Option<Vec<f32>>> = scored.iter().map(|r| {
                emb_map.get(r.rid.as_str())
                    .map(|blob| crate::serde_helpers::deserialize_f32(blob))
            }).collect();

            // Greedy MMR: λ * relevance - (1-λ) * max_sim_to_selected
            const LAMBDA: f64 = 0.7;
            const SIM_THRESHOLD: f64 = 0.98; // skip only near-exact duplicates

            let mut selected: Vec<usize> = Vec::with_capacity(top_k);
            let mut selected_embeddings: Vec<&[f32]> = Vec::with_capacity(top_k);

            // Always pick the top-scored candidate first
            if !scored.is_empty() {
                selected.push(0);
                if let Some(Some(ref emb)) = pool_embeddings.first() {
                    selected_embeddings.push(emb);
                }
            }

            // Greedily select remaining candidates
            for _ in 1..top_k {
                let mut best_idx = None;
                let mut best_mmr = f64::NEG_INFINITY;

                for (idx, result) in scored.iter().enumerate() {
                    if selected.contains(&idx) { continue; }

                    let relevance = result.score;
                    let max_sim = if let Some(Some(ref cand_emb)) = pool_embeddings.get(idx) {
                        selected_embeddings.iter()
                            .map(|sel_emb| crate::consolidate::cosine_similarity(cand_emb, sel_emb) as f64)
                            .fold(0.0f64, f64::max)
                    } else {
                        0.0
                    };

                    // Skip near-duplicates entirely
                    if max_sim > SIM_THRESHOLD { continue; }

                    let mmr = LAMBDA * relevance - (1.0 - LAMBDA) * max_sim;
                    if mmr > best_mmr {
                        best_mmr = mmr;
                        best_idx = Some(idx);
                    }
                }

                match best_idx {
                    Some(idx) => {
                        selected.push(idx);
                        if let Some(Some(ref emb)) = pool_embeddings.get(idx) {
                            selected_embeddings.push(emb);
                        }
                    }
                    None => break, // No more candidates pass the threshold
                }
            }

            // Rebuild scored from selected indices, preserving order
            let mut diverse_results = Vec::with_capacity(selected.len());
            for i in selected {
                diverse_results.push(scored[i].clone());
            }
            scored = diverse_results;
        } else {
            scored.truncate(top_k);
        }

        // Step 5: Hydrate final top_k with text + metadata from SQLite
        let final_rids: Vec<&str> = scored.iter().map(|r| r.rid.as_str()).collect();
        let text_meta = {
            let _span = tracing::debug_span!("hydrate", count = final_rids.len()).entered();
            self.fetch_text_metadata_by_rids(&final_rids)?
        };
        for result in &mut scored {
            if let Some(tm) = text_meta.get(&result.rid) {
                result.text = tm.text.clone();
                result.metadata = serde_json::from_str(&tm.metadata)
                    .unwrap_or(serde_json::Value::Object(Default::default()));
            }
        }

        // Reinforce accessed memories (spaced repetition)
        if !skip_reinforce {
            for r in &scored {
                self.reinforce(&r.rid)?;
            }
        }

        Ok(scored)
    }

    /// Execute a recall query built with the `RecallQuery` builder.
    ///
    /// ```rust,ignore
    /// let results = db.query(embedding)
    ///     .top_k(10)
    ///     .memory_type("episodic")
    ///     .namespace("work")
    ///     .execute(&db)?;
    /// ```
    pub fn query(&self, q: RecallQuery) -> Result<Vec<RecallResult>> {
        self.recall(
            &q.embedding,
            q.top_k,
            q.time_window,
            q.memory_type.as_deref(),
            q.include_consolidated,
            q.expand_entities,
            q.query_text.as_deref(),
            q.skip_reinforce,
            q.namespace.as_deref(),
            q.domain.as_deref(),
            q.source.as_deref(),
        )
    }

    /// Recall with full response including confidence scoring and refinement hints.
    ///
    /// Wraps `recall()` and computes a `RecallResponse` with confidence level,
    /// retrieval summary, and hints for the calling agent to refine queries.
    pub fn recall_with_response(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
        expand_entities: bool,
        query_text: Option<&str>,
        skip_reinforce: bool,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
    ) -> Result<RecallResponse> {
        let results = self.recall(
            query_embedding, top_k, time_window, memory_type,
            include_consolidated, expand_entities, query_text,
            skip_reinforce, namespace, domain, source,
        )?;

        // Determine which retrieval sources were used
        let mut sources_used = vec!["hnsw".to_string()];
        if query_text.is_some() && !self.is_encrypted() {
            sources_used.push("fts5".to_string());
        }
        if expand_entities {
            sources_used.push("graph".to_string());
        }
        let query_sentiment = query_text
            .map(crate::scoring::detect_query_sentiment)
            .unwrap_or(0.0);
        if query_sentiment.abs() > 0.01 {
            sources_used.push("valence".to_string());
        }

        // Candidate count: approximate from scoring cache filtered by same criteria
        let candidate_count = {
            let cache = self.scoring_cache.read().unwrap();
            cache.values().filter(|row| {
                let status_ok = if include_consolidated {
                    row.consolidation_status == "active" || row.consolidation_status == "consolidated"
                } else {
                    row.consolidation_status == "active"
                };
                status_ok
                    && memory_type.map_or(true, |mt| row.memory_type == mt)
                    && namespace.map_or(true, |ns| row.namespace == ns)
                    && domain.map_or(true, |d| row.domain == d)
                    && source.map_or(true, |s| row.source == s)
            }).count()
        };

        // Compute retrieval summary
        let top_similarity = results.first()
            .map(|r| r.scores.similarity)
            .unwrap_or(0.0);
        let score_spread = if results.len() >= 2 {
            results.first().unwrap().score - results.last().unwrap().score
        } else {
            0.0
        };

        let summary = RetrievalSummary {
            top_similarity,
            score_spread,
            sources_used: sources_used.clone(),
            candidate_count,
        };

        // Compute confidence from 4 signals with detailed reasons
        let signal_sim = top_similarity;
        let signal_gap = if results.len() >= 3 {
            results[0].score - results[2].score
        } else if results.len() == 2 {
            results[0].score - results[1].score
        } else {
            0.0
        };
        let signal_diversity = sources_used.len() as f64 / 4.0; // max 4 sources
        let signal_density = (results.len() as f64 / top_k as f64).min(1.0);

        let confidence = (0.35 * signal_sim + 0.25 * signal_gap + 0.20 * signal_diversity + 0.20 * signal_density)
            .clamp(0.0, 1.0);

        // Build certainty reasons explaining the confidence score
        let mut certainty_reasons = Vec::new();
        if signal_sim >= 0.7 {
            certainty_reasons.push(format!(
                "Strong semantic match (top similarity: {:.0}%)", signal_sim * 100.0
            ));
        } else if signal_sim >= 0.4 {
            certainty_reasons.push(format!(
                "Moderate semantic match (top similarity: {:.0}%)", signal_sim * 100.0
            ));
        } else if signal_sim > 0.0 {
            certainty_reasons.push(format!(
                "Weak semantic match (top similarity: {:.0}%) — query may be outside stored knowledge",
                signal_sim * 100.0
            ));
        } else {
            certainty_reasons.push("No matching memories found".to_string());
        }

        if results.is_empty() {
            certainty_reasons.push(format!(
                "No results from {} candidates", candidate_count
            ));
        } else if signal_density < 0.5 {
            certainty_reasons.push(format!(
                "Sparse results: only {}/{} slots filled", results.len(), top_k
            ));
        }

        if signal_gap > 0.3 {
            certainty_reasons.push(
                "Clear winner: top result stands out from the rest".to_string()
            );
        } else if signal_gap < 0.05 && results.len() >= 2 {
            certainty_reasons.push(
                "Ambiguous: multiple results scored similarly — consider refining query".to_string()
            );
        }

        if sources_used.contains(&"graph".to_string()) {
            certainty_reasons.push(
                "Graph expansion contributed entity-linked memories".to_string()
            );
        }

        // Check for stale results (last accessed > 30 days ago)
        let ts = now();
        let stale_count = results.iter().filter(|r| {
            // Use created_at as a proxy since we don't have last_access in RecallResult
            ts - r.created_at > 30.0 * 86400.0
        }).count();
        if stale_count > results.len() / 2 && !results.is_empty() {
            certainty_reasons.push(format!(
                "Note: {}/{} results are older than 30 days — information may be outdated",
                stale_count, results.len()
            ));
        }

        // Check for low-certainty memories in results
        let low_certainty_count = results.iter().filter(|r| r.certainty < 0.5).count();
        if low_certainty_count > 0 {
            certainty_reasons.push(format!(
                "{}/{} results have low memory certainty (<50%) — treat with caution",
                low_certainty_count, results.len()
            ));
        }

        // Generate hints when confidence < 0.60
        let hints = if confidence < 0.60 {
            self.generate_hints(query_text, query_embedding, &results, &summary)
        } else {
            vec![]
        };

        Ok(RecallResponse {
            results,
            confidence,
            certainty_reasons,
            retrieval_summary: summary,
            hints,
        })
    }

    /// Generate refinement hints based on the recall results and query context.
    fn generate_hints(
        &self,
        query_text: Option<&str>,
        _query_embedding: &[f32],
        results: &[RecallResult],
        summary: &RetrievalSummary,
    ) -> Vec<RefinementHint> {
        let mut hints = Vec::new();

        // Hint 1: Specificity — if query is very short, suggest adding detail
        if let Some(qt) = query_text {
            let word_count = qt.split_whitespace().count();
            if word_count <= 3 {
                hints.push(RefinementHint {
                    hint_type: "specificity".to_string(),
                    suggestion: "Try adding more context — who, when, or where?".to_string(),
                    related_entities: vec![],
                });
            }
        }

        // Hint 2: Entity hints — find entities in the graph near the query
        if let Some(qt) = query_text {
            let query_tokens = crate::graph::tokenize(qt);
            let gi = self.graph_index.read().unwrap();
            let matched = gi.entity_matches_query(&query_tokens);
            // Suggest entities that matched the query but whose memories aren't in results
            let result_rids: std::collections::HashSet<&str> = results.iter().map(|r| r.rid.as_str()).collect();
            let mut entity_suggestions = Vec::new();
            for (name, _etype, _score) in &matched {
                // Find memories linked to this entity
                let linked = gi.memories_for_entities(&[name.as_str()]);
                let has_result = linked.iter().any(|rid| result_rids.contains(rid.as_str()));
                if !has_result && entity_suggestions.len() < 5 {
                    entity_suggestions.push(name.clone());
                }
            }
            if !entity_suggestions.is_empty() {
                hints.push(RefinementHint {
                    hint_type: "entity".to_string(),
                    suggestion: format!(
                        "Related entities found but not in results: {}. Try mentioning them.",
                        entity_suggestions.join(", ")
                    ),
                    related_entities: entity_suggestions,
                });
            }
        }

        // Hint 3: Time range — if results span a wide time range
        if results.len() >= 2 {
            let min_ts = results.iter().map(|r| r.created_at).fold(f64::INFINITY, f64::min);
            let max_ts = results.iter().map(|r| r.created_at).fold(f64::NEG_INFINITY, f64::max);
            let span_days = (max_ts - min_ts) / 86400.0;
            if span_days > 30.0 {
                hints.push(RefinementHint {
                    hint_type: "time_range".to_string(),
                    suggestion: "Results span a wide time range. Try specifying a time period.".to_string(),
                    related_entities: vec![],
                });
            }
        }

        // Hint 4: Low similarity — if even the best result has low similarity
        if summary.top_similarity < 0.25 {
            hints.push(RefinementHint {
                hint_type: "keyword".to_string(),
                suggestion: "The query may use different words than the stored memories. Try rephrasing with synonyms or related terms.".to_string(),
                related_entities: vec![],
            });
        }

        // Hint 5: Domain diversity — if all results from one domain but DB has others
        if results.len() >= 3 {
            let result_domains: std::collections::HashSet<&str> =
                results.iter().map(|r| r.domain.as_str()).collect();
            if result_domains.len() == 1 {
                // Check if other domains exist in the DB
                let cache = self.scoring_cache.read().unwrap();
                let all_domains: std::collections::HashSet<&str> =
                    cache.values().map(|r| r.domain.as_str()).collect();
                let other_domains: Vec<&str> = all_domains
                    .difference(&result_domains)
                    .filter(|d| **d != "general")
                    .copied()
                    .take(5)
                    .collect();
                if !other_domains.is_empty() {
                    hints.push(RefinementHint {
                        hint_type: "domain".to_string(),
                        suggestion: format!(
                            "Results only from '{}' domain. Other domains available: {}. \
                             Consider cross-domain search if relevant.",
                            result_domains.iter().next().unwrap_or(&"unknown"),
                            other_domains.join(", ")
                        ),
                        related_entities: vec![],
                    });
                }
            }
        }

        // Hint 6: Procedural memory available — if query looks like a "how to" question
        if let Some(qt) = query_text {
            let qt_lower = qt.to_lowercase();
            let is_procedural_query = qt_lower.starts_with("how ")
                || qt_lower.contains("how do")
                || qt_lower.contains("how to")
                || qt_lower.contains("best way")
                || qt_lower.contains("approach")
                || qt_lower.contains("strategy");
            if is_procedural_query {
                let has_procedural = results.iter().any(|r| r.memory_type == "procedural");
                if !has_procedural {
                    // Check if procedural memories exist at all
                    let cache = self.scoring_cache.read().unwrap();
                    let procedural_count = cache.values()
                        .filter(|r| r.memory_type == "procedural")
                        .count();
                    if procedural_count > 0 {
                        hints.push(RefinementHint {
                            hint_type: "memory_type".to_string(),
                            suggestion: format!(
                                "{} procedural memories exist but weren't retrieved. \
                                 Try filtering by memory_type='procedural'.",
                                procedural_count
                            ),
                            related_entities: vec![],
                        });
                    }
                }
            }
        }

        hints
    }

    /// Refine a previous recall by combining original + refinement embeddings.
    ///
    /// The AI agent calls this after receiving hints from `recall_with_response()`.
    /// It combines the original query embedding with a refinement text embedding
    /// (weighted: 0.4 original + 0.6 refinement) and excludes already-seen RIDs.
    pub fn recall_refine(
        &self,
        original_query_embedding: &[f32],
        refinement_embedding: &[f32],
        original_rids: &[String],
        top_k: usize,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
    ) -> Result<RecallResponse> {
        // Combine embeddings: 0.4 * original + 0.6 * refinement
        let dim = original_query_embedding.len().min(refinement_embedding.len());
        let mut combined = vec![0.0f32; dim];
        for i in 0..dim {
            combined[i] = 0.4 * original_query_embedding[i] + 0.6 * refinement_embedding[i];
        }
        // Normalize
        let norm: f32 = combined.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for v in &mut combined {
                *v /= norm;
            }
        }

        // Run recall with combined embedding
        let mut response = self.recall_with_response(
            &combined, top_k, None, None,
            false, false, None, true,  // skip_reinforce=true for refinement
            namespace, domain, source,
        )?;

        // Exclude already-seen RIDs
        let exclude: std::collections::HashSet<&str> = original_rids.iter().map(|s| s.as_str()).collect();
        response.results.retain(|r| !exclude.contains(r.rid.as_str()));

        Ok(response)
    }

    /// Profiled version of recall() that returns per-phase timing breakdown.
    ///
    /// Mirrors `recall()` exactly but wraps each phase in timing instrumentation.
    #[cfg(feature = "profiling")]
    pub fn recall_profiled(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
        expand_entities: bool,
        query_text: Option<&str>,
        skip_reinforce: bool,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
    ) -> Result<RecallProfiledResult> {
        use std::time::Instant;
        let t_start = Instant::now();

        let learned_weights = self.load_learned_weights()?;

        let query_sentiment = query_text
            .map(scoring::detect_query_sentiment)
            .unwrap_or(0.0);

        // ── Phase 1: Vector search (HNSW) ──
        let t_vec = Instant::now();
        let ts = now();
        let fetch_k = (top_k * 20).min(500);
        let vec_results = self.vec_index.read().unwrap().search(query_embedding, fetch_k)?;
        let vec_search_ms = t_vec.elapsed().as_secs_f64() * 1000.0;
        let candidate_count = vec_results.len();

        if vec_results.is_empty() {
            return Ok(RecallProfiledResult {
                results: vec![],
                timings: RecallTimings {
                    vec_search_ms,
                    cache_score_ms: 0.0,
                    fetch_ms: 0.0,
                    scoring_ms: 0.0,
                    graph_ms: 0.0,
                    reinforce_ms: 0.0,
                    sort_truncate_ms: 0.0,
                    total_ms: t_start.elapsed().as_secs_f64() * 1000.0,
                    candidate_count: 0,
                    graph_expansion_count: 0,
                },
            });
        }

        // ── Phase 2: Score from in-memory cache ──
        let t_cache_score = Instant::now();
        let mut scored: Vec<RecallResult> = Vec::new();
        {
            let cache = self.scoring_cache.read().unwrap();
            for (rid, distance) in &vec_results {
                let Some(row) = cache.get(rid) else { continue };

                let status_ok = if include_consolidated {
                    row.consolidation_status == "active" || row.consolidation_status == "consolidated"
                } else {
                    row.consolidation_status == "active"
                };
                if !status_ok { continue; }
                if let Some(mt) = memory_type {
                    if row.memory_type != mt { continue; }
                }
                if let Some((start, end)) = time_window {
                    if row.created_at < start || row.created_at > end { continue; }
                }
                if let Some(ns) = namespace {
                    if row.namespace != ns { continue; }
                }
                if let Some(d) = domain {
                    if row.domain != d { continue; }
                }
                if let Some(s) = source {
                    if row.source != s { continue; }
                }

                let sim_score = (1.0 - distance).max(0.0);
                let elapsed = ts - row.last_access;
                let decay = scoring::decay_score(row.importance, row.half_life, elapsed);
                let age = ts - row.created_at;
                let recency = scoring::recency_score(age);
                let composite = scoring::adaptive_composite_score(sim_score, decay, recency, row.importance, row.valence, query_sentiment, &learned_weights);
                let why = scoring::build_why(sim_score, recency, decay, row.valence);
                let contributions = scoring::adaptive_contributions(sim_score, decay, recency, row.importance, &learned_weights);
                let valence_multiplier = scoring::query_valence_boost(row.valence, query_sentiment);

                scored.push(RecallResult {
                    rid: rid.clone(),
                    memory_type: row.memory_type.clone(),
                    text: String::new(),
                    created_at: row.created_at,
                    importance: row.importance,
                    valence: row.valence,
                    score: composite,
                    scores: ScoreBreakdown {
                        similarity: sim_score,
                        decay,
                        recency,
                        importance: row.importance,
                        graph_proximity: 0.0,
                        contributions,
                        valence_multiplier,
                    },
                    why_retrieved: why,
                    metadata: serde_json::Value::Null,
                    namespace: row.namespace.clone(),
                    certainty: row.certainty,
                    domain: row.domain.clone(),
                    source: row.source.clone(),
                    emotional_state: row.emotional_state.clone(),
                });
            }
        }
        let cache_score_ms = t_cache_score.elapsed().as_secs_f64() * 1000.0;

        // ── Phase 2.5: High-importance memory fallback (similarity-gated) ──
        let t_fallback = Instant::now();
        {
            let total_memories = self.scoring_cache.read().unwrap().len();
            let high_imp_threshold = if total_memories > 5000 { 0.5 } else { 0.7 };
            let min_sim_for_fallback = if total_memories > 5000 { 0.15 } else { 0.20 };
            let existing_rids: std::collections::HashSet<&str> =
                scored.iter().map(|r| r.rid.as_str()).collect();
            let important_rids: Vec<String> = {
                let cache = self.scoring_cache.read().unwrap();
                cache
                    .iter()
                    .filter(|(rid, row)| {
                        row.importance >= high_imp_threshold
                            && row.consolidation_status == "active"
                            && !existing_rids.contains(rid.as_str())
                            && memory_type.map_or(true, |mt| row.memory_type == mt)
                            && time_window.map_or(true, |(s, e)| {
                                row.created_at >= s && row.created_at <= e
                            })
                            && namespace.map_or(true, |ns| row.namespace == ns)
                            && domain.map_or(true, |d| row.domain == d)
                            && source.map_or(true, |s| row.source == s)
                    })
                    .map(|(rid, _)| rid.clone())
                    .collect()
            };

            if !important_rids.is_empty() {
                let rid_refs: Vec<&str> = important_rids.iter().map(|r| r.as_str()).collect();
                let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;
                let cache = self.scoring_cache.read().unwrap();
                for rid in &important_rids {
                    let Some(row) = cache.get(rid) else { continue };
                    let Some(emb_blob) = emb_map.get(rid.as_str()) else { continue };
                    let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
                    let sim_score =
                        crate::consolidate::cosine_similarity(query_embedding, &mem_emb) as f64;
                    if sim_score < min_sim_for_fallback { continue; }

                    let elapsed = ts - row.last_access;
                    let decay = scoring::decay_score(row.importance, row.half_life, elapsed);
                    let age = ts - row.created_at;
                    let recency = scoring::recency_score(age);
                    let composite = scoring::adaptive_composite_score(
                        sim_score, decay, recency, row.importance, row.valence, query_sentiment, &learned_weights,
                    );
                    let why = scoring::build_why(sim_score, recency, decay, row.valence);
                    let contributions =
                        scoring::adaptive_contributions(sim_score, decay, recency, row.importance, &learned_weights);
                    let valence_multiplier = scoring::query_valence_boost(row.valence, query_sentiment);

                    scored.push(RecallResult {
                        rid: rid.clone(),
                        memory_type: row.memory_type.clone(),
                        text: String::new(),
                        created_at: row.created_at,
                        importance: row.importance,
                        valence: row.valence,
                        score: composite,
                        scores: ScoreBreakdown {
                            similarity: sim_score,
                            decay,
                            recency,
                            importance: row.importance,
                            graph_proximity: 0.0,
                            contributions,
                            valence_multiplier,
                        },
                        why_retrieved: why,
                        metadata: serde_json::Value::Null,
                        namespace: row.namespace.clone(),
                        certainty: row.certainty,
                        domain: row.domain.clone(),
                        source: row.source.clone(),
                        emotional_state: row.emotional_state.clone(),
                    });
                }
            }
        }
        let fallback_ms = t_fallback.elapsed().as_secs_f64() * 1000.0;

        // ── Phase 1.5: FTS5 keyword fallback ──
        // (mirrors recall() Step 1.5 — see comments there for full rationale)
        let t_fts = Instant::now();
        if !self.is_encrypted() {
            const FTS_MIN_SIM: f64 = 0.05;

            const STOPWORDS: &[&str] = &[
                "a", "an", "the", "is", "are", "am", "was", "were", "be", "been",
                "what", "who", "how", "when", "where", "which", "why",
                "do", "did", "does", "have", "has", "had",
                "i", "me", "my", "mine", "we", "our", "you", "your",
                "to", "of", "in", "on", "at", "by", "for", "with", "from",
                "about", "tell", "and", "or", "but", "not", "no",
                "it", "its", "that", "this", "there", "s",
                "she", "her", "he", "his", "they", "them",
                "most", "each", "any", "all", "every", "been", "being",
                "up", "out", "so", "if", "than", "very", "just", "also",
            ];

            {
                if let Some(qt) = query_text {
                    let raw_keywords: Vec<String> = qt
                        .split(|c: char| !c.is_alphanumeric())
                        .filter(|s| !s.is_empty() && s.len() > 1)
                        .filter(|s| !STOPWORDS.contains(&s.to_lowercase().as_str()))
                        .map(|s| s.to_string())
                        .collect();

                    // (mirrors recall() — see comments there for full rationale)
                    let mut keywords: Vec<String> = {
                        let gi = self.graph_index.read().unwrap();
                        let query_tokens = crate::graph::tokenize(qt);
                        let matched = gi.entity_matches_query(&query_tokens);
                        let person_matches: Vec<_> = matched
                            .into_iter()
                            .filter(|(_, etype, _)| etype == "person")
                            .collect();
                        if person_matches.is_empty() {
                            raw_keywords
                        } else {
                            let person_tokens: std::collections::HashSet<String> = person_matches
                                .into_iter()
                                .flat_map(|(name, _, _)| {
                                    let mut tokens = vec![name.to_lowercase()];
                                    for token in name.split_whitespace() {
                                        tokens.push(token.to_lowercase());
                                    }
                                    tokens
                                })
                                .collect();
                            let filtered: Vec<String> = raw_keywords
                                .iter()
                                .filter(|kw| !person_tokens.contains(&kw.to_lowercase()))
                                .cloned()
                                .collect();
                            if filtered.is_empty() { raw_keywords } else { filtered }
                        }
                    };

                    // Entity-seeded FTS for group/aggregation queries (mirrors recall())
                    {
                        const GROUP_FTS_WORDS: &[&str] = &[
                            "team", "group", "colleagues", "coworkers", "friends",
                            "family", "staff", "members", "people",
                        ];
                        let qt_lower = qt.to_lowercase();
                        if GROUP_FTS_WORDS.iter().any(|kw| qt_lower.contains(kw)) {
                            let gi = self.graph_index.read().unwrap();
                            let query_tokens = crate::graph::tokenize(qt);
                            let matched = gi.entity_matches_query(&query_tokens);
                            if !matched.is_empty() {
                                let seed_names: Vec<&str> =
                                    matched.iter().map(|(n, _, _)| n.as_str()).collect();
                                let expanded = gi.expand_bfs(&seed_names, 2, 30);
                                let mut injected = 0usize;
                                for (name, hops, _) in &expanded {
                                    if *hops == 0 || injected >= 15 {
                                        continue;
                                    }
                                    if gi.entity_type(name).map_or(false, |t| t == "person") {
                                        for part in name.split_whitespace() {
                                            if part.len() > 1
                                                && !keywords
                                                    .iter()
                                                    .any(|k| k.eq_ignore_ascii_case(part))
                                            {
                                                keywords.push(part.to_string());
                                            }
                                        }
                                        injected += 1;
                                    }
                                }
                            }
                        }
                    }

                    if !keywords.is_empty() {
                        // Build FTS5 query with AND conjunction (mirrors recall())
                        let mut keyword_groups: Vec<String> = Vec::new();
                        for kw in &keywords {
                            let kw_lower = kw.to_lowercase();
                            let mut parts: Vec<String> = Vec::new();
                            parts.push(format!("\"{}\"", kw.replace('"', "")));
                            if let Some(stem) = simple_stem(&kw_lower) {
                                parts.push(format!("{}*", stem));
                            }
                            if let Some(alts) = irregular_verb_forms(&kw_lower) {
                                for alt in alts {
                                    parts.push(format!("\"{}\"", alt));
                                }
                            }
                            keyword_groups.push(if parts.len() == 1 {
                                parts[0].clone()
                            } else {
                                format!("({})", parts.join(" OR "))
                            });
                        }
                        let fts_query_and = if keyword_groups.len() >= 2 {
                            Some(keyword_groups.join(" AND "))
                        } else {
                            None
                        };
                        let fts_query_or = keyword_groups.join(" OR ");
                        let fts_query = fts_query_and.as_deref()
                            .unwrap_or(&fts_query_or);

                        let total_memories = self.scoring_cache.read().unwrap().len();
                        let fts_limit = (total_memories / 100).max(30).min(200);

                        let mean_importance = {
                            let cache = self.scoring_cache.read().unwrap();
                            if cache.is_empty() {
                                0.5
                            } else {
                                let sum: f64 = cache.values().map(|r| r.importance).sum();
                                let mean = sum / cache.len() as f64;
                                (mean * 0.7).max(0.25)
                            }
                        };

                        let fts_sql = if memory_type.is_some() {
                            format!(
                                "SELECT m.rid FROM memories m \
                                 JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                 WHERE memories_fts MATCH ?1 \
                                 AND m.consolidation_status = 'active' \
                                 AND m.type = ?2 \
                                 {} \
                                 ORDER BY rank * (0.5 + m.importance) \
                                 LIMIT {}",
                                if namespace.is_some() { "AND m.namespace = ?3" } else { "" },
                                fts_limit,
                            )
                        } else {
                            format!(
                                "SELECT m.rid FROM memories m \
                                 JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                 WHERE memories_fts MATCH ?1 \
                                 AND m.consolidation_status = 'active' \
                                 {} \
                                 ORDER BY rank * (0.5 + m.importance) \
                                 LIMIT {}",
                                if namespace.is_some() { "AND m.namespace = ?2" } else { "" },
                                fts_limit,
                            )
                        };

                        let run_fts_phase1 = |q: &str| -> Vec<String> {
                            let conn = self.conn.lock().unwrap();
                            let mut stmt = conn.prepare_cached(&fts_sql).ok();
                            if let Some(ref mut stmt) = stmt {
                                let result: std::result::Result<Vec<String>, _> = if let Some(mt) = memory_type {
                                    if let Some(ns) = namespace {
                                        stmt.query_map(
                                            params![q, mt, ns],
                                            |row| row.get::<_, String>(0),
                                        ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    } else {
                                        stmt.query_map(
                                            params![q, mt],
                                            |row| row.get::<_, String>(0),
                                        ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    }
                                } else if let Some(ns) = namespace {
                                    stmt.query_map(
                                        params![q, ns],
                                        |row| row.get::<_, String>(0),
                                    ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                } else {
                                    stmt.query_map(
                                        params![q],
                                        |row| row.get::<_, String>(0),
                                    ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                };
                                result.unwrap_or_default()
                            } else {
                                vec![]
                            }
                        };

                        let mut fts_rids = run_fts_phase1(fts_query);
                        if fts_rids.len() < 5 && fts_query_and.is_some() {
                            fts_rids = run_fts_phase1(&fts_query_or);
                        }

                        // Phase 2: Importance-filtered FTS (mirrors recall())
                        // Uses AND first, falls back to OR when AND is too strict.
                        {
                            let imp_fts_sql = if memory_type.is_some() {
                                format!(
                                    "SELECT m.rid FROM memories m \
                                     JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                     WHERE memories_fts MATCH ?1 \
                                     AND m.consolidation_status = 'active' \
                                     AND m.importance > ?2 \
                                     AND m.type = ?3 \
                                     {} \
                                     ORDER BY m.importance DESC \
                                     LIMIT 100",
                                    if namespace.is_some() { "AND m.namespace = ?4" } else { "" },
                                )
                            } else {
                                format!(
                                    "SELECT m.rid FROM memories m \
                                     JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                     WHERE memories_fts MATCH ?1 \
                                     AND m.consolidation_status = 'active' \
                                     AND m.importance > ?2 \
                                     {} \
                                     ORDER BY m.importance DESC \
                                     LIMIT 100",
                                    if namespace.is_some() { "AND m.namespace = ?3" } else { "" },
                                )
                            };

                            let run_fts_phase2 = |q: &str| -> Vec<String> {
                                let conn = self.conn.lock().unwrap();
                                let mut stmt = conn.prepare_cached(&imp_fts_sql).ok();
                                if let Some(ref mut stmt) = stmt {
                                    let result: std::result::Result<Vec<String>, _> = if let Some(mt) = memory_type {
                                        if let Some(ns) = namespace {
                                            stmt.query_map(
                                                params![q, mean_importance, mt, ns],
                                                |row| row.get::<_, String>(0),
                                            ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                        } else {
                                            stmt.query_map(
                                                params![q, mean_importance, mt],
                                                |row| row.get::<_, String>(0),
                                            ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                        }
                                    } else if let Some(ns) = namespace {
                                        stmt.query_map(
                                            params![q, mean_importance, ns],
                                            |row| row.get::<_, String>(0),
                                        ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    } else {
                                        stmt.query_map(
                                            params![q, mean_importance],
                                            |row| row.get::<_, String>(0),
                                        ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    };
                                    result.unwrap_or_default()
                                } else {
                                    vec![]
                                }
                            };

                            // Run AND first, fall back to OR if too few results.
                            let mut imp_rids = run_fts_phase2(fts_query);
                            if imp_rids.len() < 10 && fts_query_and.is_some() {
                                let or_rids = run_fts_phase2(&fts_query_or);
                                let existing: std::collections::HashSet<String> =
                                    imp_rids.iter().cloned().collect();
                                imp_rids.extend(
                                    or_rids.into_iter().filter(|r| !existing.contains(r))
                                );
                            }

                            let existing_set: std::collections::HashSet<&str> =
                                fts_rids.iter().map(|r| r.as_str()).collect();
                            for rid in imp_rids {
                                if !existing_set.contains(rid.as_str()) {
                                    fts_rids.push(rid);
                                }
                            }
                        }

                        // Phase 2.5: Per-keyword anchor scan (mirrors recall())
                        if keyword_groups.len() >= 2 {
                            let anchor_fts_sql = if memory_type.is_some() {
                                format!(
                                    "SELECT m.rid FROM memories m \
                                     JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                     WHERE memories_fts MATCH ?1 \
                                     AND m.consolidation_status = 'active' \
                                     AND m.importance > 0.5 \
                                     AND m.type = ?2 \
                                     {} \
                                     ORDER BY m.importance DESC \
                                     LIMIT 10",
                                    if namespace.is_some() { "AND m.namespace = ?3" } else { "" },
                                )
                            } else {
                                format!(
                                    "SELECT m.rid FROM memories m \
                                     JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                     WHERE memories_fts MATCH ?1 \
                                     AND m.consolidation_status = 'active' \
                                     AND m.importance > 0.5 \
                                     {} \
                                     ORDER BY m.importance DESC \
                                     LIMIT 10",
                                    if namespace.is_some() { "AND m.namespace = ?2" } else { "" },
                                )
                            };

                            let existing_fts: std::collections::HashSet<String> =
                                fts_rids.iter().cloned().collect();

                            for group in &keyword_groups {
                                let anchor_rids: Vec<String> = {
                                    let conn = self.conn.lock().unwrap();
                                    let mut stmt = conn.prepare_cached(&anchor_fts_sql).ok();
                                    if let Some(ref mut stmt) = stmt {
                                        let result: std::result::Result<Vec<String>, _> = if let Some(mt) = memory_type {
                                            if let Some(ns) = namespace {
                                                stmt.query_map(
                                                    params![group, mt, ns],
                                                    |row| row.get::<_, String>(0),
                                                ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                            } else {
                                                stmt.query_map(
                                                    params![group, mt],
                                                    |row| row.get::<_, String>(0),
                                                ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                            }
                                        } else if let Some(ns) = namespace {
                                            stmt.query_map(
                                                params![group, ns],
                                                |row| row.get::<_, String>(0),
                                            ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                        } else {
                                            stmt.query_map(
                                                params![group],
                                                |row| row.get::<_, String>(0),
                                            ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                        };
                                        result.unwrap_or_default()
                                    } else {
                                        vec![]
                                    }
                                };

                                for rid in anchor_rids {
                                    if !existing_fts.contains(&rid) {
                                        fts_rids.push(rid);
                                    }
                                }
                            }
                        }

                        {
                            let fts_rid_set: std::collections::HashSet<&str> =
                                fts_rids.iter().map(|r| r.as_str()).collect();
                            for result in &mut scored {
                                if fts_rid_set.contains(result.rid.as_str())
                                    && !result.why_retrieved.iter().any(|w| w == "keyword_match")
                                {
                                    let sim = result.scores.similarity;
                                    let boost = learned_weights.keyword_boost * (1.0 - sim).max(0.2);
                                    result.score += boost;
                                    result.why_retrieved.push("keyword_match".to_string());
                                }
                            }
                        }

                        let existing_rids: std::collections::HashSet<String> =
                            scored.iter().map(|r| r.rid.clone()).collect();
                        let new_fts_rids: Vec<String> = fts_rids.into_iter()
                            .filter(|r| !existing_rids.contains(r))
                            .collect();

                        if !new_fts_rids.is_empty() {
                            let rid_refs: Vec<&str> = new_fts_rids.iter().map(|r| r.as_str()).collect();
                            let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;

                            let cache = self.scoring_cache.read().unwrap();
                            for rid in &new_fts_rids {
                                let Some(row) = cache.get(rid) else { continue };
                                let status_ok = if include_consolidated {
                                    row.consolidation_status == "active" || row.consolidation_status == "consolidated"
                                } else {
                                    row.consolidation_status == "active"
                                };
                                if !status_ok { continue; }
                                if let Some((start, end)) = time_window {
                                    if row.created_at < start || row.created_at > end { continue; }
                                }

                                let Some(emb_blob) = emb_map.get(rid.as_str()) else { continue };
                                let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
                                let sim_score = crate::consolidate::cosine_similarity(
                                    query_embedding, &mem_emb,
                                ) as f64;

                                if sim_score < FTS_MIN_SIM { continue; }

                                let elapsed = ts - row.last_access;
                                let decay = scoring::decay_score(row.importance, row.half_life, elapsed);
                                let age = ts - row.created_at;
                                let recency = scoring::recency_score(age);
                                let composite = scoring::adaptive_composite_score(
                                    sim_score, decay, recency, row.importance, row.valence, query_sentiment, &learned_weights,
                                );
                                let kw_boost = learned_weights.keyword_boost * (1.0 - sim_score).max(0.2);
                                let mut why = scoring::build_why(sim_score, recency, decay, row.valence);
                                why.push("keyword_match".to_string());
                                why.push("fts_sourced".to_string());
                                let contributions = scoring::adaptive_contributions(
                                    sim_score, decay, recency, row.importance, &learned_weights,
                                );
                                let valence_multiplier = scoring::query_valence_boost(row.valence, query_sentiment);

                                scored.push(RecallResult {
                                    rid: rid.clone(),
                                    memory_type: row.memory_type.clone(),
                                    text: String::new(),
                                    created_at: row.created_at,
                                    importance: row.importance,
                                    valence: row.valence,
                                    score: composite + kw_boost,
                                    scores: ScoreBreakdown {
                                        similarity: sim_score,
                                        decay,
                                        recency,
                                        importance: row.importance,
                                        graph_proximity: 0.0,
                                        contributions,
                                        valence_multiplier,
                                    },
                                    why_retrieved: why,
                                    metadata: serde_json::Value::Null,
                                    namespace: row.namespace.clone(),
                                    certainty: row.certainty,
                                    domain: row.domain.clone(),
                                    source: row.source.clone(),
                                    emotional_state: row.emotional_state.clone(),
                                });
                            }
                        }
                    }
                }
            }
        }
        let fts_ms = t_fts.elapsed().as_secs_f64() * 1000.0;

        // ── Phase 2.7: Valence-based retrieval for emotional queries ──
        // (mirrors recall() Step 2.7)
        if query_sentiment.abs() > 0.5 {
            const VALENCE_SCAN_THRESHOLD: f64 = 0.4;
            const VALENCE_SCAN_MAX: usize = 30;
            const VALENCE_MIN_SIM: f64 = 0.02;

            let existing_rids: std::collections::HashSet<&str> =
                scored.iter().map(|r| r.rid.as_str()).collect();

            let valence_rids: Vec<String> = {
                let cache = self.scoring_cache.read().unwrap();
                let mut candidates: Vec<(String, f64)> = cache
                    .iter()
                    .filter(|(rid, row)| {
                        row.consolidation_status == "active"
                            && !existing_rids.contains(rid.as_str())
                            && row.valence.abs() >= VALENCE_SCAN_THRESHOLD
                            && (query_sentiment * row.valence > 0.0
                                || (query_sentiment < 0.0 && row.valence < -0.2))
                            && row.importance >= 0.5
                            && memory_type.map_or(true, |mt| row.memory_type == mt)
                            && time_window.map_or(true, |(s, e)| {
                                row.created_at >= s && row.created_at <= e
                            })
                            && namespace.map_or(true, |ns| row.namespace == ns)
                    })
                    .map(|(rid, row)| {
                        let rank = row.valence.abs() * row.importance;
                        (rid.clone(), rank)
                    })
                    .collect();
                candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                candidates.into_iter().take(VALENCE_SCAN_MAX).map(|(rid, _)| rid).collect()
            };

            if !valence_rids.is_empty() {
                let rid_refs: Vec<&str> = valence_rids.iter().map(|r| r.as_str()).collect();
                let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;
                let cache = self.scoring_cache.read().unwrap();
                for rid in &valence_rids {
                    let Some(row) = cache.get(rid) else { continue };
                    let Some(emb_blob) = emb_map.get(rid.as_str()) else { continue };
                    let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
                    let sim_score =
                        crate::consolidate::cosine_similarity(query_embedding, &mem_emb) as f64;

                    if sim_score < VALENCE_MIN_SIM { continue; }

                    let elapsed = ts - row.last_access;
                    let decay = scoring::decay_score(row.importance, row.half_life, elapsed);
                    let age = ts - row.created_at;
                    let recency = scoring::recency_score(age);
                    let composite = scoring::adaptive_composite_score(
                        sim_score, decay, recency, row.importance, row.valence, query_sentiment, &learned_weights,
                    );
                    let valence_additive = 0.20 * row.valence.abs() * row.importance;
                    let mut why = scoring::build_why(sim_score, recency, decay, row.valence);
                    why.push("valence_match".to_string());
                    let contributions =
                        scoring::adaptive_contributions(sim_score, decay, recency, row.importance, &learned_weights);
                    let valence_multiplier = scoring::query_valence_boost(row.valence, query_sentiment);

                    scored.push(RecallResult {
                        rid: rid.clone(),
                        memory_type: row.memory_type.clone(),
                        text: String::new(),
                        created_at: row.created_at,
                        importance: row.importance,
                        valence: row.valence,
                        score: composite + valence_additive,
                        scores: ScoreBreakdown {
                            similarity: sim_score,
                            decay,
                            recency,
                            importance: row.importance,
                            graph_proximity: 0.0,
                            contributions,
                            valence_multiplier,
                        },
                        why_retrieved: why,
                        metadata: serde_json::Value::Null,
                        namespace: row.namespace.clone(),
                        certainty: row.certainty,
                        domain: row.domain.clone(),
                        source: row.source.clone(),
                        emotional_state: row.emotional_state.clone(),
                    });
                }
            }
        }

        // ── Step 2.9: Cold memory fallback (mirrors recall()) ──
        if !self.is_encrypted() {
            if let Some(qt) = query_text {
                let best_score = scored.iter().map(|r| r.score).fold(0.0f64, f64::max);
                const COLD_ACTIVATION_THRESHOLD: f64 = 0.55;
                const COLD_MIN_SIM: f64 = 0.10;
                const COLD_MAX_CANDIDATES: usize = 30;

                if best_score < COLD_ACTIVATION_THRESHOLD {
                    let cold_keywords: Vec<String> = qt
                        .split(|c: char| !c.is_alphanumeric())
                        .filter(|s| !s.is_empty() && s.len() > 1)
                        .filter(|s| {
                            const STOP: &[&str] = &[
                                "a","an","the","is","are","am","was","were","be","been",
                                "what","who","how","when","where","which","why",
                                "do","did","does","have","has","had",
                                "i","me","my","mine","we","our","you","your",
                                "to","of","in","on","at","by","for","with","from",
                                "about","tell","and","or","but","not","no",
                                "it","its","that","this","there","s",
                                "she","her","he","his","they","them",
                            ];
                            !STOP.contains(&s.to_lowercase().as_str())
                        })
                        .map(|s| s.to_string())
                        .collect();

                    if !cold_keywords.is_empty() {
                        let mut fts_parts: Vec<String> = Vec::new();
                        for kw in &cold_keywords {
                            let kw_lower = kw.to_lowercase();
                            fts_parts.push(format!("\"{}\"", kw.replace('"', "")));
                            if let Some(stem) = simple_stem(&kw_lower) {
                                fts_parts.push(format!("{}*", stem));
                            }
                            if let Some(alts) = irregular_verb_forms(&kw_lower) {
                                for alt in alts {
                                    fts_parts.push(format!("\"{}\"", alt));
                                }
                            }
                        }
                        let cold_fts = fts_parts.join(" OR ");

                        let cold_sql = if memory_type.is_some() {
                            format!(
                                "SELECT m.rid FROM memories m \
                                 JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                 WHERE memories_fts MATCH ?1 \
                                 AND m.consolidation_status = 'active' \
                                 AND m.access_count = 0 \
                                 AND m.type = ?2 \
                                 {} \
                                 ORDER BY m.importance DESC \
                                 LIMIT {}",
                                if namespace.is_some() { "AND m.namespace = ?3" } else { "" },
                                COLD_MAX_CANDIDATES,
                            )
                        } else {
                            format!(
                                "SELECT m.rid FROM memories m \
                                 JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                 WHERE memories_fts MATCH ?1 \
                                 AND m.consolidation_status = 'active' \
                                 AND m.access_count = 0 \
                                 {} \
                                 ORDER BY m.importance DESC \
                                 LIMIT {}",
                                if namespace.is_some() { "AND m.namespace = ?2" } else { "" },
                                COLD_MAX_CANDIDATES,
                            )
                        };

                        let cold_rids: Vec<String> = {
                            let conn = self.conn.lock().unwrap();
                            let mut stmt = conn.prepare_cached(&cold_sql).ok();
                            if let Some(ref mut stmt) = stmt {
                                let result: std::result::Result<Vec<String>, _> = if let Some(mt) = memory_type {
                                    if let Some(ns) = namespace {
                                        stmt.query_map(
                                            params![cold_fts, mt, ns],
                                            |row| row.get::<_, String>(0),
                                        ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    } else {
                                        stmt.query_map(
                                            params![cold_fts, mt],
                                            |row| row.get::<_, String>(0),
                                        ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    }
                                } else if let Some(ns) = namespace {
                                    stmt.query_map(
                                        params![cold_fts, ns],
                                        |row| row.get::<_, String>(0),
                                    ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                } else {
                                    stmt.query_map(
                                        params![cold_fts],
                                        |row| row.get::<_, String>(0),
                                    ).map(|rows| rows.filter_map(|r| r.ok()).collect())
                                };
                                result.unwrap_or_default()
                            } else {
                                vec![]
                            }
                        };

                        let existing_rids: std::collections::HashSet<String> =
                            scored.iter().map(|r| r.rid.clone()).collect();
                        let new_cold: Vec<String> = cold_rids.into_iter()
                            .filter(|r| !existing_rids.contains(r))
                            .collect();

                        if !new_cold.is_empty() {
                            let rid_refs: Vec<&str> = new_cold.iter().map(|r| r.as_str()).collect();
                            let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;
                            let cache = self.scoring_cache.read().unwrap();

                            for rid in &new_cold {
                                let Some(row) = cache.get(rid) else { continue };
                                let Some(emb_blob) = emb_map.get(rid.as_str()) else { continue };
                                let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
                                let sim_score = crate::consolidate::cosine_similarity(
                                    query_embedding, &mem_emb,
                                ) as f64;

                                if sim_score < COLD_MIN_SIM { continue; }

                                let elapsed = ts - row.last_access;
                                let decay = scoring::decay_score(row.importance, row.half_life, elapsed);
                                let age = ts - row.created_at;
                                let recency = scoring::recency_score(age);
                                let composite = scoring::adaptive_composite_score(
                                    sim_score, decay, recency, row.importance, row.valence, query_sentiment, &learned_weights,
                                );
                                let cold_boost = 0.15 * row.importance;
                                let mut why = scoring::build_why(sim_score, recency, decay, row.valence);
                                why.push("cold_memory".to_string());
                                let contributions = scoring::adaptive_contributions(
                                    sim_score, decay, recency, row.importance, &learned_weights,
                                );
                                let valence_multiplier = scoring::query_valence_boost(row.valence, query_sentiment);

                                scored.push(RecallResult {
                                    rid: rid.clone(),
                                    memory_type: row.memory_type.clone(),
                                    text: String::new(),
                                    created_at: row.created_at,
                                    importance: row.importance,
                                    valence: row.valence,
                                    score: composite + cold_boost,
                                    scores: ScoreBreakdown {
                                        similarity: sim_score,
                                        decay,
                                        recency,
                                        importance: row.importance,
                                        graph_proximity: 0.0,
                                        contributions,
                                        valence_multiplier,
                                    },
                                    why_retrieved: why,
                                    metadata: serde_json::Value::Null,
                                    namespace: row.namespace.clone(),
                                    certainty: row.certainty,
                                    domain: row.domain.clone(),
                                    source: row.source.clone(),
                                    emotional_state: row.emotional_state.clone(),
                                });
                            }
                        }
                    }
                }
            }
        }

        // ── Phase 3: Graph expansion ──
        let t_graph = Instant::now();
        let mut graph_expansion_count = 0usize;
        if expand_entities {
            let gi = self.graph_index.read().unwrap();
            let query_entities: Vec<(String, String, u32)> = if let Some(qt) = query_text {
                let query_tokens = crate::graph::tokenize(qt);
                gi.entity_matches_query(&query_tokens)
            } else {
                vec![]
            };

            let (mut base_boost, mut seed_entities, entity_idfs): (f64, Vec<String>, std::collections::HashMap<String, f64>) = if !query_entities.is_empty() {
                let has_person = query_entities.iter().any(|(_, etype, _)| etype == "person");
                let factor = if has_person { 0.20 } else if query_entities.len() >= 2 { 0.15 } else { 0.12 };
                let idfs: std::collections::HashMap<String, f64> = query_entities
                    .iter()
                    .map(|(name, _, mc)| {
                        let idf = 1.0 / (1.0 + (*mc as f64).max(1.0).ln());
                        (name.to_lowercase(), idf)
                    })
                    .collect();
                let names: Vec<String> = query_entities.into_iter().map(|(n, _, _)| n).collect();
                (factor, names, idfs)
            } else if query_text.is_none() {
                // Embedding-only search (no query text): seed from top results
                let mut seed_sorted = scored.clone();
                seed_sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
                let seed_count = 3.min(seed_sorted.len());
                let seed_rids: Vec<&str> = seed_sorted[..seed_count].iter().map(|r| r.rid.as_str()).collect();
                let seeds = gi.entities_for_memories(&seed_rids);
                (0.05, seeds, std::collections::HashMap::new())
            } else {
                (0.0, vec![], std::collections::HashMap::new())
            };

            // Group query expansion: if query mentions "team", "family", etc.,
            // seed expansion with person entities CONNECTED to query entities.
            //
            // Previous approach: entities_by_type("person").take(15) grabbed
            // arbitrary person entities, often filler characters unrelated to
            // the query. New approach: BFS from query entities finds people
            // actually connected to the subject (e.g., Priya → Arjun, Meera,
            // Appa, Amma for "family"; Priya → Deepa, Neha for "team").
            const GROUP_KEYWORDS: &[&str] = &[
                "team", "group", "colleagues", "coworkers", "friends", "family",
                "staff", "members", "people",
            ];
            if let Some(qt) = query_text {
                let qt_lower = qt.to_lowercase();
                if GROUP_KEYWORDS.iter().any(|kw| qt_lower.contains(kw)) {
                    if !seed_entities.is_empty() {
                        // BFS from query entities to find connected person entities
                        let query_seeds: Vec<&str> = seed_entities.iter().map(|s| s.as_str()).collect();
                        let nearby = gi.expand_bfs(&query_seeds, 2, 50);
                        for (name, hops, _) in &nearby {
                            if *hops > 0
                                && gi.entity_type(name).map_or(false, |t| t == "person")
                                && !seed_entities.contains(&name.to_string())
                            {
                                seed_entities.push(name.clone());
                            }
                        }
                    } else {
                        // No query entities — fall back to type-based expansion
                        let person_entities = gi.entities_by_type("person");
                        for person in person_entities.into_iter().take(15) {
                            if !seed_entities.contains(&person) {
                                seed_entities.push(person);
                            }
                        }
                    }
                    base_boost = base_boost.max(0.20_f64);
                }
            }

            const MAX_BOOST_PER_MEMORY: f64 = 0.25;
            const MAX_GRAPH_FRACTION: f64 = 1.0;
            const MAX_SEED_ENTITIES: usize = 8;

            // Cap seed entities to prevent graph explosion with many entities
            if seed_entities.len() > MAX_SEED_ENTITIES {
                seed_entities.truncate(MAX_SEED_ENTITIES);
            }

            if !seed_entities.is_empty() && base_boost > 0.0 {
                let seed_refs: Vec<&str> = seed_entities.iter().map(|s| s.as_str()).collect();
                let expanded = gi.expand_bfs(&seed_refs, 2, 30);
                let expanded_map: std::collections::HashMap<String, (u8, f64)> = expanded
                    .iter()
                    .map(|(name, hops, weight)| (name.clone(), (*hops, *weight)))
                    .collect();

                for result in &mut scored {
                    let prox = gi.graph_proximity(&result.rid, &expanded_map);
                    if prox > 0.0 {
                        let mem_entities: Vec<String> = gi.entities_for_memory(&result.rid).into_iter().map(|s| s.to_string()).collect();
                        let mut best_idf = 1.0f64;
                        let mut connecting_entity = String::new();
                        for entity in &mem_entities {
                            if expanded_map.contains_key(entity) {
                                let idf = entity_idfs.get(&entity.to_lowercase()).copied().unwrap_or(1.0);
                                if connecting_entity.is_empty() || idf > best_idf {
                                    best_idf = idf;
                                    connecting_entity = entity.clone();
                                }
                            }
                        }
                        let consolidation_factor = {
                            let cache = self.scoring_cache.read().unwrap();
                            cache.get(&result.rid)
                                .map(|r| if r.consolidation_status == "consolidated" { 0.5 } else { 1.0 })
                                .unwrap_or(1.0)
                        };
                        let boost = (base_boost * prox * best_idf * consolidation_factor).min(MAX_BOOST_PER_MEMORY);
                        result.scores.graph_proximity = prox;
                        result.score += boost;
                        if !connecting_entity.is_empty() {
                            result.why_retrieved.push(format!("graph-connected via {connecting_entity}"));
                        }
                    }
                }

                let max_graph_only = ((MAX_GRAPH_FRACTION * top_k as f64).ceil() as usize).max(1);
                let all_entity_names: Vec<&str> = expanded.iter().map(|(n, _, _)| n.as_str()).collect();
                let graph_rids = gi.memories_for_entities(&all_entity_names);
                let existing_rids: std::collections::HashSet<&str> = scored.iter().map(|r| r.rid.as_str()).collect();

                let preselect_pool = max_graph_only * 5;
                let new_rids: Vec<String> = {
                    let cache = self.scoring_cache.read().unwrap();
                    let mut candidates: Vec<(String, f64)> = graph_rids
                        .into_iter()
                        .filter(|r| !existing_rids.contains(r.as_str()))
                        .filter_map(|r| {
                            let row = cache.get(r.as_str())?;
                            let status_ok = if include_consolidated {
                                row.consolidation_status == "active" || row.consolidation_status == "consolidated"
                            } else {
                                row.consolidation_status == "active"
                            };
                            if !status_ok { return None; }
                            if let Some(mt) = memory_type {
                                if row.memory_type != mt { return None; }
                            }
                            if let Some((start, end)) = time_window {
                                if row.created_at < start || row.created_at > end { return None; }
                            }
                            if let Some(ns) = namespace {
                                if row.namespace != ns { return None; }
                            }
                            let prox = gi.graph_proximity(&r, &expanded_map);
                            let rank = row.importance * (0.3 + 0.7 * prox);
                            Some((r, rank))
                        })
                        .collect();
                    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    candidates.into_iter().take(preselect_pool).map(|(rid, _)| rid).collect()
                };
                graph_expansion_count = new_rids.len();

                if !new_rids.is_empty() {
                    let new_rid_refs: Vec<&str> = new_rids.iter().map(|r| r.as_str()).collect();
                    let emb_map = self.fetch_embeddings_by_rids(&new_rid_refs)?;

                    let cache = self.scoring_cache.read().unwrap();
                    for rid in &new_rids {
                        if let (Some(row), Some(emb_blob)) = (cache.get(rid.as_str()), emb_map.get(rid.as_str())) {
                            let mem_embedding = crate::serde_helpers::deserialize_f32(emb_blob);
                            let sim_score = crate::consolidate::cosine_similarity(query_embedding, &mem_embedding) as f64;
                            let elapsed = ts - row.last_access;
                            let decay = scoring::decay_score(row.importance, row.half_life, elapsed);
                            let age = ts - row.created_at;
                            let recency = scoring::recency_score(age);
                            let prox = gi.graph_proximity(rid, &expanded_map);
                            let composite = scoring::adaptive_graph_composite_score(sim_score, decay, recency, row.importance, row.valence, prox, query_sentiment, &learned_weights);
                            let mut why = scoring::build_why(sim_score, recency, decay, row.valence);

                            let mem_entities: Vec<String> = gi.entities_for_memory(rid).into_iter().map(|s| s.to_string()).collect();
                            for entity in &mem_entities {
                                if expanded_map.contains_key(entity) {
                                    why.push(format!("graph-connected via {entity}"));
                                    break;
                                }
                            }

                            let contributions = scoring::adaptive_graph_contributions(sim_score, decay, recency, row.importance, prox, &learned_weights);
                            let valence_multiplier = scoring::query_valence_boost(row.valence, query_sentiment);

                            scored.push(RecallResult {
                                rid: rid.clone(),
                                memory_type: row.memory_type.clone(),
                                text: String::new(),
                                created_at: row.created_at,
                                importance: row.importance,
                                valence: row.valence,
                                score: composite,
                                scores: ScoreBreakdown {
                                    similarity: sim_score,
                                    decay,
                                    recency,
                                    importance: row.importance,
                                    graph_proximity: prox,
                                    contributions,
                                    valence_multiplier,
                                },
                                why_retrieved: why,
                                metadata: serde_json::Value::Null,
                                namespace: row.namespace.clone(),
                                certainty: row.certainty,
                                domain: row.domain.clone(),
                                source: row.source.clone(),
                                emotional_state: row.emotional_state.clone(),
                            });
                        }
                    }
                }
            }
        }
        let graph_ms = t_graph.elapsed().as_secs_f64() * 1000.0;

        // ── Phase 3.5: Keyword slot reservation (mirrors recall()) ──
        {
            scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            let cutoff_idx = top_k.min(scored.len()).saturating_sub(1);
            let cutoff_score = scored.get(cutoff_idx).map(|r| r.score).unwrap_or(0.0);

            const KEYWORD_RESERVE_SLOTS: usize = 3;
            const KEYWORD_RESERVE_MIN_SIM: f64 = 0.25;

            let mut kw_below: Vec<(usize, f64)> = scored.iter().enumerate()
                .filter(|(_, r)| {
                    r.why_retrieved.iter().any(|w| w == "keyword_match")
                        && r.scores.similarity >= KEYWORD_RESERVE_MIN_SIM
                        && r.score < cutoff_score
                })
                .map(|(i, r)| (i, r.scores.similarity))
                .collect();
            kw_below.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (idx, _) in kw_below.into_iter().take(KEYWORD_RESERVE_SLOTS) {
                scored[idx].score = cutoff_score + 0.001;
                scored[idx].why_retrieved.push("keyword_reserved".to_string());
            }
        }

        // ── Phase 4: MMR diversity selection ──
        let t_sort = Instant::now();
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        let min_pool_for_mmr = (top_k * 3).max(20);
        if scored.len() > top_k && scored.len() >= min_pool_for_mmr {
            let pool_size = scored.len().min(top_k * 10);
            scored.truncate(pool_size);

            let pool_rids: Vec<&str> = scored.iter().map(|r| r.rid.as_str()).collect();
            let emb_map = self.fetch_embeddings_by_rids(&pool_rids)?;

            let pool_embeddings: Vec<Option<Vec<f32>>> = scored.iter().map(|r| {
                emb_map.get(r.rid.as_str())
                    .map(|blob| crate::serde_helpers::deserialize_f32(blob))
            }).collect();

            const LAMBDA: f64 = 0.7;
            const SIM_THRESHOLD: f64 = 0.98;

            let mut selected: Vec<usize> = Vec::with_capacity(top_k);
            let mut selected_embeddings: Vec<&[f32]> = Vec::with_capacity(top_k);

            if !scored.is_empty() {
                selected.push(0);
                if let Some(Some(ref emb)) = pool_embeddings.first() {
                    selected_embeddings.push(emb);
                }
            }

            for _ in 1..top_k {
                let mut best_idx = None;
                let mut best_mmr = f64::NEG_INFINITY;

                for (idx, result) in scored.iter().enumerate() {
                    if selected.contains(&idx) { continue; }

                    let relevance = result.score;
                    let max_sim = if let Some(Some(ref cand_emb)) = pool_embeddings.get(idx) {
                        selected_embeddings.iter()
                            .map(|sel_emb| crate::consolidate::cosine_similarity(cand_emb, sel_emb) as f64)
                            .fold(0.0f64, f64::max)
                    } else {
                        0.0
                    };

                    if max_sim > SIM_THRESHOLD { continue; }

                    let mmr = LAMBDA * relevance - (1.0 - LAMBDA) * max_sim;
                    if mmr > best_mmr {
                        best_mmr = mmr;
                        best_idx = Some(idx);
                    }
                }

                match best_idx {
                    Some(idx) => {
                        selected.push(idx);
                        if let Some(Some(ref emb)) = pool_embeddings.get(idx) {
                            selected_embeddings.push(emb);
                        }
                    }
                    None => break,
                }
            }

            let mut diverse_results = Vec::with_capacity(selected.len());
            for i in selected {
                diverse_results.push(scored[i].clone());
            }
            scored = diverse_results;
        } else {
            scored.truncate(top_k);
        }
        let sort_truncate_ms = t_sort.elapsed().as_secs_f64() * 1000.0;

        // ── Phase 5: Hydrate final top_k with text + metadata ──
        let t_fetch = Instant::now();
        {
            let final_rids: Vec<&str> = scored.iter().map(|r| r.rid.as_str()).collect();
            let text_map = self.fetch_text_metadata_by_rids(&final_rids)?;
            for result in &mut scored {
                if let Some(tm) = text_map.get(result.rid.as_str()) {
                    result.text = tm.text.clone();
                    result.metadata = serde_json::from_str(&tm.metadata)
                        .unwrap_or(serde_json::Value::Object(Default::default()));
                }
            }
        }
        let fetch_ms = t_fetch.elapsed().as_secs_f64() * 1000.0;

        // ── Phase 6: Reinforce ──
        let t_reinforce = Instant::now();
        if !skip_reinforce {
            for r in &scored {
                self.reinforce(&r.rid)?;
            }
        }
        let reinforce_ms = t_reinforce.elapsed().as_secs_f64() * 1000.0;

        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        Ok(RecallProfiledResult {
            results: scored,
            timings: RecallTimings {
                vec_search_ms,
                cache_score_ms: cache_score_ms + fallback_ms + fts_ms,
                fetch_ms,
                scoring_ms: 0.0,
                graph_ms,
                reinforce_ms,
                sort_truncate_ms,
                total_ms,
                candidate_count,
                graph_expansion_count,
            },
        })
    }

    /// Fetch only text and metadata for a set of RIDs (post-scoring hydration).
    pub(crate) fn fetch_text_metadata_by_rids(
        &self,
        rids: &[&str],
    ) -> Result<HashMap<String, TextMetadataRow>> {
        if rids.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders: String = (0..rids.len())
            .map(|i| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT rid, type, text, metadata FROM memories WHERE rid IN ({placeholders})"
        );
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        for r in rids {
            param_values.push(Box::new(r.to_string()));
        }
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params_ref.as_slice(), |row| {
                Ok((
                    row.get::<_, String>("rid")?,
                    row.get::<_, String>("text")?,
                    row.get::<_, String>("metadata")?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        drop(stmt);
        drop(conn);

        let mut map = HashMap::new();
        for (rid, stored_text, stored_meta) in rows {
            let text = self.decrypt_text(&stored_text)?;
            let metadata = self.decrypt_text(&stored_meta)?;
            map.insert(rid.clone(), TextMetadataRow { rid, text, metadata });
        }
        Ok(map)
    }

    /// Batch-fetch embeddings for a set of RIDs (for graph-only candidate scoring).
    pub(crate) fn fetch_embeddings_by_rids(
        &self,
        rids: &[&str],
    ) -> Result<HashMap<String, Vec<u8>>> {
        if rids.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders: String = (0..rids.len())
            .map(|i| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT rid, embedding FROM memories WHERE rid IN ({placeholders})"
        );
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        for r in rids {
            param_values.push(Box::new(r.to_string()));
        }
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params_ref.as_slice(), |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        drop(stmt);
        drop(conn);

        let mut map = HashMap::new();
        for (rid, stored_emb) in rows {
            let emb = self.decrypt_embedding(&stored_emb)?;
            map.insert(rid, emb);
        }
        Ok(map)
    }

    /// Reinforce a memory on access — increase half_life, update last_access,
    /// and increment access_count.
    fn reinforce(&self, rid: &str) -> Result<()> {
        let ts = now();

        // Read half_life from cache (eliminates SELECT query)
        let current_half_life = {
            let cache = self.scoring_cache.read().unwrap();
            cache.get(rid).map(|r| r.half_life)
        };
        let new_half_life = match current_half_life {
            Some(hl) => (hl * 1.2_f64).min(31536000.0),
            None => 604800.0, // fallback if not in cache
        };

        {
            let conn = self.conn.lock().unwrap();
            conn.execute(
                "UPDATE memories SET last_access = ?1, half_life = ?2, \
                 access_count = access_count + 1 WHERE rid = ?3",
                params![ts, new_half_life, rid],
            )?;
        } // drop conn before taking write lock on scoring_cache

        // Update cache with new values
        {
            let mut cache = self.scoring_cache.write().unwrap();
            if let Some(row) = cache.get_mut(rid) {
                row.last_access = ts;
                row.half_life = new_half_life;
                row.access_count += 1;
            }
        }

        self.log_op(
            "reinforce",
            Some(rid),
            &serde_json::json!({
                "rid": rid,
                "last_access": ts,
                "half_life": new_half_life,
                "local_only": true,
            }),
            None,
        )?;

        Ok(())
    }
}
