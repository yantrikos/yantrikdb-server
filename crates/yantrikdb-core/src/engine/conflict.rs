use rusqlite::params;

use crate::error::{YantrikDbError, Result};
use crate::types::*;

use super::{now, YantrikDB};

/// Common English stopwords that should never be added to substitution categories.
const RECLASSIFY_STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "must", "need",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "his", "her", "its", "this", "that", "these",
    "those", "who", "what", "which", "where", "when", "how", "why",
    "and", "or", "but", "if", "then", "else", "so", "yet", "nor",
    "not", "no", "yes", "all", "any", "some", "every", "each",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "up",
    "about", "into", "over", "after", "before", "between", "under",
    "again", "further", "more", "most", "other", "such", "than",
    "too", "very", "just", "also", "now", "here", "there", "out",
    "only", "own", "same", "both", "few", "many", "much", "well",
    "back", "even", "still", "way", "new", "old", "one", "two",
    "first", "last", "long", "great", "little", "right", "big",
    "high", "low", "small", "large", "next", "early", "late",
    "use", "uses", "used", "using", "like", "make", "made", "get",
    "got", "take", "took", "come", "came", "go", "went", "see", "saw",
    "know", "knew", "think", "thought", "want", "give", "gave",
    "tell", "told", "work", "works", "call", "try", "ask", "put",
    "keep", "let", "begin", "seem", "help", "show", "hear", "play",
    "run", "move", "live", "believe", "bring", "happen", "write",
    "provide", "sit", "stand", "lose", "pay", "meet", "include",
    "continue", "set", "learn", "change", "lead", "understand",
    "watch", "follow", "stop", "create", "speak", "read", "add",
    "spend", "grow", "open", "walk", "win", "offer", "remember",
    "love", "consider", "appear", "buy", "wait", "serve", "die",
    "send", "expect", "build", "stay", "fall", "cut", "reach",
    "remain", "suggest", "raise", "pass", "sell", "require",
    "report", "decide", "pull", "develop", "always", "never",
    "sometimes", "often", "usually", "really", "actually",
    "probably", "already", "quite", "rather", "pretty",
    "solves", "solved", "solving", "started", "starting", "finished",
    "finishing", "before", "after", "during", "while", "until", "since",
];

impl YantrikDB {
    // ── Conflict resolution API (V2) ──

    /// Get all conflicts, optionally filtered.
    pub fn get_conflicts(
        &self,
        status: Option<&str>,
        conflict_type: Option<&str>,
        entity: Option<&str>,
        priority: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Conflict>> {
        let mut sql = String::from("SELECT * FROM conflicts WHERE 1=1");
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut idx = 1;

        if let Some(s) = status {
            sql.push_str(&format!(" AND status = ?{idx}"));
            param_values.push(Box::new(s.to_string()));
            idx += 1;
        }
        if let Some(ct) = conflict_type {
            sql.push_str(&format!(" AND conflict_type = ?{idx}"));
            param_values.push(Box::new(ct.to_string()));
            idx += 1;
        }
        if let Some(e) = entity {
            sql.push_str(&format!(" AND entity = ?{idx}"));
            param_values.push(Box::new(e.to_string()));
            idx += 1;
        }
        if let Some(p) = priority {
            sql.push_str(&format!(" AND priority = ?{idx}"));
            param_values.push(Box::new(p.to_string()));
            let _ = idx;
        }

        sql.push_str(
            " ORDER BY
            CASE priority
                WHEN 'critical' THEN 0
                WHEN 'high' THEN 1
                WHEN 'medium' THEN 2
                WHEN 'low' THEN 3
            END,
            detected_at DESC",
        );
        sql.push_str(&format!(" LIMIT {limit}"));

        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let conn = self.conn();
        let mut stmt = conn.prepare(&sql)?;
        let conflicts = stmt
            .query_map(params_ref.as_slice(), |row| {
                Ok(Conflict {
                    conflict_id: row.get("conflict_id")?,
                    conflict_type: row.get("conflict_type")?,
                    priority: row.get("priority")?,
                    status: row.get("status")?,
                    memory_a: row.get("memory_a")?,
                    memory_b: row.get("memory_b")?,
                    entity: row.get("entity")?,
                    rel_type: row.get("rel_type")?,
                    detected_at: row.get("detected_at")?,
                    detected_by: row.get("detected_by")?,
                    detection_reason: row.get("detection_reason")?,
                    resolved_at: row.get("resolved_at")?,
                    resolved_by: row.get("resolved_by")?,
                    strategy: row.get("strategy")?,
                    winner_rid: row.get("winner_rid")?,
                    resolution_note: row.get("resolution_note")?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(conflicts)
    }

    /// Get a single conflict by ID.
    pub fn get_conflict(&self, conflict_id: &str) -> Result<Option<Conflict>> {
        let conn = self.conn();
        let result = conn.query_row(
            "SELECT * FROM conflicts WHERE conflict_id = ?1",
            params![conflict_id],
            |row| {
                Ok(Conflict {
                    conflict_id: row.get("conflict_id")?,
                    conflict_type: row.get("conflict_type")?,
                    priority: row.get("priority")?,
                    status: row.get("status")?,
                    memory_a: row.get("memory_a")?,
                    memory_b: row.get("memory_b")?,
                    entity: row.get("entity")?,
                    rel_type: row.get("rel_type")?,
                    detected_at: row.get("detected_at")?,
                    detected_by: row.get("detected_by")?,
                    detection_reason: row.get("detection_reason")?,
                    resolved_at: row.get("resolved_at")?,
                    resolved_by: row.get("resolved_by")?,
                    strategy: row.get("strategy")?,
                    winner_rid: row.get("winner_rid")?,
                    resolution_note: row.get("resolution_note")?,
                })
            },
        );

        match result {
            Ok(c) => Ok(Some(c)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Resolve a conflict with a chosen strategy.
    ///
    /// Strategies:
    ///   - keep_a: tombstone memory_b, keep memory_a
    ///   - keep_b: tombstone memory_a, keep memory_b
    ///   - keep_both: mark resolved, keep both memories
    ///   - merge: create new memory (new_text required), tombstone both
    pub fn resolve_conflict(
        &self,
        conflict_id: &str,
        strategy: &str,
        winner_rid: Option<&str>,
        new_text: Option<&str>,
        resolution_note: Option<&str>,
    ) -> Result<ConflictResolutionResult> {
        let conflict = self
            .get_conflict(conflict_id)?
            .ok_or_else(|| YantrikDbError::NotFound(format!("conflict: {}", conflict_id)))?;

        if conflict.status != "open" {
            return Err(YantrikDbError::SyncError(format!(
                "conflict {} is already {}",
                conflict_id, conflict.status
            )));
        }

        let ts = now();
        let actor_id = self.actor_id.clone();
        let mut loser_tombstoned = false;
        let mut new_memory_rid = None;

        let (effective_winner, loser_rid) = match strategy {
            "keep_a" => {
                let winner = winner_rid.unwrap_or(&conflict.memory_a);
                let loser = if winner == conflict.memory_a {
                    &conflict.memory_b
                } else {
                    &conflict.memory_a
                };
                self.forget(loser)?;
                loser_tombstoned = true;
                (Some(winner.to_string()), Some(loser.to_string()))
            }
            "keep_b" => {
                let winner = winner_rid.unwrap_or(&conflict.memory_b);
                let loser = if winner == conflict.memory_b {
                    &conflict.memory_a
                } else {
                    &conflict.memory_b
                };
                self.forget(loser)?;
                loser_tombstoned = true;
                (Some(winner.to_string()), Some(loser.to_string()))
            }
            "keep_both" => (None, None),
            "merge" => {
                let text = new_text.ok_or_else(|| {
                    YantrikDbError::SyncError("merge strategy requires new_text".to_string())
                })?;
                let mem_a = self.get(&conflict.memory_a)?;
                let mem_b = self.get(&conflict.memory_b)?;
                let imp_a = mem_a.as_ref().map(|m| m.importance).unwrap_or(0.5);
                let imp_b = mem_b.as_ref().map(|m| m.importance).unwrap_or(0.5);
                let merged_importance = imp_a.max(imp_b);

                let zero_emb = vec![0.0f32; self.embedding_dim];
                let meta = serde_json::json!({
                    "merged_from": [conflict.memory_a, conflict.memory_b],
                    "conflict_id": conflict_id,
                });
                let merge_ns = mem_a.as_ref().map(|m| m.namespace.as_str()).unwrap_or("default");
                let rid = self.record(
                    text,
                    "semantic",
                    merged_importance,
                    0.0,
                    604800.0,
                    &meta,
                    &zero_emb,
                    merge_ns,
                    0.8,
                    "general",
                    "user",
                    None,
                )?;
                new_memory_rid = Some(rid.clone());

                self.forget(&conflict.memory_a)?;
                self.forget(&conflict.memory_b)?;
                loser_tombstoned = true;

                (Some(rid), None)
            }
            _ => {
                return Err(YantrikDbError::SyncError(format!(
                    "unknown resolution strategy: {}",
                    strategy
                )));
            }
        };

        // Update the conflict record
        self.conn().execute(
            "UPDATE conflicts SET
             status = 'resolved',
             resolved_at = ?1,
             resolved_by = ?2,
             strategy = ?3,
             winner_rid = ?4,
             resolution_note = ?5
             WHERE conflict_id = ?6",
            params![ts, actor_id, strategy, effective_winner, resolution_note, conflict_id],
        )?;

        // Log to oplog for replication
        self.log_op(
            "conflict_resolve",
            Some(conflict_id),
            &serde_json::json!({
                "conflict_id": conflict_id,
                "strategy": strategy,
                "winner_rid": effective_winner,
                "loser_rid": loser_rid,
                "new_text": new_text,
                "resolution_note": resolution_note,
                "resolved_at": ts,
                "resolved_by": actor_id,
            }),
            None,
        )?;

        Ok(ConflictResolutionResult {
            conflict_id: conflict_id.to_string(),
            strategy: strategy.to_string(),
            winner_rid: effective_winner,
            loser_tombstoned,
            new_memory_rid,
        })
    }

    // ── Substitution category APIs (V14) ──

    /// Reclassify a conflict and learn from the diff tokens.
    ///
    /// When a user says "this redundancy is actually a conflict," this method:
    /// 1. Extracts differing tokens between the two memories
    /// 2. Learns them into substitution categories (creates/extends categories)
    /// 3. Updates the conflict type and priority
    pub fn reclassify_conflict(
        &self,
        conflict_id: &str,
        new_type: &str,
    ) -> Result<ReclassifyResult> {
        let conflict = self
            .get_conflict(conflict_id)?
            .ok_or_else(|| YantrikDbError::NotFound(format!("conflict: {}", conflict_id)))?;

        let old_type = conflict.conflict_type.clone();

        // Get memory texts
        let mem_a = self.get(&conflict.memory_a)?;
        let mem_b = self.get(&conflict.memory_b)?;
        let text_a = mem_a.map(|m| m.text).unwrap_or_default();
        let text_b = mem_b.map(|m| m.text).unwrap_or_default();

        // Decrypt if needed
        let text_a = self.decrypt_text(&text_a).unwrap_or(text_a);
        let text_b = self.decrypt_text(&text_b).unwrap_or(text_b);

        // Extract differing tokens (symmetric difference)
        let words_a: std::collections::HashSet<String> = text_a
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|w| !w.is_empty())
            .collect();
        let words_b: std::collections::HashSet<String> = text_b
            .split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|w| !w.is_empty())
            .collect();

        let diff_a: Vec<String> = words_a.difference(&words_b).cloned().collect();
        let diff_b: Vec<String> = words_b.difference(&words_a).cloned().collect();

        let ts = now();
        let hlc_ts = self.tick_hlc();
        let hlc_bytes = hlc_ts.to_bytes().to_vec();
        let actor = self.actor_id.clone();
        let mut learned_members = Vec::new();
        let mut category_created = None;

        // Pre-classify: find which diff tokens already belong to categories
        let cats_a: Vec<(String, Option<(String, String)>)> = diff_a
            .iter()
            .map(|t| (t.clone(), self.find_member_category(t)))
            .collect();
        let cats_b: Vec<(String, Option<(String, String)>)> = diff_b
            .iter()
            .map(|t| (t.clone(), self.find_member_category(t)))
            .collect();

        // Separate known-category tokens from unknown tokens
        let known_a: Vec<(&str, &str, &str)> = cats_a.iter()
            .filter_map(|(t, c)| c.as_ref().map(|(id, name)| (t.as_str(), id.as_str(), name.as_str())))
            .collect();
        let known_b: Vec<(&str, &str, &str)> = cats_b.iter()
            .filter_map(|(t, c)| c.as_ref().map(|(id, name)| (t.as_str(), id.as_str(), name.as_str())))
            .collect();

        // Track which tokens have been processed to avoid double-adding
        let mut processed: std::collections::HashSet<String> = std::collections::HashSet::new();

        // Strategy 1: Both sides have known category members → reinforce or cross-learn
        for &(token_a, cat_id_a, cat_name_a) in &known_a {
            for &(token_b, cat_id_b, _cat_name_b) in &known_b {
                if cat_id_a == cat_id_b {
                    // Same category — reinforce confidence
                    self.conn().execute(
                        "UPDATE substitution_members SET confidence = 1.0, source = 'user_confirmed', updated_at = ?1
                         WHERE category_id = ?2 AND (token_normalized = ?3 OR token_normalized = ?4)",
                        params![ts, cat_id_a, token_a, token_b],
                    )?;
                    if processed.insert(token_a.to_string()) {
                        learned_members.push(LearnedMember {
                            token: token_a.to_string(),
                            category_name: cat_name_a.to_string(),
                            is_new: false,
                        });
                    }
                    if processed.insert(token_b.to_string()) {
                        learned_members.push(LearnedMember {
                            token: token_b.to_string(),
                            category_name: cat_name_a.to_string(),
                            is_new: false,
                        });
                    }
                }
                // Different categories: don't auto-merge
            }
        }

        // Strategy 2: One side has a known member, other side doesn't
        // Only add the BEST matching unknown token per category (not all unknowns)
        // Filter out stopwords and very short tokens
        let is_meaningful = |t: &str| -> bool {
            t.len() >= 3 && !RECLASSIFY_STOPWORDS.contains(&t)
        };

        // known_a tokens → find best unknown match in diff_b
        for &(token_a, cat_id_a, cat_name_a) in &known_a {
            if processed.contains(token_a) { continue; }
            let unknown_b: Vec<&str> = diff_b.iter()
                .map(|s| s.as_str())
                .filter(|t| is_meaningful(t) && !processed.contains(*t) && self.find_member_category(t).is_none())
                .collect();
            // Only learn if there's exactly one meaningful unknown — ambiguity = skip
            if unknown_b.len() == 1 {
                let token_b = unknown_b[0];
                self.add_member_to_category(
                    cat_id_a, token_b, token_b,
                    1.0, "user_confirmed", ts, &hlc_bytes, &actor,
                )?;
                processed.insert(token_b.to_string());
                learned_members.push(LearnedMember {
                    token: token_b.to_string(),
                    category_name: cat_name_a.to_string(),
                    is_new: true,
                });
            }
        }

        // known_b tokens → find best unknown match in diff_a
        for &(token_b, cat_id_b, cat_name_b) in &known_b {
            if processed.contains(token_b) { continue; }
            let unknown_a: Vec<&str> = diff_a.iter()
                .map(|s| s.as_str())
                .filter(|t| is_meaningful(t) && !processed.contains(*t) && self.find_member_category(t).is_none())
                .collect();
            if unknown_a.len() == 1 {
                let token_a = unknown_a[0];
                self.add_member_to_category(
                    cat_id_b, token_a, token_a,
                    1.0, "user_confirmed", ts, &hlc_bytes, &actor,
                )?;
                processed.insert(token_a.to_string());
                learned_members.push(LearnedMember {
                    token: token_a.to_string(),
                    category_name: cat_name_b.to_string(),
                    is_new: true,
                });
            }
        }

        // Strategy 3: Neither side known — only create provisional category
        // for recurring meaningful-token pairs (requires 2+ prior occurrences)
        if known_a.is_empty() && known_b.is_empty() {
            let meaningful_a: Vec<&str> = diff_a.iter()
                .map(|s| s.as_str())
                .filter(|t| is_meaningful(t))
                .collect();
            let meaningful_b: Vec<&str> = diff_b.iter()
                .map(|s| s.as_str())
                .filter(|t| is_meaningful(t))
                .collect();

            if meaningful_a.len() == 1 && meaningful_b.len() == 1 {
                let ta = meaningful_a[0];
                let tb = meaningful_b[0];
                let recurrence = self.count_reclassify_pair_occurrences(ta, tb);
                if recurrence >= 1 {
                    let prov_name = format!("learned_{}_{}", ta, tb);
                    let cat_id = crate::id::new_id();
                    self.conn().execute(
                        "INSERT OR IGNORE INTO substitution_categories
                         (id, name, conflict_mode, status, created_at, updated_at, hlc, origin_actor)
                         VALUES (?1, ?2, 'exclusive', 'provisional', ?3, ?3, ?4, ?5)",
                        params![cat_id, prov_name, ts, hlc_bytes, actor],
                    )?;
                    self.add_member_to_category(
                        &cat_id, ta, ta, 1.0, "user_confirmed", ts, &hlc_bytes, &actor,
                    )?;
                    self.add_member_to_category(
                        &cat_id, tb, tb, 1.0, "user_confirmed", ts, &hlc_bytes, &actor,
                    )?;
                    category_created = Some(prov_name.clone());
                    learned_members.push(LearnedMember {
                        token: ta.to_string(), category_name: prov_name.clone(), is_new: true,
                    });
                    learned_members.push(LearnedMember {
                        token: tb.to_string(), category_name: prov_name, is_new: true,
                    });
                }
            }
        }

        // Update conflict type and priority
        let new_conflict_type = ConflictType::from_str(new_type);
        let new_priority = new_conflict_type.default_priority();
        self.conn().execute(
            "UPDATE conflicts SET conflict_type = ?1, priority = ?2 WHERE conflict_id = ?3",
            params![new_type, new_priority, conflict_id],
        )?;

        // Log to oplog
        self.log_op(
            "conflict_reclassify",
            Some(conflict_id),
            &serde_json::json!({
                "conflict_id": conflict_id,
                "old_type": old_type,
                "new_type": new_type,
                "diff_a": diff_a,
                "diff_b": diff_b,
                "learned_members": learned_members.iter().map(|m| {
                    serde_json::json!({"token": m.token, "category": m.category_name, "is_new": m.is_new})
                }).collect::<Vec<_>>(),
                "category_created": category_created,
            }),
            None,
        )?;

        Ok(ReclassifyResult {
            conflict_id: conflict_id.to_string(),
            old_type,
            new_type: new_type.to_string(),
            learned_members,
            category_created,
        })
    }

    /// List all substitution categories with member counts.
    pub fn substitution_categories(&self) -> Result<Vec<SubstitutionCategory>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT c.id, c.name, c.conflict_mode, c.status,
                    (SELECT COUNT(*) FROM substitution_members m
                     WHERE m.category_id = c.id AND m.status = 'active') as member_count
             FROM substitution_categories c
             ORDER BY c.name"
        )?;

        let cats = stmt
            .query_map([], |row| {
                Ok(SubstitutionCategory {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    conflict_mode: row.get(2)?,
                    status: row.get(3)?,
                    member_count: row.get(4)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(cats)
    }

    /// List members of a specific substitution category.
    pub fn substitution_members(&self, category_name: &str) -> Result<Vec<SubstitutionMember>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT m.id, c.name, m.token_normalized, m.token_display,
                    m.confidence, m.source, m.status
             FROM substitution_members m
             JOIN substitution_categories c ON c.id = m.category_id
             WHERE c.name = ?1
             ORDER BY m.confidence DESC, m.token_normalized"
        )?;

        let members = stmt
            .query_map(params![category_name], |row| {
                Ok(SubstitutionMember {
                    id: row.get(0)?,
                    category_name: row.get(1)?,
                    token_normalized: row.get(2)?,
                    token_display: row.get(3)?,
                    confidence: row.get(4)?,
                    source: row.get(5)?,
                    status: row.get(6)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(members)
    }

    /// Ingest new members into a category (from LLM gossip or manual input).
    ///
    /// Creates the category if it doesn't exist. Returns number of new members added.
    pub fn learn_category_members(
        &self,
        category_name: &str,
        members: &[(String, f64)],
        source: &str,
    ) -> Result<usize> {
        let ts = now();
        let hlc_ts = self.tick_hlc();
        let hlc_bytes = hlc_ts.to_bytes().to_vec();
        let actor = self.actor_id.clone();

        // Find or create category
        let cat_id = match self.conn().query_row(
            "SELECT id FROM substitution_categories WHERE name = ?1",
            params![category_name],
            |row| row.get::<_, String>(0),
        ) {
            Ok(id) => id,
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                let id = crate::id::new_id();
                self.conn().execute(
                    "INSERT INTO substitution_categories
                     (id, name, conflict_mode, status, created_at, updated_at, hlc, origin_actor)
                     VALUES (?1, ?2, 'exclusive', 'active', ?3, ?3, ?4, ?5)",
                    params![id, category_name, ts, hlc_bytes, actor],
                )?;
                self.log_op(
                    "category_create",
                    None,
                    &serde_json::json!({
                        "category_id": id,
                        "name": category_name,
                        "source": source,
                    }),
                    None,
                )?;
                id
            }
            Err(e) => return Err(e.into()),
        };

        let status = if source == "llm_suggested" { "pending" } else { "active" };
        let mut added = 0;

        for (token, confidence) in members {
            let normalized = token.to_lowercase();
            let display = token.clone();
            let was_added = self.add_member_to_category(
                &cat_id, &normalized, &display,
                *confidence, source, ts, &hlc_bytes, &actor,
            )?;
            if was_added {
                added += 1;
            }
        }

        // Log to oplog
        self.log_op(
            "member_add",
            None,
            &serde_json::json!({
                "category_id": cat_id,
                "category_name": category_name,
                "source": source,
                "members_added": added,
                "total_submitted": members.len(),
            }),
            None,
        )?;

        Ok(added)
    }

    /// Reset a substitution category to its seed state by removing all non-seed members.
    /// Returns the number of members removed.
    pub fn reset_category_to_seed(&self, category_name: &str) -> Result<usize> {
        let conn = self.conn();
        let cat_id: String = conn.query_row(
            "SELECT id FROM substitution_categories WHERE name = ?1",
            params![category_name],
            |row| row.get(0),
        ).map_err(|_| YantrikDbError::NotFound(format!("category: {}", category_name)))?;

        let removed = conn.execute(
            "DELETE FROM substitution_members
             WHERE category_id = ?1 AND source != 'seed'",
            params![cat_id],
        )?;
        drop(conn);

        self.log_op(
            "category_reset",
            None,
            &serde_json::json!({
                "category_id": cat_id,
                "category_name": category_name,
                "members_removed": removed,
            }),
            None,
        )?;

        Ok(removed)
    }

    // ── Internal helpers for substitution categories ──

    fn find_member_category(&self, token: &str) -> Option<(String, String)> {
        self.conn().query_row(
            "SELECT c.id, c.name FROM substitution_members m
             JOIN substitution_categories c ON c.id = m.category_id
             WHERE m.token_normalized = ?1 AND m.status = 'active'
             LIMIT 1",
            params![token],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        ).ok()
    }

    fn add_member_to_category(
        &self,
        cat_id: &str,
        normalized: &str,
        display: &str,
        confidence: f64,
        source: &str,
        ts: f64,
        hlc_bytes: &[u8],
        actor: &str,
    ) -> Result<bool> {
        let member_id = crate::id::new_id();
        let status = if source == "llm_suggested" { "pending" } else { "active" };
        let rows = self.conn().execute(
            "INSERT OR IGNORE INTO substitution_members
             (id, category_id, token_normalized, token_display, confidence,
              source, status, context_hint, created_at, updated_at, hlc, origin_actor)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, NULL, ?8, ?8, ?9, ?10)",
            params![member_id, cat_id, normalized, display, confidence, source, status, ts, hlc_bytes, actor],
        )?;
        Ok(rows > 0)
    }

    fn count_reclassify_pair_occurrences(&self, token_a: &str, token_b: &str) -> usize {
        // Count how many times this pair appeared in conflict_reclassify oplog events
        let count: i64 = self.conn().query_row(
            "SELECT COUNT(*) FROM oplog
             WHERE op_type = 'conflict_reclassify'
               AND (json_extract(payload, '$.diff_a') LIKE ?1
                    OR json_extract(payload, '$.diff_b') LIKE ?1)
               AND (json_extract(payload, '$.diff_a') LIKE ?2
                    OR json_extract(payload, '$.diff_b') LIKE ?2)",
            params![
                format!("%{}%", token_a),
                format!("%{}%", token_b),
            ],
            |row| row.get(0),
        ).unwrap_or(0);
        count as usize
    }

    /// Dismiss a conflict (mark as not-a-conflict).
    pub fn dismiss_conflict(&self, conflict_id: &str, note: Option<&str>) -> Result<()> {
        let ts = now();
        let actor_id = self.actor_id.clone();

        self.conn().execute(
            "UPDATE conflicts SET
             status = 'dismissed',
             resolved_at = ?1,
             resolved_by = ?2,
             strategy = 'keep_both',
             resolution_note = ?3
             WHERE conflict_id = ?4 AND status = 'open'",
            params![ts, actor_id, note, conflict_id],
        )?;

        self.log_op(
            "conflict_resolve",
            Some(conflict_id),
            &serde_json::json!({
                "conflict_id": conflict_id,
                "strategy": "keep_both",
                "resolution_note": note,
                "resolved_at": ts,
                "resolved_by": actor_id,
                "dismissed": true,
            }),
            None,
        )?;

        Ok(())
    }
}
