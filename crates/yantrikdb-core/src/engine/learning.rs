//! Adaptive weight optimization via coordinate descent on recall feedback.
//!
//! When enough feedback has been accumulated (>= 20 rows), the learning loop
//! perturbs each weight ±5%, keeps changes that reduce the ranking loss, and
//! stores updated weights back to the database. Weight changes are capped at
//! ±0.05 per generation to prevent instability.
//!
//! The loss function uses **pairwise ranking**: within each query, every
//! (relevant, irrelevant) pair contributes a hinge loss that penalizes the
//! irrelevant memory scoring above the relevant one. This is more robust than
//! absolute threshold loss because it doesn't require knowing the "right"
//! score values — only that relevant results should rank above irrelevant ones.

use std::collections::HashMap;
use rusqlite::params;

use crate::error::Result;
use crate::scoring;
use crate::types::LearnedWeights;

use super::{now, YantrikDB};

/// Minimum feedback rows before learning kicks in.
const MIN_FEEDBACK: i64 = 20;

/// Maximum weight delta per generation.
const MAX_DELTA: f64 = 0.05;

/// Perturbation factor for coordinate descent.
const PERTURB: f64 = 0.05;

/// Margin for pairwise ranking loss: relevant should beat irrelevant by this much.
const RANKING_MARGIN: f64 = 0.05;

/// A feedback row loaded for the optimization loop.
struct FeedbackRow {
    query_text: Option<String>,
    rid: String,
    feedback: String,        // "relevant" or "irrelevant"
    score_at_retrieval: Option<f64>,
}

/// Pre-computed feature set for a feedback entry, used by the loss function.
struct FeedbackFeatures {
    estimated_sim: f64,
    decay: f64,
    recency: f64,
    importance: f64,
    valence: f64,
    feedback: String,
    query_group: usize,      // index into query groups for pairwise loss
}

impl YantrikDB {
    /// Run the adaptive learning loop if enough feedback exists.
    ///
    /// Returns true if weights were updated, false if skipped.
    pub fn run_learning(&self) -> Result<bool> {
        let count = self.feedback_count()?;

        if count < MIN_FEEDBACK {
            return Ok(false);
        }

        let current = self.load_learned_weights()?;
        let feedback = self.load_feedback()?;

        if feedback.is_empty() {
            return Ok(false);
        }

        // Assign query group indices
        let mut query_groups: HashMap<String, usize> = HashMap::new();
        for fb in &feedback {
            let key = fb.query_text.clone().unwrap_or_default();
            let next_id = query_groups.len();
            query_groups.entry(key).or_insert(next_id);
        }

        // Pre-compute features for all feedback entries
        let features = self.build_features(&current, &feedback, &query_groups);

        if features.is_empty() {
            return Ok(false);
        }

        // Compute current loss
        let current_loss = compute_loss(&current, &features);

        // Coordinate descent: try perturbing each weight
        let mut best = current.clone();
        let mut best_loss = current_loss;

        let perturbations: &[(&str, f64)] = &[
            ("w_sim", PERTURB),
            ("w_sim", -PERTURB),
            ("w_decay", PERTURB),
            ("w_decay", -PERTURB),
            ("w_recency", PERTURB),
            ("w_recency", -PERTURB),
            ("gate_tau", PERTURB),
            ("gate_tau", -PERTURB),
            ("alpha_imp", PERTURB),
            ("alpha_imp", -PERTURB),
        ];

        for &(field, delta) in perturbations {
            let mut candidate = best.clone();
            match field {
                "w_sim" => candidate.w_sim = (candidate.w_sim * (1.0 + delta)).clamp(0.05, 0.90),
                "w_decay" => candidate.w_decay = (candidate.w_decay * (1.0 + delta)).clamp(0.05, 0.90),
                "w_recency" => candidate.w_recency = (candidate.w_recency * (1.0 + delta)).clamp(0.05, 0.90),
                "gate_tau" => candidate.gate_tau = (candidate.gate_tau + delta * 0.1).clamp(0.10, 0.50),
                "alpha_imp" => candidate.alpha_imp = (candidate.alpha_imp * (1.0 + delta)).clamp(0.10, 1.50),
                _ => {}
            }

            // Normalize w_sim + w_decay + w_recency = 1.0
            let sum = candidate.w_sim + candidate.w_decay + candidate.w_recency;
            if sum > 0.0 {
                candidate.w_sim /= sum;
                candidate.w_decay /= sum;
                candidate.w_recency /= sum;
            }

            let loss = compute_loss(&candidate, &features);
            if loss < best_loss {
                best = candidate;
                best_loss = loss;
            }
        }

        // Cap weight changes at MAX_DELTA
        best.w_sim = clamp_delta(best.w_sim, current.w_sim, MAX_DELTA);
        best.w_decay = clamp_delta(best.w_decay, current.w_decay, MAX_DELTA);
        best.w_recency = clamp_delta(best.w_recency, current.w_recency, MAX_DELTA);
        best.gate_tau = clamp_delta(best.gate_tau, current.gate_tau, MAX_DELTA);
        best.alpha_imp = clamp_delta(best.alpha_imp, current.alpha_imp, MAX_DELTA);

        // Re-normalize after clamping
        let sum = best.w_sim + best.w_decay + best.w_recency;
        if sum > 0.0 {
            best.w_sim /= sum;
            best.w_decay /= sum;
            best.w_recency /= sum;
        }

        // Only save if loss improved
        if best_loss < current_loss {
            best.generation = current.generation + 1;
            self.save_learned_weights(&best)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Load feedback rows from the database.
    fn load_feedback(&self) -> Result<Vec<FeedbackRow>> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare(
            "SELECT query_text, rid, feedback, score_at_retrieval FROM recall_feedback \
             ORDER BY created_at DESC LIMIT 500",
        )?;

        let rows = stmt.query_map([], |row| {
            Ok(FeedbackRow {
                query_text: row.get(0)?,
                rid: row.get(1)?,
                feedback: row.get(2)?,
                score_at_retrieval: row.get(3)?,
            })
        })?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Build pre-computed feature vectors for all feedback entries.
    ///
    /// For entries with a stored `score_at_retrieval`, we back-calculate the
    /// original similarity using the initial weights:
    ///   base = score / (1 + alpha_imp * importance)
    ///   sim  = (base - w_decay * decay - w_recency * recency) / w_sim
    fn build_features(
        &self,
        initial_weights: &LearnedWeights,
        feedback: &[FeedbackRow],
        query_groups: &HashMap<String, usize>,
    ) -> Vec<FeedbackFeatures> {
        let cache = self.scoring_cache.read();
        let ts = now();
        let mut features = Vec::with_capacity(feedback.len());

        for fb in feedback {
            let Some(row) = cache.get(&fb.rid) else { continue };

            let elapsed = ts - row.last_access;
            let decay = scoring::decay_score(row.importance, row.half_life, elapsed);
            let age = ts - row.created_at;
            let recency = scoring::recency_score(age);

            let estimated_sim = if let Some(score) = fb.score_at_retrieval {
                // Back-calculate similarity from stored composite score
                let imp_gate = 1.0 + initial_weights.alpha_imp * row.importance;
                let base = if imp_gate > 0.0 { score / imp_gate } else { score };
                let sim = if initial_weights.w_sim > 0.0 {
                    (base - initial_weights.w_decay * decay - initial_weights.w_recency * recency)
                        / initial_weights.w_sim
                } else {
                    0.5
                };
                sim.clamp(-1.0, 1.0)
            } else {
                0.5
            };

            let key = fb.query_text.clone().unwrap_or_default();
            let group = *query_groups.get(&key).unwrap_or(&0);

            features.push(FeedbackFeatures {
                estimated_sim,
                decay,
                recency,
                importance: row.importance,
                valence: row.valence,
                feedback: fb.feedback.clone(),
                query_group: group,
            });
        }
        features
    }

    /// Save updated weights to the database.
    fn save_learned_weights(&self, weights: &LearnedWeights) -> Result<()> {
        let ts = now();
        let conn = self.conn.lock();
        conn.execute(
            "UPDATE learned_weights SET \
             w_sim = ?1, w_decay = ?2, w_recency = ?3, \
             gate_tau = ?4, alpha_imp = ?5, keyword_boost = ?6, \
             updated_at = ?7, generation = ?8 \
             WHERE id = 1",
            params![
                weights.w_sim,
                weights.w_decay,
                weights.w_recency,
                weights.gate_tau,
                weights.alpha_imp,
                weights.keyword_boost,
                ts,
                weights.generation,
            ],
        )?;
        Ok(())
    }
}

/// Compute pairwise ranking loss from pre-computed features.
///
/// For each query group, generate all (relevant, irrelevant) pairs.
/// Loss = mean of max(0, margin - (score_relevant - score_irrelevant))^2
///
/// This directly optimizes for ranking: relevant memories should score
/// at least `margin` above irrelevant ones within the same query.
fn compute_loss(weights: &LearnedWeights, features: &[FeedbackFeatures]) -> f64 {
    // Group features by query
    let mut groups: HashMap<usize, (Vec<usize>, Vec<usize>)> = HashMap::new();
    for (i, f) in features.iter().enumerate() {
        let entry = groups.entry(f.query_group).or_insert_with(|| (Vec::new(), Vec::new()));
        match f.feedback.as_str() {
            "relevant" => entry.0.push(i),
            "irrelevant" => entry.1.push(i),
            _ => {}
        }
    }

    // Pre-compute all scores once
    let scores: Vec<f64> = features.iter().map(|f| {
        scoring::adaptive_composite_score(
            f.estimated_sim, f.decay, f.recency, f.importance, f.valence, 0.0, weights,
        )
    }).collect();

    let mut total_loss = 0.0;
    let mut pair_count = 0;

    for (_group, (relevant_indices, irrelevant_indices)) in &groups {
        // Only compute loss for groups that have both relevant and irrelevant
        if relevant_indices.is_empty() || irrelevant_indices.is_empty() {
            continue;
        }

        for &ri in relevant_indices {
            for &ii in irrelevant_indices {
                let gap = RANKING_MARGIN - (scores[ri] - scores[ii]);
                if gap > 0.0 {
                    total_loss += gap * gap;
                }
                pair_count += 1;
            }
        }
    }

    if pair_count > 0 {
        total_loss / pair_count as f64
    } else {
        0.0
    }
}

/// Clamp `new` so it's within `max_delta` of `original`.
fn clamp_delta(new: f64, original: f64, max_delta: f64) -> f64 {
    new.clamp(original - max_delta, original + max_delta)
}
