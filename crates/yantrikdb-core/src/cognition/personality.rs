use rusqlite::params;

use crate::engine::YantrikDB;
use crate::error::Result;
use crate::types::{PersonalityProfile, PersonalityTrait};

/// EMA blending factor — personality shifts gradually.
const EMA_ALPHA: f64 = 0.15;

/// Derive personality traits from memory signals and update the DB via EMA blending.
/// Returns the updated personality profile.
pub fn derive_personality(db: &YantrikDB) -> Result<PersonalityProfile> {
    let ts = crate::engine::now();

    let warmth = derive_warmth(db)?;
    let depth = derive_depth(db)?;
    let energy = derive_energy(db)?;
    let attentiveness = derive_attentiveness(db)?;

    let raw_traits = [
        ("warmth", warmth),
        ("depth", depth),
        ("energy", energy),
        ("attentiveness", attentiveness),
    ];

    let mut profile_traits = Vec::with_capacity(4);

    let conn = db.conn();
    for (name, raw_score) in &raw_traits {
        // Read existing trait
        let (old_score, old_count): (f64, i64) = conn.query_row(
            "SELECT score, sample_count FROM personality_traits WHERE trait_name = ?1",
            params![name],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;

        let new_count = old_count + 1;
        // EMA blend: gradually shift toward raw signal
        let blended = if old_count == 0 {
            *raw_score // First derivation: use raw directly
        } else {
            EMA_ALPHA * raw_score + (1.0 - EMA_ALPHA) * old_score
        };
        let blended = blended.clamp(0.0, 1.0);

        // Confidence grows with samples, capped at 1.0
        let confidence = (new_count as f64 / 20.0).min(1.0);

        conn.execute(
            "UPDATE personality_traits SET score = ?1, confidence = ?2, sample_count = ?3, updated_at = ?4 \
             WHERE trait_name = ?5",
            params![blended, confidence, new_count, ts, name],
        )?;

        profile_traits.push(PersonalityTrait {
            trait_name: name.to_string(),
            score: blended,
            confidence,
            sample_count: new_count,
            updated_at: ts,
        });
    }

    Ok(PersonalityProfile {
        traits: profile_traits,
        updated_at: ts,
    })
}

/// Read the current personality profile from the DB without recomputing.
pub fn get_personality(db: &YantrikDB) -> Result<PersonalityProfile> {
    let conn = db.conn();
    let mut stmt = conn.prepare(
        "SELECT trait_name, score, confidence, sample_count, updated_at \
         FROM personality_traits ORDER BY trait_name",
    )?;

    let traits: Vec<PersonalityTrait> = stmt
        .query_map([], |row| {
            Ok(PersonalityTrait {
                trait_name: row.get(0)?,
                score: row.get(1)?,
                confidence: row.get(2)?,
                sample_count: row.get(3)?,
                updated_at: row.get(4)?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let updated_at = traits.iter().map(|t| t.updated_at).fold(0.0_f64, f64::max);

    Ok(PersonalityProfile { traits, updated_at })
}

/// Manually set a personality trait score (for testing/override).
pub fn set_personality_trait(db: &YantrikDB, name: &str, score: f64) -> Result<bool> {
    let ts = crate::engine::now();
    let score = score.clamp(0.0, 1.0);
    let changes = db.conn().execute(
        "UPDATE personality_traits SET score = ?1, updated_at = ?2 WHERE trait_name = ?3",
        params![score, ts, name],
    )?;
    Ok(changes > 0)
}

// ── Trait derivation functions ──

/// Warmth: average valence of recent memories mapped from [-1,1] to [0,1].
fn derive_warmth(db: &YantrikDB) -> Result<f64> {
    let result: f64 = db.conn().query_row(
        "SELECT COALESCE(AVG(valence), 0.0) FROM (
            SELECT valence FROM memories
            WHERE consolidation_status = 'active'
            ORDER BY created_at DESC LIMIT 100
        )",
        [],
        |row| row.get(0),
    )?;
    // Map [-1, 1] → [0, 1]
    Ok((result + 1.0) / 2.0)
}

/// Depth: distinct domain count + entity relationship complexity.
fn derive_depth(db: &YantrikDB) -> Result<f64> {
    // Count distinct domains in last 100 memories
    let domain_count: i64 = db.conn().query_row(
        "SELECT COUNT(DISTINCT domain) FROM (
            SELECT domain FROM memories
            WHERE consolidation_status = 'active'
            ORDER BY created_at DESC LIMIT 100
        )",
        [],
        |row| row.get(0),
    )?;

    // Count distinct entities mentioned
    let entity_count: i64 = db.conn().query_row(
        "SELECT COUNT(*) FROM entities",
        [],
        |row| row.get(0),
    )?;

    // Map: 1 domain → 0.2, 5+ → 0.9
    let domain_score = ((domain_count as f64 - 1.0) / 4.0 * 0.7 + 0.2).clamp(0.1, 0.9);

    // Entity factor: 0-50 entities scales 0.0-0.3 bonus
    let entity_factor = (entity_count as f64 / 50.0 * 0.3).min(0.3);

    Ok((domain_score + entity_factor).clamp(0.0, 1.0))
}

/// Energy: interaction frequency (memories per day over last 7 days).
fn derive_energy(db: &YantrikDB) -> Result<f64> {
    let ts = crate::engine::now();
    let seven_days_ago = ts - 7.0 * 86400.0;

    let count: i64 = db.conn().query_row(
        "SELECT COUNT(*) FROM memories WHERE created_at > ?1 AND consolidation_status = 'active'",
        params![seven_days_ago],
        |row| row.get(0),
    )?;

    let per_day = count as f64 / 7.0;
    // Map: 0 msgs/day → 0.1, 10+ → 0.9
    Ok((per_day / 10.0 * 0.8 + 0.1).clamp(0.1, 0.9))
}

/// Attentiveness: active pattern count + conflict resolution rate.
fn derive_attentiveness(db: &YantrikDB) -> Result<f64> {
    // Count active patterns with confidence > 0.5
    let pattern_count: i64 = db.conn().query_row(
        "SELECT COUNT(*) FROM patterns WHERE status = 'active' AND confidence > 0.5",
        [],
        |row| row.get(0),
    )?;

    // Conflict resolution rate
    let total_conflicts: i64 = db.conn().query_row(
        "SELECT COUNT(*) FROM conflicts",
        [],
        |row| row.get(0),
    )?;
    let resolved_conflicts: i64 = db.conn().query_row(
        "SELECT COUNT(*) FROM conflicts WHERE status = 'resolved'",
        [],
        |row| row.get(0),
    )?;

    // Map: 0 patterns → 0.2, 5+ → 0.9
    let pattern_score = ((pattern_count as f64) / 5.0 * 0.7 + 0.2).clamp(0.2, 0.9);

    // Resolution rate factor (0.0-0.2 bonus)
    let resolution_factor = if total_conflicts > 0 {
        (resolved_conflicts as f64 / total_conflicts as f64) * 0.2
    } else {
        0.0
    };

    Ok((pattern_score + resolution_factor).clamp(0.0, 1.0))
}
