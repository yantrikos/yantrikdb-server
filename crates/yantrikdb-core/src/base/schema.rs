pub const SCHEMA_VERSION: i32 = 13;

pub const SCHEMA_SQL: &str = "
-- Memory records: the source of truth
CREATE TABLE IF NOT EXISTS memories (
    rid TEXT PRIMARY KEY,                -- UUIDv7, stable across devices
    type TEXT NOT NULL DEFAULT 'episodic', -- episodic | semantic | procedural | emotional
    text TEXT NOT NULL,                  -- raw memory content
    embedding BLOB,                     -- vector embedding (float32 array)

    -- Temporal
    created_at REAL NOT NULL,           -- unix timestamp (float for sub-second)
    updated_at REAL NOT NULL,

    -- Decay parameters (stored, not continuously updated)
    importance REAL NOT NULL DEFAULT 0.5,  -- base importance I0 [0, 1]
    half_life REAL NOT NULL DEFAULT 604800.0, -- seconds (default: 7 days)
    last_access REAL NOT NULL,            -- unix timestamp of last recall/reinforce
    access_count INTEGER NOT NULL DEFAULT 0, -- number of times retrieved via recall
    valence REAL NOT NULL DEFAULT 0.0,    -- emotional weight [-1, 1]

    -- Consolidation tracking
    consolidated_into TEXT,              -- rid of the semantic memory this was merged into
    consolidation_status TEXT DEFAULT 'active', -- active | consolidated | tombstoned

    -- Storage tier
    storage_tier TEXT NOT NULL DEFAULT 'hot', -- hot | cold

    -- Metadata
    metadata TEXT DEFAULT '{}',          -- JSON blob for extensibility

    -- Namespace for memory isolation
    namespace TEXT NOT NULL DEFAULT 'default',

    -- Cognitive dimensions (V10)
    certainty REAL NOT NULL DEFAULT 0.8,     -- confidence in accuracy [0, 1]
    domain TEXT NOT NULL DEFAULT 'general',   -- topic domain (work, health, family, finance, etc.)
    source TEXT NOT NULL DEFAULT 'user',      -- origin (user, system, document, inference)
    emotional_state TEXT,                     -- rich emotion label (joy, sadness, anger, fear, etc.)

    -- Session & temporal (V13)
    session_id TEXT,                          -- FK to sessions.session_id (nullable)
    due_at REAL,                              -- unix timestamp for upcoming() queries
    temporal_kind TEXT                         -- deadline | reminder | event | follow_up
);

-- Session tracking (V13)
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'default',
    client_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    started_at REAL NOT NULL,
    ended_at REAL,
    summary TEXT,
    avg_valence REAL,
    memory_count INTEGER NOT NULL DEFAULT 0,
    topics TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    hlc BLOB,
    origin_actor TEXT
);

-- Entity relationship graph
CREATE TABLE IF NOT EXISTS edges (
    edge_id TEXT PRIMARY KEY,            -- UUIDv7
    src TEXT NOT NULL,                   -- entity name or memory rid
    dst TEXT NOT NULL,                   -- entity name or memory rid
    rel_type TEXT NOT NULL,              -- relationship type (e.g., \"is_about\", \"related_to\")
    weight REAL NOT NULL DEFAULT 1.0,    -- relationship strength [0, 1]
    created_at REAL NOT NULL,
    tombstoned INTEGER NOT NULL DEFAULT 0,

    UNIQUE(src, dst, rel_type)
);

-- Entities extracted from memories
CREATE TABLE IF NOT EXISTS entities (
    name TEXT PRIMARY KEY,               -- normalized entity name
    entity_type TEXT DEFAULT 'unknown',  -- person | place | thing | concept | etc.
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1,
    metadata TEXT DEFAULT '{}'
);

-- Append-only operation log (CRDT replication)
CREATE TABLE IF NOT EXISTS oplog (
    op_id TEXT PRIMARY KEY,              -- UUIDv7
    op_type TEXT NOT NULL,               -- record | relate | consolidate | decay | forget | update
    timestamp REAL NOT NULL,             -- when the operation occurred
    target_rid TEXT,                     -- primary memory affected
    payload TEXT NOT NULL DEFAULT '{}',  -- JSON: full operation details
    actor_id TEXT DEFAULT 'local',       -- device/agent identifier
    hlc BLOB,                           -- hybrid logical clock timestamp (16 bytes)
    embedding_hash BLOB,                -- BLAKE3 hash of embedding (if applicable)
    origin_actor TEXT NOT NULL DEFAULT 'local', -- which device originally created this op
    applied INTEGER NOT NULL DEFAULT 1  -- 1 = materialized locally, 0 = pending
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Peer tracking for delta sync
CREATE TABLE IF NOT EXISTS sync_peers (
    peer_actor TEXT PRIMARY KEY,
    last_synced_hlc BLOB NOT NULL,
    last_synced_op_id TEXT NOT NULL,
    last_sync_time REAL NOT NULL
);

-- Consolidation membership (set-union CRDT)
CREATE TABLE IF NOT EXISTS consolidation_members (
    consolidation_rid TEXT NOT NULL,     -- the consolidated memory
    source_rid TEXT NOT NULL,            -- original memory
    hlc BLOB NOT NULL,                  -- when this consolidation happened
    actor_id TEXT NOT NULL,             -- which device did it
    PRIMARY KEY (consolidation_rid, source_rid)
);

-- Conflict tracking (first-class data)
CREATE TABLE IF NOT EXISTS conflicts (
    conflict_id TEXT PRIMARY KEY,           -- UUIDv7
    conflict_type TEXT NOT NULL,            -- identity_fact | preference | temporal | consolidation | minor
    priority TEXT NOT NULL DEFAULT 'medium',-- low | medium | high | critical
    status TEXT NOT NULL DEFAULT 'open',    -- open | resolved | dismissed
    memory_a TEXT NOT NULL,                 -- rid of first conflicting memory
    memory_b TEXT NOT NULL,                 -- rid of second conflicting memory
    entity TEXT,                            -- entity name (nullable)
    rel_type TEXT,                          -- relationship type in conflict (nullable)
    detected_at REAL NOT NULL,
    detected_by TEXT NOT NULL,              -- actor_id that detected it
    detection_reason TEXT NOT NULL,
    resolved_at REAL,
    resolved_by TEXT,
    strategy TEXT,                          -- keep_a | keep_b | keep_both | merge | correct
    winner_rid TEXT,
    resolution_note TEXT,
    hlc BLOB NOT NULL,
    origin_actor TEXT NOT NULL
);

-- Persisted triggers with lifecycle tracking
CREATE TABLE IF NOT EXISTS trigger_log (
    trigger_id TEXT PRIMARY KEY,
    trigger_type TEXT NOT NULL,
    urgency REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    reason TEXT NOT NULL,
    suggested_action TEXT NOT NULL,
    source_rids TEXT NOT NULL DEFAULT '[]',
    context TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    delivered_at REAL,
    acknowledged_at REAL,
    acted_at REAL,
    expires_at REAL,
    cooldown_key TEXT,
    hlc BLOB NOT NULL,
    origin_actor TEXT NOT NULL
);

-- Detected patterns across memories
CREATE TABLE IF NOT EXISTS patterns (
    pattern_id TEXT PRIMARY KEY,
    pattern_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    confidence REAL NOT NULL,
    description TEXT NOT NULL,
    evidence_rids TEXT NOT NULL DEFAULT '[]',
    entity_names TEXT NOT NULL DEFAULT '[]',
    context TEXT NOT NULL DEFAULT '{}',
    first_seen REAL NOT NULL,
    last_confirmed REAL NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    hlc BLOB NOT NULL,
    origin_actor TEXT NOT NULL
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_memories_consolidation ON memories(consolidation_status);
CREATE INDEX IF NOT EXISTS idx_memories_storage_tier ON memories(storage_tier);
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
CREATE INDEX IF NOT EXISTS idx_memories_access_count ON memories(access_count);
CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain);
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
CREATE INDEX IF NOT EXISTS idx_memories_emotional_state ON memories(emotional_state);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(namespace, session_id);
CREATE INDEX IF NOT EXISTS idx_memories_due_at ON memories(namespace, due_at) WHERE due_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_last_access ON memories(last_access);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_one_active ON sessions(namespace, client_id) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_sessions_client_started ON sessions(namespace, client_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(rel_type);
CREATE INDEX IF NOT EXISTS idx_oplog_timestamp ON oplog(timestamp);
CREATE INDEX IF NOT EXISTS idx_oplog_target ON oplog(target_rid);
CREATE INDEX IF NOT EXISTS idx_oplog_hlc ON oplog(hlc);
CREATE INDEX IF NOT EXISTS idx_oplog_actor ON oplog(origin_actor);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_consolidation_source ON consolidation_members(source_rid);
CREATE INDEX IF NOT EXISTS idx_conflicts_status ON conflicts(status);
CREATE INDEX IF NOT EXISTS idx_conflicts_type ON conflicts(conflict_type);
CREATE INDEX IF NOT EXISTS idx_conflicts_priority ON conflicts(priority);
CREATE INDEX IF NOT EXISTS idx_conflicts_entity ON conflicts(entity);
CREATE INDEX IF NOT EXISTS idx_conflicts_memory_a ON conflicts(memory_a);
CREATE INDEX IF NOT EXISTS idx_conflicts_memory_b ON conflicts(memory_b);
CREATE INDEX IF NOT EXISTS idx_trigger_log_status ON trigger_log(status);
CREATE INDEX IF NOT EXISTS idx_trigger_log_type ON trigger_log(trigger_type);
CREATE INDEX IF NOT EXISTS idx_trigger_log_created ON trigger_log(created_at);
CREATE INDEX IF NOT EXISTS idx_trigger_log_cooldown ON trigger_log(cooldown_key);
CREATE INDEX IF NOT EXISTS idx_trigger_log_urgency ON trigger_log(urgency DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_status ON patterns(status);
CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON patterns(confidence DESC);

-- Memory-entity join table for graph-augmented recall
CREATE TABLE IF NOT EXISTS memory_entities (
    memory_rid TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    PRIMARY KEY (memory_rid, entity_name)
);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity_name);
CREATE INDEX IF NOT EXISTS idx_memory_entities_rid ON memory_entities(memory_rid);

-- FTS5 for full-text search on memories
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text, content=memories, content_rowid=rowid);

-- Auto-sync triggers for FTS5
CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_delete BEFORE DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE OF text ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
END;

-- Normalized join tables for trigger/pattern JSON arrays
CREATE TABLE IF NOT EXISTS trigger_source_rids (
    trigger_id TEXT NOT NULL,
    rid TEXT NOT NULL,
    PRIMARY KEY (trigger_id, rid)
);
CREATE INDEX IF NOT EXISTS idx_trigger_source_rids_rid ON trigger_source_rids(rid);

CREATE TABLE IF NOT EXISTS pattern_evidence (
    pattern_id TEXT NOT NULL,
    rid TEXT NOT NULL,
    PRIMARY KEY (pattern_id, rid)
);
CREATE INDEX IF NOT EXISTS idx_pattern_evidence_rid ON pattern_evidence(rid);

CREATE TABLE IF NOT EXISTS pattern_entities (
    pattern_id TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    PRIMARY KEY (pattern_id, entity_name)
);
CREATE INDEX IF NOT EXISTS idx_pattern_entities_entity ON pattern_entities(entity_name);

-- Recall feedback for adaptive learning (V10)
CREATE TABLE IF NOT EXISTS recall_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT,
    query_embedding BLOB,
    rid TEXT NOT NULL,
    feedback TEXT NOT NULL,              -- 'relevant' | 'irrelevant'
    score_at_retrieval REAL,
    rank_at_retrieval INTEGER,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON recall_feedback(created_at);

-- Learned scoring weights (singleton row, V10)
CREATE TABLE IF NOT EXISTS learned_weights (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    w_sim REAL NOT NULL DEFAULT 0.50,
    w_decay REAL NOT NULL DEFAULT 0.20,
    w_recency REAL NOT NULL DEFAULT 0.30,
    gate_tau REAL NOT NULL DEFAULT 0.25,
    alpha_imp REAL NOT NULL DEFAULT 0.80,
    keyword_boost REAL NOT NULL DEFAULT 0.31,
    updated_at REAL,
    feedback_count INTEGER DEFAULT 0,
    generation INTEGER DEFAULT 0
);
INSERT OR IGNORE INTO learned_weights (id) VALUES (1);

-- Personality traits (V11)
CREATE TABLE IF NOT EXISTS personality_traits (
    trait_name TEXT PRIMARY KEY,
    score REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.0,
    sample_count INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0.0
);
INSERT OR IGNORE INTO personality_traits (trait_name, score, confidence, sample_count, updated_at)
    VALUES ('warmth', 0.5, 0.0, 0, 0.0),
           ('depth', 0.5, 0.0, 0, 0.0),
           ('energy', 0.5, 0.0, 0, 0.0),
           ('attentiveness', 0.5, 0.0, 0, 0.0);

-- Cognitive State Graph: Nodes (V12)
CREATE TABLE IF NOT EXISTS cognitive_nodes (
    node_id INTEGER PRIMARY KEY,            -- compact NodeId (4-bit kind + 28-bit seq)
    kind TEXT NOT NULL,                      -- node kind string (entity, belief, goal, etc.)
    label TEXT NOT NULL,                     -- human-readable label
    -- Universal cognitive attributes
    confidence REAL NOT NULL DEFAULT 0.5,
    activation REAL NOT NULL DEFAULT 0.0,
    salience REAL NOT NULL DEFAULT 0.5,
    persistence REAL NOT NULL DEFAULT 0.5,
    valence REAL NOT NULL DEFAULT 0.0,
    urgency REAL NOT NULL DEFAULT 0.0,
    novelty REAL NOT NULL DEFAULT 1.0,
    volatility REAL NOT NULL DEFAULT 0.1,
    provenance TEXT NOT NULL DEFAULT 'observed',
    evidence_count INTEGER NOT NULL DEFAULT 1,
    last_updated_ms INTEGER NOT NULL,
    -- Kind-specific payload (JSON)
    payload TEXT NOT NULL DEFAULT '{}',
    -- Metadata (JSON)
    metadata TEXT NOT NULL DEFAULT '{}',
    -- Lifecycle
    created_at REAL NOT NULL,
    tombstoned INTEGER NOT NULL DEFAULT 0,
    -- Replication
    hlc BLOB,
    origin_actor TEXT
);
CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_kind ON cognitive_nodes(kind);
CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_activation ON cognitive_nodes(activation);
CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_urgency ON cognitive_nodes(urgency);

-- Cognitive State Graph: Edges (V12)
CREATE TABLE IF NOT EXISTS cognitive_edges (
    src_id INTEGER NOT NULL,                 -- source NodeId
    dst_id INTEGER NOT NULL,                 -- destination NodeId
    kind TEXT NOT NULL,                      -- edge kind string (supports, contradicts, etc.)
    weight REAL NOT NULL DEFAULT 0.5,        -- edge weight [-1.0, 1.0]
    confidence REAL NOT NULL DEFAULT 0.5,
    observation_count INTEGER NOT NULL DEFAULT 1,
    created_at_ms INTEGER NOT NULL,
    last_confirmed_ms INTEGER NOT NULL,
    tombstoned INTEGER NOT NULL DEFAULT 0,
    hlc BLOB,
    origin_actor TEXT,
    PRIMARY KEY (src_id, dst_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_cognitive_edges_dst ON cognitive_edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_cognitive_edges_kind ON cognitive_edges(kind);

-- High-water marks for NodeId allocator (V12)
CREATE TABLE IF NOT EXISTS cognitive_node_hwm (
    kind TEXT PRIMARY KEY,                   -- node kind string
    high_water_mark INTEGER NOT NULL DEFAULT 0
);
";

/// SQL to migrate from schema V1 to V2.
pub const MIGRATE_V1_TO_V2: &str = "
ALTER TABLE oplog ADD COLUMN hlc BLOB;
ALTER TABLE oplog ADD COLUMN embedding_hash BLOB;
ALTER TABLE oplog ADD COLUMN origin_actor TEXT NOT NULL DEFAULT 'local';
ALTER TABLE oplog ADD COLUMN applied INTEGER NOT NULL DEFAULT 1;

CREATE INDEX IF NOT EXISTS idx_oplog_hlc ON oplog(hlc);
CREATE INDEX IF NOT EXISTS idx_oplog_actor ON oplog(origin_actor);

CREATE TABLE IF NOT EXISTS sync_peers (
    peer_actor TEXT PRIMARY KEY,
    last_synced_hlc BLOB NOT NULL,
    last_synced_op_id TEXT NOT NULL,
    last_sync_time REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS consolidation_members (
    consolidation_rid TEXT NOT NULL,
    source_rid TEXT NOT NULL,
    hlc BLOB NOT NULL,
    actor_id TEXT NOT NULL,
    PRIMARY KEY (consolidation_rid, source_rid)
);
CREATE INDEX IF NOT EXISTS idx_consolidation_source ON consolidation_members(source_rid);
";

/// SQL to migrate from schema V2 to V3.
pub const MIGRATE_V2_TO_V3: &str = "
CREATE TABLE IF NOT EXISTS conflicts (
    conflict_id TEXT PRIMARY KEY,
    conflict_type TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'medium',
    status TEXT NOT NULL DEFAULT 'open',
    memory_a TEXT NOT NULL,
    memory_b TEXT NOT NULL,
    entity TEXT,
    rel_type TEXT,
    detected_at REAL NOT NULL,
    detected_by TEXT NOT NULL,
    detection_reason TEXT NOT NULL,
    resolved_at REAL,
    resolved_by TEXT,
    strategy TEXT,
    winner_rid TEXT,
    resolution_note TEXT,
    hlc BLOB NOT NULL,
    origin_actor TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conflicts_status ON conflicts(status);
CREATE INDEX IF NOT EXISTS idx_conflicts_type ON conflicts(conflict_type);
CREATE INDEX IF NOT EXISTS idx_conflicts_priority ON conflicts(priority);
CREATE INDEX IF NOT EXISTS idx_conflicts_entity ON conflicts(entity);
CREATE INDEX IF NOT EXISTS idx_conflicts_memory_a ON conflicts(memory_a);
CREATE INDEX IF NOT EXISTS idx_conflicts_memory_b ON conflicts(memory_b);
";

/// SQL to migrate from schema V3 to V4.
pub const MIGRATE_V3_TO_V4: &str = "
CREATE TABLE IF NOT EXISTS trigger_log (
    trigger_id TEXT PRIMARY KEY,
    trigger_type TEXT NOT NULL,
    urgency REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    reason TEXT NOT NULL,
    suggested_action TEXT NOT NULL,
    source_rids TEXT NOT NULL DEFAULT '[]',
    context TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    delivered_at REAL,
    acknowledged_at REAL,
    acted_at REAL,
    expires_at REAL,
    cooldown_key TEXT,
    hlc BLOB NOT NULL,
    origin_actor TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patterns (
    pattern_id TEXT PRIMARY KEY,
    pattern_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    confidence REAL NOT NULL,
    description TEXT NOT NULL,
    evidence_rids TEXT NOT NULL DEFAULT '[]',
    entity_names TEXT NOT NULL DEFAULT '[]',
    context TEXT NOT NULL DEFAULT '{}',
    first_seen REAL NOT NULL,
    last_confirmed REAL NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    hlc BLOB NOT NULL,
    origin_actor TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trigger_log_status ON trigger_log(status);
CREATE INDEX IF NOT EXISTS idx_trigger_log_type ON trigger_log(trigger_type);
CREATE INDEX IF NOT EXISTS idx_trigger_log_created ON trigger_log(created_at);
CREATE INDEX IF NOT EXISTS idx_trigger_log_cooldown ON trigger_log(cooldown_key);
CREATE INDEX IF NOT EXISTS idx_trigger_log_urgency ON trigger_log(urgency DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_status ON patterns(status);
CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON patterns(confidence DESC);
";

/// SQL to migrate from schema V4 to V5.
pub const MIGRATE_V4_TO_V5: &str = "
CREATE TABLE IF NOT EXISTS memory_entities (
    memory_rid TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    PRIMARY KEY (memory_rid, entity_name)
);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity_name);
CREATE INDEX IF NOT EXISTS idx_memory_entities_rid ON memory_entities(memory_rid);
";

/// SQL to migrate from schema V5 to V6.
pub const MIGRATE_V5_TO_V6: &str = "
ALTER TABLE memories ADD COLUMN storage_tier TEXT NOT NULL DEFAULT 'hot';
CREATE INDEX IF NOT EXISTS idx_memories_storage_tier ON memories(storage_tier);
";

/// SQL to migrate from schema V6 to V7.
pub const MIGRATE_V6_TO_V7: &str = "
-- FTS5 for full-text search on memories
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text, content=memories, content_rowid=rowid);

-- Populate FTS5 from existing data
INSERT INTO memories_fts(memories_fts) VALUES('rebuild');

-- Auto-sync triggers for FTS5
CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_delete BEFORE DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE OF text ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
END;

-- Normalized join tables
CREATE TABLE IF NOT EXISTS trigger_source_rids (
    trigger_id TEXT NOT NULL,
    rid TEXT NOT NULL,
    PRIMARY KEY (trigger_id, rid)
);
CREATE INDEX IF NOT EXISTS idx_trigger_source_rids_rid ON trigger_source_rids(rid);

CREATE TABLE IF NOT EXISTS pattern_evidence (
    pattern_id TEXT NOT NULL,
    rid TEXT NOT NULL,
    PRIMARY KEY (pattern_id, rid)
);
CREATE INDEX IF NOT EXISTS idx_pattern_evidence_rid ON pattern_evidence(rid);

CREATE TABLE IF NOT EXISTS pattern_entities (
    pattern_id TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    PRIMARY KEY (pattern_id, entity_name)
);
CREATE INDEX IF NOT EXISTS idx_pattern_entities_entity ON pattern_entities(entity_name);

-- Backfill join tables from JSON columns
INSERT OR IGNORE INTO trigger_source_rids (trigger_id, rid)
    SELECT trigger_id, json_each.value FROM trigger_log, json_each(source_rids)
    WHERE source_rids IS NOT NULL AND source_rids != '[]';

INSERT OR IGNORE INTO pattern_evidence (pattern_id, rid)
    SELECT pattern_id, json_each.value FROM patterns, json_each(evidence_rids)
    WHERE evidence_rids IS NOT NULL AND evidence_rids != '[]';

INSERT OR IGNORE INTO pattern_entities (pattern_id, entity_name)
    SELECT pattern_id, json_each.value FROM patterns, json_each(entity_names)
    WHERE entity_names IS NOT NULL AND entity_names != '[]';
";

/// SQL to migrate from schema V7 to V8.
pub const MIGRATE_V7_TO_V8: &str = "
ALTER TABLE memories ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
";

/// SQL to migrate from schema V8 to V9.
pub const MIGRATE_V8_TO_V9: &str = "
ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_memories_access_count ON memories(access_count);
";

/// SQL to migrate from schema V9 to V10.
pub const MIGRATE_V9_TO_V10: &str = "
-- New cognitive dimension columns
ALTER TABLE memories ADD COLUMN certainty REAL NOT NULL DEFAULT 0.8;
ALTER TABLE memories ADD COLUMN domain TEXT NOT NULL DEFAULT 'general';
ALTER TABLE memories ADD COLUMN source TEXT NOT NULL DEFAULT 'user';
ALTER TABLE memories ADD COLUMN emotional_state TEXT;
CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain);
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
CREATE INDEX IF NOT EXISTS idx_memories_emotional_state ON memories(emotional_state);

-- Recall feedback for adaptive learning
CREATE TABLE IF NOT EXISTS recall_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT,
    query_embedding BLOB,
    rid TEXT NOT NULL,
    feedback TEXT NOT NULL,
    score_at_retrieval REAL,
    rank_at_retrieval INTEGER,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON recall_feedback(created_at);

-- Learned scoring weights (singleton)
CREATE TABLE IF NOT EXISTS learned_weights (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    w_sim REAL NOT NULL DEFAULT 0.50,
    w_decay REAL NOT NULL DEFAULT 0.20,
    w_recency REAL NOT NULL DEFAULT 0.30,
    gate_tau REAL NOT NULL DEFAULT 0.25,
    alpha_imp REAL NOT NULL DEFAULT 0.80,
    keyword_boost REAL NOT NULL DEFAULT 0.31,
    updated_at REAL,
    feedback_count INTEGER DEFAULT 0,
    generation INTEGER DEFAULT 0
);
INSERT OR IGNORE INTO learned_weights (id) VALUES (1);
";

/// SQL to migrate from schema V10 to V11.
pub const MIGRATE_V10_TO_V11: &str = "
-- Personality traits derived from memory signals
CREATE TABLE IF NOT EXISTS personality_traits (
    trait_name TEXT PRIMARY KEY,
    score REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.0,
    sample_count INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0.0
);
INSERT OR IGNORE INTO personality_traits (trait_name, score, confidence, sample_count, updated_at)
    VALUES ('warmth', 0.5, 0.0, 0, 0.0),
           ('depth', 0.5, 0.0, 0, 0.0),
           ('energy', 0.5, 0.0, 0, 0.0),
           ('attentiveness', 0.5, 0.0, 0, 0.0);
";

/// SQL to migrate from schema V11 to V12.
pub const MIGRATE_V11_TO_V12: &str = "
-- Cognitive State Graph: Nodes
CREATE TABLE IF NOT EXISTS cognitive_nodes (
    node_id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    activation REAL NOT NULL DEFAULT 0.0,
    salience REAL NOT NULL DEFAULT 0.5,
    persistence REAL NOT NULL DEFAULT 0.5,
    valence REAL NOT NULL DEFAULT 0.0,
    urgency REAL NOT NULL DEFAULT 0.0,
    novelty REAL NOT NULL DEFAULT 1.0,
    volatility REAL NOT NULL DEFAULT 0.1,
    provenance TEXT NOT NULL DEFAULT 'observed',
    evidence_count INTEGER NOT NULL DEFAULT 1,
    last_updated_ms INTEGER NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    tombstoned INTEGER NOT NULL DEFAULT 0,
    hlc BLOB,
    origin_actor TEXT
);
CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_kind ON cognitive_nodes(kind);
CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_activation ON cognitive_nodes(activation);
CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_urgency ON cognitive_nodes(urgency);

-- Cognitive State Graph: Edges
CREATE TABLE IF NOT EXISTS cognitive_edges (
    src_id INTEGER NOT NULL,
    dst_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.5,
    observation_count INTEGER NOT NULL DEFAULT 1,
    created_at_ms INTEGER NOT NULL,
    last_confirmed_ms INTEGER NOT NULL,
    tombstoned INTEGER NOT NULL DEFAULT 0,
    hlc BLOB,
    origin_actor TEXT,
    PRIMARY KEY (src_id, dst_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_cognitive_edges_dst ON cognitive_edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_cognitive_edges_kind ON cognitive_edges(kind);

-- High-water marks for NodeId allocator
CREATE TABLE IF NOT EXISTS cognitive_node_hwm (
    kind TEXT PRIMARY KEY,
    high_water_mark INTEGER NOT NULL DEFAULT 0
);
";

/// SQL to migrate from schema V12 to V13.
pub const MIGRATE_V12_TO_V13: &str = "
-- Session tracking
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'default',
    client_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    started_at REAL NOT NULL,
    ended_at REAL,
    summary TEXT,
    avg_valence REAL,
    memory_count INTEGER NOT NULL DEFAULT 0,
    topics TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    hlc BLOB,
    origin_actor TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_one_active
    ON sessions(namespace, client_id) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_sessions_client_started
    ON sessions(namespace, client_id, started_at DESC);

-- Memories: session & temporal columns
ALTER TABLE memories ADD COLUMN session_id TEXT;
ALTER TABLE memories ADD COLUMN due_at REAL;
ALTER TABLE memories ADD COLUMN temporal_kind TEXT;
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(namespace, session_id);
CREATE INDEX IF NOT EXISTS idx_memories_due_at ON memories(namespace, due_at)
    WHERE due_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_last_access ON memories(last_access);
";
