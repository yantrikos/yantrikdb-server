pub const SCHEMA_VERSION: i32 = 3;

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
    valence REAL NOT NULL DEFAULT 0.0,    -- emotional weight [-1, 1]

    -- Consolidation tracking
    consolidated_into TEXT,              -- rid of the semantic memory this was merged into
    consolidation_status TEXT DEFAULT 'active', -- active | consolidated | tombstoned

    -- Metadata
    metadata TEXT DEFAULT '{}'           -- JSON blob for extensibility
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

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_memories_consolidation ON memories(consolidation_status);
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
