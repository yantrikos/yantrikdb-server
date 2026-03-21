/* tslint:disable */
/* eslint-disable */

/**
 * In-browser YantrikDB instance. All data lives in memory (SQLite :memory:).
 */
export class WasmYantrikDB {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Forget (tombstone) a memory by RID.
     */
    forget(rid: string): boolean;
    /**
     * Get edges for an entity. Returns JSON array.
     */
    get_edges(entity: string): any;
    /**
     * Create a new in-memory YantrikDB instance.
     */
    constructor(embedding_dim: number);
    /**
     * Recall memories by embedding similarity. Returns JSON array.
     */
    recall(embedding: Float32Array, top_k: number): any;
    /**
     * Store a memory with its embedding vector.
     */
    record(text: string, embedding: Float32Array, importance: number, valence: number, memory_type: string): string;
    /**
     * Create a relationship between entities.
     */
    relate(src: string, dst: string, rel_type: string, weight: number): string;
    /**
     * Search entities by name pattern. Returns JSON array.
     */
    search_entities(pattern: string | null | undefined, limit: number): any;
    /**
     * Get engine statistics. Returns JSON.
     */
    stats(): any;
    /**
     * Run the cognition loop. Returns JSON with triggers, conflicts, patterns.
     */
    think(): any;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmyantrikdb_free: (a: number, b: number) => void;
    readonly wasmyantrikdb_forget: (a: number, b: number, c: number, d: number) => void;
    readonly wasmyantrikdb_get_edges: (a: number, b: number, c: number, d: number) => void;
    readonly wasmyantrikdb_new: (a: number, b: number) => void;
    readonly wasmyantrikdb_recall: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly wasmyantrikdb_record: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number) => void;
    readonly wasmyantrikdb_relate: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
    readonly wasmyantrikdb_search_entities: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly wasmyantrikdb_stats: (a: number, b: number) => void;
    readonly wasmyantrikdb_think: (a: number, b: number) => void;
    readonly rust_zstd_wasm_shim_calloc: (a: number, b: number) => number;
    readonly rust_zstd_wasm_shim_free: (a: number) => void;
    readonly rust_zstd_wasm_shim_malloc: (a: number) => number;
    readonly rust_zstd_wasm_shim_memcmp: (a: number, b: number, c: number) => number;
    readonly rust_zstd_wasm_shim_memcpy: (a: number, b: number, c: number) => number;
    readonly rust_zstd_wasm_shim_memmove: (a: number, b: number, c: number) => number;
    readonly rust_zstd_wasm_shim_memset: (a: number, b: number, c: number) => number;
    readonly rust_zstd_wasm_shim_qsort: (a: number, b: number, c: number, d: number) => void;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export4: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
