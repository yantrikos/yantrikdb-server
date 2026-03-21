use wasm_bindgen::prelude::*;

// Provide sqlite3_os_init for SQLITE_OS_OTHER builds.
// SQLite calls this during initialization to register the default VFS.
// For :memory: databases, SQLite's built-in memdb VFS handles storage,
// but we need at least a registered VFS for the open path to succeed.
extern "C" {
    fn sqlite3_vfs_register(vfs: *const u8, make_default: i32) -> i32;
}

// Minimal VFS struct layout matching SQLite's sqlite3_vfs (version 1).
// We only need the structure to be registered — :memory: bypasses most methods.
#[repr(C)]
struct MinimalVfs {
    i_version: i32,
    sz_os_file: i32,
    mx_pathname: i32,
    p_next: *const u8,
    z_name: *const u8,
    p_app_data: *const u8,
    x_open: extern "C" fn(*const u8, *const u8, *mut u8, i32, *mut i32) -> i32,
    x_delete: extern "C" fn(*const u8, *const u8, i32) -> i32,
    x_access: extern "C" fn(*const u8, *const u8, i32, *mut i32) -> i32,
    x_full_pathname: extern "C" fn(*const u8, *const u8, i32, *mut u8) -> i32,
    x_dl_open: *const u8,
    x_dl_error: *const u8,
    x_dl_sym: *const u8,
    x_dl_close: *const u8,
    x_randomness: extern "C" fn(*const u8, i32, *mut u8) -> i32,
    x_sleep: extern "C" fn(*const u8, i32) -> i32,
    x_current_time: extern "C" fn(*const u8, *mut f64) -> i32,
    x_get_last_error: extern "C" fn(*const u8, i32, *mut u8) -> i32,
}

// io_methods for file handles
#[repr(C)]
struct MinimalIoMethods {
    i_version: i32,
    x_close: extern "C" fn(*mut u8) -> i32,
    x_read: extern "C" fn(*mut u8, *mut u8, i32, i64) -> i32,
    x_write: extern "C" fn(*mut u8, *const u8, i32, i64) -> i32,
    x_truncate: extern "C" fn(*mut u8, i64) -> i32,
    x_sync: extern "C" fn(*mut u8, i32) -> i32,
    x_file_size: extern "C" fn(*mut u8, *mut i64) -> i32,
    x_lock: extern "C" fn(*mut u8, i32) -> i32,
    x_unlock: extern "C" fn(*mut u8, i32) -> i32,
    x_check_reserved_lock: extern "C" fn(*mut u8, *mut i32) -> i32,
    x_file_control: extern "C" fn(*mut u8, i32, *mut u8) -> i32,
    x_sector_size: extern "C" fn(*mut u8) -> i32,
    x_device_characteristics: extern "C" fn(*mut u8) -> i32,
}

// SAFETY: The VFS structs contain function pointers and null pointers only.
// They are read-only after initialization. WASM is single-threaded.
unsafe impl Sync for MinimalVfs {}
unsafe impl Sync for MinimalIoMethods {}

extern "C" fn vfs_open(_vfs: *const u8, _name: *const u8, file: *mut u8, _flags: i32, out_flags: *mut i32) -> i32 {
    unsafe {
        // Zero the file struct, then set pMethods pointer at offset 0
        core::ptr::write_bytes(file, 0, 64);
        let methods_ptr = &IO_METHODS as *const MinimalIoMethods as *const u8;
        core::ptr::copy_nonoverlapping(&methods_ptr as *const *const u8 as *const u8, file, core::mem::size_of::<*const u8>());
        if !out_flags.is_null() { *out_flags = _flags; }
    }
    0 // SQLITE_OK
}
extern "C" fn vfs_delete(_vfs: *const u8, _name: *const u8, _sync: i32) -> i32 { 0 }
extern "C" fn vfs_access(_vfs: *const u8, _name: *const u8, _flags: i32, out: *mut i32) -> i32 {
    unsafe { *out = 0; }
    0
}
extern "C" fn vfs_fullpathname(_vfs: *const u8, input: *const u8, n: i32, out: *mut u8) -> i32 {
    unsafe {
        let mut i = 0;
        while i < n - 1 {
            let c = *input.add(i as usize);
            *out.add(i as usize) = c;
            if c == 0 { break; }
            i += 1;
        }
        *out.add(i as usize) = 0;
    }
    0
}
extern "C" fn vfs_randomness(_vfs: *const u8, n: i32, out: *mut u8) -> i32 {
    static mut SEED: u32 = 0x12345678;
    unsafe {
        for i in 0..n {
            SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
            *out.add(i as usize) = (SEED >> 16) as u8;
        }
    }
    n
}
extern "C" fn vfs_sleep(_vfs: *const u8, microseconds: i32) -> i32 { microseconds }
extern "C" fn vfs_current_time(_vfs: *const u8, time: *mut f64) -> i32 {
    unsafe { *time = 2460000.5; } // Approximate Julian day — SQLite uses for default timestamps
    0
}
extern "C" fn vfs_get_last_error(_vfs: *const u8, _n: i32, _buf: *mut u8) -> i32 { 0 }

// IO methods — all stubs since :memory: bypasses file I/O
extern "C" fn io_close(_f: *mut u8) -> i32 { 0 }
extern "C" fn io_read(_f: *mut u8, _buf: *mut u8, _n: i32, _off: i64) -> i32 { 10 /* SQLITE_IOERR_READ */ }
extern "C" fn io_write(_f: *mut u8, _buf: *const u8, _n: i32, _off: i64) -> i32 { 778 /* SQLITE_IOERR_WRITE */ }
extern "C" fn io_truncate(_f: *mut u8, _size: i64) -> i32 { 0 }
extern "C" fn io_sync(_f: *mut u8, _flags: i32) -> i32 { 0 }
extern "C" fn io_file_size(_f: *mut u8, size: *mut i64) -> i32 { unsafe { *size = 0; } 0 }
extern "C" fn io_lock(_f: *mut u8, _n: i32) -> i32 { 0 }
extern "C" fn io_unlock(_f: *mut u8, _n: i32) -> i32 { 0 }
extern "C" fn io_check_lock(_f: *mut u8, out: *mut i32) -> i32 { unsafe { *out = 0; } 0 }
extern "C" fn io_file_control(_f: *mut u8, _op: i32, _arg: *mut u8) -> i32 { 12 /* SQLITE_NOTFOUND */ }
extern "C" fn io_sector_size(_f: *mut u8) -> i32 { 512 }
extern "C" fn io_device_char(_f: *mut u8) -> i32 { 0 }

static IO_METHODS: MinimalIoMethods = MinimalIoMethods {
    i_version: 1,
    x_close: io_close,
    x_read: io_read,
    x_write: io_write,
    x_truncate: io_truncate,
    x_sync: io_sync,
    x_file_size: io_file_size,
    x_lock: io_lock,
    x_unlock: io_unlock,
    x_check_reserved_lock: io_check_lock,
    x_file_control: io_file_control,
    x_sector_size: io_sector_size,
    x_device_characteristics: io_device_char,
};

static VFS_NAME: &[u8] = b"wasm\0";

static WASM_VFS: MinimalVfs = MinimalVfs {
    i_version: 1,
    sz_os_file: 64, // Generous size for the file struct
    mx_pathname: 256,
    p_next: core::ptr::null(),
    z_name: VFS_NAME.as_ptr(),
    p_app_data: core::ptr::null(),
    x_open: vfs_open,
    x_delete: vfs_delete,
    x_access: vfs_access,
    x_full_pathname: vfs_fullpathname,
    x_dl_open: core::ptr::null(),
    x_dl_error: core::ptr::null(),
    x_dl_sym: core::ptr::null(),
    x_dl_close: core::ptr::null(),
    x_randomness: vfs_randomness,
    x_sleep: vfs_sleep,
    x_current_time: vfs_current_time,
    x_get_last_error: vfs_get_last_error,
};

#[no_mangle]
pub extern "C" fn sqlite3_os_init() -> i32 {
    unsafe {
        sqlite3_vfs_register(&WASM_VFS as *const MinimalVfs as *const u8, 1)
    }
}

#[no_mangle]
pub extern "C" fn sqlite3_os_end() -> i32 {
    0
}

/// In-browser YantrikDB instance. All data lives in memory (SQLite :memory:).
#[wasm_bindgen]
pub struct WasmYantrikDB {
    inner: yantrikdb::YantrikDB,
}

#[wasm_bindgen]
impl WasmYantrikDB {
    /// Create a new in-memory YantrikDB instance.
    #[wasm_bindgen(constructor)]
    pub fn new(embedding_dim: usize) -> Result<WasmYantrikDB, JsError> {
        let db = yantrikdb::YantrikDB::new(":memory:", embedding_dim)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmYantrikDB { inner: db })
    }

    /// Store a memory with its embedding vector.
    pub fn record(
        &self,
        text: &str,
        embedding: Vec<f32>,
        importance: f64,
        valence: f64,
        memory_type: &str,
    ) -> Result<String, JsError> {
        let meta = serde_json::json!({});
        self.inner
            .record(
                text,
                memory_type,
                importance,
                valence,
                604800.0,  // 7-day half-life default
                &meta,     // metadata
                &embedding,
                "default", // namespace
                0.8,       // certainty
                "general", // domain
                "user",    // source
                None,      // emotional_state
            )
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Recall memories by embedding similarity. Returns JSON array.
    pub fn recall(
        &self,
        embedding: Vec<f32>,
        top_k: usize,
    ) -> Result<JsValue, JsError> {
        let results = self.inner
            .recall(&embedding, top_k, None, None, false, false, None, false, None, None, None)
            .map_err(|e| JsError::new(&e.to_string()))?;

        serde_wasm_bindgen::to_value(&results)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Create a relationship between entities.
    pub fn relate(
        &self,
        src: &str,
        dst: &str,
        rel_type: &str,
        weight: f64,
    ) -> Result<String, JsError> {
        self.inner
            .relate(src, dst, rel_type, weight)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Run the cognition loop. Returns JSON with triggers, conflicts, patterns.
    pub fn think(&self) -> Result<JsValue, JsError> {
        let config = yantrikdb::ThinkConfig::default();
        let result = self.inner
            .think(&config)
            .map_err(|e| JsError::new(&e.to_string()))?;

        serde_wasm_bindgen::to_value(&result)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get engine statistics. Returns JSON.
    pub fn stats(&self) -> Result<JsValue, JsError> {
        let stats = self.inner
            .stats(None)
            .map_err(|e| JsError::new(&e.to_string()))?;

        serde_wasm_bindgen::to_value(&stats)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Search entities by name pattern. Returns JSON array.
    pub fn search_entities(
        &self,
        pattern: Option<String>,
        limit: usize,
    ) -> Result<JsValue, JsError> {
        let results = self.inner
            .search_entities(pattern.as_deref(), None, limit)
            .map_err(|e| JsError::new(&e.to_string()))?;

        serde_wasm_bindgen::to_value(&results)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Get edges for an entity. Returns JSON array.
    pub fn get_edges(&self, entity: &str) -> Result<JsValue, JsError> {
        let edges = self.inner
            .get_edges(entity)
            .map_err(|e| JsError::new(&e.to_string()))?;

        serde_wasm_bindgen::to_value(&edges)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Forget (tombstone) a memory by RID.
    pub fn forget(&self, rid: &str) -> Result<bool, JsError> {
        self.inner
            .forget(rid)
            .map_err(|e| JsError::new(&e.to_string()))
    }
}
