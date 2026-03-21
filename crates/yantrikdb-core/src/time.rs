/// Platform-aware time utilities.
/// On native targets, uses `std::time::SystemTime` / `Instant`.
/// On wasm32, uses `js_sys::Date::now()`.

#[cfg(not(target_arch = "wasm32"))]
pub fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

#[cfg(not(target_arch = "wasm32"))]
pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(target_arch = "wasm32")]
pub fn now_secs() -> f64 {
    js_sys::Date::now() / 1000.0
}

#[cfg(target_arch = "wasm32")]
pub fn now_ms() -> u64 {
    js_sys::Date::now() as u64
}

/// Monotonic-ish instant for measuring elapsed time.
#[cfg(not(target_arch = "wasm32"))]
pub struct Instant(std::time::Instant);

#[cfg(not(target_arch = "wasm32"))]
impl Instant {
    pub fn now() -> Self {
        Instant(std::time::Instant::now())
    }
    pub fn elapsed_ms(&self) -> u64 {
        self.0.elapsed().as_millis() as u64
    }
}

#[cfg(target_arch = "wasm32")]
pub struct Instant(f64);

#[cfg(target_arch = "wasm32")]
impl Instant {
    pub fn now() -> Self {
        Instant(js_sys::Date::now())
    }
    pub fn elapsed_ms(&self) -> u64 {
        (js_sys::Date::now() - self.0).max(0.0) as u64
    }
}
