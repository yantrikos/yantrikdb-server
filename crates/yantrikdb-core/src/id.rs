/// Platform-aware unique ID generation.
/// On native targets, uses uuid7 (time-ordered UUIDv7).
/// On wasm32, uses a custom generator with js_sys::Date for time.

#[cfg(not(target_arch = "wasm32"))]
pub fn new_id() -> String {
    uuid7::uuid7().to_string()
}

#[cfg(target_arch = "wasm32")]
pub fn new_id() -> String {
    use rand::Rng;
    // Generate UUIDv7-like: 48-bit timestamp + 80-bit random
    let ts_ms = js_sys::Date::now() as u64;
    let mut rng = rand::thread_rng();
    let rand_a: u16 = rng.gen::<u16>() & 0x0FFF; // 12 bits
    let rand_b: u64 = rng.gen::<u64>() & 0x3FFFFFFFFFFFFFFF; // 62 bits

    // UUIDv7 format: tttttttt-tttt-7rrr-Nrrr-rrrrrrrrrrrr
    let time_high = (ts_ms >> 16) as u32;
    let time_mid = (ts_ms & 0xFFFF) as u16;
    let ver_rand = 0x7000 | rand_a;
    let var_rand_high = 0x8000 | ((rand_b >> 48) as u16 & 0x3FFF);
    let rand_low = rand_b & 0xFFFFFFFFFFFF;

    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        time_high, time_mid, ver_rand, var_rand_high, rand_low
    )
}
