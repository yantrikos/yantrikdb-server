/// Wire protocol opcodes for YantrikDB.
///
/// Each opcode is a single byte identifying the command type.
/// Direction conventions:
///   C→S = Client to Server
///   S→C = Server to Client
///   Both = Either direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum OpCode {
    // --- Auth (0x01–0x03) ---
    Auth = 0x01,
    AuthOk = 0x02,
    AuthFail = 0x03,

    // --- Database (0x10–0x12) ---
    SelectDb = 0x10,
    CreateDb = 0x11,
    DbOk = 0x12,
    ListDb = 0x13,
    ListDbResult = 0x14,

    // --- Remember (0x20–0x22) ---
    Remember = 0x20,
    RememberOk = 0x21,
    RememberBatch = 0x22,

    // --- Recall (0x30–0x32) ---
    Recall = 0x30,
    RecallResult = 0x31,
    RecallEnd = 0x32,

    // --- Graph (0x40–0x43) ---
    Relate = 0x40,
    RelateOk = 0x41,
    Edges = 0x42,
    EdgesResult = 0x43,

    // --- Forget (0x50–0x51) ---
    Forget = 0x50,
    ForgetOk = 0x51,

    // --- Session (0x60–0x62) ---
    SessionStart = 0x60,
    SessionEnd = 0x61,
    SessionOk = 0x62,

    // --- Think (0x70–0x71) ---
    Think = 0x70,
    ThinkResult = 0x71,

    // --- Events (0x80–0x82) ---
    Subscribe = 0x80,
    Event = 0x81,
    Unsubscribe = 0x82,

    // --- Conflicts (0x90–0x92) ---
    Conflicts = 0x90,
    Resolve = 0x91,
    ConflictResult = 0x92,

    // --- Info (0xA0–0xA2) ---
    Personality = 0xA0,
    Stats = 0xA1,
    InfoResult = 0xA2,

    // --- Control (0xF0–0xF2) ---
    Error = 0xF0,
    Ping = 0xF1,
    Pong = 0xF2,
}

impl OpCode {
    pub fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            0x01 => Some(Self::Auth),
            0x02 => Some(Self::AuthOk),
            0x03 => Some(Self::AuthFail),

            0x10 => Some(Self::SelectDb),
            0x11 => Some(Self::CreateDb),
            0x12 => Some(Self::DbOk),
            0x13 => Some(Self::ListDb),
            0x14 => Some(Self::ListDbResult),

            0x20 => Some(Self::Remember),
            0x21 => Some(Self::RememberOk),
            0x22 => Some(Self::RememberBatch),

            0x30 => Some(Self::Recall),
            0x31 => Some(Self::RecallResult),
            0x32 => Some(Self::RecallEnd),

            0x40 => Some(Self::Relate),
            0x41 => Some(Self::RelateOk),
            0x42 => Some(Self::Edges),
            0x43 => Some(Self::EdgesResult),

            0x50 => Some(Self::Forget),
            0x51 => Some(Self::ForgetOk),

            0x60 => Some(Self::SessionStart),
            0x61 => Some(Self::SessionEnd),
            0x62 => Some(Self::SessionOk),

            0x70 => Some(Self::Think),
            0x71 => Some(Self::ThinkResult),

            0x80 => Some(Self::Subscribe),
            0x81 => Some(Self::Event),
            0x82 => Some(Self::Unsubscribe),

            0x90 => Some(Self::Conflicts),
            0x91 => Some(Self::Resolve),
            0x92 => Some(Self::ConflictResult),

            0xA0 => Some(Self::Personality),
            0xA1 => Some(Self::Stats),
            0xA2 => Some(Self::InfoResult),

            0xF0 => Some(Self::Error),
            0xF1 => Some(Self::Ping),
            0xF2 => Some(Self::Pong),

            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_all_opcodes() {
        let all = [
            OpCode::Auth,
            OpCode::AuthOk,
            OpCode::AuthFail,
            OpCode::SelectDb,
            OpCode::CreateDb,
            OpCode::DbOk,
            OpCode::ListDb,
            OpCode::ListDbResult,
            OpCode::Remember,
            OpCode::RememberOk,
            OpCode::RememberBatch,
            OpCode::Recall,
            OpCode::RecallResult,
            OpCode::RecallEnd,
            OpCode::Relate,
            OpCode::RelateOk,
            OpCode::Edges,
            OpCode::EdgesResult,
            OpCode::Forget,
            OpCode::ForgetOk,
            OpCode::SessionStart,
            OpCode::SessionEnd,
            OpCode::SessionOk,
            OpCode::Think,
            OpCode::ThinkResult,
            OpCode::Subscribe,
            OpCode::Event,
            OpCode::Unsubscribe,
            OpCode::Conflicts,
            OpCode::Resolve,
            OpCode::ConflictResult,
            OpCode::Personality,
            OpCode::Stats,
            OpCode::InfoResult,
            OpCode::Error,
            OpCode::Ping,
            OpCode::Pong,
        ];
        for op in all {
            let byte = op as u8;
            let decoded = OpCode::from_u8(byte)
                .unwrap_or_else(|| panic!("failed to decode opcode 0x{byte:02X}"));
            assert_eq!(decoded, op);
        }
    }
}
