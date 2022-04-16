//! Packets involved in the authentication&connection handshake
//!
//! The basic flow is:
//! 1. Client sends [PktCSConnectionRequestPayload]
//! 2. Server calculates an authorization message and challenge, and sends it in a [PktSCConnectionRequestAckPayload]
//! 3. Client solves the challenge and sends it along with the original request in a [PktCSHandshakePayload]
//! 4. Server confirms an established connection in a [PktSCHandshakeAckPayload]
//! 5. From this point on, a shared session key is established and all packets are authenticated&encrypted
//!
//! This is designed so that UDP packets with forged source addresses have minimal possible impact on the server,
//! as they don't need to allocate any persistent structures until a solved challenge is received.
use bxw_util::sodiumoxide::crypto::{box_, kx, hash::sha256 as hash};
use std::sync::Arc;
use num_enum::*;
use serde::*;

#[repr(u8)]
#[derive(
    Copy, Clone, Debug, Hash, Eq, PartialEq, IntoPrimitive, TryFromPrimitive, Deserialize, Serialize,
)]
#[serde(try_from = "u8", into = "u8")]
pub enum ClientConnectionType {
    /// Doesn't expect a connection, just server details
    QueryServerInfo = 1,
    /// Connects to the server intending to play the game
    GameClient = 16,
}

#[repr(u8)]
#[derive(
    Copy, Clone, Debug, Hash, Eq, PartialEq, IntoPrimitive, TryFromPrimitive, Deserialize, Serialize,
)]
#[serde(try_from = "u8", into = "u8")]
pub enum PacketTypeHandshake {
    CSConnectionRequest = 10,
    SCConnectionRequestAck = 11,
    CSHandshake = 12,
    SCHandshakeAck = 13,
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Deserialize, Serialize)]
#[repr(transparent)]
pub struct ConnectionCookie(pub [u8; 32]);

impl ConnectionCookie {
    pub fn new_random() -> Self {
        let mut c = Self(Default::default());
        bxw_util::sodiumoxide::randombytes::randombytes_into(&mut c.0[..]);
        c
    }
}

/// Client->Server first handshake packet
#[derive(Clone, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct PktCSConnectionRequestPayload {
    /// Version of the client, must match `super::PACKET_PROTOCOL_CURRENT_VERSION`
    pub c_version_id: u32,
    /// Client's permanent identifying public key
    pub c_player_id: box_::PublicKey,
    /// Client connection type
    pub c_type: ClientConnectionType,
    /// Random buffer identifying this specific request
    pub random_cookie: ConnectionCookie,
}

/// Server->Client first handshake ack packet
#[derive(Clone, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct PktSCConnectionRequestAckPayload {
    /// Version of the server, must match `super::PACKET_PROTOCOL_CURRENT_VERSION`
    pub s_version_id: u32,
    /// Server's permanent identifying public key
    pub s_server_id: box_::PublicKey,
    /// Server name
    pub s_name: Arc<str>,
    /// Server's response to the connection request
    pub s_response: ConnectionResponse,
    /// Cookie from the client request
    pub random_cookie: ConnectionCookie,
    /// Token allowing this client to connect (encrypted by server's local key)
    pub connection_token: Box<[u8]>,
    /// Number of leading zero bits for the proof-of-work anti-ddos algorithm
    pub connection_leading_zeros: u8,
}

/// Client->Server second handshake packet
#[derive(Clone, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct PktCSHandshakePayload {
    /// Original request, to allow for stateless processing of half-open connections
    pub request: PktCSConnectionRequestPayload,
    /// Token for connection
    pub connection_token: Box<[u8]>,
    /// A buffer that when appended to the `connection_token` produces a number of leading zero bits in its binary representation
    pub proof_of_work: [u8; hash::DIGESTBYTES],
    /// Client's key exchange public key for this session
    pub c_kx_public: kx::PublicKey,
}

/// Server->Client second handshake ack packet
#[derive(Clone, Hash, Eq, PartialEq, Deserialize, Serialize)]
pub struct PktSCHandshakeAckPayload {
    /// Version of the server, must match `super::PACKET_PROTOCOL_CURRENT_VERSION`
    pub s_version_id: u32,
    /// Server's key exchange public key for this session
    pub s_kx_public: kx::PublicKey,
    /// Server's permanent identifying public key
    pub s_server_id: box_::PublicKey,
    /// Server's response to the connection request
    pub s_response: ConnectionResponse,
    /// Cookie from the client request
    pub random_cookie: ConnectionCookie,
}

#[repr(u8)]
#[derive(
    Copy, Clone, Debug, Hash, Eq, PartialEq, IntoPrimitive, TryFromPrimitive, Deserialize, Serialize,
)]
#[serde(try_from = "u8", into = "u8")]
pub enum ConnectionResponse {
    Accepted = 1,
    BadVersion,
    Blocked,
    NoSlots,
    AlreadyPresent,
}
