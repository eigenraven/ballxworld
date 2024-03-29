//! Authentication and packet encoding (as these are tied together by encryption)

use crate::network::packets::auth::{
    ClientConnectionType, ConnectionResponse, PacketTypeHandshake, PktCSConnectionRequestPayload,
    PktSCConnectionRequestAckPayload,
};
use crate::network::packets::PACKET_PROTOCOL_CURRENT_VERSION;
use bxw_util::rmp_serde;
use bxw_util::sodiumoxide::crypto::{box_, sealedbox, secretbox};
use bxw_util::sodiumoxide::padding;
use bxw_util::zstd;
use num_enum::*;
use serde::*;
use std::borrow::Cow;
use std::cell::RefCell;
use std::convert::{TryFrom, TryInto};
use std::fmt::Debug;
use std::sync::Arc;

pub fn current_net_timestamp() -> u64 {
    use std::time;
    let dur = time::SystemTime::now()
        .duration_since(time::UNIX_EPOCH)
        .expect("Invalid system time set, check your date/time settings");
    dur.as_secs() * 1000 + dur.subsec_millis() as u64
}

// Allow up to 10s clock difference, then drop packets
const MAX_NET_TIMESTAMP_DELTA: u64 = 10_000;

pub const NET_KEEPALIVE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(20);

pub fn net_timestamp_delta(a: u64, b: u64) -> u64 {
    if a > b {
        a - b
    } else {
        b - a
    }
}

thread_local! {
static ZSTD_COMPRESSOR: RefCell<zstd::bulk::Compressor<'static>> = RefCell::new(zstd::bulk::Compressor::new(NET_ZSTD_COMPRESS_LEVEL).unwrap());
static ZSTD_DECOMPRESSOR: RefCell<zstd::bulk::Decompressor<'static>> = RefCell::new(zstd::bulk::Decompressor::new().unwrap());
}
const NET_ZSTD_COMPRESS_LEVEL: i32 = 3;
const NET_ZSTD_DECOMPRESS_SIZE_LIMIT: usize = 4 * 1024 * 1024;
const NET_ZSTD_COMPRESS_LEN_THRESHOLD: usize = 128;

pub fn net_zstd_compress(data: &[u8]) -> Vec<u8> {
    ZSTD_COMPRESSOR
        .with(|c| c.borrow_mut().compress(data))
        .expect("Unexpected compression error")
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DecompressError {
    UnknownError,
}

pub fn net_zstd_decompress(
    data: &[u8],
    custom_size_limit: Option<usize>,
) -> Result<Vec<u8>, DecompressError> {
    ZSTD_DECOMPRESSOR
        .with(|c| {
            c.borrow_mut().decompress(
                data,
                custom_size_limit.unwrap_or(NET_ZSTD_DECOMPRESS_SIZE_LIMIT),
            )
        })
        .map_err(|_| DecompressError::UnknownError)
}

pub fn net_mpack_serialize<M: Serialize>(m: &M) -> Vec<u8> {
    rmp_serde::to_vec(m).unwrap_or_else(|e| {
        panic!(
            "Serialization error {:?} occured when serializing message of type {:?}",
            e,
            std::any::type_name::<M>()
        )
    })
}

pub use rmp_serde::decode::Error as PacketDeserializeError;

use super::packets::auth::ConnectionCookie;

pub fn net_mpack_deserialize<'a, M: Deserialize<'a>>(
    m: &'a [u8],
) -> Result<M, PacketDeserializeError> {
    rmp_serde::from_read_ref::<_, M>(m)
}

#[repr(u8)]
#[derive(
    Copy, Clone, Debug, Hash, Eq, PartialEq, IntoPrimitive, TryFromPrimitive, Deserialize, Serialize,
)]
#[serde(try_from = "u8", into = "u8")]
pub enum PacketFormatVersion {
    V1 = 1,
}

/**
 * Packet formats, sent as first byte of a packet
 * V1 = Version 1 of the network protocol (current)
 * E1 = Encrypted version 1 (current)
 */
#[repr(u8)]
#[derive(
    Copy, Clone, Debug, Hash, Eq, PartialEq, IntoPrimitive, TryFromPrimitive, Deserialize, Serialize,
)]
#[serde(try_from = "u8", into = "u8")]
pub enum PacketFormat {
    /// Unencrypted, first packet send client->server
    ConnectionRequestV1 = 0xB0,
    /// First server->client packet, encrypted with the client's public key
    ConnectionRequestAckV1 = 0xB1,
    /// Encrypted with server's public key, connection request with proof of owning the receiving address
    ConnectionHandshakeV1 = 0xB2,
    /// Encrypted with client's public key, confirmation of an established connection with the session keys established
    ConnectionHandshakeAckV1 = 0xB3,
    /// All other packets use this format (or the compressed variant), encrypted with the established session key
    EncryptedV1 = 0xBA,
    /// All other packets use this format (or the uncompressed variant), encrypted with the established session key, the message field is also zstd-compressed
    EncryptedCompressedV1 = 0xBB,
}

impl PacketFormat {
    pub fn version(self) -> PacketFormatVersion {
        PacketFormatVersion::V1
    }
}

/**
 * Stream ID for the packet
 */
#[repr(u8)]
#[derive(
    Copy, Clone, Debug, Hash, Eq, PartialEq, IntoPrimitive, TryFromPrimitive, Deserialize, Serialize,
)]
#[serde(try_from = "u8", into = "u8")]
pub enum PacketStream {
    Handshake = 0x01,
    /// Keepalive, disconnect, etc. commands
    ConnectionControl = 0x02,
    GameMessages = 0x03,
}

#[derive(Clone, Debug, Hash, Eq, PartialEq)]
#[repr(C)]
pub struct PacketV1<'d> {
    /// Unencrypted
    pub format: PacketFormat,
    /// Encrypted
    pub stream: PacketStream,
    /// Encrypted, value interpretation dependent on `stream`
    pub packet_id: u8,
    /// Encrypted, sequential number of the packet in the stream
    pub seq_id: u32,
    /// Encrypted, millisecond unix timestamp at the moment of sending - packets are rejected with too big of an offset from now
    pub sent_time: u64,
    /// Encrypted (and potentially compressed), the attached message(s)
    pub message: Cow<'d, [u8]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PacketEncodeError {
    OutputBufferTooSmall,
    MessageTooLong,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PacketDecodeError {
    TooShort,
    InvalidPadding,
    UnexpectedFieldValue(&'static str, u64),
    TooBigTimeDelta,
    DecryptionError,
}

#[derive(Debug)]
pub enum PacketProcessingError {
    Encode(PacketEncodeError),
    Decode(PacketDecodeError),
    Deserialize(PacketDeserializeError),
    Decompress(DecompressError),
    UntrustedCrypto,
    /// Invalid identifying information, possibly from a previous connection attempt
    StrayPacket,
    UnsupportedVersion,
}

impl From<PacketEncodeError> for PacketProcessingError {
    fn from(e: PacketEncodeError) -> Self {
        Self::Encode(e)
    }
}

impl From<PacketDecodeError> for PacketProcessingError {
    fn from(e: PacketDecodeError) -> Self {
        Self::Decode(e)
    }
}

impl From<PacketDeserializeError> for PacketProcessingError {
    fn from(e: PacketDeserializeError) -> Self {
        Self::Deserialize(e)
    }
}

impl From<DecompressError> for PacketProcessingError {
    fn from(e: DecompressError) -> Self {
        Self::Decompress(e)
    }
}

const HANDSHAKE_MSG_PADSIZE: usize = 1024;

impl<'d> PacketV1<'d> {
    fn to_owned_message(&self) -> PacketV1<'static> {
        let owned_msg: Vec<u8> = (*self.message).to_owned();
        PacketV1 {
            format: self.format,
            stream: self.stream,
            packet_id: self.packet_id,
            seq_id: self.seq_id,
            sent_time: self.sent_time,
            message: Cow::from(owned_msg),
        }
    }

    fn fields_from_decrypted_bytes(
        format: PacketFormat,
        raw_fields: &'d [u8],
    ) -> Result<Self, PacketDecodeError> {
        if raw_fields.len() < 14 {
            Err(PacketDecodeError::TooShort)
        } else {
            Ok(Self {
                format,
                stream: PacketStream::try_from(raw_fields[0]).map_err(|_| {
                    PacketDecodeError::UnexpectedFieldValue("stream", raw_fields[0] as u64)
                })?,
                packet_id: raw_fields[1],
                seq_id: u32::from_le_bytes(raw_fields[2..6].try_into().unwrap()),
                sent_time: u64::from_le_bytes(raw_fields[6..14].try_into().unwrap()),
                message: Cow::Borrowed(&raw_fields[14..]),
            })
        }
    }

    fn fields_from_decrypted_bytes_owned(
        format: PacketFormat,
        mut raw_fields: Vec<u8>,
    ) -> Result<PacketV1<'static>, PacketDecodeError> {
        if raw_fields.len() < 14 {
            Err(PacketDecodeError::TooShort)
        } else {
            let mut pkt = PacketV1 {
                format,
                stream: PacketStream::try_from(raw_fields[0]).map_err(|_| {
                    PacketDecodeError::UnexpectedFieldValue("stream", raw_fields[0] as u64)
                })?,
                packet_id: raw_fields[1],
                seq_id: u32::from_le_bytes(raw_fields[2..6].try_into().unwrap()),
                sent_time: u64::from_le_bytes(raw_fields[6..14].try_into().unwrap()),
                message: Cow::Borrowed(&[]),
            };
            raw_fields.drain(..14);
            pkt.message = Cow::Owned(raw_fields);
            Ok(pkt)
        }
    }

    fn write_decrypted_fields<'o>(
        &self,
        output: &'o mut [u8],
    ) -> Result<&'o mut [u8], PacketEncodeError> {
        if output.len() < 14 + self.message.len() {
            Err(PacketEncodeError::OutputBufferTooSmall)
        } else {
            output[0] = self.stream.into();
            output[1] = self.packet_id;
            output[2..6].copy_from_slice(&self.seq_id.to_le_bytes());
            output[6..14].copy_from_slice(&self.sent_time.to_le_bytes());
            output[14..14 + self.message.len()].copy_from_slice(&self.message);
            Ok(&mut output[0..14 + self.message.len()])
        }
    }

    fn write_decrypted_fields_vec(&self) -> Vec<u8> {
        let mut v = vec![0u8; self.decrypted_fields_len()];
        self.write_decrypted_fields(&mut v).unwrap();
        v
    }

    fn decrypted_fields_len(&self) -> usize {
        14 + self.message.len()
    }
    const fn minimum_fields_len() -> usize {
        14
    }

    pub fn encode_connection_request(message: &[u8]) -> Result<Vec<u8>, PacketEncodeError> {
        let mut packet: Vec<u8> = Vec::with_capacity(HANDSHAKE_MSG_PADSIZE + 32);
        packet.push(PacketFormat::ConnectionRequestV1.into());
        if message.len() >= HANDSHAKE_MSG_PADSIZE - 4 {
            return Err(PacketEncodeError::MessageTooLong);
        }
        let mut pad_buffer = [0u8; HANDSHAKE_MSG_PADSIZE];
        pad_buffer[0..message.len()].copy_from_slice(message);
        let padded_message_len =
            padding::pad(&mut pad_buffer, message.len(), HANDSHAKE_MSG_PADSIZE)
                .map_err(|_| PacketEncodeError::MessageTooLong)?;
        let padded_message = &pad_buffer[0..padded_message_len];
        let pkt = PacketV1 {
            format: PacketFormat::ConnectionRequestV1,
            stream: PacketStream::Handshake,
            packet_id: PacketTypeHandshake::CSConnectionRequest.into(),
            seq_id: 0,
            sent_time: current_net_timestamp(),
            message: Cow::Borrowed(padded_message),
        };
        packet.resize(pkt.decrypted_fields_len() + 1, 0);
        pkt.write_decrypted_fields(&mut packet[1..])?;
        Ok(packet)
    }

    pub fn try_decode_connection_request(raw: &'d [u8]) -> Result<Self, PacketDecodeError> {
        // UDP amplification prevention - require the connection-starting packet to be large
        if raw.len() < HANDSHAKE_MSG_PADSIZE {
            return Err(PacketDecodeError::TooShort);
        }
        let format = PacketFormat::try_from(raw[0])
            .map_err(|_| PacketDecodeError::UnexpectedFieldValue("format", raw[0] as u64))?;
        if format != PacketFormat::ConnectionRequestV1 {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "format@connRequest",
                raw[0] as u64,
            ));
        }
        let mut packet = Self::fields_from_decrypted_bytes(format, &raw[1..])?;
        if packet.stream != PacketStream::Handshake {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "stream@connRequest",
                packet.stream as u64,
            ));
        }
        if packet.seq_id != 0 {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "seq_id@connRequest",
                packet.seq_id as u64,
            ));
        }
        if PacketTypeHandshake::try_from(packet.packet_id)
            != Ok(PacketTypeHandshake::CSConnectionRequest)
        {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "packet_id@connRequest",
                packet.packet_id as u64,
            ));
        }
        if net_timestamp_delta(packet.sent_time, current_net_timestamp()) > MAX_NET_TIMESTAMP_DELTA
        {
            return Err(PacketDecodeError::TooBigTimeDelta);
        }
        let msg_unpad_len =
            padding::unpad(&packet.message, packet.message.len(), HANDSHAKE_MSG_PADSIZE)
                .map_err(|_| PacketDecodeError::InvalidPadding)?;
        packet.message = match packet.message {
            Cow::Borrowed(original) => Cow::Borrowed(&original[0..msg_unpad_len]),
            Cow::Owned(mut original) => {
                original.resize(msg_unpad_len, 0u8);
                Cow::Owned(original)
            }
        };
        Ok(packet)
    }

    pub fn encode_connection_request_ack(
        message: &[u8],
        client_pubkey: &box_::PublicKey,
    ) -> Result<Vec<u8>, PacketEncodeError> {
        let mut packet: Vec<u8> = Vec::with_capacity(256);
        packet.push(PacketFormat::ConnectionRequestAckV1.into());
        let pkt = PacketV1 {
            format: PacketFormat::ConnectionRequestAckV1,
            stream: PacketStream::Handshake,
            packet_id: PacketTypeHandshake::SCConnectionRequestAck.into(),
            seq_id: 1,
            sent_time: current_net_timestamp(),
            message: Cow::Borrowed(message),
        };
        let fields = pkt.write_decrypted_fields_vec();
        let mut encrypted_fields = sealedbox::seal(&fields, client_pubkey);
        packet.append(&mut encrypted_fields);
        Ok(packet)
    }

    pub fn try_decode_connection_request_ack(
        raw: &[u8],
        client_keypair: (&box_::PublicKey, &box_::SecretKey),
    ) -> Result<PacketV1<'static>, PacketDecodeError> {
        let format = PacketFormat::try_from(raw[0])
            .map_err(|_| PacketDecodeError::UnexpectedFieldValue("format", raw[0] as u64))?;
        if format != PacketFormat::ConnectionRequestAckV1 {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "format@connRequestAck",
                raw[0] as u64,
            ));
        }
        let encr = &raw[1..];
        let decr = sealedbox::open(encr, client_keypair.0, client_keypair.1)
            .map_err(|_| PacketDecodeError::DecryptionError)?;
        let packet = PacketV1::fields_from_decrypted_bytes(format, &decr)?;
        if packet.stream != PacketStream::Handshake {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "stream@connRequestAck",
                packet.stream as u64,
            ));
        }
        if packet.seq_id != 1 {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "seq_id@connRequestAck",
                packet.seq_id as u64,
            ));
        }
        if PacketTypeHandshake::try_from(packet.packet_id)
            != Ok(PacketTypeHandshake::SCConnectionRequestAck)
        {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "packet_id@connRequestAck",
                packet.packet_id as u64,
            ));
        }
        if net_timestamp_delta(packet.sent_time, current_net_timestamp()) > MAX_NET_TIMESTAMP_DELTA
        {
            return Err(PacketDecodeError::TooBigTimeDelta);
        }
        let packet = packet.to_owned_message();
        Ok(packet)
    }

    pub fn encode_connection_handshake(
        message: &[u8],
        server_pubkey: &box_::PublicKey,
    ) -> Result<Vec<u8>, PacketEncodeError> {
        let mut packet: Vec<u8> = Vec::with_capacity(256);
        packet.push(PacketFormat::ConnectionHandshakeV1.into());
        if message.len() >= HANDSHAKE_MSG_PADSIZE - 4 {
            return Err(PacketEncodeError::MessageTooLong);
        }
        let pkt = PacketV1 {
            format: PacketFormat::ConnectionHandshakeV1,
            stream: PacketStream::Handshake,
            packet_id: PacketTypeHandshake::CSHandshake.into(),
            seq_id: 2,
            sent_time: current_net_timestamp(),
            message: Cow::Borrowed(message),
        };
        let fields = pkt.write_decrypted_fields_vec();
        let mut encrypted_fields = sealedbox::seal(&fields, server_pubkey);
        packet.append(&mut encrypted_fields);
        Ok(packet)
    }

    pub fn try_decode_connection_handshake(
        raw: &'d [u8],
        server_keypair: (&box_::PublicKey, &box_::SecretKey),
    ) -> Result<Self, PacketDecodeError> {
        let format = PacketFormat::try_from(raw[0])
            .map_err(|_| PacketDecodeError::UnexpectedFieldValue("format", raw[0] as u64))?;
        if format != PacketFormat::ConnectionRequestAckV1 {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "format@handshake",
                raw[0] as u64,
            ));
        }
        let encr = &raw[1..];
        let decr = sealedbox::open(encr, server_keypair.0, server_keypair.1)
            .map_err(|_| PacketDecodeError::DecryptionError)?;
        let packet = PacketV1::fields_from_decrypted_bytes(format, &decr)?;
        if packet.stream != PacketStream::Handshake {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "stream@handshake",
                packet.stream as u64,
            ));
        }
        if packet.seq_id != 2 {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "seq_id@handshake",
                packet.seq_id as u64,
            ));
        }
        if PacketTypeHandshake::try_from(packet.packet_id) != Ok(PacketTypeHandshake::CSHandshake) {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "packet_id@handshake",
                packet.packet_id as u64,
            ));
        }
        if net_timestamp_delta(packet.sent_time, current_net_timestamp()) > MAX_NET_TIMESTAMP_DELTA
        {
            return Err(PacketDecodeError::TooBigTimeDelta);
        }
        let packet = packet.to_owned_message();
        Ok(packet)
    }

    pub fn encode_connection_handshake_ack(
        message: &[u8],
        client_pubkey: &box_::PublicKey,
    ) -> Result<Vec<u8>, PacketEncodeError> {
        let mut packet: Vec<u8> = Vec::with_capacity(256);
        packet.push(PacketFormat::ConnectionHandshakeAckV1.into());
        let pkt = PacketV1 {
            format: PacketFormat::ConnectionHandshakeAckV1,
            stream: PacketStream::Handshake,
            packet_id: PacketTypeHandshake::SCHandshakeAck.into(),
            seq_id: 3,
            sent_time: current_net_timestamp(),
            message: Cow::Borrowed(message),
        };
        let fields = pkt.write_decrypted_fields_vec();
        let mut encrypted_fields = sealedbox::seal(&fields, client_pubkey);
        packet.append(&mut encrypted_fields);
        Ok(packet)
    }

    pub fn try_decode_connection_handshake_ack(
        raw: &'d [u8],
        client_keypair: (&box_::PublicKey, &box_::SecretKey),
    ) -> Result<Self, PacketDecodeError> {
        let format = PacketFormat::try_from(raw[0])
            .map_err(|_| PacketDecodeError::UnexpectedFieldValue("format", raw[0] as u64))?;
        if format != PacketFormat::ConnectionHandshakeAckV1 {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "format@handshake",
                raw[0] as u64,
            ));
        }
        let encr = &raw[1..];
        let decr = sealedbox::open(encr, client_keypair.0, client_keypair.1)
            .map_err(|_| PacketDecodeError::DecryptionError)?;
        let packet = PacketV1::fields_from_decrypted_bytes(format, &decr)?;
        if packet.stream != PacketStream::Handshake {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "stream@handshake",
                packet.stream as u64,
            ));
        }
        if packet.seq_id != 3 {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "seq_id@handshake",
                packet.seq_id as u64,
            ));
        }
        if PacketTypeHandshake::try_from(packet.packet_id)
            != Ok(PacketTypeHandshake::SCHandshakeAck)
        {
            return Err(PacketDecodeError::UnexpectedFieldValue(
                "packet_id@handshake",
                packet.packet_id as u64,
            ));
        }
        if net_timestamp_delta(packet.sent_time, current_net_timestamp()) > MAX_NET_TIMESTAMP_DELTA
        {
            return Err(PacketDecodeError::TooBigTimeDelta);
        }
        let packet = packet.to_owned_message();
        Ok(packet)
    }

    pub fn encode_established(
        stream: PacketStream,
        packet_id: u8,
        seq_id: u32,
        message: &[u8],
        tx_key: &secretbox::Key,
    ) -> Vec<u8> {
        let mut packet: Vec<u8> = Vec::with_capacity(128);
        let mut decrypted_payload = Cow::Borrowed(message);
        let mut format = PacketFormat::EncryptedV1;
        if message.len() > NET_ZSTD_COMPRESS_LEN_THRESHOLD {
            let compressed = net_zstd_compress(message);
            if compressed.len() < message.len() {
                decrypted_payload = Cow::Owned(compressed);
                format = PacketFormat::EncryptedCompressedV1;
            }
        }
        packet.push(format.into());
        let pkt = PacketV1 {
            format,
            stream,
            packet_id,
            seq_id,
            sent_time: current_net_timestamp(),
            message: decrypted_payload,
        };
        let fields = pkt.write_decrypted_fields_vec();
        let nonce = secretbox::gen_nonce();
        let mut encrypted_fields = secretbox::seal(&fields, &nonce, tx_key);
        packet.reserve_exact(nonce.0.len() + encrypted_fields.len());
        packet.extend_from_slice(&nonce.0);
        packet.append(&mut encrypted_fields);
        packet
    }

    /// Returns a `PacketV1` with an unencrypted, decompressed `message` field.
    pub fn decode_established(
        raw: &[u8],
        rx_key: &secretbox::Key,
    ) -> Result<PacketV1<'static>, PacketProcessingError> {
        if raw.len() < 1 + PacketV1::minimum_fields_len() + secretbox::NONCEBYTES {
            return Err(PacketDecodeError::TooShort.into());
        }
        let format: PacketFormat = raw[0]
            .try_into()
            .map_err(|_| PacketDecodeError::UnexpectedFieldValue("format", raw[0] as u64))?;
        if !(format == PacketFormat::EncryptedV1 || format == PacketFormat::EncryptedCompressedV1) {
            return Err(PacketDecodeError::UnexpectedFieldValue("format", format as u64).into());
        }
        let nonce = secretbox::Nonce::from_slice(&raw[1..1 + secretbox::NONCEBYTES]).unwrap();
        let fields = &raw[1 + secretbox::NONCEBYTES..];
        let decrypted = secretbox::open(fields, &nonce, rx_key)
            .map_err(|_| PacketDecodeError::DecryptionError)?;
        let mut pkt = PacketV1::fields_from_decrypted_bytes_owned(format, decrypted)?;
        if net_timestamp_delta(pkt.sent_time, current_net_timestamp()) > MAX_NET_TIMESTAMP_DELTA {
            return Err(PacketDecodeError::TooBigTimeDelta.into());
        }
        if format == PacketFormat::EncryptedCompressedV1 {
            let compressed = std::mem::replace(&mut pkt.message, Cow::Borrowed(&[])).into_owned();
            pkt.message = Cow::Owned(net_zstd_decompress(&compressed, None)?);
        }
        Ok(pkt)
    }
}

pub struct ClientHandshakeState1 {
    random_cookie: ConnectionCookie,
}

pub fn authflow_client_connection_request_packet(
    my_pubkey: &box_::PublicKey,
    conn_type: ClientConnectionType,
) -> Result<(Vec<u8>, ClientHandshakeState1), PacketEncodeError> {
    //let (kx_pk, kx_sk) = kx::gen_keypair();
    let random_cookie = ConnectionCookie::new_random();
    let smsg = PktCSConnectionRequestPayload {
        c_version_id: PACKET_PROTOCOL_CURRENT_VERSION,
        c_player_id: *my_pubkey,
        c_type: conn_type,
        random_cookie,
    };
    let msg = net_mpack_serialize(&smsg);
    let pkt = PacketV1::encode_connection_request(&msg)?;
    Ok((pkt, ClientHandshakeState1 { random_cookie }))
}

pub struct ServerConnectionRequestState<'handler> {
    request: PktCSConnectionRequestPayload,
    source: &'handler [u8],
    server_token_key: &'handler secretbox::Key,
}

impl<'h> ServerConnectionRequestState<'h> {
    pub fn request(&self) -> &PktCSConnectionRequestPayload {
        &self.request
    }
}

pub fn authflow_server_try_accept_connection_request<'h>(
    packet: &'h [u8],
    source: &'h [u8],
    server_token_key: &'h secretbox::Key,
) -> Result<ServerConnectionRequestState<'h>, PacketProcessingError> {
    let decoded = PacketV1::try_decode_connection_request(packet)?;
    let msg: PktCSConnectionRequestPayload = net_mpack_deserialize(&decoded.message)?;
    if msg.c_version_id != PACKET_PROTOCOL_CURRENT_VERSION {
        return Err(PacketProcessingError::UnsupportedVersion);
    }
    Ok(ServerConnectionRequestState {
        request: msg,
        source,
        server_token_key,
    })
}

const DDOS_EASY: u8 = 10;
const DDOS_HARD: u8 = 20;

pub fn authflow_server_respond_to_connection_request(
    state: ServerConnectionRequestState,
    my_pubkey: &box_::PublicKey,
    _my_seckey: &box_::SecretKey,
    server_name: Arc<str>,
    response: ConnectionResponse,
    ddos_protection_hard: bool,
) -> Result<Vec<u8>, PacketProcessingError> {
    let ServerConnectionRequestState {
        request,
        source,
        server_token_key,
    } = state;
    let mut connection_token_dec: Vec<u8> = Vec::with_capacity(32);
    connection_token_dec.extend_from_slice(&request.random_cookie.0);
    connection_token_dec.extend_from_slice(source);
    let nonce = secretbox::gen_nonce();
    let mut connection_token = secretbox::seal(&connection_token_dec, &nonce, server_token_key);
    connection_token.extend_from_slice(&nonce.0);
    let smsg = PktSCConnectionRequestAckPayload {
        s_version_id: PACKET_PROTOCOL_CURRENT_VERSION,
        s_server_id: *my_pubkey,
        s_name: server_name,
        s_response: response,
        random_cookie: request.random_cookie,
        connection_token: connection_token.into(),
        connection_leading_zeros: if ddos_protection_hard {
            DDOS_HARD
        } else {
            DDOS_EASY
        },
    };
    let msg = net_mpack_serialize(&smsg);
    Ok(PacketV1::encode_connection_request_ack(
        &msg,
        &request.c_player_id,
    )?)
}

pub struct ServerHandshakeState1 {
    packet: PktCSConnectionRequestPayload,
}

impl ServerHandshakeState1 {
    pub fn get_request(&self) -> &PktCSConnectionRequestPayload {
        &self.packet
    }
}

pub fn authflow_server_try_accept_handshake_packet(
    _packet: &[u8],
    _source: &[u8],
    _server_token_key: &secretbox::Key,
    _ddos_protection_hard: bool,
) -> Result<ServerHandshakeState1, PacketProcessingError> {
    /*let decoded = PacketV1::try_decode_connection_handshake(packet)?;
    let msg: PktCSHandshakePayload = net_mpack_deserialize(&decoded.message)?;
    // conn token verification
    {
        let client_conn_token = &msg.connection_token[..];
        if client_conn_token.len() < secretbox::NONCEBYTES + secretbox::MACBYTES + 32 {
            return Err(PacketProcessingError::UntrustedCrypto);
        }
        let nonce = secretbox::Nonce::try_from(
            &client_conn_token[client_conn_token.len() - secretbox::NONCEBYTES..],
        )
        .map_err(|| PacketProcessingError::UntrustedCrypto)?;
        let mut expected_conn_token_dec = Vec::with_capacity(32);
        expected_conn_token_dec.extend_from_slice(&msg.request.random_cookie.0);
        expected_conn_token_dec.extend_from_slice(source);
        let expected_conn_token =
            secretbox::seal(&expected_conn_token_dec, &nonce, server_token_key);
        let client_conn_token_no_nonce =
            &client_conn_token[..client_conn_token.len() - secretbox::NONCEBYTES];
        if !bxw_util::sodiumoxide::utils::memcmp(&expected_conn_token, &client_conn_token_no_nonce)
        {
            return Err(PacketProcessingError::UntrustedCrypto);
        }
    }
    // anti-ddos verification
    {
        let pow_token = Vec::with_capacity(msg.connection_token.len() + msg.proof_of_work.len());
        pow_token.extend_from_slice(&msg.connection_token[..]);
        pow_token.extend_from_slice(&msg.proof_of_work[..]);
        let h = hash::hash(&pow_token);
        let hb = &h.0[..];
        let mut leading_zeros = 0;
        for byte_val in hb.iter() {
            let byte_leading = byte_val.leading_zeros();
            leading_zeros += byte_leading;
            if byte_leading != 8 {
                break;
            }
        }
        let required_leading_zeros = if ddos_protection_hard {
            DDOS_HARD
        } else {
            DDOS_EASY
        };
        if leading_zeros < required_leading_zeros as u32 {
            return Err(PacketProcessingError::UntrustedCrypto);
        }
    }
    Ok(ServerHandshakeState1 { packet: msg })
     */
    todo!()
}

pub fn authflow_server_respond_to_handshake_packet(
    _state: ServerHandshakeState1,
    _my_pubkey: &box_::PublicKey,
    _my_seckey: &box_::SecretKey,
    _my_name: String,
    _response: ConnectionResponse,
) -> Result<(Vec<u8>, ServersideConnectionCryptoState), PacketProcessingError> {
    /*let ServerHandshakeState1 {
        packet: initial_packet,
    } = state;
    let (kx_pk, kx_sk) = kx::gen_keypair();
    let smsg = PktSCHandshakeAckPayload {
        s_version_id: PACKET_PROTOCOL_CURRENT_VERSION,
        s_kx_public: kx_pk,
        s_server_id: *my_pubkey,
        s_response: response,
        random_cookie: initial_packet.random_cookie,
    };
    let msg = net_mpack_serialize(&smsg);
    let (rx, tx) = kx::server_session_keys(&kx_pk, &kx_sk, &initial_packet.c_kx_public)
        .map_err(|_| PacketProcessingError::UntrustedCrypto)?;
    let pkt = PacketV1::encode_connection_request_ack(&msg, &initial_packet.c_player_id)?;
    Ok((
        pkt,
        ServersideConnectionCryptoState {
            rx_key: secretbox::Key::from_slice(&rx.0).unwrap(),
            tx_key: secretbox::Key::from_slice(&tx.0).unwrap(),
            client_id: initial_packet.c_player_id,
            client_type: initial_packet.c_type,
            client_version_id: initial_packet.c_version_id,
        },
    ))*/
    todo!()
}

pub struct ServersideConnectionCryptoState {
    pub rx_key: secretbox::Key,
    pub tx_key: secretbox::Key,
    pub client_id: box_::PublicKey,
    pub client_type: ClientConnectionType,
    pub client_version_id: u32,
}

pub struct ClientsideConnectionCryptoState {
    pub rx_key: secretbox::Key,
    pub tx_key: secretbox::Key,
    pub server_id: box_::PublicKey,
    pub server_name: String,
    pub server_version_id: u32,
}

pub fn authflow_client_try_accept_handshake_ack(
    _state: &ClientHandshakeState1,
    _packet: &[u8],
    _my_keypair: (&box_::PublicKey, &box_::SecretKey),
) -> Result<(ConnectionResponse, ClientsideConnectionCryptoState), PacketProcessingError> {
    /*
    let ClientHandshakeState1 {
        kx_pk,
        kx_sk,
        random_cookie,
    } = state;
    let decoded = PacketV1::try_decode_connection_request_ack(packet, my_keypair)?;
    let msg: PktSCConnectionRequestAckPayload = net_mpack_deserialize(&decoded.message)?;
    if msg.random_cookie != *random_cookie {
        return Err(PacketProcessingError::StrayPacket);
    }
    let decr_cookie = box_::open(
        &msg.crypted_cookie.1,
        &msg.crypted_cookie.0,
        &msg.s_server_id,
        my_keypair.1,
    )
    .map_err(|_| PacketProcessingError::UntrustedCrypto)?;
    let bytes_cookie = state.random_cookie.to_le_bytes();
    if decr_cookie != bytes_cookie {
        return Err(PacketProcessingError::UntrustedCrypto);
    }
    let (rx, tx) = kx::client_session_keys(kx_pk, kx_sk, &msg.s_kx_public)
        .map_err(|_| PacketProcessingError::UntrustedCrypto)?;
    Ok((
        msg.s_response,
        ClientsideConnectionCryptoState {
            rx_key: secretbox::Key::from_slice(&rx.0).unwrap(),
            tx_key: secretbox::Key::from_slice(&tx.0).unwrap(),
            server_id: msg.s_server_id,
            server_name: msg.s_name.clone(),
            server_version_id: msg.s_version_id,
        },
    ))
     */
    todo!()
}

#[cfg(test)]
mod test {
    use super::*;
    use bxw_util::sodiumoxide::crypto::box_;

    #[test]
    fn test_packet_v1_decrypted_encoding() {
        let msg = b"Testing packet encoding";
        let pkt = PacketV1 {
            format: PacketFormat::ConnectionRequestV1,
            stream: PacketStream::ConnectionControl,
            packet_id: 8,
            seq_id: 123456,
            sent_time: current_net_timestamp(),
            message: Cow::Borrowed(msg),
        };
        let mut obuf = [255u8; 96];
        let oref: &[u8] = pkt
            .write_decrypted_fields(&mut obuf)
            .expect("Obuf too small?");
        let oref_len = oref.len();
        let dpkt = PacketV1::fields_from_decrypted_bytes(pkt.format, oref)
            .expect("Couldn't re-decode packet");
        assert_eq!(dpkt, pkt);
        let owned_buf = Vec::from(oref);
        let owpkt = PacketV1::fields_from_decrypted_bytes_owned(pkt.format, owned_buf)
            .expect("Couldn't re-decode owned packet");
        assert_eq!(owpkt, pkt);
        let unencoded: &[u8] = &obuf[oref_len..];
        assert!(unencoded.iter().all(|x| *x == 255u8));
        PacketV1::fields_from_decrypted_bytes(pkt.format, &[0, 1, 2])
            .expect_err("Short packet mustn't decode correctly");
    }

    #[test]
    fn test_packet_v1_encrypted_encoding() {
        bxw_util::sodiumoxide::init().unwrap();
        let skey = secretbox::gen_key();
        let short_msg = b"Short message" as &[u8];
        let mut long_msg = vec![121u8; 1024];
        long_msg[10] = 77;
        long_msg[103] = 12;
        for &msg in &[short_msg, &long_msg] {
            let stream = PacketStream::ConnectionControl;
            let packet_id = 13;
            let seq_id = 24;
            let encoded = PacketV1::encode_established(stream, packet_id, seq_id, msg, &skey);
            let decoded =
                PacketV1::decode_established(&encoded, &skey).expect("Couldn't decode packet");
            assert_eq!(decoded.stream, stream);
            assert_eq!(decoded.packet_id, packet_id);
            assert_eq!(decoded.seq_id, seq_id);
            assert_eq!(decoded.message, msg);
            // ensure compression works
            assert!(encoded.len() < 256);
        }
    }

    #[test]
    fn test_packet_v1_handshake_symmetry() {
        let hs_msg = b"Hello, world\0!";
        let encoded =
            PacketV1::encode_connection_request(hs_msg).expect("Error creating handshake message");
        let decoded = PacketV1::try_decode_connection_request(&encoded)
            .expect("Error decoding handshake message");
        assert_eq!(&*decoded.message, hs_msg);
        assert!(decoded.message.ends_with(b"!"));
        assert_eq!(
            PacketV1::try_decode_connection_request(&encoded[0..encoded.len() / 2]),
            Err(PacketDecodeError::TooShort)
        );
    }

    #[test]
    fn test_packet_v1_handshake_ack_symmetry() {
        bxw_util::sodiumoxide::init().unwrap();
        let (client_pk, client_sk) = box_::gen_keypair();
        let hs_msg = b"Hello, world\0!";
        let mut encoded = PacketV1::encode_connection_request_ack(hs_msg, &client_pk)
            .expect("Error creating handshake ack message");
        let decoded =
            PacketV1::try_decode_connection_request_ack(&encoded, (&client_pk, &client_sk))
                .expect("Error decoding handshake ack message");
        assert_eq!(&*decoded.message, hs_msg);
        assert!(decoded.message.ends_with(b"!"));
        assert_eq!(
            PacketV1::try_decode_connection_request_ack(
                &encoded[0..encoded.len() / 2],
                (&client_pk, &client_sk),
            ),
            Err(PacketDecodeError::DecryptionError)
        );
        encoded[8] = encoded[8].wrapping_add(1);
        assert_eq!(
            PacketV1::try_decode_connection_request_ack(&encoded, (&client_pk, &client_sk)),
            Err(PacketDecodeError::DecryptionError)
        );
    }

    #[test]
    fn test_simple_authflow() {
        bxw_util::sodiumoxide::init().unwrap();
        let (client_pk, client_sk) = box_::gen_keypair();
        let (server_pk, server_sk) = box_::gen_keypair();
        let server_tk = secretbox::gen_key();
        let server_name = String::from("Authflow test server");
        let client_addr = [0u8; 16];
        // Client:
        let (cs0, chs) =
            authflow_client_connection_request_packet(&client_pk, ClientConnectionType::GameClient)
                .expect("Error in client handshake authflow");
        // Server:
        let shs =
            authflow_server_try_accept_handshake_packet(&cs0, &client_addr, &server_tk, false)
                .expect("Error in server handshake accept authflow");
        assert_eq!(shs.get_request().c_type, ClientConnectionType::GameClient);
        assert_eq!(shs.get_request().c_player_id, client_pk);
        assert_eq!(
            shs.get_request().c_version_id,
            PACKET_PROTOCOL_CURRENT_VERSION
        );
        let server_response = ConnectionResponse::Accepted;
        let (sc1, shs) = authflow_server_respond_to_handshake_packet(
            shs,
            &server_pk,
            &server_sk,
            server_name.clone(),
            server_response,
        )
        .expect("Error in server handshake respond authflow");
        assert_eq!(shs.client_id, client_pk);
        assert_eq!(shs.client_type, ClientConnectionType::GameClient);
        // Client:
        let (sr, chs) =
            authflow_client_try_accept_handshake_ack(&chs, &sc1, (&client_pk, &client_sk))
                .expect("Error in client handshake accept authflow");
        assert_eq!(sr, server_response);
        assert_eq!(chs.server_name, server_name);
        assert_eq!(chs.tx_key, shs.rx_key);
        assert_eq!(chs.rx_key, shs.tx_key);
        assert_eq!(chs.server_id, server_pk);
        assert_eq!(chs.server_version_id, PACKET_PROTOCOL_CURRENT_VERSION);
    }
}
