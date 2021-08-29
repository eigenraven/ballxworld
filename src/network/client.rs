use crate::config::ConfigHandle;
use crate::network::packets;
use crate::network::packets::auth::{ClientConnectionType, ConnectionResponse};
use crate::network::protocol::{
    authflow_client_handshake_packet, authflow_client_try_accept_handshake_ack, PacketStream,
    PacketV1, NET_KEEPALIVE_INTERVAL,
};
use bxw_util::flume;
use bxw_util::log;
use bxw_util::sodiumoxide::crypto::box_;
use num_enum::{FromPrimitive, IntoPrimitive};
use std::io::ErrorKind;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time;

#[derive(Debug)]
pub enum ClientCreationError {
    SocketConnectionError {
        error: std::io::Error,
    },
    Timeout {
        duration: time::Duration,
    },
    ConnectionRejected {
        reason: packets::auth::ConnectionResponse,
    },
}

#[derive(Clone, Debug, Hash)]
pub enum ClientControlMessage {
    Disconnect,
    SendChat(String),
}

pub struct ClientConfig {
    pub id_keys: (box_::PublicKey, box_::SecretKey),
}

#[derive(Clone, PartialEq, Eq)]
pub struct ServerDetails {
    pub name: String,
    pub address: SocketAddr,
}

#[repr(u32)]
#[derive(
    FromPrimitive, IntoPrimitive, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash,
)]
pub enum ClientConnectionState {
    #[default]
    Unknown,
    StartingThread,
    SendingHandshake1,
    WaitingForHandshakeAck,
    Connected,
    Shutdown,
}

pub struct ClientSharedData {
    server: Arc<ServerDetails>,
    config: Arc<ClientConfig>,
    connection_state: AtomicU32,
    ms_since_last_recv: AtomicU32,
    ms_since_last_send: AtomicU32,
}

pub struct Client {
    client_thread: thread::JoinHandle<()>,
    client_control: flume::Sender<ClientControlMessage>,
    shared_data: Arc<ClientSharedData>,
}

const CLIENT_CONTROL_CHANNEL_BOUND: usize = 256;

impl Client {
    pub fn new(
        cfg: ConfigHandle,
        ccfg: Arc<ClientConfig>,
        server: Arc<ServerDetails>,
    ) -> Result<Self, ClientCreationError> {
        let (ccon_tx, ccon_rx) = flume::bounded(CLIENT_CONTROL_CHANNEL_BOUND);
        use std::net::{Ipv4Addr, Ipv6Addr, SocketAddrV4, SocketAddrV6};
        let bindaddr = match &server.address {
            SocketAddr::V4(_) => SocketAddr::V4(SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, 0)),
            SocketAddr::V6(_) => SocketAddr::V6(SocketAddrV6::new(Ipv6Addr::UNSPECIFIED, 0, 0, 0)),
        };
        let socket = std::net::UdpSocket::bind(bindaddr)
            .map_err(|error| ClientCreationError::SocketConnectionError { error })?;
        socket
            .connect(server.address)
            .map_err(|error| ClientCreationError::SocketConnectionError { error })?;
        socket
            .set_nonblocking(false)
            .map_err(|error| ClientCreationError::SocketConnectionError { error })?;
        // TODO: Save/load identifying keys
        let shared_data = Arc::new(ClientSharedData {
            server,
            config: ccfg,
            connection_state: AtomicU32::new(ClientConnectionState::StartingThread.into()),
            ms_since_last_recv: AtomicU32::new(0),
            ms_since_last_send: AtomicU32::new(0),
        });
        let shared_state_copy = Arc::clone(&shared_data);
        let client_thread = thread::Builder::new()
            .name("bxw-client-netio".to_owned())
            .stack_size(2 * 1024 * 1024)
            .spawn(move || {
                if let Err(e) = client_netmain(cfg, ccon_rx, socket, shared_state_copy) {
                    log::error!("Client netmain terminated with error: {:?}", e);
                }
            })
            .expect("Couldn't start main client network thread");
        Ok(Self {
            client_thread,
            client_control: ccon_tx,
            shared_data,
        })
    }

    pub fn send_control_message(&self, msg: ClientControlMessage) {
        match self.client_control.send(msg.clone()) {
            Ok(_nrecv) => {}
            Err(_) => {
                log::error!(
                    "NET CONTROL MSG failed to send - no listening interface handlers: {:?}",
                    msg
                );
            }
        }
    }

    pub fn wait_for_shutdown(self) {
        self.client_thread
            .join()
            .expect("Error when shutting down server network thread");
    }
}

const NET_CLIENT_CONNECTION_RETRIES: u32 = 5;
/// Effective timeout is this multiplied by `NET_CLIENT_CONNECTION_RETRIES`
const NET_CLIENT_CONNECTION_HANDSHAKE_TIMEOUT: time::Duration = time::Duration::from_millis(500);
const NET_CLIENT_CONNECTION_TIMEOUT: time::Duration = time::Duration::from_secs(15);
/// How often the control message queue is checked if there's no network traffic,
/// in practice the timeout to recvfrom in the network thread.
const NET_CLIENT_POLL_MQ_INTERVAL: time::Duration = time::Duration::from_millis(5);

fn client_netmain(
    cfg: ConfigHandle,
    control: flume::Receiver<ClientControlMessage>,
    socket: std::net::UdpSocket,
    shared_data: Arc<ClientSharedData>,
) -> std::io::Result<()> {
    socket.set_read_timeout(Some(NET_CLIENT_POLL_MQ_INTERVAL))?;
    socket.set_write_timeout(Some(NET_CLIENT_POLL_MQ_INTERVAL))?;
    let mut recvbuf: Box<[u8; 65536]> = bxw_util::bytemuck::zeroed_box();
    log::info!(
        "Attempting connection to {:?} from {:?}",
        shared_data.server.address,
        socket.local_addr()
    );
    let (hs0pkt, hs0state) = authflow_client_handshake_packet(
        &shared_data.config.id_keys.0,
        ClientConnectionType::GameClient,
    )
    .expect("Couldn't encode handshake packet");
    let mut hs0resp = None;
    'handshake_retries: for _retries in 0..NET_CLIENT_CONNECTION_RETRIES {
        socket.send(&hs0pkt)?;
        let sendtime = time::Instant::now();
        let timeout = sendtime + NET_CLIENT_CONNECTION_HANDSHAKE_TIMEOUT;
        loop {
            match socket.recv(&mut recvbuf[..]) {
                Err(e)
                    if matches!(
                        e.kind(),
                        ErrorKind::TimedOut | ErrorKind::WouldBlock | ErrorKind::Interrupted
                    ) => {}
                Err(e) => {
                    return Err(e);
                }
                Ok(bytes_received) => {
                    let msg = &recvbuf[0..bytes_received];
                    match authflow_client_try_accept_handshake_ack(
                        &hs0state,
                        msg,
                        (&shared_data.config.id_keys.0, &shared_data.config.id_keys.1),
                    ) {
                        Ok(resp) => {
                            hs0resp = Some(resp);
                            break 'handshake_retries;
                        }
                        Err(err) => {
                            // could be spoofed packets, so just ignore them (but log)
                            log::warn!("Potential spoofed/wrong packed during network handshake -- error code: {:?}", err);
                            continue;
                        }
                    }
                }
            }
            if time::Instant::now() >= timeout {
                break;
            }
            if control.is_disconnected() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "Connection to server cancelled",
                ));
            }
            while let Ok(msg) = control.try_recv() {
                if let ClientControlMessage::Disconnect = msg {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "Connection to server cancelled",
                    ));
                }
            }
        }
    }
    drop((hs0pkt, hs0state));
    let (connresp, cccstate) = hs0resp.ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::TimedOut,
            "Couldn't connect to server, check log for details",
        )
    })?;
    if connresp != ConnectionResponse::Accepted {
        log::error!("Server didn't accept our connection: {:?}", connresp);
        return Err(std::io::Error::new(
            std::io::ErrorKind::ConnectionRefused,
            "Server didn't accept connection, check log for details",
        ));
    }
    let mut current_server_address = shared_data.server.address;
    log::info!(
        "Connected to server at {:?}: Name(`{}`), Version({}), ID({})",
        shared_data.server.address,
        &cccstate.server_name,
        &cccstate.server_version_id,
        bxw_util::sodiumoxide::hex::encode(&cccstate.server_id)
    );
    let mut last_recv_time = time::Instant::now();
    let mut last_send_time = last_recv_time;
    let mut send_keepalive = true;
    // Main network loop
    'netloop: loop {
        if control.is_disconnected() {
            break;
        }
        while let Ok(msg) = control.try_recv() {
            match msg {
                ClientControlMessage::Disconnect => {
                    break 'netloop;
                }
                ClientControlMessage::SendChat(_) => todo!(),
            }
        }
        {
            let now = time::Instant::now();
            let since_recv = now - last_recv_time;
            let since_send = now - last_send_time;
            // monitoring only, so can be relaxed
            shared_data.ms_since_last_recv.store(
                since_recv.as_millis().max(u32::MAX as u128) as u32,
                Ordering::Relaxed,
            );
            shared_data.ms_since_last_send.store(
                since_send.as_millis().max(u32::MAX as u128) as u32,
                Ordering::Relaxed,
            );
            if since_recv >= NET_CLIENT_CONNECTION_TIMEOUT {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    "Server connection timed out",
                ));
            }
            if since_send >= NET_KEEPALIVE_INTERVAL {
                send_keepalive = true;
            }
        }
        if send_keepalive {
            // TODO: Keepalive/ACK vectors
        }
        match socket.recv_from(&mut recvbuf[..]) {
            Err(e)
                if matches!(
                    e.kind(),
                    ErrorKind::TimedOut | ErrorKind::WouldBlock | ErrorKind::Interrupted
                ) => {}
            Err(e) => {
                // TODO: Resilient error handling
                return Err(e);
            }
            Ok((bytes_received, received_from)) => {
                let msg = &recvbuf[0..bytes_received];
                if msg.len() < 32 {
                    continue;
                }
                let pkt = match PacketV1::decode_established(msg, &cccstate.rx_key) {
                    Ok(pkt) => {
                        last_recv_time = time::Instant::now();
                        if current_server_address != received_from {
                            log::warn!(
                                "Server address changed to {:?} from {:?}",
                                received_from,
                                current_server_address
                            );
                            current_server_address = received_from;
                        }
                        pkt
                    }
                    Err(e) => {
                        //TODO:Slow warns
                        log::warn!("Packet processing error {:?}", e);
                        continue;
                    }
                };
                match pkt.stream {
                    PacketStream::Handshake => { /* ignore */ }
                    PacketStream::ConnectionControl => {
                        //
                    }
                    _ => { /* hand-off to game processing */ }
                }
            }
        }
    }
    log::info!("Client socket handler terminating");
    Ok(())
}
