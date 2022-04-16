use crate::config::ConfigHandle;
use crate::network::packets;
use crate::network::protocol;
use bxw_util::flume;
use bxw_util::log;
use bxw_util::parking_lot::RwLock;
use bxw_util::smallvec::SmallVec;
use bxw_util::sodiumoxide::crypto::box_;
use bxw_util::sodiumoxide::crypto::secretbox;
use std::collections::HashMap;
use std::convert::TryInto;
use std::io::ErrorKind;
use std::iter::FromIterator;
use std::net::SocketAddr;
use std::sync::atomic::AtomicU32;
use std::sync::Arc;
use std::thread;
use std::time;

#[derive(Debug)]
pub enum ServerCreationError {
    SocketBindError { addr: String, error: std::io::Error },
}

#[derive(Clone, Debug, Hash)]
pub enum ServerControlMessage {
    Stop,
    BroadcastChat(String),
}

pub struct NetServer {
    server_thread: thread::JoinHandle<()>,
    server_control: flume::Sender<ServerControlMessage>,
    shared_state: Arc<NetServerSharedState>,
}

type ConnectionSource = (usize, std::net::SocketAddr);

pub struct ConnectedClient {
    source: ConnectionSource,
    cryptostate: protocol::ServersideConnectionCryptoState,
}

pub struct NetServerSharedState {
    connection_raw_count: AtomicU32,
    server_id_keys: (box_::PublicKey, box_::SecretKey),
    server_token_key: secretbox::Key,
    server_name: RwLock<String>,
}

impl NetServerSharedState {
    pub fn new(server_id_keys: (box_::PublicKey, box_::SecretKey), server_name: String) -> Self {
        Self {
            connection_raw_count: AtomicU32::new(0),
            server_id_keys,
            server_token_key: secretbox::gen_key(),
            server_name: RwLock::new(server_name),
        }
    }
}

const SERVER_CONTROL_CHANNEL_BOUND: usize = 1024;

impl NetServer {
    pub fn new(cfg: ConfigHandle) -> Result<Self, ServerCreationError> {
        let cfg_clone = cfg.clone();
        let (scon_tx, scon_rx) = flume::bounded(SERVER_CONTROL_CHANNEL_BOUND);
        let sockets = {
            let mut v: Vec<mio::net::UdpSocket> =
                Vec::with_capacity(cfg.read().server_listen_addresses.len());
            for addr in cfg.read().server_listen_addresses.iter() {
                match mio::net::UdpSocket::bind(*addr) {
                    Ok(sock) => {
                        v.push(sock);
                        log::info!("Bound socket #{} to address `{}`", v.len() - 1, addr);
                    }
                    Err(error) => {
                        log::error!(
                            "Error binding socket #{} to address `{}`: {}",
                            v.len(),
                            addr,
                            error
                        );
                        return Err(ServerCreationError::SocketBindError {
                            addr: format!("{}", addr),
                            error,
                        });
                    }
                }
            }
            v
        };
        // TODO: Save/load identifying keys
        let shared_state = Arc::new(NetServerSharedState::new(
            box_::gen_keypair(),
            String::from("BXW Server"),
        ));
        let shared_state_copy = Arc::clone(&shared_state);
        let server_thread = thread::Builder::new()
            .name("bxw-server-netio-main".to_owned())
            .stack_size(2 * 1024 * 1024)
            .spawn(move || ServerNetmain::new(cfg_clone, scon_rx, sockets, shared_state_copy).run())
            .expect("Couldn't start main server network thread");
        Ok(Self {
            server_thread,
            server_control: scon_tx,
            shared_state,
        })
    }

    pub fn send_control_message(&self, msg: ServerControlMessage) {
        match self.server_control.send(msg.clone()) {
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
        self.server_thread
            .join()
            .expect("Error when shutting down server network thread");
    }
}

#[derive(Clone, Eq, PartialEq, Debug, Hash)]
struct RawPacket {
    source: (usize, SocketAddr),
    data: Box<[u8]>,
}
/*
async fn server_connection_handler(
    source: (usize, SocketAddr),
    initial_hs_state: protocol::ServerHandshakeState1,
    mut packet_stream: mpsc::Receiver<RawPacket>,
    shared_state: Arc<NetServerSharedState>,
    socket: Arc<net::UdpSocket>,
) {
    let target = source.1;
    let _connguard =
        bxw_util::scopeguard::guard(Arc::clone(&shared_state.connection_raw_count), |cc| {
            cc.fetch_sub(1, SeqCst);
        });
    log::info!(
        "New connection from {:?} - version {}, id {}",
        source,
        initial_hs_state.get_request().c_version_id,
        bxw_util::sodiumoxide::hex::encode(&initial_hs_state.get_request().c_player_id)
    );
    let connresponse = packets::auth::ConnectionResponse::Accepted;
    let (hsack_packet, _ssccs) = match authflow_server_respond_to_handshake_packet(
        initial_hs_state,
        &shared_state.server_id_keys.0,
        &shared_state.server_id_keys.1,
        shared_state.server_name.read().clone(),
        connresponse,
    ) {
        Ok(x) => x,
        Err(e) => {
            log::warn!("Error from {:?}: {:?}", e, source);
            packet_stream.close();
            return;
        }
    };
    match socket.send_to(&hsack_packet, target).await {
        Ok(_) => {}
        Err(e) => {
            log::warn!(
                "Error sending handshake ack to {:?}, terminating connection: {:?}",
                source,
                e
            );
            packet_stream.close();
            return;
        }
    }
}*/

const NET_SERVER_CONNECTION_TIMEOUT: time::Duration = time::Duration::from_secs(15);
/// How often the control message queue is checked if there's no network traffic,
/// in practice the timeout to mio::Poll::poll in the network thread.
const NET_SERVER_POLL_MQ_INTERVAL: time::Duration = time::Duration::from_millis(5);

struct ServerNetmain {
    cfg: ConfigHandle,
    control: flume::Receiver<ServerControlMessage>,
    sockets: Vec<mio::net::UdpSocket>,
    shared_state: Arc<NetServerSharedState>,
    mtu: u16,
    recv_buf: Box<[u8; 65536]>,
    send_bytes: Box<[slice_deque::SliceDeque<u8>]>,
    send_bytes_offset: Box<[u64]>,
    send_queue: Box<[slice_deque::SliceDeque<(std::net::SocketAddr, u64, usize)>]>,
    connected_clients: HashMap<ConnectionSource, ConnectedClient>,
}

impl ServerNetmain {
    /// Don't use directly, try sending over the socket first,
    /// and fall back to this if the socket was blocked
    fn push_send_bytes(&mut self, socket: usize, addr: std::net::SocketAddr, bytes: &[u8]) {
        let start = self.send_bytes_offset[socket] + self.send_bytes[socket].len() as u64;
        self.send_bytes[socket].extend_from_slice(bytes);
        self.send_queue[socket].push_back((addr, start, bytes.len()));
    }

    fn get_send_bytes_range(&self, socket: usize, qentry: usize) -> std::ops::Range<usize> {
        let (_, start, len) = &self.send_queue[socket][qentry];
        let start: usize = (start - self.send_bytes_offset[socket]).try_into().unwrap();
        start..start + len
    }

    fn pop_send_bytes_range(&mut self, socket: usize) {
        let (_, start, len) = &self.send_queue[socket].front().unwrap();
        let (start, len) = (*start, *len);
        assert_eq!(0, start - self.send_bytes_offset[socket]);
        assert!(self.send_bytes[socket].len() >= len);
        self.send_bytes_offset[socket] += len as u64;
        self.send_bytes[socket].truncate_front(self.send_bytes[socket].len() - len);
        self.send_queue[socket].pop_front();
    }

    fn new(
        cfg: ConfigHandle,
        control: flume::Receiver<ServerControlMessage>,
        sockets: Vec<mio::net::UdpSocket>,
        shared_state: Arc<NetServerSharedState>,
    ) -> Self {
        let mtu = cfg.read().server_mtu;
        let num_socks = sockets.len();
        Self {
            cfg,
            control,
            sockets,
            shared_state,
            mtu,
            recv_buf: bxw_util::bytemuck::zeroed_box(),
            send_bytes: Box::from_iter(std::iter::repeat(Default::default()).take(num_socks)),
            send_bytes_offset: Box::from_iter(std::iter::repeat(0).take(num_socks)),
            send_queue: Box::from_iter(std::iter::repeat(Default::default()).take(num_socks)),
            connected_clients: HashMap::with_capacity(32),
        }
    }

    fn run(&mut self) {
        let mut events = mio::Events::with_capacity(128);
        let mut poll = mio::Poll::new().expect("Cannot create network socket poller");
        for (i, sock) in self.sockets.iter_mut().enumerate() {
            poll.registry()
                .register(
                    sock,
                    mio::Token(i),
                    mio::Interest::READABLE | mio::Interest::WRITABLE,
                )
                .expect("Cannot register network sockets with poll object");
        }
        'netloop: loop {
            if self.control.is_disconnected() {
                break;
            }
            while let Ok(msg) = self.control.try_recv() {
                match msg {
                    ServerControlMessage::Stop => {
                        break 'netloop;
                    }
                    ServerControlMessage::BroadcastChat(_) => todo!(),
                }
            }
            self.nettick_clients();
            match poll.poll(&mut events, Some(NET_SERVER_POLL_MQ_INTERVAL)) {
                Err(e)
                    if matches!(
                        e.kind(),
                        ErrorKind::TimedOut | ErrorKind::WouldBlock | ErrorKind::Interrupted
                    ) => {}
                Err(e) => {
                    // TODO: Resilient error handling
                    log::error!("Error polling sockets: {:?}", e);
                    return;
                }
                Ok(()) => {
                    for event in events.iter() {
                        self.handle_event(poll.registry(), event);
                    }
                }
            }
        }
    }

    fn nettick_clients(&mut self) {
        //
    }

    fn handle_event(&mut self, registry: &mio::Registry, ev: &mio::event::Event) {
        let socket_id = ev.token().0;
        if ev.is_writable() {
            let mut needs_more_events = false;
            while let Some((sto, _, _)) = self.send_queue[socket_id].front() {
                let bytes = &self.send_bytes[socket_id][self.get_send_bytes_range(socket_id, 0)];
                match self.sockets[socket_id].send_to(bytes, *sto) {
                    Ok(sent) => {
                        assert_eq!(sent, bytes.len());
                        self.pop_send_bytes_range(socket_id);
                    }
                    Err(e) if matches!(e.kind(), ErrorKind::WouldBlock) => {
                        needs_more_events = true;
                        break;
                    }
                    Err(e) if matches!(e.kind(), ErrorKind::TimedOut | ErrorKind::Interrupted) => {
                        continue;
                    }
                    Err(_) => {
                        // TODO: Resilient error handling
                        log::error!(
                            "Could not send packet S#{}->{:?}, {} bytes",
                            socket_id,
                            sto,
                            bytes.len()
                        );
                        break;
                    }
                }
            }
            if !needs_more_events {
                registry
                    .reregister(
                        &mut self.sockets[socket_id],
                        mio::Token(socket_id),
                        mio::Interest::READABLE,
                    )
                    .expect("Couldn't re-register network socket with poll");
            }
        }
        if ev.is_readable() {
            loop {
                match self.sockets[socket_id].recv_from(&mut self.recv_buf[..]) {
                    Ok((byte_count, from)) => {
                        self.handle_packet(socket_id, byte_count, from);
                    }
                    Err(e) if matches!(e.kind(), ErrorKind::WouldBlock) => {
                        break;
                    }
                    Err(e) if matches!(e.kind(), ErrorKind::TimedOut | ErrorKind::Interrupted) => {
                        continue;
                    }
                    Err(_) => {
                        // TODO: Resilient error handling
                        log::error!("Could not receive packet on socket S#{}", socket_id);
                        break;
                    }
                }
            }
        }
    }

    fn handle_packet(&mut self, socket_id: usize, byte_count: usize, from: SocketAddr) {
        let msg = &self.recv_buf[0..byte_count];
        let csrc = (socket_id, from);
        let mut source: SmallVec<[u8; 32]> = SmallVec::new();
        source.extend_from_slice(&socket_id.to_ne_bytes());
        //TODO: ip source.extend_from_slice(from);
        let conntable_entry = self.connected_clients.entry(csrc);
        if let std::collections::hash_map::Entry::Occupied(_conntable_entry) = conntable_entry {
            //
        } else {
            let shs1 = match protocol::authflow_server_try_accept_handshake_packet(
                msg,
                &source,
                &self.shared_state.server_token_key,
                false,
            ) {
                Ok(x) => x,
                Err(_) => {
                    // TODO: Slow log
                    log::info!(
                        "Unrecognized packet from S{}-{:?} of size {}",
                        socket_id,
                        from,
                        byte_count
                    );
                    return;
                }
            };
            let response = packets::auth::ConnectionResponse::Accepted;
            let sname = self.shared_state.server_name.read().clone();
            match protocol::authflow_server_respond_to_handshake_packet(
                shs1,
                &self.shared_state.server_id_keys.0,
                &self.shared_state.server_id_keys.1,
                sname,
                response,
            ) {
                Ok(_) => {}
                Err(e) => {
                    log::error!(
                        "Couldn't respond to handshake packet from S{}-{:?} of size {}: {:?}",
                        socket_id,
                        from,
                        byte_count,
                        e
                    );
                }
            }
            // TODO:Make stateless
        }
    }
}
