use crate::ecs::*;
use crate::*;
use bxw_util::fnv::*;
use bxw_util::itertools::*;
use bxw_util::math::*;
use bxw_util::smallvec::*;
use bxw_util::taskpool::{Task, TaskPool};
use bxw_util::*;
use parking_lot::*;
use std::any::*;
use std::cell::Cell;
use std::sync::atomic::*;
use std::sync::mpsc::*;

#[repr(u8)]
#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq, Hash, Debug)]
pub enum ChunkDataState {
    Unloaded = 0,
    Waiting = 1,
    Loaded = 2,
    Updating = 3,
}

impl ChunkDataState {
    pub fn requires_load_request(self) -> bool {
        self == ChunkDataState::Unloaded
    }
}

pub const CHUNK_BLOCK_DATA: usize = 0;
pub const CHUNK_LIGHT_DATA: usize = 1;
pub const CHUNK_MESH_DATA: usize = 2;

pub type ChunkBlockData = Option<Arc<VChunk>>;
pub type ChunkLightData = Option<Arc<()>>;
type KindsSmallVec = SmallVec<[usize; 6]>;

pub type AnyChunkData = Option<Arc<dyn Any + Send + Sync>>;

pub trait ChunkDataHandler {
    fn status_array(&self) -> &Vec<ChunkDataState>;
    fn status_array_mut(&mut self) -> &mut Vec<ChunkDataState>;
    /// (ChunkDataKind, requires 26 populated neighbors)
    fn get_dependency(&self) -> Option<(usize, bool)>;
    fn get_data(&self, world: &World, index: usize) -> AnyChunkData;
    /// Returns the old data
    fn swap_data(&mut self, world: &World, index: usize, new_data: AnyChunkData) -> AnyChunkData;
    /// Extend with unloaded states or shrink unloaded states (it will never be called on non-Unloaded chunks)
    fn resize_data(&mut self, world: &World, new_size: usize);
    fn create_chunk_update_task(
        &mut self,
        world: &World,
        cpos: ChunkPosition,
        index: usize,
    ) -> Option<Task>;
    fn needs_loading_for_anchor(&self, anchor: &CLoadAnchor) -> bool;
}

#[derive(Clone, Default, Debug)]
pub struct NoopChunkDataHandler {
    pub statuses: Vec<ChunkDataState>,
}

impl ChunkDataHandler for NoopChunkDataHandler {
    fn status_array(&self) -> &Vec<ChunkDataState> {
        &self.statuses
    }

    fn status_array_mut(&mut self) -> &mut Vec<ChunkDataState> {
        &mut self.statuses
    }

    fn get_dependency(&self) -> Option<(usize, bool)> {
        None
    }

    fn get_data(&self, _world: &World, _index: usize) -> AnyChunkData {
        None
    }

    fn swap_data(
        &mut self,
        _world: &World,
        _index: usize,
        _new_data: AnyChunkData,
    ) -> AnyChunkData {
        None
    }

    fn resize_data(&mut self, _world: &World, _new_size: usize) {}

    fn create_chunk_update_task(
        &mut self,
        _world: &World,
        _cpos: ChunkPosition,
        index: usize,
    ) -> Option<Task> {
        self.statuses[index] = ChunkDataState::Loaded;
        None
    }

    fn needs_loading_for_anchor(&self, _anchor: &CLoadAnchor) -> bool {
        false
    }
}

pub type SynchronousUpdateTask = Box<dyn FnOnce(&mut World)>;

pub struct World {
    allocation: FnvHashMap<ChunkPosition, usize>,
    chunk_positions: Vec<Option<ChunkPosition>>,
    free_indices: Vec<usize>,
    data: Vec<RefCell<Box<dyn ChunkDataHandler>>>,
    entities: Arc<RwLock<ECS>>,
    load_data: Arc<Mutex<LoadData>>,
    load_data_busy: Arc<AtomicBool>,
    tasks_in_pool: Arc<AtomicI32>,
    remaining_deltas: Vec<ChunkDelta>,
    sync_task_queue: (
        SyncSender<SynchronousUpdateTask>,
        Receiver<SynchronousUpdateTask>,
    ),
}

#[derive(Clone, Eq, PartialEq, Hash)]
struct LoadAnchor {
    eid: u64,
    cpos: ChunkPosition,
    radius: u32,
    requested_kinds: KindsSmallVec,
}

impl Ord for LoadAnchor {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        Ord::cmp(&self.eid, &other.eid)
    }
}

impl PartialOrd for LoadAnchor {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        PartialOrd::partial_cmp(&self.eid, &other.eid)
    }
}

#[derive(Clone, Eq, PartialEq)]
struct ChunkDelta {
    unload: bool,
    cpos: ChunkPosition,
    handlers: KindsSmallVec,
    min_anchor_distance: i32,
}

impl Ord for ChunkDelta {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering::*;
        // unloads before loads
        if self.unload != other.unload {
            if self.unload {
                Less
            } else {
                Greater
            }
        } else if self.unload {
            // Unload farthest first
            Ord::cmp(&other.min_anchor_distance, &self.min_anchor_distance)
        } else {
            // Load nearest first
            Ord::cmp(&self.min_anchor_distance, &other.min_anchor_distance)
        }
    }
}

impl PartialOrd for ChunkDelta {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(Ord::cmp(self, other))
    }
}

struct LoadData {
    allocation: FnvHashMap<ChunkPosition, usize>,
    chunk_positions: Vec<Option<ChunkPosition>>,
    busy: Arc<AtomicBool>,
    status_arrays: Vec<Vec<ChunkDataState>>,
    anchors: Vec<LoadAnchor>,
    remaining_deltas: Vec<ChunkDelta>,
    dependencies: Vec<Option<(usize, bool)>>,
}

impl Default for World {
    fn default() -> Self {
        let default_capacity = 64;
        let busy_arc = Arc::new(AtomicBool::new(false));
        let (tx, rx) = sync_channel(256);
        let mut handlers = Vec::new();
        for _ in 0..8 {
            handlers.push(RefCell::new(
                Box::new(NoopChunkDataHandler::default()) as Box<dyn ChunkDataHandler>
            ));
        }
        Self {
            allocation: FnvHashMap::with_capacity_and_hasher(default_capacity, Default::default()),
            chunk_positions: Vec::with_capacity(default_capacity),
            free_indices: Vec::with_capacity(default_capacity),
            data: handlers,
            entities: Arc::new(Default::default()),
            load_data: Arc::new(Mutex::new(LoadData {
                allocation: Default::default(),
                chunk_positions: Default::default(),
                busy: busy_arc.clone(),
                status_arrays: Vec::new(),
                anchors: Default::default(),
                remaining_deltas: Default::default(),
                dependencies: Default::default(),
            })),
            load_data_busy: busy_arc,
            tasks_in_pool: Arc::new(Default::default()),
            remaining_deltas: Vec::new(),
            sync_task_queue: (tx, rx),
        }
    }
}

impl World {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn get_sync_task_channel(&self) -> SyncSender<SynchronousUpdateTask> {
        self.sync_task_queue.0.clone()
    }

    fn extend_allocations(&mut self) {
        let ocap = self.chunk_positions.len();
        let ncap = (ocap + 1) * 2;
        if self.allocation.capacity() < ncap {
            self.allocation.reserve(ncap - self.allocation.capacity());
        }
        self.chunk_positions.resize(ncap, None);
        if self.free_indices.capacity() < ncap {
            self.free_indices.reserve(ncap);
        }
        for idx in ocap..ncap {
            self.free_indices.push(idx);
        }
        for d in self.data.iter() {
            let mut d = d.borrow_mut();
            d.status_array_mut().resize(ncap, ChunkDataState::Unloaded);
            d.resize_data(self, ncap);
        }
    }

    fn get_or_allocate_chunk(&mut self, cpos: ChunkPosition) -> usize {
        if let Some(&idx) = self.allocation.get(&cpos) {
            idx
        } else {
            if self.free_indices.is_empty() {
                self.extend_allocations();
            }
            let fidx = self.free_indices.pop().unwrap();
            self.chunk_positions[fidx] = Some(cpos);
            self.allocation.insert(cpos, fidx);
            fidx
        }
    }

    pub fn main_loop_tick(&mut self, task_pool: &TaskPool) {
        self.check_load_deltas(task_pool);
        for _taskn in 0..256 {
            match self.sync_task_queue.1.try_recv() {
                Ok(task) => (task)(self),
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
        }
        let mut remaining = 256 - self.tasks_in_pool.load(Ordering::SeqCst);
        let mut tasks = Vec::with_capacity(32);
        for _deltan in 0..512 {
            if let Some(delta) = self.remaining_deltas.pop() {
                let cid = match self.allocation.get(&delta.cpos).copied() {
                    Some(cid) => cid,
                    None => {
                        if delta.unload {
                            continue;
                        } else {
                            self.get_or_allocate_chunk(delta.cpos)
                        }
                    }
                };
                if delta.unload {
                    for kind in delta.handlers {
                        let mut kind = self.data[kind].borrow_mut();
                        if kind.status_array()[cid] != ChunkDataState::Unloaded {
                            let arc = kind.swap_data(self, cid, None);
                            kind.status_array_mut()[cid] = ChunkDataState::Unloaded;
                            let cnt = self.tasks_in_pool.clone();
                            tasks.push(Task::new(
                                move || {
                                    drop(arc);
                                    cnt.fetch_sub(1, Ordering::SeqCst);
                                },
                                false,
                                false,
                            ));
                        }
                    }
                } else {
                    for kind in delta.handlers {
                        let mut kind = self.data[kind].borrow_mut();
                        if kind.status_array()[cid] == ChunkDataState::Loaded {
                            continue;
                        } else {
                            let task = kind.create_chunk_update_task(self, delta.cpos, cid);
                            if let Some(task) = task {
                                remaining -= 1;
                                self.tasks_in_pool.fetch_add(1, Ordering::SeqCst);
                                let cnt = self.tasks_in_pool.clone();
                                tasks.push(task.compose_with(Task::new(
                                    move || {
                                        cnt.fetch_sub(1, Ordering::SeqCst);
                                    },
                                    false,
                                    false,
                                )));
                            }
                        }
                    }
                }
            } else {
                break;
            }
            if remaining <= 0 {
                break;
            }
        }
    }

    fn check_load_deltas(&mut self, task_pool: &TaskPool) {
        if !self.load_data_busy.load(Ordering::Acquire) {
            let mut load_data = self.load_data.lock();
            if !load_data.remaining_deltas.is_empty() {
                std::mem::swap(&mut self.remaining_deltas, &mut load_data.remaining_deltas);
                load_data.remaining_deltas.clear();
            }
            let ecs = self.entities.read();
            let mut new_anchors = Vec::with_capacity(load_data.anchors.len());
            let it = ECSHandler::<CLoadAnchor>::iter(&*ecs);

            for anchor in it {
                let mut anchor_kinds = SmallVec::new();
                for (kid, kind) in self.data.iter().enumerate() {
                    let kind = kind.borrow();
                    if kind.needs_loading_for_anchor(anchor) {
                        anchor_kinds.push(kid);
                    }
                }
                let loc: Option<&CLocation> = ecs.get_component(anchor.entity_id());
                if loc.is_none() {
                    continue;
                }
                let loc = loc.unwrap();
                new_anchors.push(LoadAnchor {
                    eid: anchor.entity_id().u64(),
                    cpos: chunkpos_from_blockpos(blockpos_from_worldpos(loc.position)),
                    radius: anchor.radius,
                    requested_kinds: anchor_kinds,
                });
            }
            drop(ecs);
            new_anchors.sort();
            //if new_anchors != load_data.anchors {
            {
                load_data
                    .status_arrays
                    .resize_with(self.data.len(), Vec::new);
                load_data.chunk_positions.clone_from(&self.chunk_positions);
                load_data.allocation.clone_from(&self.allocation);
                load_data.dependencies = self
                    .data
                    .iter()
                    .map(|d| d.borrow().get_dependency())
                    .collect();
                for (i, status_array) in load_data.status_arrays.iter_mut().enumerate() {
                    let orig = self.data[i].borrow_mut();
                    status_array.clone_from(orig.status_array());
                }
                self.load_data_busy.store(true, Ordering::SeqCst);
                load_data.anchors = new_anchors;
                drop(load_data);
                let task_load_data = self.load_data.clone();
                task_pool.push_tasks(std::iter::once(Task::new(
                    move || recalculate_load_deltas(task_load_data),
                    true,
                    false,
                )));
            }
        }
    }
}

fn recalculate_load_deltas(load_data: Arc<Mutex<LoadData>>) {
    let mut load_data = load_data.lock();
    let mut chunks_to_unload: FnvHashMap<ChunkPosition, (KindsSmallVec, i32)> = Default::default();
    chunks_to_unload.reserve(64);
    for (&cpos, &cid) in load_data.allocation.iter() {
        let mut kind_should_be_loaded = [false; 8];
        let mut min_dist = i32::max_value();
        for anchor in load_data.anchors.iter() {
            let dist: i32 = (cpos - anchor.cpos).iter().map(|x| x * x).sum();
            min_dist = dist.min(min_dist);
            if does_anchor_load_coords(anchor, cpos) {
                for &kind in anchor.requested_kinds.iter() {
                    kind_should_be_loaded[kind] = true;
                }
            }
        }
        for (kind, statuses) in load_data.status_arrays.iter().enumerate() {
            let is_loaded = !ChunkDataState::requires_load_request(statuses[cid]);
            if is_loaded && !kind_should_be_loaded[kind] {
                chunks_to_unload.insert(cpos, (Default::default(), min_dist));
            }
        }
    }
    let mut chunks_to_load: FnvHashMap<ChunkPosition, (KindsSmallVec, Cell<i32>)> =
        Default::default();
    chunks_to_load.reserve(64);
    for anchor in load_data.anchors.iter() {
        for cpos in iter_coords_to_load(anchor) {
            let cid = load_data.allocation.get(&cpos).copied();
            let missing: KindsSmallVec = if let Some(cid) = cid {
                let mut missing = SmallVec::new();
                for &kind in anchor.requested_kinds.iter() {
                    if ChunkDataState::requires_load_request(load_data.status_arrays[kind][cid]) {
                        missing.push(kind);
                    }
                }
                missing
            } else {
                anchor.requested_kinds.clone()
            };
            for kind in missing {
                if can_request_load(&load_data, kind, cpos, cid) {
                    let dist: i32 = (cpos - anchor.cpos).iter().map(|x| x * x).sum();
                    let (rkinds, rmindist) = chunks_to_load
                        .entry(cpos)
                        .or_insert((Default::default(), Cell::new(dist)));
                    if rkinds.contains(&kind) {
                        rmindist.set(rmindist.get().min(dist));
                    } else {
                        rkinds.push(kind);
                    }
                }
            }
        }
    }
    load_data.remaining_deltas.clear();
    let old_cap = load_data.remaining_deltas.capacity();
    let new_cap = chunks_to_load.len();
    if old_cap < new_cap {
        load_data.remaining_deltas.reserve(new_cap - old_cap);
    }
    for (cpos, (unload_kinds, min_anchor_distance)) in chunks_to_unload {
        load_data.remaining_deltas.push(ChunkDelta {
            unload: true,
            cpos,
            handlers: unload_kinds,
            min_anchor_distance,
        });
    }
    for (cpos, (load_kinds, min_anchor_distance)) in chunks_to_load {
        load_data.remaining_deltas.push(ChunkDelta {
            unload: false,
            cpos,
            handlers: load_kinds,
            min_anchor_distance: min_anchor_distance.get(),
        });
    }
    load_data.remaining_deltas.sort_unstable();
    load_data.remaining_deltas.reverse();
    load_data.busy.store(false, Ordering::SeqCst);
}

fn iter_coords_to_load(loader: &LoadAnchor) -> Box<dyn Iterator<Item = ChunkPosition>> {
    let origin = loader.cpos;
    let r = loader.radius as i32;
    let r2 = r * r;
    let rrange = -r..=r;
    let cube_range = iproduct!(rrange.clone(), rrange.clone(), rrange);
    let sphere_range = cube_range.filter(move |(x, y, z)| x * x + y * y + z * z <= r2);
    Box::new(sphere_range.map(move |(x, y, z)| origin + vec3(x, y, z)))
}

fn does_anchor_load_coords(loader: &LoadAnchor, cpos: ChunkPosition) -> bool {
    (cpos - loader.cpos).map(|x| x * x).sum() <= loader.radius as i32
}

fn iter_neighbors(cpos: ChunkPosition) -> Box<dyn Iterator<Item = ChunkPosition>> {
    Box::new(
        iproduct!(
            cpos.x - 1..=cpos.x + 1,
            cpos.y - 1..=cpos.y + 1,
            cpos.z - 1..=cpos.z + 1
        )
        .filter(move |&(x, y, z)| x != cpos.x || y != cpos.y || z != cpos.z)
        .map(|(x, y, z)| vec3(x, y, z)),
    )
}

fn can_request_load(
    load_data: &LoadData,
    kind: usize,
    cpos: ChunkPosition,
    cid: Option<usize>,
) -> bool {
    if let Some((depkind, neighbours)) = load_data.dependencies[kind] {
        let cid = if let Some(cid) = cid {
            cid
        } else {
            return false;
        };
        let statuses = &load_data.status_arrays[depkind];
        if neighbours {
            !iter_neighbors(cpos).any(|npos| {
                load_data
                    .allocation
                    .get(&npos)
                    .copied()
                    .map(|nid| ChunkDataState::requires_load_request(statuses[nid]))
                    .unwrap_or(true)
            })
        } else {
            !ChunkDataState::requires_load_request(statuses[cid])
        }
    } else {
        true
    }
}
