use crate::world::ecs::ECS;
use crate::world::registry::VoxelRegistry;
use crate::world::{ChunkPosition, VoxelChunk, VoxelChunkRef, VOXEL_CHUNK_DIM};
use cgmath::prelude::*;
use cgmath::{vec3, Vector3};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::sync::{Arc, RwLock};

pub trait WorldGenerator {
    fn generate_chunk(&self, cref: VoxelChunkRef, registry: &VoxelRegistry);
}

#[derive(Debug, Clone)]
pub struct LoadAnchor {
    pub position: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub chunk_radius: i32,
}

impl Default for LoadAnchor {
    fn default() -> Self {
        LoadAnchor {
            position: Vector3::zero(),
            velocity: Vector3::zero(),
            chunk_radius: 1,
        }
    }
}

type ClientWorld = crate::client::world::ClientWorld;

pub struct World {
    pub name: String,
    pub loaded_chunks: HashMap<ChunkPosition, Arc<RwLock<VoxelChunk>>>,
    pub loading_queue: HashSet<ChunkPosition>,
    pub load_anchor: LoadAnchor,
    pub worldgen: Option<Arc<dyn WorldGenerator>>,
    pub registry: Arc<VoxelRegistry>,
    pub entities: Arc<RwLock<ECS>>,
    pub client_world: Option<ClientWorld>,
}

impl Debug for World {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "World {{ name: {}, loaded_chunks: {:?}, loading_queue: {:?} }}",
            &self.name,
            self.loaded_chunks.keys(),
            self.loading_queue
        )
    }
}

impl World {
    pub fn new(name: String, registry: Arc<VoxelRegistry>) -> World {
        Self {
            name,
            loaded_chunks: HashMap::new(),
            loading_queue: HashSet::new(),
            load_anchor: Default::default(),
            worldgen: None,
            registry,
            entities: Arc::new(RwLock::new(ECS::new())),
            client_world: None,
        }
    }

    pub fn change_generator(&mut self, new_generator: Arc<dyn WorldGenerator>) {
        self.worldgen = Some(new_generator);
    }

    pub fn physics_tick(&mut self) {
        // Chunk loading
        if self.worldgen.is_some() {
            let mut req_positions: HashSet<ChunkPosition> = HashSet::new();

            {
                let anchor = &self.load_anchor;
                let r = anchor.chunk_radius;
                let pos: Vector3<i32> = anchor
                    .position
                    .div_element_wise(VOXEL_CHUNK_DIM as f32)
                    .map(|c| c as i32);
                for xoff in 0..r {
                    for yoff in 0..r {
                        for zoff in 0..r {
                            if xoff * xoff + yoff * yoff + zoff * zoff <= r * r {
                                req_positions.insert(pos + vec3(xoff, yoff, zoff));

                                req_positions.insert(pos + vec3(-xoff, yoff, zoff));
                                req_positions.insert(pos + vec3(xoff, -yoff, zoff));
                                req_positions.insert(pos + vec3(xoff, yoff, -zoff));

                                req_positions.insert(pos + vec3(-xoff, -yoff, zoff));
                                req_positions.insert(pos + vec3(xoff, -yoff, -zoff));
                                req_positions.insert(pos + vec3(-xoff, yoff, -zoff));

                                req_positions.insert(pos + vec3(-xoff, -yoff, -zoff));
                            }
                        }
                    }
                }
            }

            let mut pos_to_load: HashSet<ChunkPosition> = HashSet::new();
            req_positions
                .iter()
                .filter(|p| !self.loaded_chunks.contains_key(p) && !self.loading_queue.contains(p))
                .for_each(|p| {
                    pos_to_load.insert(*p);
                });

            let mut pos_to_unload: HashSet<ChunkPosition> = HashSet::new();
            self.loaded_chunks
                .keys()
                .filter(|p| !req_positions.contains(p))
                .for_each(|p| {
                    pos_to_unload.insert(*p);
                });

            let worldgen = self.worldgen.as_mut().unwrap();
            for p in pos_to_load {
                let chunk = Arc::new(RwLock::new(VoxelChunk::new()));
                let cref = VoxelChunkRef {
                    chunk: Arc::downgrade(&chunk),
                    position: p,
                };
                worldgen.generate_chunk(cref, &self.registry);
                chunk.write().unwrap().dirty += 1;
                self.loaded_chunks.insert(p, chunk);
            }
            for p in pos_to_unload {
                self.loaded_chunks.remove(&p);
            }
        }
    }
}
