use crate::client::render::voxrender::vox::{ChunkBuffers, VoxelVertex};
use crate::world::registry::VoxelRegistry;
use crate::world::{Direction, VoxelChunk, VOXEL_CHUNK_DIM};
use cgmath::{vec3, Vector3};

pub fn mesh_from_chunk(chunk: &VoxelChunk, registry: &VoxelRegistry) -> Option<ChunkBuffers> {
    let mut vbuf: Vec<VoxelVertex> = Vec::new();
    let mut ibuf: Vec<u32> = Vec::new();
    let neighbors: Vec<_> = chunk.neighbor.iter().map(|n| n.upgrade()).collect();
    if !neighbors.iter().all(|n| n.is_some()) {
        return None;
    }
    let neighbors_locks: Vec<_> = neighbors.into_iter().map(|o| o.unwrap()).collect();
    let neighbors: Vec<_> = neighbors_locks.iter().map(|l| l.read().unwrap()).collect();

    struct CubeSide {
        // counter-clockwise coords of the face
        pub verts: [f32; 3 * 4],
        // what to add to position to find neighbor
        pub ioffset: Vector3<i32>,
        // texture coordinates (u,v) matched with vertex coordinates
        pub texcs: [f32; 2 * 4],
    }

    const SIDES: [CubeSide; 6] = [
        // x+ -> "right"
        CubeSide {
            verts: [
                0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5,
            ],
            ioffset: vec3(1, 0, 0),
            texcs: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        // x- -> "left"
        CubeSide {
            verts: [
                -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5,
            ],
            ioffset: vec3(-1, 0, 0),
            texcs: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        // y+ -> "bottom"
        CubeSide {
            verts: [
                -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5,
            ],
            ioffset: vec3(0, 1, 0),
            texcs: [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        },
        // y- -> "top"
        CubeSide {
            verts: [
                0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5,
            ],
            ioffset: vec3(0, -1, 0),
            texcs: [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        },
        // z+ -> "back"
        CubeSide {
            verts: [
                0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5,
            ],
            ioffset: vec3(0, 0, 1),
            texcs: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
        // z- -> "front"
        CubeSide {
            verts: [
                -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5,
            ],
            ioffset: vec3(0, 0, -1),
            texcs: [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
    ];

    for (vidx, vox) in chunk.data.iter().enumerate() {
        let vdef = registry.get_definition_from_id(&vox);

        let ipos = vec3(
            (vidx % VOXEL_CHUNK_DIM) as i32,
            ((vidx / VOXEL_CHUNK_DIM) % VOXEL_CHUNK_DIM) as i32,
            ((vidx / VOXEL_CHUNK_DIM / VOXEL_CHUNK_DIM) % VOXEL_CHUNK_DIM) as i32,
        );
        let x = ipos.x as f32;
        let y = ipos.y as f32;
        let z = ipos.z as f32;

        if !vdef.has_mesh {
            continue;
        }

        for side in &SIDES {
            let touchpos = ipos + side.ioffset;
            let tdef;
            if ![touchpos.x, touchpos.y, touchpos.z]
                .iter()
                .any(|c| (*c) < 0 || (*c) >= VOXEL_CHUNK_DIM as i32)
            {
                let a = VOXEL_CHUNK_DIM as i32;
                let tidx = (touchpos[0] + touchpos[1] * a + touchpos[2] * a * a) as usize;
                tdef = registry.get_definition_from_id(&chunk.data[tidx]);
                if tdef.has_mesh {
                    continue;
                }
            } else {
                let tdir = Direction::try_from_vec(side.ioffset).unwrap();
                let tchunk = &neighbors[tdir as usize];
                let touchpos = touchpos - 32 * side.ioffset;
                let a = VOXEL_CHUNK_DIM as i32;
                let tidx = (touchpos[0] + touchpos[1] * a + touchpos[2] * a * a) as usize;
                tdef = registry.get_definition_from_id(&tchunk.data[tidx]);
                if tdef.has_mesh {
                    continue;
                }
            }
            let voff = vbuf.len() as u32;
            for t in 0..4 {
                let position: [f32; 4] = [
                    x + side.verts[t * 3],
                    y + side.verts[t * 3 + 1],
                    z + side.verts[t * 3 + 2],
                    1.0,
                ];
                let texid;
                use crate::world::TextureMapping;
                match &vdef.texture_mapping {
                    TextureMapping::TiledSingle(t) => {
                        texid = *t;
                    }
                    TextureMapping::TiledTSB {
                        top,
                        side: tside,
                        bottom,
                    } => {
                        if side.ioffset.y == 1 {
                            texid = *top;
                        } else if side.ioffset.y == -1 {
                            texid = *bottom;
                        } else {
                            texid = *tside;
                        }
                    }
                }
                vbuf.push(VoxelVertex {
                    position,
                    color: [
                        vdef.debug_color[0],
                        vdef.debug_color[1],
                        vdef.debug_color[2],
                        1.0,
                    ],
                    texcoord: [side.texcs[t * 2], side.texcs[t * 2 + 1], texid as f32],
                });
            }
            ibuf.extend([voff, voff + 1, voff + 2, voff + 2, voff + 3, voff].iter());
        }
    }

    Some(ChunkBuffers {
        vertices: vbuf,
        indices: ibuf,
    })
}
