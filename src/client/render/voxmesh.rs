use crate::client::render::voxrender::vox::{ChunkBuffers, VoxelVertex};
use crate::world::registry::VoxelRegistry;
use crate::world::{VoxelChunk, VOXEL_CHUNK_DIM};
use cgmath::{vec3, Vector3};

pub fn mesh_from_chunk(chunk: &VoxelChunk, registry: &VoxelRegistry) -> ChunkBuffers {
    let mut vbuf: Vec<VoxelVertex> = Vec::new();
    let mut ibuf: Vec<u32> = Vec::new();

    struct CubeSide {
        // counter-clockwise coords of the face
        pub verts: [f32; 3 * 4],
        pub ioffset: Vector3<i32>,
    }

    const SIDES: [CubeSide; 6] = [
        // x+ -> "right"
        CubeSide {
            verts: [
                0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5,
            ],
            ioffset: vec3(1, 0, 0),
        },
        // x- -> "left"
        CubeSide {
            verts: [
                -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -0.5, -0.5,
            ],
            ioffset: vec3(-1, 0, 0),
        },
        // y+ -> "bottom"
        CubeSide {
            verts: [
                -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5,
            ],
            ioffset: vec3(0, 1, 0),
        },
        // y- -> "top"
        CubeSide {
            verts: [
                0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5,
            ],
            ioffset: vec3(0, -1, 0),
        },
        // z+ -> "back"
        CubeSide {
            verts: [
                0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5,
            ],
            ioffset: vec3(0, 0, 1),
        },
        // z- -> "front"
        CubeSide {
            verts: [
                -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5,
            ],
            ioffset: vec3(0, 0, -1),
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
            let touchpos: [i32; 3] = (ipos + side.ioffset).into();
            if !touchpos
                .iter()
                .any(|c| (*c) < 0 || (*c) >= VOXEL_CHUNK_DIM as i32)
            {
                let a = VOXEL_CHUNK_DIM as i32;
                let tidx = (touchpos[0] + touchpos[1] * a + touchpos[2] * a * a) as usize;
                let tdef = registry.get_definition_from_id(&chunk.data[tidx]);
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
                let shade = (t + 10) as f32 / 13.0;
                vbuf.push(VoxelVertex {
                    position,
                    color: [
                        vdef.debug_color[0] * shade,
                        vdef.debug_color[1] * shade,
                        vdef.debug_color[2] * shade,
                        1.0,
                    ],
                });
            }
            ibuf.extend([voff, voff + 1, voff + 2, voff + 2, voff + 3, voff].iter());
        }
    }

    ChunkBuffers {
        vertices: vbuf,
        indices: ibuf,
    }
}
