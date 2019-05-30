use sdl2::event::{Event, WindowEvent};

use super::super::world;
use super::vulkan::RenderingContext;
use crate::client::voxmesh::mesh_from_chunk;
use rand::Rng;

const PHYSICS_FRAME_TIME: f64 = 1.0 / 60.0;

pub fn client_main() {
    let sdl_ctx = sdl2::init().unwrap();
    let sdl_vid = sdl_ctx.video().unwrap();
    let mut sdl_timer = sdl_ctx.timer().unwrap();
    let mut gfx = RenderingContext::new(&sdl_vid);

    let mut vxreg = world::VoxelRegistry::new();
    vxreg
        .build_definition()
        .name("core:green")
        .debug_color(0.1, 0.8, 0.1)
        .has_physical_properties()
        .finish()
        .unwrap();
    vxreg
        .build_definition()
        .name("core:stone")
        .debug_color(0.45, 0.4, 0.4)
        .has_physical_properties()
        .finish()
        .unwrap();
    let mut chunk = world::VoxelChunk::new();
    for ch in chunk.data.iter_mut() {
        ch.id = rand::thread_rng().gen_range(0, 3);
    }
    gfx.d_reset_buffers(mesh_from_chunk(&chunk, &vxreg));

    let pf_mult = 1.0 / sdl_timer.performance_frequency() as f64;
    let mut previous_frame_time = sdl_timer.performance_counter() as f64 * pf_mult;
    let mut physics_accum_time = 0.0f64;
    //
    let mut event_pump = sdl_ctx.event_pump().unwrap();
    'running: loop {
        let current_frame_time = sdl_timer.performance_counter() as f64 * pf_mult;
        // avoid any nasty divisions by 0
        let frame_delta_time = 1.0e-6f64.max(current_frame_time - previous_frame_time);
        physics_accum_time += frame_delta_time;
        previous_frame_time = current_frame_time;
        let physics_frames = (physics_accum_time / PHYSICS_FRAME_TIME) as i32;
        if physics_frames > 0 {
            physics_accum_time -= f64::from(physics_frames) * PHYSICS_FRAME_TIME;
            for _pfrm in 0..physics_frames {
                // do physics tick
            }
        }

        gfx.draw_next_frame(frame_delta_time);

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'running,
                Event::Window { win_event, .. } => match win_event {
                    WindowEvent::Resized(_w, _h) => {
                        gfx.outdated_swapchain = true;
                    }
                    WindowEvent::SizeChanged(_w, _h) => {
                        gfx.outdated_swapchain = true;
                    }
                    _ => {}
                },
                Event::KeyDown {keycode, ..} => {
                    match keycode {
                        Some(sdl2::keyboard::Keycode::A) => {
                            gfx.position.x += 10.0;
                        },
                        Some(sdl2::keyboard::Keycode::D) => {
                            gfx.position.x -= 10.0;
                        },
                        Some(sdl2::keyboard::Keycode::W) => {
                            gfx.position.z += 10.0;
                        },
                        Some(sdl2::keyboard::Keycode::S) => {
                            gfx.position.z -= 10.0;
                        },
                        Some(sdl2::keyboard::Keycode::Q) => {
                            gfx.position.y -= 10.0;
                        },
                        Some(sdl2::keyboard::Keycode::E) => {
                            gfx.position.y += 10.0;
                        },
                        _ => {}
                    }
                },
                _ => {}
            }
        }

        sdl_timer.delay(16);
    }
}
