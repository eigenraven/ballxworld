#![allow(unused_variables)]

use crate::client::input::InputManager;
use crate::client::render::resources::RenderingResources;
use crate::client::render::ui;
use crate::client::render::ui::z::*;
use crate::client::render::ui::{
    GuiCmd, GuiControlStyle, GuiCoord, GuiOrderedCmd, GuiRect, GuiRenderer, GuiVec2, GUI_WHITE,
};
use crate::client::render::{RenderingContext, VoxelRenderer};
use crate::client::world::{CameraSettings, ClientWorld};
use crate::config::Config;
use bxw_util::debug_data::DEBUG_DATA;
use bxw_util::math::*;
use bxw_util::*;
use bxw_world::blocks::register_standard_blocks;
use bxw_world::ecs::*;
use bxw_world::entities::player::PLAYER_EYE_HEIGHT;
use bxw_world::BlockPosition;
use std::cell::RefCell;
use std::f64::consts::PI;
use std::rc::Rc;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::client::render::egui_ash_sdl::EguiIntegration;
use crate::client::render::voxrender::MeshDataHandler;
use crate::client::screens::player_inventory::UiPlayerInventory;
use crate::client::screens::UiScreen;
use crate::network::client::{Client, ClientConfig, ClientControlMessage, ServerDetails};
use bxw_util::change::Change;
use bxw_util::collider::AABB;
use bxw_util::direction::OctahedralOrientation;
use bxw_world::blocks::stdshapes::StdMeta;
use bxw_world::physics::SMALL_V_CUTOFF;
use bxw_world::physics::TIMESTEP as PHYSICS_FRAME_TIME;
use bxw_world::storage::WorldSave;

#[derive(Debug, Clone, Default)]
struct InputState {
    /// X (-Left,Right+) Y (-Down,Up+)
    walk: (f32, f32),
    look: (f32, f32),
    capture_mouse: bool,
    do_normal: bool,
}

pub fn client_main() {
    let sdl_ctx = sdl2::init().unwrap();
    let sdl_vid = sdl_ctx.video().unwrap();

    let cfg = Config::standard_load();
    let use_netclient = std::env::args().any(|a| a == "-netclient");
    if std::env::args().any(|a| a == "-renderdoc") {
        let mut cfg = cfg.write();
        cfg.debugging.renderdoc = true;
        cfg.debugging.vk_debug_layers = false;
        log::warn!("Adjusting settings for renderdoc");
    }

    let task_pool = bxw_util::taskpool::TaskPool::new(cfg.read().performance.threads as usize);
    let mut rctx = Box::new(RenderingContext::new(&sdl_vid, &cfg.read()));
    let rres = Arc::new(RenderingResources::load(&cfg.read(), &mut rctx));
    let vctx = Rc::new(RefCell::new(VoxelRenderer::new(
        &cfg.read(),
        &mut rctx,
        rres.clone(),
    )));
    let mut guictx = Box::new(GuiRenderer::new(&cfg.read(), &mut rctx, rres.clone()));
    let mut egui = Box::new(EguiIntegration::new(&mut rctx));

    let mut vxreg: Box<bxw_world::voxregistry::VoxelRegistry> = Box::default();
    {
        let vctx = vctx.borrow();
        register_standard_blocks(&mut vxreg, &|nm| vctx.get_texture_id(nm));
    }
    let vxreg: Arc<bxw_world::voxregistry::VoxelRegistry> = Arc::from(vxreg);
    let savefile = {
        let name = "clientworld";
        if let Some(ws) = WorldSave::list_existing()
            .expect("Couldn't list world savefiles")
            .into_iter()
            .find(|ws| ws.name() == name)
        {
            ws
        } else {
            WorldSave::new(name).expect("Couldn't create a new world savefile")
        }
    };
    let (mut world, mut client_world) = ClientWorld::new_local_world(vxreg.clone(), &savefile)
        .expect("Couldn't create a new world");
    {
        let lp = client_world.local_player;
        let ents = world.ecs();
        let anchor: &CLoadAnchor = ents.get_component(lp).unwrap();
        let mut new_anchor = anchor.clone();
        new_anchor.radius = cfg.read().performance.draw_distance;
        let change = [EntityChange {
            kind: EntityChangeKind::UpdateEntity(anchor.entity_id()),
            load_anchor: Change::Update {
                old: anchor.clone(),
                new: new_anchor,
            },
            ..Default::default()
        }];
        world.apply_entity_changes(&change);
    }
    world.replace_handler(
        bxw_world::worldmgr::CHUNK_MESH_DATA,
        Box::new(MeshDataHandler::new(vctx.clone(), &rctx.handles)),
    );

    let mut previous_frame_time = Instant::now();
    let mut physics_accum_time = 0.0f64;
    //

    let mut input_mgr = InputManager::new(&sdl_ctx);
    input_mgr.input_state.capture_input_requested = true;

    let mut look_pos = BlockPosition::default();
    let mut look_precise_pos: Vector3<f64> = zero();
    let mut click_pos: Option<BlockPosition> = None;
    let mut click_datum: bxw_world::VoxelDatum = Default::default();
    let mut click_place = false;

    let mut event_pump = sdl_ctx.event_pump().unwrap();

    let i_placeable = [
        vxreg.get_definition_from_name("core:grass").unwrap(),
        vxreg.get_definition_from_name("core:snow_grass").unwrap(),
        vxreg.get_definition_from_name("core:dirt").unwrap(),
        vxreg.get_definition_from_name("core:stone").unwrap(),
        vxreg.get_definition_from_name("core:diamond_ore").unwrap(),
        vxreg.get_definition_from_name("core:debug").unwrap(),
        vxreg.get_definition_from_name("core:table").unwrap(),
    ];
    let mut i_place = 4;
    let mut i_orientation = 0;
    let i_destroy = vxreg.get_definition_from_name("core:void").unwrap();

    let netclient = if use_netclient {
        let addr = std::net::SocketAddr::V4("127.0.0.1:20138".parse().unwrap());
        let ccfg = ClientConfig {
            id_keys: sodiumoxide::crypto::box_::gen_keypair(),
        };
        let server = ServerDetails {
            name: "Server".to_owned(),
            address: addr,
        };
        Some(
            Client::new(cfg.clone(), Arc::new(ccfg), Arc::new(server))
                .expect("Couldn't create netclient"),
        )
    } else {
        None
    };

    'running: loop {
        let current_frame_time = Instant::now();

        // avoid any nasty divisions by 0
        let frame_delta_time = 1.0e-6f64.max(
            current_frame_time
                .saturating_duration_since(previous_frame_time)
                .as_secs_f64(),
        );
        DEBUG_DATA.frame_times.push_sec(frame_delta_time);
        DEBUG_DATA
            .fps
            .store((1.0 / frame_delta_time) as u32, Ordering::Release);

        physics_accum_time += frame_delta_time;
        previous_frame_time = current_frame_time;
        let physics_frames = (physics_accum_time / PHYSICS_FRAME_TIME) as i32;
        {
            let (dyaw, dpitch) = (
                input_mgr.input_state.look.x as f64,
                input_mgr.input_state.look.y as f64,
            );
            input_mgr.input_state.look = zero();
            let CameraSettings::FPS { pitch, yaw } = &mut client_world.camera_settings;
            *pitch -= dpitch / 60.0;
            *pitch = f64::min(f64::max(*pitch, -PI / 4.0 + 0.01), PI / 4.0 - 0.01);
            *yaw -= dyaw / 60.0;
            *yaw %= 2.0 * PI;
            let (pitch, yaw) = (*pitch, *yaw);
        }
        if physics_frames > 0 {
            let _p_physics_zone = bxw_util::tracy_client::span!("Physics frames handling", 4);
            physics_accum_time -= f64::from(physics_frames) * PHYSICS_FRAME_TIME;
            let physics_frames = if physics_frames > 10 {
                log::warn!(
                    "Physics lagging behind, skipping {} ticks",
                    physics_frames - 1
                );
                1
            } else {
                physics_frames
            };
            let local_player = client_world.local_player;

            if let Some(bpos) = click_pos {
                click_pos = None;
                // place block
                let i_used = if click_place {
                    i_placeable[i_place]
                } else {
                    i_destroy
                };
                let shape = if input_mgr
                    .pressed_keys
                    .contains(&sdl2::keyboard::Keycode::Num1)
                {
                    1
                } else if input_mgr
                    .pressed_keys
                    .contains(&sdl2::keyboard::Keycode::Num2)
                {
                    2
                } else if input_mgr
                    .pressed_keys
                    .contains(&sdl2::keyboard::Keycode::Num3)
                {
                    3
                } else {
                    0
                };
                let meta = StdMeta::from_parts(shape, i_orientation).unwrap().to_meta();
                let change = bxw_world::worldmgr::VoxelChange {
                    bpos,
                    from: click_datum,
                    to: bxw_world::VoxelDatum::new(i_used.id(), meta),
                };
                world.apply_voxel_changes(&[change]);
            }

            for _pfrm in 0..physics_frames {
                let _p_physics_zone = bxw_util::tracy_client::span!("Physics tick", 4);
                // do physics tick
                let entities = world.ecs();
                let lp_loc: &CLocation = entities.get_component(local_player).unwrap();
                let mut new_loc: CLocation = lp_loc.clone();
                // position
                let &CameraSettings::FPS { pitch, yaw } = &client_world.camera_settings;
                let qyaw = Quaternion::from_polar_decomposition(1.0, yaw, Vector3::y_axis());
                let qpitch = Quaternion::from_polar_decomposition(1.0, pitch, Vector3::x_axis());
                new_loc.orientation = UnitQuaternion::new_normalize(qpitch * qyaw);

                let mview = glm::quat_to_mat3(&new_loc.orientation).transpose();

                let mut wvel = Vector3::new(
                    input_mgr.input_state.walk.x as f64,
                    0.0,
                    input_mgr.input_state.walk.y as f64,
                );
                wvel *= 6.0; // Walk 6m/s
                if input_mgr.input_state.noclip {
                    wvel *= 10.0;
                }
                if input_mgr.input_state.sprint.is_active() {
                    wvel *= 3.0;
                }
                let mut tvel = mview * wvel;
                let tspeed = tvel.magnitude();

                if input_mgr.input_state.noclip {
                    new_loc.position += tvel * PHYSICS_FRAME_TIME;
                    new_loc.velocity = tvel;
                } else {
                    tvel.y = 0.0;
                }
                let lp_phys: &CPhysics = entities.get_component(local_player).unwrap();
                let mut new_phys: CPhysics = lp_phys.clone();
                new_phys.control_target_velocity = if tspeed < SMALL_V_CUTOFF {
                    zero()
                } else {
                    tvel.normalize() * tspeed
                };
                new_phys.frozen = input_mgr.input_state.noclip;
                if input_mgr.input_state.jump.is_active() && new_phys.against_wall[2] {
                    new_phys.control_frame_impulse.y = 300.0;
                }
                let change = [EntityChange {
                    kind: EntityChangeKind::UpdateEntity(lp_loc.entity_id()),
                    location: Change::Update {
                        old: lp_loc.clone(),
                        new: new_loc,
                    },
                    physics: Change::Update {
                        old: lp_phys.clone(),
                        new: new_phys,
                    },
                    ..Default::default()
                }];
                world.apply_entity_changes(&change);
                bxw_world::physics::world_physics_tick(&mut world);
            }
        }

        {
            let _p_zone = bxw_util::tracy_client::span!("Main world loop tick", 4);
            world.main_loop_tick(&task_pool);
        }
        {
            let _p_zone = bxw_util::tracy_client::span!("Task pool loop tick", 4);
            task_pool.main_thread_tick();
        }

        let _p_span_prepass = bxw_util::tracy_client::span!("Render prepass", 4);
        if let Some(mut fc) = rctx.frame_begin_prepass(&cfg.read(), &client_world, frame_delta_time)
        {
            let mut vctx = vctx.borrow_mut();
            fc.begin_region([0.7, 0.7, 0.1, 1.0], || "vctx.prepass_draw");
            vctx.prepass_draw(&mut fc, &world);
            fc.end_region();
            fc.begin_region([0.5, 0.5, 0.5, 1.0], || "gui.prepass_draw");
            egui.prepass_draw(fc.cmd, &mut fc, |ctx| {
                egui::Window::new("Debug info")
                    .resizable(true)
                    .vscroll(true)
                    .show(ctx, |ui| {
                        ui.label(DEBUG_DATA.hud_format());
                        let pl_x = DEBUG_DATA.local_player_x.load(Ordering::Relaxed) as f64 / 10.0;
                        let pl_y = DEBUG_DATA.local_player_y.load(Ordering::Relaxed) as f64 / 10.0;
                        let pl_z = DEBUG_DATA.local_player_z.load(Ordering::Relaxed) as f64 / 10.0;
                        let pl_chunk = bxw_world::ChunkPosition::from(
                            bxw_world::BlockPosition::new(pl_x as i32, pl_y as i32, pl_z as i32),
                        );
                        let pl_cidx = world.get_chunk_index(pl_chunk);
                        ui.horizontal(|ui| {
                            ui.label("Player position:");
                            ui.label(format!("X: {:.1}", pl_x));
                            ui.label(format!("Y: {:.1}", pl_y));
                            ui.label(format!("Z: {:.1}", pl_z));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Player chunk position:");
                            ui.label(format!("X: {}", pl_chunk.0.x));
                            ui.label(format!("Y: {}", pl_chunk.0.y));
                            ui.label(format!("Z: {}", pl_chunk.0.z));
                            ui.label(format!(
                                "Index: {}",
                                pl_cidx.map(|x| x as i64).unwrap_or(-1)
                            ));
                        });
                        ui.horizontal(|ui| {
                            ui.label("Looking at:");
                            ui.label(format!("X: {} ({:.2})", look_pos.0.x, look_precise_pos.x));
                            ui.label(format!("Y: {} ({:.2})", look_pos.0.y, look_precise_pos.y));
                            ui.label(format!("Z: {} ({:.2})", look_pos.0.z, look_precise_pos.z));
                        });
                        ui.label(format!(
                            "Place orientation: {} {:?}",
                            i_orientation,
                            OctahedralOrientation::from_index(i_orientation as usize).unwrap()
                        ));
                        if let Some(pl_cidx) = pl_cidx {
                            let handler_statuses = [
                                bxw_world::worldmgr::CHUNK_BLOCK_DATA,
                                bxw_world::worldmgr::CHUNK_MESH_DATA,
                            ]
                            .into_iter()
                            .map(|hid| world.get_handler(hid).borrow().status_array()[pl_cidx]);
                            ui.label("Chunk data states: ");
                            ui.horizontal(|ui| {
                                for cds in handler_statuses {
                                    ui.label(format!("{:?}, ", cds));
                                }
                            });
                        }
                        ui.allocate_space(ui.available_size());
                    });
            });
            let gui = guictx.prepass_draw(&mut fc);
            gui.push_cmd(GuiOrderedCmd {
                z_index: GUI_Z_LAYER_BACKGROUND + GUI_Z_OFFSET_CONTROL,
                color: GUI_WHITE,
                cmd: GuiCmd::Rectangle {
                    style: GuiControlStyle::Crosshair,
                    rect: GuiControlStyle::Crosshair
                        .gui_rect_centered(GuiVec2(GuiCoord(0.5, 0.0), GuiCoord(0.5, 0.0))),
                },
            });
            if input_mgr.capturing_input() {
                let bsize = ui::theme::SLOT_SIZE;
                let bgap = ui::theme::SLOT_GAP;
                let mut x = -(i_placeable.len() as f32) / 2.0 * (bsize + bgap);
                let y = -bsize - bgap - 8.0;
                for (i, &def) in i_placeable.iter().enumerate() {
                    gui.push_cmd(GuiOrderedCmd {
                        z_index: GUI_Z_LAYER_HUD + GUI_Z_OFFSET_CONTROL,
                        color: GUI_WHITE,
                        cmd: GuiCmd::VoxelPreview {
                            texture: def.texture_mapping,
                            rect: GuiRect::from_xywh(
                                (0.5, x + bgap / 2.0),
                                (1.0, y),
                                (0.0, bsize),
                                (0.0, bsize),
                            ),
                        },
                    });
                    gui.push_cmd(GuiOrderedCmd {
                        z_index: GUI_Z_LAYER_HUD + GUI_Z_OFFSET_BG,
                        color: GUI_WHITE,
                        cmd: GuiCmd::Rectangle {
                            style: if i == i_place {
                                GuiControlStyle::Window
                            } else {
                                GuiControlStyle::Button
                            },
                            rect: GuiRect::from_xywh(
                                (0.5, x),
                                (1.0, y - bgap / 2.0),
                                (0.0, bsize + bgap),
                                (0.0, bsize + bgap),
                            ),
                        },
                    });
                    x += bsize + bgap;
                }
            } else if false {
                let mut iscreen = UiPlayerInventory {};
                iscreen.draw(gui, Some((&world, &client_world)));
            }
            fc.end_region();
            drop(_p_span_prepass);
            let _p_span_inpass = bxw_util::tracy_client::span!("Render inpass", 4);
            let mut fc = RenderingContext::frame_goto_pass(fc);
            fc.begin_region([0.3, 0.3, 0.8, 1.0], || "vctx.inpass_draw");
            vctx.inpass_draw(&mut fc, &world);
            fc.end_region();

            fc.begin_region([0.5, 0.5, 0.5, 1.0], || "gui.inpass_draw");
            guictx.inpass_draw(&mut fc);
            egui.inpass_draw(fc.cmd, &mut fc);
            fc.end_region();

            drop(_p_span_inpass);
            let _p_span_postpass = bxw_util::tracy_client::span!("Render postpass", 4);
            let fc = RenderingContext::frame_goto_postpass(fc);
            fc.insert_label([0.1, 0.8, 0.1, 1.0], || "frame_finish");
            drop(_p_span_postpass);
            let _p_frame_finish = bxw_util::tracy_client::span!("Frame finish", 4);
            RenderingContext::frame_finish(fc);
            bxw_util::tracy_client::frame_mark();
        }

        let _p_span_input = bxw_util::tracy_client::span!("Input processing", 4);
        input_mgr.pre_process();
        for event in event_pump.poll_iter() {
            egui.handle_event(&event, &mut rctx);
            input_mgr.process(&mut rctx, event);
        }
        input_mgr.post_events_update(&mut rctx);
        i_place = ((-input_mgr.input_state.scroller % i_placeable.len() as i32
            + i_placeable.len() as i32) as usize)
            % i_placeable.len();

        if input_mgr.input_state.requesting_exit {
            input_mgr.input_state.requesting_exit = false;
            break 'running;
        }

        if input_mgr
            .just_pressed_keys
            .contains(&sdl2::keyboard::Keycode::Q)
        {
            i_orientation = (i_orientation + 1) % 24;
        } else if input_mgr
            .just_pressed_keys
            .contains(&sdl2::keyboard::Keycode::E)
        {
            i_orientation = (i_orientation + 23) % 24;
        }

        let player_pos;
        let player_ang;
        {
            let entities = world.ecs();
            let lp_loc: &CLocation = entities.get_component(client_world.local_player).unwrap();
            player_pos = lp_loc.position;
            player_ang = lp_loc.orientation;
            DEBUG_DATA
                .local_player_x
                .store((lp_loc.position.x * 10.0) as i64, Ordering::Release);
            DEBUG_DATA
                .local_player_y
                .store((lp_loc.position.y * 10.0) as i64, Ordering::Release);
            DEBUG_DATA
                .local_player_z
                .store((lp_loc.position.z * 10.0) as i64, Ordering::Release);

            let primary = input_mgr.input_state.primary_action.get_and_reset_pressed();
            let secondary = input_mgr
                .input_state
                .secondary_action
                .get_and_reset_pressed();
            use bxw_world::raycast;
            let mview = glm::quat_to_mat3(&player_ang).transpose();
            let fwd = mview * vec3(0.0, 0.0, 1.0);
            let rc = raycast::RaycastQuery::new_directed(
                player_pos + vec3(0.0, PLAYER_EYE_HEIGHT / 2.0, 0.0),
                fwd,
                32.0,
                &world,
                true,
                false,
            )
            .execute();
            if let raycast::Hit::Voxel { position, .. } = &rc.hit {
                look_pos = *position;
                look_precise_pos = rc.hit_point;
            }
            if primary | secondary {
                click_place = secondary;
                if let raycast::Hit::Voxel {
                    position,
                    datum,
                    normal,
                    normal_datum,
                } = &rc.hit
                {
                    if !click_place {
                        click_pos = Some(*position);
                        click_datum = *datum;
                    } else if normal_datum
                        .map(|d| vxreg.get_definition_from_datum(d).selection_shape.is_none())
                        .unwrap_or(false)
                    {
                        let place_pos = *position + BlockPosition(normal.to_vec());
                        let player_aabb = lp_loc
                            .bounding_shape
                            .aabb(lp_loc.position)
                            .inflate(-bxw_world::physics::TOUCH_EPSILON);
                        let voxel_aabb = i_placeable[i_place]
                            .collision_shape
                            .unwrap_or_default()
                            .translate(place_pos.0.map(|c| c as f64));
                        let intersecting = AABB::intersection(player_aabb, voxel_aabb).is_some();
                        if !intersecting {
                            click_pos = Some(place_pos);
                            click_datum = normal_datum.unwrap();
                        }
                    }
                }
            }
        }
        drop(_p_span_input);

        let _p_fps_keeper = bxw_util::tracy_client::span!("FPS Target Keeper", 4);
        if let Some(fps) = cfg.read().render.fps_lock {
            let end_current_frame_time = Instant::now() + Duration::from_micros(100);
            let target_ft = Duration::from_secs_f64(1.0 / f64::from(fps));
            let elapsed_ft = end_current_frame_time.saturating_duration_since(current_frame_time);
            if target_ft > elapsed_ft {
                std::thread::sleep(target_ft - elapsed_ft);
            }
        }
    }

    if let Some(nc) = netclient {
        nc.send_control_message(ClientControlMessage::Disconnect);
        log::info!("Waiting for netclient shutdown");
        nc.wait_for_shutdown();
        log::info!("Done netclient shutdown");
    }

    drop(world);
    drop(task_pool);
    let vctx = Rc::try_unwrap(vctx)
        .ok()
        .expect("Remaining references to VoxelRenderer")
        .into_inner();
    vctx.destroy(&rctx.handles);
    egui.destroy(&rctx.handles);
    guictx.destroy(&rctx.handles);
    Arc::try_unwrap(rres)
        .unwrap_or_else(|_| panic!("Handle still held to resource manager"))
        .destroy(&rctx.handles);
    rctx.destroy();
}
