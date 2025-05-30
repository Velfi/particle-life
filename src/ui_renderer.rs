use crate::app_settings::AppSettings;
use crate::physics::{ExtendedPhysics, PhysicsSnapshot};
use crate::rendering;
use crate::ui::{Clock, SelectionManager};

#[allow(clippy::too_many_arguments)]
pub fn render_ui(
    ctx: &egui::Context,
    world_texture_id: egui::TextureId,
    config_width: f32,
    config_height: f32,
    show_gui: bool,
    show_graphics_window: &mut bool,
    show_controls_window: &mut bool,
    show_about_window: &mut bool,
    tile_fade_strength: &mut f32,
    traces_user_enabled: &mut bool,
    camera_is_moving: bool,
    physics_loop_pause: &mut bool,
    physics_snapshot: &PhysicsSnapshot,
    render_clock: &Clock,
    _app_settings: &mut AppSettings,
    _palettes: &mut SelectionManager<Box<dyn rendering::Palette>>,
    _position_setters: &mut SelectionManager<Box<dyn crate::physics::PositionSetter>>,
    _type_setters: &mut SelectionManager<Box<dyn crate::physics::TypeSetter>>,
    _matrix_generators: &mut SelectionManager<Box<dyn crate::physics::MatrixGenerator>>,
    local_matrix: &mut Vec<Vec<f64>>,
    physics: &mut ExtendedPhysics,
    physics_time_avg: f64,
    trace_fade: &mut f32,
    cursor_size: &mut f64,
    cursor_strength: &mut f64,
) {
    // Display world texture as background
    ctx.layer_painter(egui::LayerId::background()).image(
        world_texture_id,
        egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(config_width, config_height),
        ),
        egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(1.0, 1.0)),
        egui::Color32::WHITE,
    );

    if !show_gui {
        return;
    }

    render_menu_bar(
        ctx,
        show_controls_window,
        show_about_window,
        show_graphics_window,
    );
    render_physics_panel(
        ctx,
        physics_loop_pause,
        render_clock,
        physics_time_avg,
        physics_snapshot,
        local_matrix,
        physics,
        _matrix_generators,
        _position_setters,
        _type_setters,
        _app_settings,
        _palettes,
        traces_user_enabled,
        camera_is_moving,
        cursor_size,
        cursor_strength,
    );
    render_graphics_window(
        ctx,
        show_graphics_window,
        render_clock,
        traces_user_enabled,
        camera_is_moving,
        tile_fade_strength,
        trace_fade,
        _app_settings,
        _palettes,
    );
    render_controls_window(ctx, show_controls_window);
    render_about_window(ctx, show_about_window);
}

pub fn render_menu_bar(
    ctx: &egui::Context,
    show_controls_window: &mut bool,
    show_about_window: &mut bool,
    show_graphics_window: &mut bool,
) {
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::menu::bar(ui, |ui| {
            ui.menu_button("Menu", |ui| {
                if ui.button("Controls").clicked() {
                    *show_controls_window = true;
                }
                if ui.button("About").clicked() {
                    *show_about_window = true;
                }
            });
            ui.menu_button("View", |ui| {
                if ui.button("Graphics Settings").clicked() {
                    *show_graphics_window = true;
                }
                if ui.button("Hide GUI [ESC]").clicked() {
                    // Can't modify show_gui from here
                }
            });
        });
    });
}

#[allow(clippy::too_many_arguments)]
pub fn render_physics_panel(
    ctx: &egui::Context,
    physics_loop_pause: &mut bool,
    render_clock: &Clock,
    physics_time_avg: f64,
    physics_snapshot: &PhysicsSnapshot,
    local_matrix: &mut Vec<Vec<f64>>,
    physics: &mut ExtendedPhysics,
    _matrix_generators: &mut SelectionManager<Box<dyn crate::physics::MatrixGenerator>>,
    _position_setters: &mut SelectionManager<Box<dyn crate::physics::PositionSetter>>,
    _type_setters: &mut SelectionManager<Box<dyn crate::physics::TypeSetter>>,
    _app_settings: &mut AppSettings,
    _palettes: &mut SelectionManager<Box<dyn rendering::Palette>>,
    traces_user_enabled: &mut bool,
    camera_is_moving: bool,
    cursor_size: &mut f64,
    cursor_strength: &mut f64,
) {
    egui::SidePanel::left("physics_panel")
        .min_width(320.0)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.heading("Particle Life Simulator");

                // Simulation status
                ui.horizontal(|ui| {
                    if *physics_loop_pause {
                        ui.colored_label(egui::Color32::RED, "⏸ PAUSED");
                    } else {
                        ui.colored_label(egui::Color32::GREEN, "▶ RUNNING");
                    }
                    ui.separator();
                    ui.label(format!(
                        "Render: {:.0} FPS",
                        render_clock.get_avg_framerate()
                    ));
                    ui.separator();
                    ui.label(format!("Physics: {:.2}ms", physics_time_avg));
                });

                // Play/Pause
                if ui
                    .button(if *physics_loop_pause { "Play" } else { "Pause" })
                    .clicked()
                {
                    *physics_loop_pause = !*physics_loop_pause;
                }

                ui.separator();

                // Physics parameters
                ui.label("Physics Parameters");
                ui.label(format!("Particles: {}", physics_snapshot.particle_count));
                ui.label(format!(
                    "Matrix Size: {}x{}",
                    physics_snapshot.type_count.len(),
                    physics_snapshot.type_count.len()
                ));

                ui.separator();

                render_matrix_editor(
                    ui,
                    physics_snapshot,
                    local_matrix,
                    physics,
                    _matrix_generators,
                );

                ui.separator();

                render_particle_setup(ui, physics, _position_setters, _type_setters);

                ui.separator();

                render_rendering_settings(
                    ui,
                    _app_settings,
                    _palettes,
                    traces_user_enabled,
                    camera_is_moving,
                );

                ui.separator();

                render_mouse_interaction(ui, cursor_size, cursor_strength);

                ui.separator();

                render_physics_settings(ui, physics, physics_snapshot);

                ui.separator();

                render_type_distribution(ui, physics_snapshot, physics, _palettes);

                ui.separator();

                render_quick_controls(ui);
            });
        });
}

pub fn render_matrix_editor(
    ui: &mut egui::Ui,
    physics_snapshot: &PhysicsSnapshot,
    local_matrix: &mut Vec<Vec<f64>>,
    physics: &mut ExtendedPhysics,
    _matrix_generators: &mut SelectionManager<Box<dyn crate::physics::MatrixGenerator>>,
) {
    ui.heading("Interaction Matrix");

    // Ensure local matrix matches current size
    let matrix_size = physics_snapshot.type_count.len();
    if local_matrix.len() != matrix_size {
        local_matrix.resize(matrix_size, vec![0.0; matrix_size]);
        for row in local_matrix.iter_mut() {
            row.resize(matrix_size, 0.0);
        }
        local_matrix.iter_mut().enumerate().for_each(|(i, row)| {
            row.iter_mut().enumerate().for_each(|(j, val)| {
                *val = physics.matrix.get(i, j);
            });
        });
    }

    if matrix_size > 0 {
        // Matrix editing grid with color styling
        let spacing = 1.0;
        let cell_size = 50.0;
        let matrix_width = matrix_size as f32 * (cell_size + spacing) - spacing;
        
        ui.allocate_ui_with_layout(
            egui::Vec2::new(matrix_width, matrix_width + 40.0),
            egui::Layout::top_down(egui::Align::Center),
            |ui| {
                ui.label("Click and drag to edit values");
                ui.small("Green = Attraction, Red = Repulsion");
                
                // Display a grid representing the matrix
                (0..matrix_size).for_each(|i| {
                    ui.horizontal(|ui| {
                        (0..matrix_size).for_each(|j| {
                            let value = local_matrix[i][j];
                            
                            // Color based on the interaction strength
                            let color = if value > 0.0 {
                                // Positive (attraction) - green
                                let intensity = (value.abs().min(1.0) * 255.0) as u8;
                                egui::Color32::from_rgb(0, intensity, 0)
                            } else {
                                // Negative (repulsion) - red
                                let intensity = (value.abs().min(1.0) * 255.0) as u8;
                                egui::Color32::from_rgb(intensity, 0, 0)
                            };
                            
                            // Create a frame with the color background
                            let frame = egui::Frame::none()
                                .fill(color)
                                .inner_margin(2.0);
                            
                            frame.show(ui, |ui| {
                                ui.allocate_ui_with_layout(
                                    egui::Vec2::new(cell_size - 4.0, cell_size - 4.0),
                                    egui::Layout::centered_and_justified(egui::Direction::TopDown),
                                    |ui| {
                                        let mut temp_value = local_matrix[i][j] as f32;
                                        let response = ui.add(
                                            egui::DragValue::new(&mut temp_value)
                                                .clamp_range(-1.0..=1.0)
                                                .speed(0.01)
                                                .custom_formatter(|n, _| {
                                                    format!("{:.2}", n)
                                                })
                                                .custom_parser(|s| {
                                                    s.parse::<f64>().ok()
                                                })
                                        );
                                        
                                        if response.changed() {
                                            local_matrix[i][j] = temp_value as f64;
                                            physics.matrix.set(i, j, local_matrix[i][j]);
                                        }
                                        
                                        response.on_hover_text(format!("Type {} -> Type {}: {:.3}\nDrag to edit, or double-click to type", i, j, temp_value));
                                    }
                                );
                            });
                        });
                    });
                });
            }
        );
    }

    // Matrix controls
    ui.horizontal(|ui| {
        ui.label("Matrix Size:");
        let mut matrix_size_input = matrix_size as i32;
        if ui
            .add(
                egui::DragValue::new(&mut matrix_size_input)
                    .clamp_range(2..=8)
                    .speed(1),
            )
            .changed()
        {
            let new_size = matrix_size_input.max(2) as usize;
            physics.set_matrix_size(new_size);
        }
    });

    ui.horizontal(|ui| {
        if ui.button("Generate Matrix").clicked() {
            physics.generate_matrix_with_generator(_matrix_generators.get_active().as_ref());
            local_matrix.iter_mut().enumerate().for_each(|(i, row)| {
                row.iter_mut().enumerate().for_each(|(j, val)| {
                    *val = physics.matrix.get(i, j);
                });
            });
        }
        if ui.button("Zero Matrix").clicked() {
            local_matrix.iter_mut().enumerate().for_each(|(i, row)| {
                row.iter_mut().enumerate().for_each(|(j, val)| {
                    *val = 0.0;
                    physics.matrix.set(i, j, 0.0);
                });
            });
        }
    });
}

pub fn render_particle_setup(
    ui: &mut egui::Ui,
    physics: &mut ExtendedPhysics,
    _position_setters: &mut SelectionManager<Box<dyn crate::physics::PositionSetter>>,
    _type_setters: &mut SelectionManager<Box<dyn crate::physics::TypeSetter>>,
) {
    ui.heading("Particle Setup");

    ui.horizontal(|ui| {
        if ui.button("Reset Positions").clicked() {
            physics.set_positions_with_setter(_position_setters.get_active().as_ref());
        }
        if ui.button("Reset Types").clicked() {
            physics.set_types_with_setter(_type_setters.get_active().as_ref());
        }
    });
}

pub fn render_rendering_settings(
    ui: &mut egui::Ui,
    _app_settings: &mut AppSettings,
    _palettes: &mut SelectionManager<Box<dyn rendering::Palette>>,
    traces_user_enabled: &mut bool,
    camera_is_moving: bool,
) {
    ui.heading("Rendering Settings");

    ui.horizontal(|ui| {
        ui.label("Particle Size:");
        ui.add(egui::Slider::new(&mut _app_settings.particle_size, 0.1..=1.0).step_by(0.01));
    });

    ui.horizontal(|ui| {
        ui.label("Particle Traces:");
        if ui.checkbox(traces_user_enabled, "Enable [T]").changed() {
            // Trace state will be updated after UI
        }
        if camera_is_moving && *traces_user_enabled {
            ui.label("(disabled during panning)");
        }
    });
}

pub fn render_mouse_interaction(
    ui: &mut egui::Ui,
    cursor_size: &mut f64,
    cursor_strength: &mut f64,
) {
    ui.heading("Mouse Interaction");

    ui.horizontal(|ui| {
        ui.label("Cursor Size:");
        ui.add(egui::Slider::new(cursor_size, 0.05..=1.0).step_by(0.05));
    });

    ui.horizontal(|ui| {
        ui.label("Cursor Strength:");
        ui.add(egui::Slider::new(cursor_strength, 0.0..=20.0).step_by(0.5));
    });

    ui.label("Left Click: Repel particles");
    ui.label("Right Click: Attract particles");
}

pub fn render_physics_settings(
    ui: &mut egui::Ui,
    physics: &mut ExtendedPhysics,
    physics_snapshot: &PhysicsSnapshot,
) {
    ui.heading("Physics Settings");

    // Particle count
    let mut particle_count_input = physics_snapshot.particle_count as i32;
    ui.horizontal(|ui| {
        ui.label("Particles:");
        if ui
            .add(
                egui::DragValue::new(&mut particle_count_input)
                    .clamp_range(100..=200000)
                    .speed(10),
            )
            .changed()
        {
            let new_count = particle_count_input.max(100) as usize;
            physics.set_particle_count(new_count);
        }
    });

    // Force multiplier
    ui.horizontal(|ui| {
        ui.label("Force:");
        let mut force = physics.settings.force as f32;
        if ui
            .add(egui::Slider::new(&mut force, 0.0..=5.0).step_by(0.1))
            .changed()
        {
            physics.settings.force = force as f64;
        }
    });

    // Friction
    ui.horizontal(|ui| {
        ui.label("Friction:");
        let mut friction = physics.settings.friction as f32;
        if ui
            .add(egui::Slider::new(&mut friction, 0.0..=1.0).step_by(0.01))
            .changed()
        {
            physics.settings.friction = friction as f64;
        }
    });

    // Boundaries
    ui.horizontal(|ui| {
        ui.label("Boundaries:");
        let mut wrap = physics.settings.wrap;
        if ui.checkbox(&mut wrap, "Wrap around").changed() {
            physics.settings.wrap = wrap;
        }
    });
}

pub fn render_type_distribution(
    ui: &mut egui::Ui,
    physics_snapshot: &PhysicsSnapshot,
    physics: &mut ExtendedPhysics,
    _palettes: &mut SelectionManager<Box<dyn rendering::Palette>>,
) {
    ui.heading("Type Distribution");

    let matrix_size = physics_snapshot.type_count.len();
    if matrix_size > 0 {
        let total_particles = physics_snapshot.particle_count;
        for (i, &type_count) in physics_snapshot
            .type_count
            .iter()
            .enumerate()
            .take(matrix_size)
        {
            let percentage = if total_particles > 0 {
                (type_count as f32 / total_particles as f32) * 100.0
            } else {
                0.0
            };

            ui.horizontal(|ui| {
                ui.label(format!("Type {}: {} ({:.1}%)", i, type_count, percentage));
                let progress = percentage / 100.0;
                ui.add(egui::ProgressBar::new(progress).desired_width(100.0));
            });
        }

        if ui.button("Equalize Types").clicked() {
            physics.set_type_count_equal();
        }
    }
}

pub fn render_quick_controls(ui: &mut egui::Ui) {
    ui.label("Quick Controls:");
    ui.label("[Space] - Play/Pause");
    ui.label("[P] - Reset positions");
    ui.label("[C] - Reset types");
    ui.label("[M] - Generate new matrix");
    ui.label("[B] - Toggle boundaries");
    ui.label("[T] - Toggle traces");
    ui.label("[ESC] - Toggle GUI");
}

#[allow(clippy::too_many_arguments)]
pub fn render_graphics_window(
    ctx: &egui::Context,
    show_graphics_window: &mut bool,
    render_clock: &Clock,
    traces_user_enabled: &mut bool,
    camera_is_moving: bool,
    tile_fade_strength: &mut f32,
    trace_fade: &mut f32,
    _app_settings: &mut AppSettings,
    _palettes: &mut SelectionManager<Box<dyn rendering::Palette>>,
) {
    if *show_graphics_window {
        egui::Window::new("Graphics")
            .open(show_graphics_window)
            .show(ctx, |ui| {
                ui.label(format!(
                    "Graphics FPS: {:.0}",
                    render_clock.get_avg_framerate()
                ));
                ui.separator();

                ui.checkbox(traces_user_enabled, "Traces [T]");
                if camera_is_moving && *traces_user_enabled {
                    ui.label("(disabled during panning)");
                }

                ui.horizontal(|ui| {
                    ui.label("Edge Fade:");
                    ui.add(egui::Slider::new(tile_fade_strength, 0.0..=1.0).step_by(0.05));
                });

                if *traces_user_enabled {
                    ui.horizontal(|ui| {
                        ui.label("Trace Fade:");
                        ui.add(egui::Slider::new(trace_fade, 0.0..=1.0).step_by(0.01));
                    });
                }

                ui.horizontal(|ui| {
                    ui.label("Particle Size:");
                    ui.add(
                        egui::Slider::new(&mut _app_settings.particle_size, 0.1..=1.0)
                            .step_by(0.01),
                    );
                });
            });
    }
}

pub fn render_controls_window(ctx: &egui::Context, show_controls_window: &mut bool) {
    if *show_controls_window {
        egui::Window::new("Controls")
            .open(show_controls_window)
            .resizable(false)
            .show(ctx, |ui| {
                ui.heading("Keyboard Controls");
                ui.separator();

                ui.label("🎮 Simulation:");
                ui.label("  [SPACE] - Play/Pause simulation");
                ui.label("  [P] - Reset particle positions");
                ui.label("  [C] - Reset particle types");
                ui.label("  [M] - Generate new random matrix");
                ui.label("  [B] - Toggle boundaries (wrap/clamp)");

                ui.separator();

                ui.label("🖥️ Display:");
                ui.label("  [ESC] - Toggle GUI visibility");
                ui.label("  [T] - Toggle particle traces");
                ui.label("  [G] - Show graphics settings window");

                ui.separator();

                ui.label("📷 Camera Controls:");
                ui.label("  • Mouse wheel - Zoom in/out");
                ui.label("  • [WASD] or Arrow Keys - Pan camera");
                ui.label("  • [Z] - Reset zoom");
                ui.label("  • [Shift+Z] - Fit to window");

                ui.separator();

                ui.label("🖱️ Mouse Interaction:");
                ui.label("  • Left Click - Repel particles");
                ui.label("  • Right Click - Attract particles");
            });
    }
}

pub fn render_about_window(ctx: &egui::Context, show_about_window: &mut bool) {
    if *show_about_window {
        egui::Window::new("About")
            .open(show_about_window)
            .resizable(false)
            .show(ctx, |ui| {
                ui.heading("🦀 Particle Life Simulator");
                ui.separator();

                ui.label("A high-performance particle simulation");
                ui.label("converted from Java to Rust");

                ui.separator();

                ui.label("🔧 Technology Stack:");
                ui.label("  • Rust programming language");
                ui.label("  • egui for immediate mode GUI");
                ui.label("  • wgpu for GPU-accelerated rendering");
                ui.label("  • nalgebra for vector mathematics");

                ui.separator();

                ui.label("Originally inspired by the Java Particle Life project");
                ui.label("Converted with Claude Code");
            });
    }
}
