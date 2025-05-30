mod physics;
mod rendering;
mod shaders;
mod ui;
mod app_settings;

use winit::{
    event::{Event, WindowEvent, KeyEvent, MouseButton, ElementState},
    event_loop::EventLoop,
    window::WindowBuilder,
    keyboard::{KeyCode, PhysicalKey},
};
use wgpu::{Surface, SurfaceConfiguration};
use egui_winit::State;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use physics::{
    ExtendedPhysics, PhysicsSnapshot, DefaultPositionSetter, DefaultMatrixGenerator, 
    RandomPositionSetter, CenterPositionSetter, RandomTypeSetter,
    // Matrix generators
    SymmetryMatrixGenerator, ChainsMatrixGenerator, Chains2MatrixGenerator, Chains3MatrixGenerator,
    SnakesMatrixGenerator, ZeroMatrixGenerator,
    // Position setters
    UniformCirclePositionSetter, CenteredCirclePositionSetter, RingPositionSetter,
    RainbowRingPositionSetter, ColorBattlePositionSetter, ColorWheelPositionSetter,
    LinePositionSetter, SpiralPositionSetter, RainbowSpiralPositionSetter,
    // Type setters  
    Randomize10PercentTypeSetter, SlicesTypeSetter, OnionTypeSetter, RotateTypeSetter,
    FlipTypeSetter, MoreOfFirstTypeSetter, KillStillTypeSetter
};
use rendering::{
    ParticleRenderer, Camera, NaturalRainbowPalette, SimpleRainbowPalette, SunsetPalette, HexPalette, Palette
};
use shaders::{ParticleShader, UniformData, FadeShader};
use ui::{SelectionManager, Clock, Loop};
use app_settings::AppSettings;

struct Application {
    // Core components
    window: Arc<winit::window::Window>,
    surface: Arc<Surface<'static>>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    config: SurfaceConfiguration,
    
    // Rendering
    particle_renderer: ParticleRenderer,
    particle_shader: ParticleShader,
    fade_shader: FadeShader,
    camera: Camera,
    world_texture: wgpu::Texture,
    world_texture_view: wgpu::TextureView,
    world_texture_id: egui::TextureId,
    
    // Physics
    physics: ExtendedPhysics,
    physics_snapshot: PhysicsSnapshot,
    physics_loop: Loop,
    new_snapshot_available: Arc<Mutex<bool>>,
    
    // UI
    egui_ctx: egui::Context,
    egui_state: State,
    egui_renderer: egui_wgpu::Renderer,
    
    // UI state
    show_gui: bool,
    show_graphics_window: bool,
    show_controls_window: bool,
    show_about_window: bool,
    tile_fade_strength: f32,
    traces: bool,
    trace_fade: f32,
    traces_user_enabled: bool,  // User's actual traces setting
    camera_is_moving: bool,     // Whether camera is currently being moved
    camera_movement_timer: std::time::Instant,  // Timer to detect when movement stops
    prev_camera_position: nalgebra::Vector3<f64>,  // Previous camera position for movement detection
    prev_camera_size: f64,      // Previous camera size for zoom detection
    
    // Matrix editing state
    local_matrix: Vec<Vec<f64>>,
    
    // Selection managers
    palettes: SelectionManager<Box<dyn rendering::Palette>>,
    position_setters: SelectionManager<Box<dyn physics::PositionSetter>>,
    matrix_generators: SelectionManager<Box<dyn physics::MatrixGenerator>>,
    type_setters: SelectionManager<Box<dyn physics::TypeSetter>>,
    
    
    // Timing
    render_clock: Clock,
    
    // Input state
    mouse_x: f64,
    mouse_y: f64,
    pmouse_x: f64,
    pmouse_y: f64,
    
    // Key states
    keys_pressed: std::collections::HashSet<KeyCode>,
    left_mouse_pressed: bool,
    right_mouse_pressed: bool,
    middle_mouse_pressed: bool,
    
    // Mouse state tracking
    cursor_moved_last_frame: bool,
    
    // App settings
    app_settings: AppSettings,
    
    // Performance tracking
    physics_time_avg: f64,
    physics_time_samples: Vec<f64>,
    
    // Debug timing
    last_debug_time: std::time::Instant,
    
    // Mouse interaction
    cursor_world_position: nalgebra::Vector3<f64>,
    cursor_size: f64,
    cursor_strength: f64,
}

impl Application {
    async fn new(event_loop: &EventLoop<()>) -> Self {
        let window = Arc::new(WindowBuilder::new()
            .with_title("Particle Life Simulator")
            .with_inner_size(winit::dpi::LogicalSize::new(1200.0, 800.0))
            .build(event_loop)
            .unwrap());

        // Initialize WGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
            flags: wgpu::InstanceFlags::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        let surface = Arc::new(instance.create_surface(window.clone()).unwrap());

        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.unwrap();

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_buffer_size: 1024 * 1024 * 1024, // 1GB buffer size limit
                    ..wgpu::Limits::default()
                },
            },
            None,
        ).await.unwrap();

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // Initialize egui
        let egui_ctx = egui::Context::default();
        let egui_state = State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            None,
            None,
        );
        let mut egui_renderer = egui_wgpu::Renderer::new(&device, surface_format, None, 1);

        // Initialize rendering
        let particle_renderer = ParticleRenderer::new();
        let particle_shader = ParticleShader::new(&device, surface_format);
        let fade_shader = FadeShader::new(&device, surface_format);
        let camera = Camera::new();
        
        // Create offscreen world texture for particle traces
        let world_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("World Texture"),
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: surface_format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let world_texture_view = world_texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Register world texture with egui
        let world_texture_id = egui_renderer.register_native_texture(&device, &world_texture_view, wgpu::FilterMode::Linear);

        // Initialize physics
        let physics = ExtendedPhysics::new(
            Box::new(DefaultPositionSetter),
            Box::new(DefaultMatrixGenerator),
        );
        
        let physics_snapshot = PhysicsSnapshot::new();
        let physics_loop = Loop::new();
        let new_snapshot_available = Arc::new(Mutex::new(false));

        // Initialize selection managers
        let mut palette_list = vec![
            ("Natural Rainbow".to_string(), Box::new(NaturalRainbowPalette) as Box<dyn rendering::Palette>),
            ("Simple Rainbow".to_string(), Box::new(SimpleRainbowPalette) as Box<dyn rendering::Palette>),
            ("Sunset".to_string(), Box::new(SunsetPalette) as Box<dyn rendering::Palette>),
        ];
        
        // Load HEX palettes from palettes directory
        if let Ok(entries) = std::fs::read_dir("palettes") {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("hex") {
                    if let Ok(hex_palette) = HexPalette::load_from_file(path.to_str().unwrap()) {
                        let name = hex_palette.name().to_string();
                        palette_list.push((name, Box::new(hex_palette) as Box<dyn rendering::Palette>));
                    }
                }
            }
        }
        
        let palettes = SelectionManager::new(palette_list);

        let position_setters = SelectionManager::new(vec![
            ("Centered".to_string(), Box::new(CenterPositionSetter) as Box<dyn physics::PositionSetter>),
            ("Uniform".to_string(), Box::new(RandomPositionSetter) as Box<dyn physics::PositionSetter>),
            ("Uniform Circle".to_string(), Box::new(UniformCirclePositionSetter) as Box<dyn physics::PositionSetter>),
            ("Centered Circle".to_string(), Box::new(CenteredCirclePositionSetter) as Box<dyn physics::PositionSetter>),
            ("Ring".to_string(), Box::new(RingPositionSetter) as Box<dyn physics::PositionSetter>),
            ("Rainbow Ring".to_string(), Box::new(RainbowRingPositionSetter) as Box<dyn physics::PositionSetter>),
            ("Color Battle".to_string(), Box::new(ColorBattlePositionSetter) as Box<dyn physics::PositionSetter>),
            ("Color Wheel".to_string(), Box::new(ColorWheelPositionSetter) as Box<dyn physics::PositionSetter>),
            ("Line".to_string(), Box::new(LinePositionSetter) as Box<dyn physics::PositionSetter>),
            ("Spiral".to_string(), Box::new(SpiralPositionSetter) as Box<dyn physics::PositionSetter>),
            ("Rainbow Spiral".to_string(), Box::new(RainbowSpiralPositionSetter) as Box<dyn physics::PositionSetter>),
        ]);

        let matrix_generators = SelectionManager::new(vec![
            ("Random".to_string(), Box::new(DefaultMatrixGenerator) as Box<dyn physics::MatrixGenerator>),
            ("Symmetry".to_string(), Box::new(SymmetryMatrixGenerator) as Box<dyn physics::MatrixGenerator>),
            ("Chains".to_string(), Box::new(ChainsMatrixGenerator) as Box<dyn physics::MatrixGenerator>),
            ("Chains 2".to_string(), Box::new(Chains2MatrixGenerator) as Box<dyn physics::MatrixGenerator>),
            ("Chains 3".to_string(), Box::new(Chains3MatrixGenerator) as Box<dyn physics::MatrixGenerator>),
            ("Snakes".to_string(), Box::new(SnakesMatrixGenerator) as Box<dyn physics::MatrixGenerator>),
            ("Zero".to_string(), Box::new(ZeroMatrixGenerator) as Box<dyn physics::MatrixGenerator>),
        ]);

        let type_setters = SelectionManager::new(vec![
            ("Random".to_string(), Box::new(RandomTypeSetter) as Box<dyn physics::TypeSetter>),
            ("Randomize 10%".to_string(), Box::new(Randomize10PercentTypeSetter) as Box<dyn physics::TypeSetter>),
            ("Slices".to_string(), Box::new(SlicesTypeSetter) as Box<dyn physics::TypeSetter>),
            ("Onion".to_string(), Box::new(OnionTypeSetter) as Box<dyn physics::TypeSetter>),
            ("Rotate".to_string(), Box::new(RotateTypeSetter) as Box<dyn physics::TypeSetter>),
            ("Flip".to_string(), Box::new(FlipTypeSetter) as Box<dyn physics::TypeSetter>),
            ("More of First".to_string(), Box::new(MoreOfFirstTypeSetter) as Box<dyn physics::TypeSetter>),
            ("Kill Still".to_string(), Box::new(KillStillTypeSetter) as Box<dyn physics::TypeSetter>),
        ]);

        // Initialize UI components
        let render_clock = Clock::new(60.0);

        // Load app settings
        let app_settings = AppSettings::load().unwrap_or_default();

        Self {
            window,
            surface,
            device,
            queue,
            config,
            particle_renderer,
            particle_shader,
            fade_shader,
            camera,
            world_texture,
            world_texture_view,
            world_texture_id,
            physics,
            physics_snapshot,
            physics_loop,
            new_snapshot_available,
            egui_ctx,
            egui_state,
            egui_renderer,
            show_gui: true,
            show_graphics_window: false,
            show_controls_window: false,
            show_about_window: false,
            tile_fade_strength: 0.7,
            traces: false,
            trace_fade: 0.95,
            traces_user_enabled: false,
            camera_is_moving: false,
            camera_movement_timer: std::time::Instant::now(),
            prev_camera_position: nalgebra::Vector3::new(0.5, 0.5, 0.0),
            prev_camera_size: 2.0,
            local_matrix: vec![vec![0.0; 4]; 4], // Initialize with 4x4 matrix
            palettes,
            position_setters,
            matrix_generators,
            type_setters,
            render_clock,
            mouse_x: 0.0,
            mouse_y: 0.0,
            pmouse_x: 0.0,
            pmouse_y: 0.0,
            keys_pressed: std::collections::HashSet::new(),
            left_mouse_pressed: false,
            right_mouse_pressed: false,
            middle_mouse_pressed: false,
            cursor_moved_last_frame: false,
            app_settings,
            physics_time_avg: 0.0,
            physics_time_samples: Vec::with_capacity(60),
            last_debug_time: std::time::Instant::now(),
            cursor_world_position: nalgebra::Vector3::new(0.0, 0.0, 0.0),
            cursor_size: 0.2,
            cursor_strength: 5.0,
        }
    }

    fn setup(&mut self) {
        // Initialize physics with some default particles
        self.physics.set_particle_count(20_000);
        self.physics.set_matrix_size(4);
        self.physics.set_positions();
        
        // Equalize type distribution
        self.physics.set_type_count_equal();
        
        // Generate an interesting interaction matrix
        self.physics.generate_matrix();
        
        // Take initial snapshot
        self.physics_snapshot = self.physics.take_snapshot();
        
        
        // Initialize local matrix copy
        let matrix_size = self.physics.matrix.size();
        self.local_matrix = vec![vec![0.0; matrix_size]; matrix_size];
        // Copy current physics matrix values using iterators
        self.local_matrix.iter_mut().enumerate().for_each(|(i, row)| {
            row.iter_mut().enumerate().for_each(|(j, val)| {
                *val = self.physics.matrix.get(i, j);
            });
        });
        
        // Buffer initial particle data (for CPU renderer fallback)
        self.particle_renderer.buffer_particle_data(
            &self.device,
            &self.queue,
            &self.physics_snapshot.positions,
            &self.physics_snapshot.velocities,
            &self.physics_snapshot.types,
            self.palettes.get_active().as_ref(),
        );
        
        println!("Initialized {} particles with {} types", 
                 self.physics_snapshot.particle_count, 
                 self.physics.matrix.size());
    }

    fn handle_event(&mut self, event: &WindowEvent) -> bool {
        let response = self.egui_state.on_window_event(&self.window, event);
        if response.consumed {
            return true;
        }

        match event {
            WindowEvent::KeyboardInput { event: KeyEvent { physical_key: PhysicalKey::Code(key_code), state, .. }, .. } => {
                match state {
                    ElementState::Pressed => {
                        self.keys_pressed.insert(*key_code);
                        self.handle_key_pressed(*key_code);
                    }
                    ElementState::Released => {
                        self.keys_pressed.remove(key_code);
                    }
                }
                true
            }
            WindowEvent::MouseInput { button, state, .. } => {
                match (button, state) {
                    (MouseButton::Left, ElementState::Pressed) => self.left_mouse_pressed = true,
                    (MouseButton::Left, ElementState::Released) => self.left_mouse_pressed = false,
                    (MouseButton::Right, ElementState::Pressed) => self.right_mouse_pressed = true,
                    (MouseButton::Right, ElementState::Released) => self.right_mouse_pressed = false,
                    (MouseButton::Middle, ElementState::Pressed) => self.middle_mouse_pressed = true,
                    (MouseButton::Middle, ElementState::Released) => self.middle_mouse_pressed = false,
                    _ => {}
                }
                false
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.pmouse_x = self.mouse_x;
                self.pmouse_y = self.mouse_y;
                self.mouse_x = position.x;
                self.mouse_y = position.y;
                self.cursor_moved_last_frame = true;
                
                // Update cursor world position
                self.update_cursor_world_position();
                false
            }
            WindowEvent::MouseWheel { delta, .. } => {
                match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => {
                        self.handle_scroll(*y as f64);
                    }
                    winit::event::MouseScrollDelta::PixelDelta(delta) => {
                        // Handle trackpad scroll for zooming only
                        if delta.y.abs() > 0.1 {
                            let y = delta.y / 120.0; // Convert pixel delta to line delta equivalent
                            self.handle_scroll(y);
                        }
                    }
                }
                false
            }
            WindowEvent::Resized(physical_size) => {
                self.config.width = physical_size.width;
                self.config.height = physical_size.height;
                self.surface.configure(&self.device, &self.config);
                
                // Recreate world texture with new size
                self.world_texture = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("World Texture"),
                    size: wgpu::Extent3d {
                        width: self.config.width,
                        height: self.config.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.config.format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                self.world_texture_view = self.world_texture.create_view(&wgpu::TextureViewDescriptor::default());
                
                // Update egui texture
                self.world_texture_id = self.egui_renderer.register_native_texture(&self.device, &self.world_texture_view, wgpu::FilterMode::Linear);
                
                false
            }
            WindowEvent::TouchpadPressure { .. } => false,
            WindowEvent::AxisMotion { .. } => false,
            _ => false,
        }
    }

    fn handle_key_pressed(&mut self, key_code: KeyCode) {
        match key_code {
            KeyCode::Escape => self.show_gui = !self.show_gui,
            KeyCode::KeyG => self.show_graphics_window = !self.show_graphics_window,
            KeyCode::KeyT => {
                self.traces_user_enabled = !self.traces_user_enabled;
                // Update actual traces state unless camera is moving
                if !self.camera_is_moving {
                    self.traces = self.traces_user_enabled;
                }
            },
            KeyCode::Space => self.physics_loop.pause = !self.physics_loop.pause,
            KeyCode::KeyP => self.physics.set_positions_with_setter(self.position_setters.get_active().as_ref()),
            KeyCode::KeyC => self.physics.set_types_with_setter(self.type_setters.get_active().as_ref()),
            KeyCode::KeyM => {
                self.physics.generate_matrix_with_generator(self.matrix_generators.get_active().as_ref());
                // Update local matrix copy for UI
                let matrix_size = self.physics.matrix.size();
                for i in 0..matrix_size {
                    for j in 0..matrix_size {
                        self.local_matrix[i][j] = self.physics.matrix.get(i, j);
                    }
                }
            },
            KeyCode::KeyB => self.physics.settings.wrap = !self.physics.settings.wrap,
            KeyCode::KeyZ => {
                if self.keys_pressed.contains(&KeyCode::ShiftLeft) || self.keys_pressed.contains(&KeyCode::ShiftRight) {
                    self.camera.reset(true, self.config.width as f32 / self.config.height as f32);
                } else {
                    self.camera.reset(false, self.config.width as f32 / self.config.height as f32);
                }
            }
            _ => {}
        }
    }

    fn handle_scroll(&mut self, delta_y: f64) {
        let ctrl_pressed = self.keys_pressed.contains(&KeyCode::ControlLeft) || 
                          self.keys_pressed.contains(&KeyCode::ControlRight);
        let shift_pressed = self.keys_pressed.contains(&KeyCode::ShiftLeft) || 
                           self.keys_pressed.contains(&KeyCode::ShiftRight);

        if ctrl_pressed && shift_pressed {
            // Change time step
            self.physics.settings.dt *= 1.2_f64.powf(-delta_y);
            self.physics.settings.dt = self.physics.settings.dt.clamp(0.001, 0.1);
        } else if shift_pressed {
            // Change particle size
            self.app_settings.particle_size *= 1.2_f32.powf(-delta_y as f32);
        } else if ctrl_pressed {
            // Change cursor size (not implemented yet)
        } else {
            // Zoom camera using DPI-aware cursor position
            let zoom_factor = 1.2_f64.powf(-delta_y);
            let scale_factor = self.window.scale_factor();
            let logical_size = self.window.inner_size();
            
            self.camera.zoom(
                self.mouse_x,
                self.mouse_y,
                zoom_factor,
                scale_factor,
                logical_size.width,
                logical_size.height,
            );
        }
    }
    
    fn update_cursor_world_position(&mut self) {
        // Convert screen coordinates to world coordinates
        let scale_factor = self.window.scale_factor();
        let logical_size = self.window.inner_size();
        let aspect_ratio = logical_size.width as f64 / logical_size.height as f64;
        
        // Mouse coordinates might be in physical pixels, convert to logical
        let logical_mouse_x = self.mouse_x / scale_factor;
        let logical_mouse_y = self.mouse_y / scale_factor;
        
        // Normalize mouse coordinates to [-0.5, 0.5] using logical size
        let mouse_x_norm = (logical_mouse_x / logical_size.width as f64) - 0.5;
        let mouse_y_norm = (logical_mouse_y / logical_size.height as f64) - 0.5;
        
        // Convert to world coordinates
        self.cursor_world_position.x = self.camera.position.x + mouse_x_norm * self.camera.size;
        self.cursor_world_position.y = self.camera.position.y + mouse_y_norm * self.camera.size / aspect_ratio;
        self.cursor_world_position.z = 0.0;
    }

    fn update(&mut self, dt: f64) {
        self.render_clock.tick();
        
        // Handle camera input (this updates target position/size)
        // Base movement speed independent of zoom level
        let base_movement_speed = self.app_settings.camera_movement_speed as f64 * dt;
        
        // Keyboard camera movement
        if self.keys_pressed.contains(&KeyCode::KeyA) || self.keys_pressed.contains(&KeyCode::ArrowLeft) {
            self.camera.pan(-base_movement_speed, 0.0);
        }
        if self.keys_pressed.contains(&KeyCode::KeyD) || self.keys_pressed.contains(&KeyCode::ArrowRight) {
            self.camera.pan(base_movement_speed, 0.0);
        }
        if self.keys_pressed.contains(&KeyCode::KeyW) || self.keys_pressed.contains(&KeyCode::ArrowUp) {
            self.camera.pan(0.0, -base_movement_speed);
        }
        if self.keys_pressed.contains(&KeyCode::KeyS) || self.keys_pressed.contains(&KeyCode::ArrowDown) {
            self.camera.pan(0.0, base_movement_speed);
        }
        
        // Mouse panning removed - only keyboard panning and mouse zooming supported
        
        // Reset cursor movement flag for next frame
        self.cursor_moved_last_frame = false;
        
        // Update camera (applies smoothing to move towards target)
        self.camera.update(self.app_settings.camera_smoothness as f64);
        
        // Check for actual camera movement by comparing current position to previous
        let position_delta = (self.camera.position - self.prev_camera_position).magnitude();
        let size_delta = (self.camera.size - self.prev_camera_size).abs();
        let movement_threshold = 0.0001; // Adjust this threshold as needed
        
        let camera_actually_moving = position_delta > movement_threshold || size_delta > movement_threshold;
        
        if camera_actually_moving {
            self.camera_is_moving = true;
            self.camera_movement_timer = std::time::Instant::now();
            // Temporarily disable trails during movement
            if self.traces_user_enabled {
                self.traces = false;
            }
        } else if self.camera_is_moving {
            // Check if camera has stopped moving for a brief period
            if self.camera_movement_timer.elapsed().as_millis() > 150 { // 150ms delay to account for smoothing
                self.camera_is_moving = false;
                // Re-enable trails if user has them enabled
                if self.traces_user_enabled {
                    self.traces = true;
                }
            }
        }
        
        // Update previous camera state for next frame
        self.prev_camera_position = self.camera.position;
        self.prev_camera_size = self.camera.size;

        // Update physics with performance tracking
        if !self.physics_loop.pause {
            let physics_start = std::time::Instant::now();
            
            // Determine cursor interaction
            let cursor_interaction = if self.left_mouse_pressed || self.right_mouse_pressed {
                // Left mouse = repel (negative strength), right mouse = attract (positive strength)
                let strength = if self.left_mouse_pressed { -self.cursor_strength } else { self.cursor_strength };
                Some((self.cursor_world_position, self.cursor_size, strength))
            } else {
                None
            };
            
            // Update physics with cursor interaction
            if let Some((pos, size, strength)) = cursor_interaction {
                self.physics.update_with_cursor(Some(pos), size, strength);
            } else {
                self.physics.update();
            }
            
            let physics_time = physics_start.elapsed().as_micros() as f64 / 1000.0; // Convert to milliseconds
            
            // Track physics timing for display
            self.physics_time_samples.push(physics_time);
            if self.physics_time_samples.len() > 60 {
                self.physics_time_samples.remove(0);
            }
            self.physics_time_avg = self.physics_time_samples.iter().sum::<f64>() / self.physics_time_samples.len() as f64;
            
            // Debug output every few seconds to verify simulation is running
            if self.last_debug_time.elapsed().as_secs() >= 5 {
                    let start_time = std::time::Instant::now();
                    
                    // Measure physics update time
                    let particle_count = self.physics.particles.len();
                    let avg_velocity = if particle_count > 0 {
                        self.physics.particles.iter()
                            .map(|p| p.velocity.norm())
                            .sum::<f64>() / particle_count as f64
                    } else {
                        0.0
                    };
                    
                    let physics_time = start_time.elapsed();
                    
                    println!("Physics: {} particles, avg velocity: {:.6}, calc time: {:.2}ms", 
                             particle_count, avg_velocity, physics_time.as_micros() as f64 / 1000.0);
                    self.last_debug_time = std::time::Instant::now();
                }
        }

        // Update physics snapshot
        self.physics_snapshot = self.physics.take_snapshot();
        *self.new_snapshot_available.lock().unwrap() = true;

        // Update particle renderer with latest data
        self.particle_renderer.buffer_particle_data(
            &self.device,
            &self.queue,
            &self.physics_snapshot.positions,
            &self.physics_snapshot.velocities,
            &self.physics_snapshot.types,
            self.palettes.get_active().as_ref(),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Render particles to offscreen world texture
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Particle Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.world_texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: if self.traces {
                            wgpu::LoadOp::Load  // Always load for fading
                        } else {
                            wgpu::LoadOp::Clear(wgpu::Color::BLACK)
                        },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Calculate camera matrices for both fade and particle shaders
            let view_proj_matrix = self.camera.get_view_projection_matrix(
                self.config.width as f32 / self.config.height as f32
            );

            // Apply fade effect when traces are enabled
            if self.traces {
                self.fade_shader.update_uniforms(&self.queue, self.trace_fade);
                self.fade_shader.render(&mut render_pass);
            }

            // Update particle shader uniforms
            let uniform_data = UniformData {
                view_proj: view_proj_matrix.into(),
                time: self.render_clock.get_dt_millis() as f32 / 1000.0,
                particle_size: self.app_settings.particle_size,
                cam_top_left: [
                    (self.camera.position.x - self.camera.size * 0.5) as f32,
                    (self.camera.position.y - self.camera.size * 0.5) as f32,
                ],
                wrap: if self.physics.settings.wrap { 1 } else { 0 },
                show_tiling: 1, // Always enable tiling
                world_size: 4.0,
                tile_fade_strength: self.tile_fade_strength,
            };

            self.particle_shader.update_uniforms(&self.queue, &uniform_data);
            self.particle_renderer.render(&mut render_pass, &self.particle_shader, self.physics_snapshot.particle_count, true);
        }


        // Render GUI
        let raw_input = self.egui_state.take_egui_input(&self.window);
        
        // Split self to avoid borrow checker issues
        let show_gui = self.show_gui;
        let show_graphics_window = &mut self.show_graphics_window;
        let show_controls_window = &mut self.show_controls_window;
        let show_about_window = &mut self.show_about_window;
        let tile_fade_strength = &mut self.tile_fade_strength;
        let traces_user_enabled = &mut self.traces_user_enabled;
        let camera_is_moving = self.camera_is_moving;
        let physics_loop_pause = &mut self.physics_loop.pause;
        let physics_snapshot = &self.physics_snapshot;
        let render_clock = &self.render_clock;
        let _app_settings = &mut self.app_settings;
        let _palettes = &mut self.palettes;
        let _position_setters = &mut self.position_setters;
        let _type_setters = &mut self.type_setters;
        let _matrix_generators = &mut self.matrix_generators;
        let local_matrix = &mut self.local_matrix;
        let physics = &mut self.physics;
        let physics_time_avg = self.physics_time_avg;
        let trace_fade = &mut self.trace_fade;
        let cursor_size = &mut self.cursor_size;
        let cursor_strength = &mut self.cursor_strength;
        
        let world_texture_id = self.world_texture_id;
        let config_width = self.config.width as f32;
        let config_height = self.config.height as f32;
        
        let full_output = self.egui_ctx.run(raw_input, |ctx| {
            // Display world texture as background
            ctx.layer_painter(egui::LayerId::background()).image(
                world_texture_id,
                egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(config_width, config_height)),
                egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(1.0, 1.0)),
                egui::Color32::WHITE,
            );
            
            if !show_gui {
                return;
            }

            // Main menu bar
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

            // Physics panel
            egui::SidePanel::left("physics_panel").min_width(320.0).show(ctx, |ui| {
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
                        ui.label(format!("Render: {:.0} FPS", render_clock.get_avg_framerate()));
                        ui.separator();
                        ui.label(format!("Physics: {:.2}ms", physics_time_avg));
                    });

                    // Play/Pause
                    if ui.button(if *physics_loop_pause { "Play" } else { "Pause" }).clicked() {
                        *physics_loop_pause = !*physics_loop_pause;
                    }

                    ui.separator();

                    // Physics parameters
                    ui.label("Physics Parameters");
                    ui.label(format!("Particles: {}", physics_snapshot.particle_count));
                    ui.label(format!("Matrix Size: {}x{}", physics_snapshot.type_count.len(), physics_snapshot.type_count.len()));
                    
                    ui.separator();
                    
                    // Interactive Matrix Editor
                    ui.heading("Interaction Matrix");
                    
                    // Ensure local matrix matches current size
                    let matrix_size = physics_snapshot.type_count.len();
                    if local_matrix.len() != matrix_size {
                        local_matrix.resize(matrix_size, vec![0.0; matrix_size]);
                        for row in local_matrix.iter_mut() {
                            row.resize(matrix_size, 0.0);
                        }
                        // Copy current physics matrix values using iterators
                        local_matrix.iter_mut().enumerate().for_each(|(i, row)| {
                            row.iter_mut().enumerate().for_each(|(j, val)| {
                                *val = physics.matrix.get(i, j);
                            });
                        });
                    }
                    
                    if matrix_size > 0 {
                        // Matrix editing grid
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
                    
                    // Matrix size control
                    ui.horizontal(|ui| {
                        ui.label("Matrix Size:");
                        let mut matrix_size_input = matrix_size as i32;
                        if ui.add(egui::DragValue::new(&mut matrix_size_input).clamp_range(2..=8).speed(1)).changed() {
                            let new_size = matrix_size_input.max(2) as usize;
                            physics.set_matrix_size(new_size);
                            println!("Set matrix size to {}x{}", new_size, new_size);
                        }
                    });
                    
                    // Matrix generator selection
                    ui.horizontal(|ui| {
                        ui.label("Generator:");
                        let current_name = _matrix_generators.get_active_name().to_string();
                        let names = _matrix_generators.get_item_names();
                        
                        let mut selected_index = _matrix_generators.get_active_index();
                        let mut changed = false;
                        egui::ComboBox::from_id_source("matrix_generator")
                            .selected_text(&current_name)
                            .show_ui(ui, |ui| {
                                for (i, name) in names.iter().enumerate() {
                                    if ui.selectable_label(i == selected_index, *name).clicked() {
                                        selected_index = i;
                                        changed = true;
                                    }
                                }
                            });
                        
                        if changed {
                            _matrix_generators.set_active(selected_index);
                            println!("Changed matrix generator to: {}", _matrix_generators.get_active_name());
                        }
                    });
                    
                    // Matrix controls
                    ui.horizontal(|ui| {
                        if ui.button("Generate Matrix").clicked() {
                            physics.generate_matrix_with_generator(_matrix_generators.get_active().as_ref());
                            // Update local copy using iterators
                            local_matrix.iter_mut().enumerate().for_each(|(i, row)| {
                                row.iter_mut().enumerate().for_each(|(j, val)| {
                                    *val = physics.matrix.get(i, j);
                                });
                            });
                            println!("Generated matrix with {}", _matrix_generators.get_active_name());
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
                    
                    ui.separator();
                    
                    // Particle Setup Controls
                    ui.heading("Particle Setup");
                    
                    // Position setter selection
                    ui.horizontal(|ui| {
                        ui.label("Position Mode:");
                        let current_name = _position_setters.get_active_name().to_string();
                        let names = _position_setters.get_item_names();
                        
                        let mut selected_index = _position_setters.get_active_index();
                        let mut changed = false;
                        egui::ComboBox::from_id_source("position_setter")
                            .selected_text(&current_name)
                            .show_ui(ui, |ui| {
                                for (i, name) in names.iter().enumerate() {
                                    if ui.selectable_label(i == selected_index, *name).clicked() {
                                        selected_index = i;
                                        changed = true;
                                    }
                                }
                            });
                        
                        if changed {
                            _position_setters.set_active(selected_index);
                            println!("Changed position setter to: {}", _position_setters.get_active_name());
                        }
                        
                        if ui.button("Reset Positions").clicked() {
                            physics.set_positions_with_setter(_position_setters.get_active().as_ref());
                            println!("Reset particle positions with {}", _position_setters.get_active_name());
                        }
                    });
                    
                    // Type setter selection  
                    ui.horizontal(|ui| {
                        ui.label("Type Mode:");
                        let current_name = _type_setters.get_active_name().to_string();
                        let names = _type_setters.get_item_names();
                        
                        let mut selected_index = _type_setters.get_active_index();
                        let mut changed = false;
                        egui::ComboBox::from_id_source("type_setter")
                            .selected_text(&current_name)
                            .show_ui(ui, |ui| {
                                for (i, name) in names.iter().enumerate() {
                                    if ui.selectable_label(i == selected_index, *name).clicked() {
                                        selected_index = i;
                                        changed = true;
                                    }
                                }
                            });
                        
                        if changed {
                            _type_setters.set_active(selected_index);
                            println!("Changed type setter to: {}", _type_setters.get_active_name());
                        }
                        
                        if ui.button("Reset Types").clicked() {
                            physics.set_types_with_setter(_type_setters.get_active().as_ref());
                            println!("Reset particle types with {}", _type_setters.get_active_name());
                        }
                    });
                    
                    ui.separator();
                    
                    // Rendering Controls
                    ui.heading("Rendering Settings");
                    
                    // Particle size control in main panel
                    ui.horizontal(|ui| {
                        ui.label("Particle Size:");
                        if ui.add(egui::Slider::new(&mut _app_settings.particle_size, 0.1..=1.0).step_by(0.01)).changed() {
                            // Particle size is updated automatically through the reference
                        }
                    });
                    
                    // Palette selection in main panel
                    ui.horizontal(|ui| {
                        ui.label("Color Palette:");
                        let current_name = _palettes.get_active_name().to_string();
                        let names = _palettes.get_item_names();
                        
                        let mut selected_index = _palettes.get_active_index();
                        egui::ComboBox::from_id_source("palette")
                            .selected_text(&current_name)
                            .show_ui(ui, |ui| {
                                for (i, name) in names.iter().enumerate() {
                                    if ui.selectable_label(i == selected_index, *name).clicked() {
                                        selected_index = i;
                                    }
                                }
                            });
                        
                        if selected_index != _palettes.get_active_index() {
                            _palettes.set_active(selected_index);
                        }
                    });
                    
                    // Traces toggle
                    ui.horizontal(|ui| {
                        ui.label("Particle Traces:");
                        if ui.checkbox(traces_user_enabled, "Enable [T]").changed() {
                            // Update actual traces state unless camera is moving
                            if !camera_is_moving {
                                // We need to update self.traces, but we can't borrow self mutably here
                                // We'll handle this after the UI section
                            }
                        }
                        if camera_is_moving && *traces_user_enabled {
                            ui.label("(disabled during panning)");
                        }
                    });
                    
                    ui.separator();
                    
                    // Mouse Interaction Controls
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
                    
                    ui.separator();
                    
                    // Physics Parameters Controls
                    ui.heading("Physics Settings");
                    
                    // Particle count
                    let mut particle_count_input = physics_snapshot.particle_count as i32;
                    ui.horizontal(|ui| {
                        ui.label("Particles:");
                        if ui.add(egui::DragValue::new(&mut particle_count_input).clamp_range(100..=200000).speed(10)).changed() {
                            let new_count = particle_count_input.max(100) as usize;
                            physics.set_particle_count(new_count);
                            println!("Set particle count to {}", new_count);
                        }
                    });
                    
                    // Force multiplier
                    ui.horizontal(|ui| {
                        ui.label("Force:");
                        let mut force = physics.settings.force as f32;
                        if ui.add(egui::Slider::new(&mut force, 0.0..=5.0).step_by(0.1)).changed() {
                            physics.settings.force = force as f64;
                        }
                    });
                    
                    // Friction
                    ui.horizontal(|ui| {
                        ui.label("Friction:");
                        let mut friction = physics.settings.friction as f32;
                        if ui.add(egui::Slider::new(&mut friction, 0.0..=1.0).step_by(0.01)).changed() {
                            physics.settings.friction = friction as f64;
                        }
                    });
                    
                    // rmax (interaction distance)
                    ui.horizontal(|ui| {
                        ui.label("Range (rmax):");
                        let mut rmax = physics.settings.rmax as f32;
                        if ui.add(egui::Slider::new(&mut rmax, 0.005..=0.2).step_by(0.005)).changed() {
                            physics.settings.rmax = rmax as f64;
                        }
                    });
                    
                    // Boundaries (wrap)
                    ui.horizontal(|ui| {
                        ui.label("Boundaries:");
                        let mut wrap = physics.settings.wrap;
                        if ui.checkbox(&mut wrap, "Wrap around").changed() {
                            physics.settings.wrap = wrap;
                        }
                    });
                    
                    ui.separator();
                    
                    // Particle Type Distribution
                    ui.heading("Type Distribution");
                    
                    if matrix_size > 0 {
                        // Display type counts as bars
                        let total_particles = physics_snapshot.particle_count;
                        physics_snapshot.type_count.iter()
                            .enumerate()
                            .take(matrix_size)
                            .for_each(|(i, &type_count)| {
                                let percentage = if total_particles > 0 { 
                                    (type_count as f32 / total_particles as f32) * 100.0 
                                } else { 
                                    0.0 
                                };
                                
                                ui.horizontal(|ui| {
                                    // Color indicator
                                    let color = _palettes.get_active().get_color(i, matrix_size);
                                    let color32 = egui::Color32::from_rgb(
                                        (color.r * 255.0) as u8,
                                        (color.g * 255.0) as u8, 
                                        (color.b * 255.0) as u8
                                    );
                                    
                                    ui.colored_label(color32, format!("Type {}:", i));
                                    ui.label(format!("{} ({:.1}%)", type_count, percentage));
                                    
                                    // Simple progress bar
                                    let progress = percentage / 100.0;
                                    ui.add(egui::ProgressBar::new(progress).desired_width(100.0));
                                });
                            });
                        
                        // Equalize button
                        if ui.button("Equalize Types").clicked() {
                            physics.set_type_count_equal();
                            println!("Equalized particle type distribution");
                        }
                    }
                    
                    ui.separator();
                    
                    ui.label("Quick Controls:");
                    ui.label("[Space] - Play/Pause");
                    ui.label("[P] - Reset positions");
                    ui.label("[C] - Reset types");
                    ui.label("[M] - Generate new matrix");
                    ui.label("[B] - Toggle boundaries");
                    ui.label("[T] - Toggle traces");
                    ui.label("[ESC] - Toggle GUI");
                    ui.label("[G] - Graphics settings");
                });
            });

            // Graphics window
            if *show_graphics_window {
                egui::Window::new("Graphics")
                    .open(show_graphics_window)
                    .show(ctx, |ui| {
                        ui.label(format!("Graphics FPS: {:.0}", render_clock.get_avg_framerate()));
                        
                        ui.separator();
                        
                        // Rendering options
                        ui.checkbox(traces_user_enabled, "Traces [T]");
                        if camera_is_moving && *traces_user_enabled {
                            ui.label("(disabled during panning)");
                        }
                        
                        ui.label("3x3 Tiling: Always enabled");
                        ui.horizontal(|ui| {
                            ui.label("Edge Fade:");
                            ui.add(egui::Slider::new(tile_fade_strength, 0.0..=1.0).step_by(0.05).text("strength"));
                        });
                        if *tile_fade_strength > 0.0 {
                            ui.label("Edge tiles fade to emphasize center tile");
                        }
                        
                        if *traces_user_enabled {
                            ui.horizontal(|ui| {
                                ui.label("Trace Fade:");
                                ui.add(egui::Slider::new(trace_fade, 0.0..=1.0).step_by(0.01).text("fade"));
                            });
                        }
                        
                        // Particle size
                        ui.horizontal(|ui| {
                            ui.label("Particle Size:");
                            ui.add(egui::Slider::new(&mut _app_settings.particle_size, 0.1..=1.0).step_by(0.01));
                        });
                        
                        // Palette selection
                        ui.horizontal(|ui| {
                            ui.label("Palette:");
                            let current_name = _palettes.get_active_name().to_string();
                            let names = _palettes.get_item_names();
                            
                            let mut selected_index = _palettes.get_active_index();
                            egui::ComboBox::from_id_source("palette_graphics")
                                .selected_text(&current_name)
                                .show_ui(ui, |ui| {
                                    for (i, name) in names.iter().enumerate() {
                                        if ui.selectable_label(i == selected_index, *name).clicked() {
                                            selected_index = i;
                                        }
                                    }
                                });
                            
                            if selected_index != _palettes.get_active_index() {
                                _palettes.set_active(selected_index);
                            }
                        });
                    });
            }

            // Controls window
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
                        
                        ui.label("📷 Camera (Pan & Zoom):");
                        ui.label("  🔍 Zoom Controls:");
                        ui.label("    • Mouse wheel - Zoom in/out (towards cursor)");
                        ui.label("    • [Z] - Reset zoom to 100%");
                        ui.label("    • [Shift+Z] - Fit simulation to window");
                        ui.label("  🖱️ Pan Controls:");
                        ui.label("    • [WASD] or Arrow Keys - Pan with keyboard");
                        ui.label("  ⚙️ Other:");
                        ui.label("    • [Shift+Scroll] - Adjust particle size");
                        ui.label("    • [Ctrl+Shift+Scroll] - Adjust time step");
                        
                        ui.separator();
                        
                        ui.label("🖱️ Mouse Interaction:");
                        ui.label("  • Left Click - Repel particles from cursor");
                        ui.label("  • Right Click - Attract particles to cursor");
                        ui.label("  • Adjust cursor size and strength in main panel");
                        
                        ui.separator();
                        
                        ui.label("🎛️ Matrix Editing:");
                        ui.label("  Drag matrix values - Fine-tune interactions");
                        ui.label("  Double-click matrix values - Type exact number");
                        ui.label("  Green = Attraction, Red = Repulsion");
                        
                        ui.separator();
                        
                        ui.label("💡 Tips:");
                        ui.small("• Traces automatically disable during camera movement");
                        ui.small("• Use different generators for interesting patterns");
                        ui.small("• Try various position setters for different layouts");
                        ui.small("• Adjust particle size with Shift+Scroll for better visibility");
                    });
            }

            // About window
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
                        
                        ui.label("🧬 Features:");
                        ui.label("  • Real-time particle physics simulation");
                        ui.label("  • Interactive matrix editing");
                        ui.label("  • Multiple particle types and colors");
                        ui.label("  • Adjustable physics parameters");
                        ui.label("  • Smooth camera controls");
                        
                        ui.separator();
                        
                        ui.small("Originally inspired by the Java Particle Life project");
                        ui.small("Converted with Claude Code");
                    });
            }
        });

        // Update actual traces state based on user setting and camera movement
        if !self.camera_is_moving {
            self.traces = self.traces_user_enabled;
        }

        self.egui_state.handle_platform_output(&self.window, full_output.platform_output);

        let tris = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(&self.device, &self.queue, *id, image_delta);
        }

        self.egui_renderer.update_buffers(&self.device, &self.queue, &mut encoder, &tris, &egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        });

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        self.egui_renderer.render(&mut render_pass, &tris, &egui_wgpu::ScreenDescriptor {
            size_in_pixels: [self.config.width, self.config.height],
            pixels_per_point: self.window.scale_factor() as f32,
        });

        drop(render_pass);

        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

}

#[tokio::main]
async fn main() {
    env_logger::init();
    
    let event_loop = EventLoop::new().unwrap();
    let mut app = Application::new(&event_loop).await;
    app.setup();
    
    let mut last_render_time = Instant::now();
    
    event_loop.run(move |event, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                control_flow.exit();
            }
            Event::WindowEvent { event, .. } => {
                match event {
                    WindowEvent::RedrawRequested => {
                        let now = Instant::now();
                        let dt = now.duration_since(last_render_time).as_secs_f64();
                        last_render_time = now;
                        
                        app.update(dt);
                        
                        match app.render() {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost) => {
                                // Reconfigure surface
                                app.surface.configure(&app.device, &app.config);
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                control_flow.exit();
                            }
                            Err(e) => {
                                eprintln!("Render error: {:?}", e);
                            }
                        }
                    }
                    _ => {
                        let _handled = app.handle_event(&event);
                    }
                }
            }
            Event::AboutToWait => {
                app.window.request_redraw();
            }
            _ => {}
        }
    }).unwrap();
}