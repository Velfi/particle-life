mod app_settings;
mod physics;
mod rendering;
mod shaders;
mod ui;
mod ui_renderer;

use egui_winit::State;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use wgpu::{Surface, SurfaceConfiguration};
use winit::{
    event::{ElementState, Event, KeyEvent, MouseButton, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::WindowBuilder,
};

use app_settings::AppSettings;
use physics::{
    CenterPositionSetter,
    CenteredCirclePositionSetter,
    Chains2MatrixGenerator,
    Chains3MatrixGenerator,
    ChainsMatrixGenerator,
    ColorBattlePositionSetter,
    ColorWheelPositionSetter,
    DefaultMatrixGenerator,
    DefaultPositionSetter,
    ExtendedPhysics,
    FlipTypeSetter,
    KillStillTypeSetter,
    LinePositionSetter,
    MoreOfFirstTypeSetter,
    OnionTypeSetter,
    PhysicsSnapshot,
    RainbowRingPositionSetter,
    RainbowSpiralPositionSetter,
    RandomPositionSetter,
    RandomTypeSetter,
    // Type setters
    Randomize10PercentTypeSetter,
    RingPositionSetter,
    RotateTypeSetter,
    SlicesTypeSetter,
    SnakesMatrixGenerator,
    SpiralPositionSetter,
    // Matrix generators
    SymmetryMatrixGenerator,
    // Position setters
    UniformCirclePositionSetter,
    ZeroMatrixGenerator,
};
use rendering::{
    Camera, HexPalette, Palette, ParticleRenderer,
};
use shaders::{FadeShader, ParticleShader, UniformData};
use ui::{Clock, Loop, SelectionManager};

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
    traces_user_enabled: bool, // User's actual traces setting
    camera_is_moving: bool,    // Whether camera is currently being moved
    camera_movement_timer: std::time::Instant, // Timer to detect when movement stops
    prev_camera_position: nalgebra::Vector3<f64>, // Previous camera position for movement detection
    prev_camera_size: f64,     // Previous camera size for zoom detection

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
        let window = Self::create_window(event_loop);
        let (device, queue, surface, config) = Self::init_wgpu(&window).await;
        let (egui_ctx, egui_state, mut egui_renderer) =
            Self::init_egui(&window, &device, config.format);
        let (particle_renderer, particle_shader, fade_shader, camera) =
            Self::init_rendering(&device, config.format);
        let (world_texture, world_texture_view, world_texture_id) =
            Self::create_world_texture(&device, &config, &mut egui_renderer);
        let (physics, physics_snapshot, physics_loop, new_snapshot_available) =
            Self::init_physics();
        let (palettes, position_setters, matrix_generators, type_setters) =
            Self::init_selection_managers();
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
            local_matrix: vec![vec![0.0; 4]; 4],
            palettes,
            position_setters,
            matrix_generators,
            type_setters,
            render_clock: Clock::new(60.0),
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

    fn create_window(event_loop: &EventLoop<()>) -> Arc<winit::window::Window> {
        Arc::new(
            WindowBuilder::new()
                .with_title("Particle Life Simulator")
                .with_inner_size(winit::dpi::LogicalSize::new(1200.0, 800.0))
                .build(event_loop)
                .unwrap(),
        )
    }

    async fn init_wgpu(
        window: &Arc<winit::window::Window>,
    ) -> (
        Arc<wgpu::Device>,
        Arc<wgpu::Queue>,
        Arc<Surface<'static>>,
        SurfaceConfiguration,
    ) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
            flags: wgpu::InstanceFlags::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        let surface = Arc::new(instance.create_surface(window.clone()).unwrap());

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_buffer_size: 1024 * 1024 * 1024,
                        ..wgpu::Limits::default()
                    },
                },
                None,
            )
            .await
            .unwrap();

        let device = Arc::new(device);
        let queue = Arc::new(queue);

        let size = window.inner_size();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
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

        (device, queue, surface, config)
    }

    fn init_egui(
        window: &Arc<winit::window::Window>,
        device: &Arc<wgpu::Device>,
        surface_format: wgpu::TextureFormat,
    ) -> (egui::Context, State, egui_wgpu::Renderer) {
        let egui_ctx = egui::Context::default();
        let egui_state = State::new(egui_ctx.clone(), egui::ViewportId::ROOT, window, None, None);
        let egui_renderer = egui_wgpu::Renderer::new(device, surface_format, None, 1);
        (egui_ctx, egui_state, egui_renderer)
    }

    fn init_rendering(
        device: &Arc<wgpu::Device>,
        surface_format: wgpu::TextureFormat,
    ) -> (ParticleRenderer, ParticleShader, FadeShader, Camera) {
        let particle_renderer = ParticleRenderer::new();
        let particle_shader = ParticleShader::new(device, surface_format);
        let fade_shader = FadeShader::new(device, surface_format);
        let camera = Camera::new();
        (particle_renderer, particle_shader, fade_shader, camera)
    }

    fn create_world_texture(
        device: &Arc<wgpu::Device>,
        config: &SurfaceConfiguration,
        egui_renderer: &mut egui_wgpu::Renderer,
    ) -> (wgpu::Texture, wgpu::TextureView, egui::TextureId) {
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
            format: config.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let world_texture_view = world_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let world_texture_id = egui_renderer.register_native_texture(
            device,
            &world_texture_view,
            wgpu::FilterMode::Linear,
        );
        (world_texture, world_texture_view, world_texture_id)
    }

    fn init_physics() -> (ExtendedPhysics, PhysicsSnapshot, Loop, Arc<Mutex<bool>>) {
        let physics = ExtendedPhysics::new(
            Box::new(DefaultPositionSetter),
            Box::new(DefaultMatrixGenerator),
        );
        let physics_snapshot = PhysicsSnapshot::new();
        let physics_loop = Loop::new();
        let new_snapshot_available = Arc::new(Mutex::new(false));
        (
            physics,
            physics_snapshot,
            physics_loop,
            new_snapshot_available,
        )
    }

    #[allow(clippy::type_complexity)]
    fn init_selection_managers() -> (
        SelectionManager<Box<dyn rendering::Palette>>,
        SelectionManager<Box<dyn physics::PositionSetter>>,
        SelectionManager<Box<dyn physics::MatrixGenerator>>,
        SelectionManager<Box<dyn physics::TypeSetter>>,
    ) {
        let palettes = Self::init_palettes();
        let position_setters = Self::init_position_setters();
        let matrix_generators = Self::init_matrix_generators();
        let type_setters = Self::init_type_setters();
        (palettes, position_setters, matrix_generators, type_setters)
    }

    fn init_palettes() -> SelectionManager<Box<dyn rendering::Palette>> {
        let mut palette_list = Vec::new();

        // Load Endesga-8 palette first to make it default
        if let Ok(hex_palette) = HexPalette::load_from_file("palettes/endesga-8.hex") {
            let name = hex_palette.name().to_string();
            palette_list.push((name, Box::new(hex_palette) as Box<dyn rendering::Palette>));
        }

        // Load other hex palettes
        if let Ok(entries) = std::fs::read_dir("palettes") {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("hex") {
                    // Skip endesga-8.hex as we already loaded it
                    if path.file_name().unwrap() == "endesga-8.hex" {
                        continue;
                    }
                    if let Ok(hex_palette) = HexPalette::load_from_file(path.to_str().unwrap()) {
                        let name = hex_palette.name().to_string();
                        palette_list.push((name, Box::new(hex_palette) as Box<dyn rendering::Palette>));
                    }
                }
            }
        }

        SelectionManager::new(palette_list)
    }

    fn init_position_setters() -> SelectionManager<Box<dyn physics::PositionSetter>> {
        SelectionManager::new(vec![
            (
                "Centered".to_string(),
                Box::new(CenterPositionSetter) as Box<dyn physics::PositionSetter>,
            ),
            (
                "Uniform".to_string(),
                Box::new(RandomPositionSetter) as Box<dyn physics::PositionSetter>,
            ),
            (
                "Uniform Circle".to_string(),
                Box::new(UniformCirclePositionSetter) as Box<dyn physics::PositionSetter>,
            ),
            (
                "Centered Circle".to_string(),
                Box::new(CenteredCirclePositionSetter) as Box<dyn physics::PositionSetter>,
            ),
            (
                "Ring".to_string(),
                Box::new(RingPositionSetter) as Box<dyn physics::PositionSetter>,
            ),
            (
                "Rainbow Ring".to_string(),
                Box::new(RainbowRingPositionSetter) as Box<dyn physics::PositionSetter>,
            ),
            (
                "Color Battle".to_string(),
                Box::new(ColorBattlePositionSetter) as Box<dyn physics::PositionSetter>,
            ),
            (
                "Color Wheel".to_string(),
                Box::new(ColorWheelPositionSetter) as Box<dyn physics::PositionSetter>,
            ),
            (
                "Line".to_string(),
                Box::new(LinePositionSetter) as Box<dyn physics::PositionSetter>,
            ),
            (
                "Spiral".to_string(),
                Box::new(SpiralPositionSetter) as Box<dyn physics::PositionSetter>,
            ),
            (
                "Rainbow Spiral".to_string(),
                Box::new(RainbowSpiralPositionSetter) as Box<dyn physics::PositionSetter>,
            ),
        ])
    }

    fn init_matrix_generators() -> SelectionManager<Box<dyn physics::MatrixGenerator>> {
        SelectionManager::new(vec![
            (
                "Random".to_string(),
                Box::new(DefaultMatrixGenerator) as Box<dyn physics::MatrixGenerator>,
            ),
            (
                "Symmetry".to_string(),
                Box::new(SymmetryMatrixGenerator) as Box<dyn physics::MatrixGenerator>,
            ),
            (
                "Chains".to_string(),
                Box::new(ChainsMatrixGenerator) as Box<dyn physics::MatrixGenerator>,
            ),
            (
                "Chains 2".to_string(),
                Box::new(Chains2MatrixGenerator) as Box<dyn physics::MatrixGenerator>,
            ),
            (
                "Chains 3".to_string(),
                Box::new(Chains3MatrixGenerator) as Box<dyn physics::MatrixGenerator>,
            ),
            (
                "Snakes".to_string(),
                Box::new(SnakesMatrixGenerator) as Box<dyn physics::MatrixGenerator>,
            ),
            (
                "Zero".to_string(),
                Box::new(ZeroMatrixGenerator) as Box<dyn physics::MatrixGenerator>,
            ),
        ])
    }

    fn init_type_setters() -> SelectionManager<Box<dyn physics::TypeSetter>> {
        SelectionManager::new(vec![
            (
                "Random".to_string(),
                Box::new(RandomTypeSetter) as Box<dyn physics::TypeSetter>,
            ),
            (
                "Randomize 10%".to_string(),
                Box::new(Randomize10PercentTypeSetter) as Box<dyn physics::TypeSetter>,
            ),
            (
                "Slices".to_string(),
                Box::new(SlicesTypeSetter) as Box<dyn physics::TypeSetter>,
            ),
            (
                "Onion".to_string(),
                Box::new(OnionTypeSetter) as Box<dyn physics::TypeSetter>,
            ),
            (
                "Rotate".to_string(),
                Box::new(RotateTypeSetter) as Box<dyn physics::TypeSetter>,
            ),
            (
                "Flip".to_string(),
                Box::new(FlipTypeSetter) as Box<dyn physics::TypeSetter>,
            ),
            (
                "More of First".to_string(),
                Box::new(MoreOfFirstTypeSetter) as Box<dyn physics::TypeSetter>,
            ),
            (
                "Kill Still".to_string(),
                Box::new(KillStillTypeSetter) as Box<dyn physics::TypeSetter>,
            ),
        ])
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
        self.local_matrix
            .iter_mut()
            .enumerate()
            .for_each(|(i, row)| {
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

        println!(
            "Initialized {} particles with {} types",
            self.physics_snapshot.particle_count,
            self.physics.matrix.size()
        );
    }

    fn handle_event(&mut self, event: &WindowEvent) -> bool {
        let response = self.egui_state.on_window_event(&self.window, event);
        if response.consumed {
            return true;
        }

        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key_code),
                        state,
                        ..
                    },
                ..
            } => {
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
                    (MouseButton::Right, ElementState::Released) => {
                        self.right_mouse_pressed = false
                    }
                    (MouseButton::Middle, ElementState::Pressed) => {
                        self.middle_mouse_pressed = true
                    }
                    (MouseButton::Middle, ElementState::Released) => {
                        self.middle_mouse_pressed = false
                    }
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
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                self.world_texture_view = self
                    .world_texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                // Update egui texture
                self.world_texture_id = self.egui_renderer.register_native_texture(
                    &self.device,
                    &self.world_texture_view,
                    wgpu::FilterMode::Linear,
                );

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
            }
            KeyCode::Space => self.physics_loop.pause = !self.physics_loop.pause,
            KeyCode::KeyP => self
                .physics
                .set_positions_with_setter(self.position_setters.get_active().as_ref()),
            KeyCode::KeyC => self
                .physics
                .set_types_with_setter(self.type_setters.get_active().as_ref()),
            KeyCode::KeyM => {
                self.physics
                    .generate_matrix_with_generator(self.matrix_generators.get_active().as_ref());
                // Update local matrix copy for UI
                let matrix_size = self.physics.matrix.size();
                for i in 0..matrix_size {
                    for j in 0..matrix_size {
                        self.local_matrix[i][j] = self.physics.matrix.get(i, j);
                    }
                }
            }
            KeyCode::KeyB => self.physics.settings.wrap = !self.physics.settings.wrap,
            KeyCode::KeyZ => {
                if self.keys_pressed.contains(&KeyCode::ShiftLeft)
                    || self.keys_pressed.contains(&KeyCode::ShiftRight)
                {
                    self.camera
                        .reset(true, self.config.width as f32 / self.config.height as f32);
                } else {
                    self.camera
                        .reset(false, self.config.width as f32 / self.config.height as f32);
                }
            }
            _ => {}
        }
    }

    fn handle_scroll(&mut self, delta_y: f64) {
        let ctrl_pressed = self.keys_pressed.contains(&KeyCode::ControlLeft)
            || self.keys_pressed.contains(&KeyCode::ControlRight);
        let shift_pressed = self.keys_pressed.contains(&KeyCode::ShiftLeft)
            || self.keys_pressed.contains(&KeyCode::ShiftRight);

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
        self.cursor_world_position.y =
            self.camera.position.y + mouse_y_norm * self.camera.size / aspect_ratio;
        self.cursor_world_position.z = 0.0;
    }

    fn update(&mut self, dt: f64) {
        self.render_clock.tick();

        // Handle camera input (this updates target position/size)
        // Base movement speed independent of zoom level
        let base_movement_speed = self.app_settings.camera_movement_speed as f64 * dt;

        // Keyboard camera movement
        if self.keys_pressed.contains(&KeyCode::KeyA)
            || self.keys_pressed.contains(&KeyCode::ArrowLeft)
        {
            self.camera.pan(-base_movement_speed, 0.0);
        }
        if self.keys_pressed.contains(&KeyCode::KeyD)
            || self.keys_pressed.contains(&KeyCode::ArrowRight)
        {
            self.camera.pan(base_movement_speed, 0.0);
        }
        if self.keys_pressed.contains(&KeyCode::KeyW)
            || self.keys_pressed.contains(&KeyCode::ArrowUp)
        {
            self.camera.pan(0.0, -base_movement_speed);
        }
        if self.keys_pressed.contains(&KeyCode::KeyS)
            || self.keys_pressed.contains(&KeyCode::ArrowDown)
        {
            self.camera.pan(0.0, base_movement_speed);
        }

        // Mouse panning removed - only keyboard panning and mouse zooming supported

        // Reset cursor movement flag for next frame
        self.cursor_moved_last_frame = false;

        // Update camera (applies smoothing to move towards target)
        self.camera
            .update(self.app_settings.camera_smoothness as f64);

        // Check for actual camera movement by comparing current position to previous
        let position_delta = (self.camera.position - self.prev_camera_position).magnitude();
        let size_delta = (self.camera.size - self.prev_camera_size).abs();
        let movement_threshold = 0.0001; // Adjust this threshold as needed

        let camera_actually_moving =
            position_delta > movement_threshold || size_delta > movement_threshold;

        if camera_actually_moving {
            self.camera_is_moving = true;
            self.camera_movement_timer = std::time::Instant::now();
            // Temporarily disable trails during movement
            if self.traces_user_enabled {
                self.traces = false;
            }
        } else if self.camera_is_moving {
            // Check if camera has stopped moving for a brief period
            if self.camera_movement_timer.elapsed().as_millis() > 150 {
                // 150ms delay to account for smoothing
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
                let strength = if self.left_mouse_pressed {
                    -self.cursor_strength
                } else {
                    self.cursor_strength
                };
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
            self.physics_time_avg = self.physics_time_samples.iter().sum::<f64>()
                / self.physics_time_samples.len() as f64;

            // Debug output every few seconds to verify simulation is running
            if self.last_debug_time.elapsed().as_secs() >= 5 {
                let start_time = std::time::Instant::now();

                // Measure physics update time
                let particle_count = self.physics.particles.len();
                let avg_velocity = if particle_count > 0 {
                    self.physics
                        .particles
                        .iter()
                        .map(|p| p.velocity.norm())
                        .sum::<f64>()
                        / particle_count as f64
                } else {
                    0.0
                };

                let physics_time = start_time.elapsed();

                println!(
                    "Physics: {} particles, avg velocity: {:.6}, calc time: {:.2}ms",
                    particle_count,
                    avg_velocity,
                    physics_time.as_micros() as f64 / 1000.0
                );
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
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
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
                            wgpu::LoadOp::Load // Always load for fading
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
            let view_proj_matrix = self
                .camera
                .get_view_projection_matrix(self.config.width as f32 / self.config.height as f32);

            // Apply fade effect when traces are enabled
            if self.traces {
                self.fade_shader
                    .update_uniforms(&self.queue, self.trace_fade);
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

            self.particle_shader
                .update_uniforms(&self.queue, &uniform_data);
            self.particle_renderer.render(
                &mut render_pass,
                &self.particle_shader,
                self.physics_snapshot.particle_count,
                true,
            );
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
            ui_renderer::render_ui(
                ctx,
                world_texture_id,
                config_width,
                config_height,
                show_gui,
                show_graphics_window,
                show_controls_window,
                show_about_window,
                tile_fade_strength,
                traces_user_enabled,
                camera_is_moving,
                physics_loop_pause,
                physics_snapshot,
                render_clock,
                _app_settings,
                _palettes,
                _position_setters,
                _type_setters,
                _matrix_generators,
                local_matrix,
                physics,
                physics_time_avg,
                trace_fade,
                cursor_size,
                cursor_strength,
            );
        });

        // Update actual traces state based on user setting and camera movement
        if !self.camera_is_moving {
            self.traces = self.traces_user_enabled;
        }

        self.egui_state
            .handle_platform_output(&self.window, full_output.platform_output);

        let tris = self
            .egui_ctx
            .tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }

        self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &tris,
            &egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point: self.window.scale_factor() as f32,
            },
        );

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

        self.egui_renderer.render(
            &mut render_pass,
            &tris,
            &egui_wgpu::ScreenDescriptor {
                size_in_pixels: [self.config.width, self.config.height],
                pixels_per_point: self.window.scale_factor() as f32,
            },
        );

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

    event_loop
        .run(move |event, control_flow| {
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
        })
        .unwrap();
}
