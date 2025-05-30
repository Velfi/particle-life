mod particle;
mod shaders;

use std::sync::Arc;
use winit::{
    event::{ElementState, Event, KeyEvent, WindowEvent},
    event_loop::EventLoop,
    keyboard::{PhysicalKey, KeyCode},
    window::WindowBuilder,
};
use std::time::{Duration, Instant};

use particle::ParticleSystem;

struct State<'window> {
    window: Arc<winit::window::Window>,
    surface: wgpu::Surface<'window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    clear_color: wgpu::Color,
    particle_system: ParticleSystem,
    frame_count: u32,
    last_fps_update: Instant,
    last_frame_time: Instant,
}

impl<'window> State<'window> {
    async fn new(window: Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
            flags: wgpu::InstanceFlags::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        let surface = instance.create_surface(window.clone()).unwrap();

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
                    required_features: wgpu::Features::VERTEX_WRITABLE_STORAGE,
                    required_limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
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

        let clear_color = wgpu::Color {
            r: 0.1,
            g: 0.1,
            b: 0.1,
            a: 1.0,
        };

        let particle_system = ParticleSystem::new(&device, 100_000);

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            clear_color,
            particle_system,
            frame_count: 0,
            last_fps_update: Instant::now(),
            last_frame_time: Instant::now(),
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { position: _position, .. } => {
                // Handle mouse movement
                true
            }
            WindowEvent::MouseInput { state: _state, button: _button, .. } => {
                // Handle mouse input
                true
            }
            _ => false,
        }
    }

    fn update(&mut self) {
        self.particle_system.update(&self.device, &self.queue);
        
        // Update FPS counter
        self.frame_count += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_fps_update);
        
        // Update FPS display every 500ms
        if elapsed >= Duration::from_millis(500) {
            let fps = self.frame_count as f64 / elapsed.as_secs_f64();
            let frame_time = now.duration_since(self.last_frame_time).as_secs_f64() * 1000.0;
            self.window.set_title(&format!("Particle Life - FPS: {:.1} (Frame Time: {:.1}ms)", fps, frame_time));
            self.frame_count = 0;
            self.last_fps_update = now;
        }
        self.last_frame_time = now;
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

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            self.particle_system.render(&mut render_pass);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new()
        .with_title("Particle Life")
        .with_inner_size(winit::dpi::LogicalSize::new(2400, 1800))
        .build(&event_loop)
        .unwrap());

    let mut state = pollster::block_on(State::new(window.clone()));

    event_loop
        .run(move |event, target| {
            match event {
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == state.window.id() => {
                    if !state.input(event) {
                        match event {
                            WindowEvent::CloseRequested
                            | WindowEvent::KeyboardInput {
                                event:
                                    KeyEvent {
                                        state: ElementState::Pressed,
                                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                                        ..
                                    },
                                ..
                            } => target.exit(),
                            WindowEvent::Resized(physical_size) => {
                                state.resize(*physical_size);
                            }
                            WindowEvent::ScaleFactorChanged { scale_factor: _scale_factor, .. } => {
                                let new_size = state.window.inner_size();
                                state.resize(new_size);
                            }
                            _ => {}
                        }
                    }
                }
                Event::AboutToWait => {
                    state.update();
                    match state.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                        Err(e) => eprintln!("{:?}", e),
                    }
                    state.window.request_redraw();
                }
                _ => {}
            }
        })
        .unwrap();
}
