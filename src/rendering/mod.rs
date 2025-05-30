use wgpu::{Device, Buffer, BufferUsages};
use nalgebra::{Matrix4, Vector3};
use crate::shaders::{ParticleShader, ParticleVertex, hsv_to_rgb};
use crate::physics::Position;

pub struct ParticleRenderer {
    vertex_buffer: Option<Buffer>,
    vertex_capacity: usize,
}

impl ParticleRenderer {
    pub fn new() -> Self {
        Self {
            vertex_buffer: None,
            vertex_capacity: 0,
        }
    }

    pub fn buffer_particle_data(
        &mut self,
        device: &Device,
        queue: &wgpu::Queue,
        positions: &[Position],
        velocities: &[nalgebra::Vector3<f64>],
        types: &[usize],
        palette: &dyn Palette,
    ) {
        if positions.is_empty() {
            return;
        }

        let vertex_count = positions.len() * 6; // 6 vertices per particle
        
        // Resize buffer if needed
        if self.vertex_capacity < vertex_count {
            self.vertex_capacity = (vertex_count * 2).max(6000); // Account for 6 vertices per particle
            self.vertex_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Particle Vertex Buffer"),
                size: (self.vertex_capacity * std::mem::size_of::<ParticleVertex>()) as u64,
                usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        // Create vertex data - 6 vertices per particle (2 triangles making a quad)
        let max_types = types.iter().max().copied().unwrap_or(0) + 1;
        let quad_vertices = [
            [-1.0, -1.0], // bottom-left
            [ 1.0, -1.0], // bottom-right 
            [-1.0,  1.0], // top-left
            [ 1.0, -1.0], // bottom-right (shared)
            [ 1.0,  1.0], // top-right
            [-1.0,  1.0], // top-left (shared)
        ];
        
        let mut vertices = Vec::with_capacity(positions.len() * 6);
        for ((pos, _vel), &particle_type) in positions.iter()
            .zip(velocities.iter())
            .zip(types.iter())
        {
            let color = palette.get_color(particle_type, max_types);
            
            // Create 6 vertices for this particle (2 triangles)
            for &quad_vertex in &quad_vertices {
                vertices.push(ParticleVertex {
                    position: [pos.x as f32, pos.y as f32, pos.z as f32],
                    color: [color.r, color.g, color.b, color.a],
                    size: 1.0,
                    quad_vertex,
                    _padding: 0.0,
                });
            }
        }

        // Upload to GPU
        if let Some(ref buffer) = self.vertex_buffer {
            if !vertices.is_empty() {
                queue.write_buffer(
                    buffer,
                    0,
                    bytemuck::cast_slice(&vertices),
                );
            }
        }
    }

    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        shader: &'a ParticleShader,
        particle_count: usize,
    ) {
        if let Some(ref vertex_buffer) = self.vertex_buffer {
            render_pass.set_pipeline(&shader.render_pipeline);
            render_pass.set_bind_group(0, &shader.bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            // Draw 6 vertices per particle (2 triangles making a quad)
            render_pass.draw(0..(particle_count * 6) as u32, 0..1);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub fn from_hsv(h: f32, s: f32, v: f32) -> Self {
        let rgb = hsv_to_rgb(h, s, v);
        Self::new(rgb[0], rgb[1], rgb[2], 1.0)
    }
}

pub trait Palette: Send + Sync {
    fn get_color(&self, type_index: usize, total_types: usize) -> Color;
    fn name(&self) -> &str;
}

pub struct NaturalRainbowPalette;

impl Palette for NaturalRainbowPalette {
    fn get_color(&self, type_index: usize, total_types: usize) -> Color {
        let hue = (type_index as f32 / total_types.max(1) as f32) * 360.0;
        Color::from_hsv(hue, 0.7, 0.9)
    }

    fn name(&self) -> &str {
        "Natural Rainbow"
    }
}

pub struct SimpleRainbowPalette;

impl Palette for SimpleRainbowPalette {
    fn get_color(&self, type_index: usize, total_types: usize) -> Color {
        let hue = (type_index as f32 / total_types.max(1) as f32) * 360.0;
        Color::from_hsv(hue, 1.0, 1.0)
    }

    fn name(&self) -> &str {
        "Simple Rainbow"
    }
}

pub struct SunsetPalette;

impl Palette for SunsetPalette {
    fn get_color(&self, type_index: usize, total_types: usize) -> Color {
        let t = type_index as f32 / total_types.max(1) as f32;
        // Transition from red to orange to yellow
        let hue = t * 60.0; // 0 to 60 degrees (red to yellow)
        Color::from_hsv(hue, 0.8, 0.9)
    }

    fn name(&self) -> &str {
        "Sunset"
    }
}

pub struct HexPalette {
    colors: Vec<Color>,
    name: String,
}

impl HexPalette {
    pub fn load_from_file(file_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(file_path)?;
        let mut colors = Vec::new();
        
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            
            // Parse hex color (expecting 6 characters: RRGGBB)
            if line.len() != 6 {
                continue;
            }
            
            let r = u8::from_str_radix(&line[0..2], 16)? as f32 / 255.0;
            let g = u8::from_str_radix(&line[2..4], 16)? as f32 / 255.0;
            let b = u8::from_str_radix(&line[4..6], 16)? as f32 / 255.0;
            
            colors.push(Color { r, g, b, a: 1.0 });
        }
        
        if colors.is_empty() {
            return Err("No valid colors found in palette file".into());
        }
        
        // Extract name from file path
        let file_name = std::path::Path::new(file_path)
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("Unknown")
            .to_string();
        
        Ok(HexPalette {
            colors,
            name: file_name,
        })
    }
}

impl Palette for HexPalette {
    fn get_color(&self, type_index: usize, _total_types: usize) -> Color {
        self.colors[type_index % self.colors.len()]
    }

    fn name(&self) -> &str {
        &self.name
    }
}

pub struct Camera {
    pub position: Vector3<f64>,
    pub size: f64,
    pub target_position: Vector3<f64>,
    pub target_size: f64,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            position: Vector3::new(0.0, 0.0, 0.0),
            size: 8.0, // Increased to show larger world by default
            target_position: Vector3::new(0.0, 0.0, 0.0),
            target_size: 8.0,
        }
    }

    pub fn get_view_projection_matrix(&self, aspect_ratio: f32) -> Matrix4<f32> {
        let left = (self.position.x - self.size * 0.5) as f32;
        let right = (self.position.x + self.size * 0.5) as f32;
        let bottom = (self.position.y + self.size * 0.5 / aspect_ratio as f64) as f32;
        let top = (self.position.y - self.size * 0.5 / aspect_ratio as f64) as f32;

        Matrix4::new_orthographic(left, right, bottom, top, -1.0, 1.0)
    }

    pub fn update(&mut self, smoothness: f64) {
        self.position = self.position.lerp(&self.target_position, smoothness);
        self.size = self.size + (self.target_size - self.size) * smoothness;
    }

    pub fn zoom(&mut self, mouse_x: f64, mouse_y: f64, zoom_factor: f64, scale_factor: f64, logical_width: u32, logical_height: u32) {
        // Convert mouse position to world coordinates using DPI-aware logic
        let aspect_ratio = logical_width as f64 / logical_height as f64;
        
        // Mouse coordinates are in physical pixels, convert to logical
        let logical_mouse_x = mouse_x / scale_factor;
        let logical_mouse_y = mouse_y / scale_factor;
        
        // Normalize mouse coordinates to [-0.5, 0.5] using logical size
        let mouse_x_norm = (logical_mouse_x / logical_width as f64) - 0.5;
        let mouse_y_norm = (logical_mouse_y / logical_height as f64) - 0.5;

        // Calculate new size with zoom limits
        let new_size = self.target_size * zoom_factor;
        
        // Zoom limits: min 0.01 (very zoomed in), max 20.0 (much further zoomed out)
        let min_zoom = 0.01;
        let max_zoom = 20.0;
        let clamped_size = new_size.clamp(min_zoom, max_zoom);
        
        // Only proceed if zoom is within limits
        if clamped_size != self.target_size {
            let size_diff = clamped_size - self.target_size;

            // Calculate new position to zoom towards mouse
            let new_x = self.target_position.x - mouse_x_norm * size_diff;
            let new_y = self.target_position.y - mouse_y_norm * size_diff / aspect_ratio as f64;

            // Apply strict bounds checking
            self.target_size = clamped_size;
            self.apply_position_bounds(new_x, new_y);
        }
    }

    pub fn pan(&mut self, delta_x: f64, delta_y: f64) {
        // Calculate new position
        let new_x = self.target_position.x + delta_x * self.target_size;
        let new_y = self.target_position.y + delta_y * self.target_size;
        
        // Apply strict bounds checking
        self.apply_position_bounds(new_x, new_y);
    }
    
    fn apply_position_bounds(&mut self, new_x: f64, new_y: f64) {
        // World bounds are -2.0 to 2.0
        let world_min = -2.0;
        let world_max = 2.0;
        let world_size = world_max - world_min; // = 4.0
        
        // Camera should never be more than half the world width from the simulation
        let max_distance = world_size * 0.5; // = 2.0
        
        // Calculate the maximum allowed camera bounds
        let min_camera_x = world_min - max_distance; // = -4.0
        let max_camera_x = world_max + max_distance;  // = 4.0
        let min_camera_y = world_min - max_distance; // = -4.0  
        let max_camera_y = world_max + max_distance;  // = 4.0
        
        // Clamp camera position to strict bounds
        self.target_position.x = new_x.clamp(min_camera_x, max_camera_x);
        self.target_position.y = new_y.clamp(min_camera_y, max_camera_y);
    }

    pub fn reset(&mut self, fit_to_window: bool, aspect_ratio: f32) {
        self.target_position = Vector3::new(0.0, 0.0, 0.0);
        if fit_to_window {
            self.target_size = (8.0_f32.min(aspect_ratio * 8.0) as f64).min(8.0 / aspect_ratio as f64);
        } else {
            self.target_size = 8.0;
        }
    }
}