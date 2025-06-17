use crate::physics::Position;
use crate::shaders::{ParticleShader, ParticleVertex};
use crate::lut_manager::LutData;
use nalgebra::{Matrix4, Vector3};
use wgpu::{Buffer, BufferUsages, Device};

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
        lut: &LutColorMapper,
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
            [1.0, -1.0],  // bottom-right
            [-1.0, 1.0],  // top-left
            [1.0, -1.0],  // bottom-right (shared)
            [1.0, 1.0],   // top-right
            [-1.0, 1.0],  // top-left (shared)
        ];

        let mut vertices = Vec::with_capacity(positions.len() * 6);
        for ((pos, _vel), &particle_type) in
            positions.iter().zip(velocities.iter()).zip(types.iter())
        {
            let color = lut.get_particle_color(particle_type, max_types);

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
                queue.write_buffer(buffer, 0, bytemuck::cast_slice(&vertices));
            }
        }
    }

    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        shader: &'a ParticleShader,
        particle_count: usize,
        show_tiling: bool,
    ) {
        if let Some(ref vertex_buffer) = self.vertex_buffer {
            render_pass.set_pipeline(&shader.render_pipeline);
            render_pass.set_bind_group(0, &shader.bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));

            // If tiling is enabled, draw 9 instances (3x3 grid), otherwise draw 1 instance
            let instance_count = if show_tiling { 9 } else { 1 };

            // Draw 6 vertices per particle (2 triangles making a quad)
            render_pass.draw(0..(particle_count * 6) as u32, 0..instance_count);
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
    pub fn from_lut_bytes(r: u8, g: u8, b: u8) -> Self {
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: 1.0,
        }
    }
}

/// LUT-based color mapper that samples equidistant points along the LUT
/// The first sample is used as background color, subsequent samples are particle colors
pub struct LutColorMapper {
    lut_data: LutData,
    background_color: Color,
    particle_colors: Vec<Color>,
    reversed: bool,
}

impl LutColorMapper {
    /// Creates a new LUT color mapper from LUT data
    /// Samples n = num_species + 1 equidistant stops along the LUT
    /// First stop is background color, other stops are particle colors
    pub fn new(lut_data: LutData, num_species: usize) -> Self {
        let n = num_species + 1;
        let mut colors = Vec::with_capacity(n);
        
        // Sample n equidistant points along the LUT (0 to 255)
        for i in 0..n {
            let index = if n == 1 {
                0
            } else {
                (i * 255) / (n - 1)
            };
            let index = index.min(255);
            
            let color = Color::from_lut_bytes(
                lut_data.red[index],
                lut_data.green[index],
                lut_data.blue[index],
            );
            colors.push(color);
        }
        
        let background_color = colors[0];
        let particle_colors = if colors.len() > 1 {
            colors[1..].to_vec()
        } else {
            vec![colors[0]] // Fallback if only one color
        };
        
        Self {
            lut_data,
            background_color,
            particle_colors,
            reversed: false,
        }
    }
    
    /// Gets a particle color by type index
    pub fn get_particle_color(&self, type_index: usize, total_types: usize) -> Color {
        if self.particle_colors.is_empty() {
            self.background_color
        } else {
            // If the number of types has changed, we need to regenerate colors
            if total_types != self.particle_colors.len() {
                // This is a temporary fix - the actual regeneration should happen elsewhere
                // For now, we'll just use modulo to cycle through available colors
                self.particle_colors[type_index % self.particle_colors.len()]
            } else {
                self.particle_colors[type_index]
            }
        }
    }
    
    /// Gets the background color from the LUT
    pub fn get_background_color(&self) -> Color {
        self.background_color
    }
    
    /// Gets the name of the LUT
    pub fn name(&self) -> &str {
        &self.lut_data.name
    }
    
    /// Gets whether the LUT is reversed
    pub fn is_reversed(&self) -> bool {
        self.reversed
    }
    
    /// Sets whether the LUT should be reversed
    pub fn set_reversed(&mut self, reversed: bool) {
        if self.reversed != reversed {
            self.reversed = reversed;
            // Regenerate colors with current species count
            let num_species = self.particle_colors.len();
            self.regenerate_colors(num_species);
        }
    }
    
    /// Regenerates colors based on current settings
    fn regenerate_colors(&mut self, num_species: usize) {
        let n = num_species + 1;
        let mut colors = Vec::with_capacity(n);
        
        // Sample n equidistant points along the LUT (0 to 255)
        for i in 0..n {
            let index = if n == 1 {
                0
            } else {
                if self.reversed {
                    // Reverse the sampling order
                    255 - (i * 255) / (n - 1)
                } else {
                    (i * 255) / (n - 1)
                }
            };
            let index = index.min(255);
            
            let color = Color::from_lut_bytes(
                self.lut_data.red[index],
                self.lut_data.green[index],
                self.lut_data.blue[index],
            );
            colors.push(color);
        }
        
        self.background_color = colors[0];
        self.particle_colors = if colors.len() > 1 {
            colors[1..].to_vec()
        } else {
            vec![colors[0]] // Fallback if only one color
        };
    }

    /// Updates the number of species and regenerates colors accordingly
    pub fn update_species_count(&mut self, num_species: usize) {
        self.regenerate_colors(num_species);
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

    pub fn zoom(
        &mut self,
        mouse_x: f64,
        mouse_y: f64,
        zoom_factor: f64,
        scale_factor: f64,
        logical_width: u32,
        logical_height: u32,
    ) {
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
            let new_y = self.target_position.y - mouse_y_norm * size_diff / aspect_ratio;

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
        let max_camera_x = world_max + max_distance; // = 4.0
        let min_camera_y = world_min - max_distance; // = -4.0
        let max_camera_y = world_max + max_distance; // = 4.0

        // Clamp camera position to strict bounds
        self.target_position.x = new_x.clamp(min_camera_x, max_camera_x);
        self.target_position.y = new_y.clamp(min_camera_y, max_camera_y);
    }

    pub fn reset(&mut self, fit_to_window: bool, aspect_ratio: f32) {
        self.target_position = Vector3::new(0.0, 0.0, 0.0);
        if fit_to_window {
            self.target_size =
                (8.0_f32.min(aspect_ratio * 8.0) as f64).min(8.0 / aspect_ratio as f64);
        } else {
            self.target_size = 8.0;
        }
    }
}
