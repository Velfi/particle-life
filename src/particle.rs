use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use crate::shaders::{COMPUTE_SHADER, RENDER_SHADER};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Particle {
    position: [f32; 2],
    velocity: [f32; 2],
    color: [f32; 4],
    species: u32,
    mass: f32,
    parameters: [f32; 4],
}

impl Particle {
    pub fn new(position: [f32; 2], species: u32) -> Self {
        Self {
            position,
            velocity: [0.0, 0.0],
            color: color_for_species(species),
            species,
            mass: 1.0,  // Initialize with default mass
            parameters: [0.5, 0.5, 0.5, 0.5],  // Initialize with neutral parameters
        }
    }
}

pub struct ParticleSystem {
    particles: Vec<Particle>,
    particle_buffer: wgpu::Buffer,
    compute_bind_group: wgpu::BindGroup,
    render_bind_group: wgpu::BindGroup,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    render_bind_group_layout: wgpu::BindGroupLayout,
    compute_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
}

impl ParticleSystem {
    pub fn new(device: &wgpu::Device, num_particles: usize) -> Self {
        // Calculate maximum particles based on device limits
        let particle_size = std::mem::size_of::<Particle>() as u32;
        let max_buffer_size = device.limits().max_storage_buffer_binding_size;
        let max_particles = (max_buffer_size / particle_size) as usize;
        
        // Use the smaller of requested particles or max supported
        let num_particles = num_particles.min(max_particles);
        
        let mut particles = Vec::with_capacity(num_particles);
        
        // Create particles for each species
        let particles_per_species = num_particles / 9;
        for species in 0..9 {
            for _ in 0..particles_per_species {
                // Generate random position within the bounds
                let x = rand::random::<f32>() * 2400.0;
                let y = rand::random::<f32>() * 1800.0;
                
                let mut p = Particle::new([x, y], species);
                
                // Add small random initial velocity
                p.velocity = [
                    (rand::random::<f32>() - 0.5) * 0.5,
                    (rand::random::<f32>() - 0.5) * 0.5,
                ];
                
                particles.push(p);
            }
        }

        // Create particle buffer
        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Buffer"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Create compute bind group layout (read-write)
        let compute_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("compute_bind_group_layout"),
        });

        // Create render bind group layout (read-only)
        let render_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("render_bind_group_layout"),
        });

        // Create compute bind group
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
            ],
            label: Some("compute_bind_group"),
        });

        // Create render bind group
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &render_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
            ],
            label: Some("render_bind_group"),
        });

        // Create compute pipeline
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(COMPUTE_SHADER.into()),
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Compute Pipeline Layout"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            })),
            module: &compute_shader,
            entry_point: "main",
        });

        // Create render pipeline
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Render Shader"),
            source: wgpu::ShaderSource::Wgsl(RENDER_SHADER.into()),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&render_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Self {
            particles,
            particle_buffer,
            compute_bind_group,
            render_bind_group,
            compute_bind_group_layout,
            render_bind_group_layout,
            compute_pipeline,
            render_pipeline,
        }
    }

    pub fn update(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            compute_pass.dispatch_workgroups((self.particles.len() as u32 + 63) / 64, 1, 1);
        }

        queue.submit(std::iter::once(encoder.finish()));
        device.poll(wgpu::Maintain::Wait);
    }

    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.render_bind_group, &[]);
        render_pass.draw(0..(self.particles.len() as u32 * 6), 0..1);
    }
}

fn color_for_species(species: u32) -> [f32; 4] {
    match species % 9 {
        0 => [1.0, 0.0, 0.0, 1.0],    // Red
        1 => [0.0, 1.0, 0.0, 1.0],    // Green
        2 => [0.0, 0.0, 1.0, 1.0],    // Blue
        3 => [1.0, 1.0, 0.0, 1.0],    // Yellow
        4 => [1.0, 0.0, 1.0, 1.0],    // Magenta
        5 => [0.0, 1.0, 1.0, 1.0],    // Cyan
        6 => [1.0, 0.5, 0.0, 1.0],    // Orange
        7 => [0.5, 0.0, 1.0, 1.0],    // Purple
        8 => [0.0, 0.5, 0.5, 1.0],    // Teal
        _ => [1.0, 1.0, 1.0, 1.0],    // White (fallback)
    }
} 