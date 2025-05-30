use wgpu::{Device, Queue, Buffer, RenderPipeline, VertexBufferLayout, VertexAttribute, BindGroup};
use bytemuck::{Pod, Zeroable};
use crate::rendering::Palette;
use crate::gpu_physics::{GpuParticle, GpuPhysicsSystem};
use crate::shaders::UniformData;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct QuadVertex {
    pub position: [f32; 2],
}

impl QuadVertex {
    const ATTRIBS: [VertexAttribute; 1] = wgpu::vertex_attr_array![
        0 => Float32x2,  // position
    ];

    pub fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<QuadVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ParticleInstance {
    pub world_position: [f32; 3],
    pub color: [f32; 4],
    pub size: f32,
    pub _padding: [f32; 3],
}

impl ParticleInstance {
    const ATTRIBS: [VertexAttribute; 3] = wgpu::vertex_attr_array![
        1 => Float32x3,  // world_position
        2 => Float32x4,  // color
        3 => Float32,    // size
    ];

    pub fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<ParticleInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &Self::ATTRIBS,
        }
    }
}

pub struct InstancedParticleRenderer {
    render_pipeline: RenderPipeline,
    uniform_buffer: Buffer,
    bind_group: BindGroup,
    quad_vertex_buffer: Buffer,
    instance_buffer: Buffer,
    instance_capacity: usize,
}

impl InstancedParticleRenderer {
    pub fn new(device: &Device, format: wgpu::TextureFormat, max_particles: usize) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Instanced Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/particle_instanced.wgsl").into()),
        });

        // Create quad vertices (single quad for all particles)
        let quad_vertices = [
            QuadVertex { position: [-1.0, -1.0] }, // bottom-left
            QuadVertex { position: [ 1.0, -1.0] }, // bottom-right
            QuadVertex { position: [-1.0,  1.0] }, // top-left
            QuadVertex { position: [ 1.0, -1.0] }, // bottom-right (duplicate)
            QuadVertex { position: [ 1.0,  1.0] }, // top-right
            QuadVertex { position: [-1.0,  1.0] }, // top-left (duplicate)
        ];

        let quad_vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Quad Vertex Buffer"),
            size: (quad_vertices.len() * std::mem::size_of::<QuadVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        
        // Map and copy data
        {
            let mut buffer_view = quad_vertex_buffer.slice(..).get_mapped_range_mut();
            buffer_view.copy_from_slice(bytemuck::cast_slice(&quad_vertices));
        }
        quad_vertex_buffer.unmap();

        // Create instance buffer
        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Instance Buffer"),
            size: (max_particles * std::mem::size_of::<ParticleInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Instanced Uniform Buffer"),
            size: std::mem::size_of::<UniformData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Instanced Particle Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Instanced Particle Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Instanced Particle Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Instanced Particle Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[QuadVertex::desc(), ParticleInstance::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
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
            render_pipeline,
            uniform_buffer,
            bind_group,
            quad_vertex_buffer,
            instance_buffer,
            instance_capacity: max_particles,
        }
    }

    pub fn update_from_gpu_buffer(
        &self,
        device: &Device,
        queue: &Queue,
        encoder: &mut wgpu::CommandEncoder,
        gpu_physics: &GpuPhysicsSystem,
        palette: &dyn Palette,
        type_count: usize,
    ) {
        let particle_count = gpu_physics.particle_count();
        if particle_count == 0 {
            return;
        }

        // Create a staging buffer to read particle data from GPU
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Staging Buffer"),
            size: (particle_count * std::mem::size_of::<GpuParticle>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy from GPU buffer to staging buffer
        encoder.copy_buffer_to_buffer(
            gpu_physics.get_particles_buffer(),
            0,
            &staging_buffer,
            0,
            (particle_count * std::mem::size_of::<GpuParticle>()) as u64,
        );

        // Note: In practice, we'd want to do this asynchronously or use a different approach
        // For now, this demonstrates the concept but would need optimization
    }

    pub fn update_instances(
        &self,
        queue: &Queue,
        particles: &[GpuParticle],
        palette: &dyn Palette,
        type_count: usize,
    ) {
        if particles.is_empty() {
            return;
        }

        let instances: Vec<ParticleInstance> = particles.iter().map(|particle| {
            let color = palette.get_color(particle.type_id as usize, type_count);
            ParticleInstance {
                world_position: particle.position,
                color: [color.r, color.g, color.b, color.a],
                size: 1.0,
                _padding: [0.0; 3],
            }
        }).collect();

        queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&instances));
    }

    pub fn update_uniforms(&self, queue: &Queue, uniform_data: &UniformData) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[*uniform_data]));
    }

    pub fn render<'a>(
        &'a self,
        render_pass: &mut wgpu::RenderPass<'a>,
        instance_count: usize,
    ) {
        if instance_count == 0 {
            return;
        }

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_vertex_buffer(0, self.quad_vertex_buffer.slice(..));
        render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        
        // Draw 6 vertices (2 triangles = 1 quad) for each instance
        render_pass.draw(0..6, 0..instance_count as u32);
    }
}