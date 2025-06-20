//! Shader system for the particle life simulator
//! 
//! This module manages the WebGPU shaders and rendering pipelines used for:
//! - Particle rendering with instanced quads
//! - Screen-space fade effects
//! - Physics computation
//! - Compositing and post-processing
//! 
//! The shader system uses WGSL (WebGPU Shading Language) for GPU-accelerated
//! rendering and computation.

use bytemuck::{Pod, Zeroable};
use wgpu::{BindGroup, Buffer, Device, RenderPipeline, VertexAttribute, VertexBufferLayout};

/// Vertex format for particle rendering
/// 
/// Each particle is rendered as a quad with:
/// - position: 3D world position
/// - color: RGBA color
/// - size: Particle size in world units
/// - quad_vertex: 2D position within the quad (-1 to 1)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ParticleVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
    pub size: f32,
    pub quad_vertex: [f32; 2],
    pub _padding: f32,
}

impl ParticleVertex {
    /// Vertex attributes for the particle shader
    const ATTRIBS: [VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x3,  // position
        1 => Float32x4,  // color
        2 => Float32,    // size
        3 => Float32x2,  // quad_vertex
    ];

    /// Returns the vertex buffer layout for particle rendering
    pub fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: std::mem::size_of::<ParticleVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// Uniform data passed to the particle shader
/// 
/// Contains:
/// - view_proj: View-projection matrix for camera
/// - time: Current simulation time
/// - particle_size: Base size of particles
/// - cam_top_left: Camera position for wrapping
/// - wrap: Whether to wrap particles around world
/// - show_tiling: Whether to show world tiling
/// - world_size: Size of the world
/// - tile_fade_strength: Strength of tile fade effect
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct UniformData {
    pub view_proj: [[f32; 4]; 4],
    pub time: f32,
    pub particle_size: f32,
    pub cam_top_left: [f32; 2],
    pub wrap: u32,
    pub show_tiling: u32,
    pub world_size: f32,
    pub tile_fade_strength: f32,
}

/// Manages the particle rendering pipeline
/// 
/// Handles:
/// - Shader compilation and pipeline creation
/// - Uniform buffer management
/// - Bind group setup
pub struct ParticleShader {
    pub render_pipeline: RenderPipeline,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl ParticleShader {
    /// Creates a new particle shader with the specified device and format
    /// 
    /// Sets up:
    /// - WGSL shader compilation
    /// - Uniform buffer creation
    /// - Bind group layout and creation
    /// - Render pipeline configuration
    pub fn new(device: &Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Particle Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("particle.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<UniformData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Particle Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Particle Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Particle Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Particle Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[ParticleVertex::desc()],
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
        }
    }

    /// Updates the uniform buffer with new data
    /// 
    /// Parameters:
    /// - queue: The WebGPU queue for buffer updates
    /// - uniform_data: New uniform data to upload
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniform_data: &UniformData) {
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[*uniform_data]),
        );
    }
}

/// Manages the screen fade effect shader
/// 
/// Handles:
/// - Fade shader compilation
/// - Uniform buffer for fade factor
/// - Render pipeline for full-screen quad
pub struct FadeShader {
    pipeline: RenderPipeline,
    bind_group: BindGroup,
    uniform_buffer: Buffer,
}

/// Uniform data for the fade shader
/// 
/// Contains:
/// - fade_factor: Strength of the fade effect (0-1)
/// - background_color: Color to fade to
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FadeUniformData {
    fade_factor: f32,
    background_color: [f32; 4],
    _padding: [f32; 3], // Add padding to match shader's 32-byte alignment
}

impl FadeShader {
    /// Creates a new fade shader with the specified device and format
    /// 
    /// Sets up:
    /// - WGSL shader compilation
    /// - Uniform buffer creation
    /// - Bind group layout and creation
    /// - Render pipeline for full-screen quad
    pub fn new(device: &Device, format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fade Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("fade.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("Fade Bind Group Layout"),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fade Uniform Buffer"),
            size: std::mem::size_of::<FadeUniformData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("Fade Bind Group"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Fade Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fade Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
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
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            bind_group,
            uniform_buffer,
        }
    }

    /// Updates the fade factor uniform
    /// 
    /// Parameters:
    /// - queue: The WebGPU queue for buffer updates
    /// - fade_factor: New fade factor (0-1)
    /// - background_color: Color to fade to
    pub fn update_uniforms(&self, queue: &wgpu::Queue, fade_factor: f32, background_color: [f32; 4]) {
        let uniform_data = FadeUniformData {
            fade_factor,
            background_color,
            _padding: [0.0; 3],
        };
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&[uniform_data]),
        );
    }

    /// Renders the fade effect
    /// 
    /// Parameters:
    /// - render_pass: The WebGPU render pass to record commands
    pub fn render<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..4, 0..1); // Fullscreen quad with triangle strip
    }
}
