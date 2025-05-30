use wgpu::{Device, Queue, Buffer, ComputePipeline, BindGroup, BindGroupLayout};
use bytemuck::{Pod, Zeroable};
use crate::physics::{Position, Velocity, PhysicsSettings, Matrix};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuParticle {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub type_id: u32,
    pub _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuPhysicsSettings {
    pub dt: f32,
    pub rmax: f32,
    pub friction: f32,
    pub force_multiplier: f32,
    pub world_size: f32,
    pub wrap_boundaries: u32,
    pub matrix_size: u32,
    pub cursor_active: u32,
    pub cursor_position: [f32; 3],
    pub cursor_size: f32,
    pub cursor_strength: f32,
    pub _padding: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SpatialGridParams {
    pub cell_size: f32,
    pub grid_size: u32,
    pub world_min: f32,
    pub _padding: f32,
}

pub struct GpuPhysicsSystem {
    // Buffers
    pub particle_buffer: Buffer,
    pub force_buffer: Buffer,
    pub settings_buffer: Buffer,
    pub matrix_buffer: Buffer,
    pub spatial_grid_buffer: Buffer,
    pub grid_counts_buffer: Buffer,
    pub grid_params_buffer: Buffer,
    
    // Compute pipelines
    clear_grid_pipeline: ComputePipeline,
    build_grid_pipeline: ComputePipeline,
    calculate_forces_pipeline: ComputePipeline,
    update_particles_pipeline: ComputePipeline,
    
    // Bind groups
    compute_bind_group: BindGroup,
    
    // Parameters
    particle_count: usize,
    grid_size: u32,
    max_particles_per_cell: u32,
}

impl GpuPhysicsSystem {
    pub fn new(device: &Device, max_particles: usize) -> Self {
        let grid_size = 64u32; // 64x64 spatial grid
        let max_particles_per_cell = 64u32;
        let world_size = 4.0f32;
        let cell_size = world_size / grid_size as f32;
        
        // Create buffers
        let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Buffer"),
            size: (max_particles * std::mem::size_of::<GpuParticle>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let force_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Force Buffer"),
            size: (max_particles * std::mem::size_of::<[f32; 3]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let settings_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Physics Settings Buffer"),
            size: std::mem::size_of::<GpuPhysicsSettings>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let matrix_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Interaction Matrix Buffer"),
            size: (64 * 64 * std::mem::size_of::<f32>()) as u64, // Max 64x64 matrix
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let total_grid_cells = (grid_size * grid_size) as usize;
        let spatial_grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spatial Grid Buffer"),
            size: (total_grid_cells * max_particles_per_cell as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let grid_counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid Counts Buffer"),
            size: (total_grid_cells * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let grid_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid Params Buffer"),
            size: std::mem::size_of::<SpatialGridParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Load compute shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Physics Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/physics_compute.wgsl").into()),
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Physics Compute Bind Group Layout"),
            entries: &[
                // Particles buffer
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
                // Forces buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Settings uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Matrix buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Spatial grid buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Grid counts buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Grid params uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create bind group
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Physics Compute Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: force_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: settings_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: matrix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: spatial_grid_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: grid_counts_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: grid_params_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create compute pipelines
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Physics Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let clear_grid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Clear Grid Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "clear_grid",
        });
        
        let build_grid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Build Grid Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "build_grid",
        });
        
        let calculate_forces_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Calculate Forces Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "calculate_forces",
        });
        
        let update_particles_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Particles Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "update_particles",
        });
        
        // Initialize grid parameters
        let grid_params = SpatialGridParams {
            cell_size,
            grid_size,
            world_min: -2.0,
            _padding: 0.0,
        };
        
        Self {
            particle_buffer,
            force_buffer,
            settings_buffer,
            matrix_buffer,
            spatial_grid_buffer,
            grid_counts_buffer,
            grid_params_buffer,
            clear_grid_pipeline,
            build_grid_pipeline,
            calculate_forces_pipeline,
            update_particles_pipeline,
            compute_bind_group,
            particle_count: 0,
            grid_size,
            max_particles_per_cell,
        }
    }
    
    pub fn initialize_particles(&mut self, queue: &Queue, particles: &[crate::physics::Particle]) {
        self.particle_count = particles.len();
        
        // Convert CPU particles to GPU format
        let gpu_particles: Vec<GpuParticle> = particles.iter().map(|p| GpuParticle {
            position: [p.position.x as f32, p.position.y as f32, p.position.z as f32],
            velocity: [p.velocity.x as f32, p.velocity.y as f32, p.velocity.z as f32],
            type_id: p.type_id as u32,
            _padding: 0,
        }).collect();
        
        queue.write_buffer(&self.particle_buffer, 0, bytemuck::cast_slice(&gpu_particles));
        
        // Initialize grid parameters
        let grid_params = SpatialGridParams {
            cell_size: 4.0 / self.grid_size as f32,
            grid_size: self.grid_size,
            world_min: -2.0,
            _padding: 0.0,
        };
        queue.write_buffer(&self.grid_params_buffer, 0, bytemuck::cast_slice(&[grid_params]));
    }
    
    pub fn update_settings(&self, queue: &Queue, settings: &PhysicsSettings, cursor_pos: Option<nalgebra::Vector3<f64>>, cursor_size: f64, cursor_strength: f64) {
        let gpu_settings = GpuPhysicsSettings {
            dt: settings.dt as f32,
            rmax: settings.rmax as f32,
            friction: settings.friction as f32,
            force_multiplier: settings.force as f32,
            world_size: 4.0,
            wrap_boundaries: if settings.wrap { 1 } else { 0 },
            matrix_size: 0, // Will be updated when matrix is set
            cursor_active: if cursor_pos.is_some() { 1 } else { 0 },
            cursor_position: if let Some(pos) = cursor_pos {
                [pos.x as f32, pos.y as f32, pos.z as f32]
            } else {
                [0.0, 0.0, 0.0]
            },
            cursor_size: cursor_size as f32,
            cursor_strength: cursor_strength as f32,
            _padding: [0.0; 3],
        };
        
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::cast_slice(&[gpu_settings]));
    }
    
    pub fn update_matrix(&self, queue: &Queue, matrix: &Matrix) {
        let size = matrix.size();
        let mut matrix_data = vec![0.0f32; 64 * 64]; // Max size buffer
        
        for i in 0..size {
            for j in 0..size {
                matrix_data[i * 64 + j] = matrix.get(i, j) as f32;
            }
        }
        
        queue.write_buffer(&self.matrix_buffer, 0, bytemuck::cast_slice(&matrix_data));
        
        // Update matrix size in settings
        let mut gpu_settings = GpuPhysicsSettings {
            dt: 0.0, rmax: 0.0, friction: 0.0, force_multiplier: 0.0, world_size: 4.0,
            wrap_boundaries: 0, matrix_size: size as u32, cursor_active: 0,
            cursor_position: [0.0; 3], cursor_size: 0.0, cursor_strength: 0.0, _padding: [0.0; 3],
        };
        // Note: This is a partial update - in practice we'd need to track current settings
    }
    
    pub fn update(&self, encoder: &mut wgpu::CommandEncoder) {
        let workgroup_size = 64;
        let total_cells = (self.grid_size * self.grid_size) as u32;
        let particle_workgroups = (self.particle_count as u32 + workgroup_size - 1) / workgroup_size;
        let grid_workgroups = (total_cells + workgroup_size - 1) / workgroup_size;
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Physics Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            
            // 1. Clear spatial grid
            compute_pass.set_pipeline(&self.clear_grid_pipeline);
            compute_pass.dispatch_workgroups(grid_workgroups, 1, 1);
            
            // 2. Build spatial grid
            compute_pass.set_pipeline(&self.build_grid_pipeline);
            compute_pass.dispatch_workgroups(particle_workgroups, 1, 1);
            
            // 3. Calculate forces
            compute_pass.set_pipeline(&self.calculate_forces_pipeline);
            compute_pass.dispatch_workgroups(particle_workgroups, 1, 1);
            
            // 4. Update particles
            compute_pass.set_pipeline(&self.update_particles_pipeline);
            compute_pass.dispatch_workgroups(particle_workgroups, 1, 1);
        }
    }
    
    pub fn get_particles_buffer(&self) -> &Buffer {
        &self.particle_buffer
    }
    
    pub fn particle_count(&self) -> usize {
        self.particle_count
    }
}