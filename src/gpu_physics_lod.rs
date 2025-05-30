use wgpu::{Device, Queue, Buffer, ComputePipeline, BindGroup};
use bytemuck::{Pod, Zeroable};
use crate::physics::{PhysicsSettings, Matrix};
use crate::rendering::Camera;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuParticleLod {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub type_id: u32,
    pub lod_level: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuPhysicsSettingsLod {
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
    // LOD settings
    pub camera_position: [f32; 3],
    pub camera_size: f32,
    pub lod_distance_0: f32,  // High detail distance
    pub lod_distance_1: f32,  // Medium detail distance  
    pub lod_distance_2: f32,  // Low detail distance
    pub enable_frustum_culling: u32,
    pub frustum_left: f32,
    pub frustum_right: f32,
    pub frustum_top: f32,
    pub frustum_bottom: f32,
    pub _padding: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SpatialGridParams {
    pub cell_size: f32,
    pub grid_size: u32,
    pub world_min: f32,
    pub _padding: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LodStats {
    pub lod0_count: u32,    // High detail
    pub lod1_count: u32,    // Medium detail
    pub lod2_count: u32,    // Low detail
    pub culled_count: u32,  // Culled/outside frustum
}

pub struct GpuPhysicsSystemLod {
    // Buffers
    pub particle_buffer: Buffer,
    pub force_buffer: Buffer,
    pub settings_buffer: Buffer,
    pub matrix_buffer: Buffer,
    pub spatial_grid_buffer: Buffer,
    pub grid_counts_buffer: Buffer,
    pub grid_params_buffer: Buffer,
    pub lod_stats_buffer: Buffer,
    pub lod_stats_staging_buffer: Buffer,
    
    // Compute pipelines
    clear_grid_pipeline: ComputePipeline,
    calculate_lod_and_build_grid_pipeline: ComputePipeline,
    calculate_forces_lod_pipeline: ComputePipeline,
    update_particles_lod_pipeline: ComputePipeline,
    
    // Bind groups
    compute_bind_group: BindGroup,
    
    // Parameters
    particle_count: usize,
    grid_size: u32,
    max_particles_per_cell: u32,
    
    // LOD configuration
    pub lod_distance_0: f32,
    pub lod_distance_1: f32,
    pub lod_distance_2: f32,
    pub enable_frustum_culling: bool,
    
    // Performance metrics
    pub current_lod_stats: LodStats,
}

impl GpuPhysicsSystemLod {
    pub fn new(device: &Device, max_particles: usize) -> Self {
        let grid_size = 64u32; // 64x64 spatial grid
        let max_particles_per_cell = 64u32;
        let world_size = 4.0f32;
        let cell_size = world_size / grid_size as f32;
        
        // Create buffers
        let particle_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD Particle Buffer"),
            size: (max_particles * std::mem::size_of::<GpuParticleLod>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let force_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD Force Buffer"),
            size: (max_particles * std::mem::size_of::<[f32; 3]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let settings_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD Physics Settings Buffer"),
            size: std::mem::size_of::<GpuPhysicsSettingsLod>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let matrix_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD Interaction Matrix Buffer"),
            size: (64 * 64 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let total_grid_cells = (grid_size * grid_size) as usize;
        let spatial_grid_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD Spatial Grid Buffer"),
            size: (total_grid_cells * max_particles_per_cell as usize * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let grid_counts_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD Grid Counts Buffer"),
            size: (total_grid_cells * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        
        let grid_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD Grid Params Buffer"),
            size: std::mem::size_of::<SpatialGridParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let lod_stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD Stats Buffer"),
            size: std::mem::size_of::<LodStats>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        let lod_stats_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LOD Stats Staging Buffer"),
            size: std::mem::size_of::<LodStats>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        // Load compute shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LOD Physics Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/physics_compute_lod.wgsl").into()),
        });
        
        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LOD Physics Compute Bind Group Layout"),
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
                // LOD stats buffer
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create bind group
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LOD Physics Compute Bind Group"),
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
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: lod_stats_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create compute pipelines
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LOD Physics Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let clear_grid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Clear Grid Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "clear_grid",
        });
        
        let calculate_lod_and_build_grid_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Calculate LOD and Build Grid Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "calculate_lod_and_build_grid",
        });
        
        let calculate_forces_lod_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Calculate Forces LOD Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "calculate_forces_lod",
        });
        
        let update_particles_lod_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Particles LOD Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "update_particles_lod",
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
            lod_stats_buffer,
            lod_stats_staging_buffer,
            clear_grid_pipeline,
            calculate_lod_and_build_grid_pipeline,
            calculate_forces_lod_pipeline,
            update_particles_lod_pipeline,
            compute_bind_group,
            particle_count: 0,
            grid_size,
            max_particles_per_cell,
            lod_distance_0: 1.0,   // High detail within 1.0 camera units
            lod_distance_1: 3.0,   // Medium detail within 3.0 camera units
            lod_distance_2: 8.0,   // Low detail within 8.0 camera units
            enable_frustum_culling: true,
            current_lod_stats: LodStats {
                lod0_count: 0,
                lod1_count: 0,
                lod2_count: 0,
                culled_count: 0,
            },
        }
    }
    
    pub fn initialize_particles(&mut self, queue: &Queue, particles: &[crate::physics::Particle]) {
        self.particle_count = particles.len();
        
        // Convert CPU particles to GPU LOD format
        let gpu_particles: Vec<GpuParticleLod> = particles.iter().map(|p| GpuParticleLod {
            position: [p.position.x as f32, p.position.y as f32, p.position.z as f32],
            velocity: [p.velocity.x as f32, p.velocity.y as f32, p.velocity.z as f32],
            type_id: p.type_id as u32,
            lod_level: 0, // Will be calculated in compute shader
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
    
    pub fn update_settings(&self, queue: &Queue, settings: &PhysicsSettings, camera: &Camera, cursor_pos: Option<nalgebra::Vector3<f64>>, cursor_size: f64, cursor_strength: f64) {
        // Calculate frustum bounds from camera
        let camera_half_size = camera.size as f32 * 0.5;
        let frustum_left = camera.position.x as f32 - camera_half_size * 1.2; // Add some margin
        let frustum_right = camera.position.x as f32 + camera_half_size * 1.2;
        let frustum_bottom = camera.position.y as f32 - camera_half_size * 1.2;
        let frustum_top = camera.position.y as f32 + camera_half_size * 1.2;
        
        let gpu_settings = GpuPhysicsSettingsLod {
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
            camera_position: [camera.position.x as f32, camera.position.y as f32, camera.position.z as f32],
            camera_size: camera.size as f32,
            lod_distance_0: self.lod_distance_0,
            lod_distance_1: self.lod_distance_1,
            lod_distance_2: self.lod_distance_2,
            enable_frustum_culling: if self.enable_frustum_culling { 1 } else { 0 },
            frustum_left,
            frustum_right,
            frustum_top,
            frustum_bottom,
            _padding: [0.0; 2],
        };
        
        queue.write_buffer(&self.settings_buffer, 0, bytemuck::cast_slice(&[gpu_settings]));
    }
    
    pub fn update_matrix(&self, queue: &Queue, matrix: &Matrix) {
        let size = matrix.size();
        let mut matrix_data = vec![0.0f32; 64 * 64];
        
        for i in 0..size {
            for j in 0..size {
                matrix_data[i * 64 + j] = matrix.get(i, j) as f32;
            }
        }
        
        queue.write_buffer(&self.matrix_buffer, 0, bytemuck::cast_slice(&matrix_data));
    }
    
    pub fn update(&self, encoder: &mut wgpu::CommandEncoder) {
        let workgroup_size = 64;
        let total_cells = (self.grid_size * self.grid_size) as u32;
        let particle_workgroups = (self.particle_count as u32 + workgroup_size - 1) / workgroup_size;
        let grid_workgroups = (total_cells + workgroup_size - 1) / workgroup_size;
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LOD Physics Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            
            // 1. Clear spatial grid and LOD stats
            compute_pass.set_pipeline(&self.clear_grid_pipeline);
            compute_pass.dispatch_workgroups(grid_workgroups, 1, 1);
            
            // 2. Calculate LOD levels and build spatial grid
            compute_pass.set_pipeline(&self.calculate_lod_and_build_grid_pipeline);
            compute_pass.dispatch_workgroups(particle_workgroups, 1, 1);
            
            // 3. Calculate forces with LOD awareness
            compute_pass.set_pipeline(&self.calculate_forces_lod_pipeline);
            compute_pass.dispatch_workgroups(particle_workgroups, 1, 1);
            
            // 4. Update particles with LOD-aware time stepping
            compute_pass.set_pipeline(&self.update_particles_lod_pipeline);
            compute_pass.dispatch_workgroups(particle_workgroups, 1, 1);
        }
        
        // Copy LOD stats for reading
        encoder.copy_buffer_to_buffer(
            &self.lod_stats_buffer,
            0,
            &self.lod_stats_staging_buffer,
            0,
            std::mem::size_of::<LodStats>() as u64,
        );
    }
    
    pub async fn read_lod_stats(&mut self, device: &Device) -> LodStats {
        let buffer_slice = self.lod_stats_staging_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        
        if receiver.receive().await.unwrap().is_ok() {
            let data = buffer_slice.get_mapped_range();
            let stats: LodStats = *bytemuck::from_bytes(&data);
            drop(data);
            self.lod_stats_staging_buffer.unmap();
            self.current_lod_stats = stats;
            stats
        } else {
            self.current_lod_stats
        }
    }
    
    pub fn get_particles_buffer(&self) -> &Buffer {
        &self.particle_buffer
    }
    
    pub fn particle_count(&self) -> usize {
        self.particle_count
    }
    
    pub fn set_lod_distances(&mut self, distance_0: f32, distance_1: f32, distance_2: f32) {
        self.lod_distance_0 = distance_0;
        self.lod_distance_1 = distance_1;
        self.lod_distance_2 = distance_2;
    }
    
    pub fn set_frustum_culling(&mut self, enabled: bool) {
        self.enable_frustum_culling = enabled;
    }
    
    pub fn get_current_lod_stats(&self) -> &LodStats {
        &self.current_lod_stats
    }
}