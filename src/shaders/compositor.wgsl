// Compositor shader for the particle life simulator
//
// This shader handles the final compositing of the simulation:
// - World-space tiling for wrapping visualization
// - Camera transform and zoom
// - Texture sampling with proper wrapping
//
// The shader renders a fullscreen quad that samples from the source texture
// with proper world-space coordinates and tiling.

/// Uniform data for compositing
struct CompositorUniforms {
    /// World offset for this tile instance
    world_offset: vec2<f32>,
    /// Size of the world (4.0 for -2.0 to 2.0)
    world_size: f32,
    /// Padding to 16 bytes
    _padding1: f32,
    /// Camera position in world space
    cam_position: vec2<f32>,
    /// Camera zoom size
    cam_size: f32,
    /// Padding to 32 bytes
    _padding2: f32,
}

// Uniform buffer for compositing settings
@group(0) @binding(0)
var<uniform> uniforms: CompositorUniforms;

// Source texture containing the rendered simulation
@group(0) @binding(1)
var source_texture: texture_2d<f32>;

// Sampler for texture sampling
@group(0) @binding(2)
var texture_sampler: sampler;

/// Vertex output to fragment shader
struct VertexOutput {
    /// Clip space position for rasterization
    @builtin(position) clip_position: vec4<f32>,
    /// UV coordinates for texture sampling
    @location(0) uv: vec2<f32>,
    /// World position for effects
    @location(1) world_pos: vec2<f32>,
}

/// Vertex shader entry point
/// 
/// Creates a fullscreen quad and transforms vertices to:
/// 1. Screen space for rasterization
/// 2. World space for camera transform
/// 3. UV space for texture sampling
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Create fullscreen triangle strip vertices  
    let x = f32((vertex_index & 1u) * 2u) - 1.0;
    let y = f32((vertex_index & 2u)) - 1.0;
    
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    
    // Convert screen position to world coordinates
    let world_x = x * uniforms.cam_size * 0.5 + uniforms.cam_position.x;
    let world_y = y * uniforms.cam_size * 0.5 + uniforms.cam_position.y;
    
    // Apply the tile offset to get the source world position
    let source_world_x = world_x - uniforms.world_offset.x;
    let source_world_y = world_y - uniforms.world_offset.y;
    
    // Convert source world position to UV coordinates (world space is -2 to +2)
    let uv_x = (source_world_x + 2.0) / 4.0;
    let uv_y = 1.0 - (source_world_y + 2.0) / 4.0;
    
    out.uv = vec2<f32>(uv_x, uv_y);
    out.world_pos = vec2<f32>(world_x, world_y);
    
    return out;
}

/// Fragment shader entry point
/// 
/// Samples the source texture with proper UV wrapping for tiling.
/// The sampler is configured to repeat, so this will wrap correctly
/// around the world boundaries.
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample the texture with proper UV wrapping for tiling
    // The sampler is set to Repeat mode, so this will wrap correctly
    return textureSample(source_texture, texture_sampler, in.uv);
}