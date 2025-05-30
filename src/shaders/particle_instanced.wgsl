// Instanced particle rendering shader

struct VertexInput {
    @location(0) position: vec2<f32>,    // Quad vertex position (-1 to 1)
}

struct InstanceInput {
    @location(1) world_position: vec3<f32>,  // Particle world position
    @location(2) color: vec4<f32>,           // Particle color
    @location(3) size: f32,                  // Particle size
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) quad_pos: vec2<f32>,
}

struct UniformData {
    view_proj: mat4x4<f32>,
    time: f32,
    particle_size: f32,
    cam_top_left: vec2<f32>,
    wrap: u32,
    _padding: vec3<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: UniformData;

@vertex
fn vs_main(
    vertex: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    
    // Calculate world position of this vertex
    let scaled_vertex = vertex.position * instance.size * uniforms.particle_size;
    let world_pos = vec4<f32>(
        instance.world_position.x + scaled_vertex.x,
        instance.world_position.y + scaled_vertex.y,
        instance.world_position.z,
        1.0
    );
    
    out.clip_position = uniforms.view_proj * world_pos;
    out.color = instance.color;
    out.quad_pos = vertex.position;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Create circular particles with smooth edges
    let dist = length(in.quad_pos);
    let alpha = smoothstep(0.9, 0.7, dist);
    
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}