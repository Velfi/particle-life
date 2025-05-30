struct FadeUniforms {
    fade_factor: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: FadeUniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Create fullscreen triangle strip in screen space (NDC coordinates)
    // Vertex 0: (-1, -1), Vertex 1: (1, -1), Vertex 2: (-1, 1), Vertex 3: (1, 1)
    let x = f32((vertex_index & 1u) * 2u) - 1.0;
    let y = f32((vertex_index & 2u)) - 1.0;
    
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x * 0.5 + 0.5, y * 0.5 + 0.5);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Apply fade effect by outputting semi-transparent black
    // fade_factor controls how much to fade (0.0 = no fade, 1.0 = complete fade)
    let alpha = 1.0 - uniforms.fade_factor;
    return vec4<f32>(0.0, 0.0, 0.0, alpha);
}