// Fade shader for the particle life simulator
//
// This shader implements a screen-space fade effect:
// - Renders a fullscreen quad
// - Applies a semi-transparent black overlay
// - Controls fade strength with a uniform
//
// The fade effect is used to create motion trails and visual effects
// by blending the current frame with previous frames.

/// Uniform data for fade effect
struct FadeUniforms {
    /// Fade strength (0.0 = no fade, 1.0 = complete fade)
    fade_factor: f32,
}

// Uniform buffer for fade settings
@group(0) @binding(0)
var<uniform> uniforms: FadeUniforms;

/// Vertex output to fragment shader
struct VertexOutput {
    /// Clip space position for rasterization
    @builtin(position) clip_position: vec4<f32>,
    /// UV coordinates for effects
    @location(0) uv: vec2<f32>,
}

/// Vertex shader entry point
/// 
/// Creates a fullscreen quad in screen space:
/// - Vertex 0: (-1, -1)
/// - Vertex 1: (1, -1)
/// - Vertex 2: (-1, 1)
/// - Vertex 3: (1, 1)
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

/// Fragment shader entry point
/// 
/// Applies a semi-transparent black overlay to create the fade effect.
/// The fade_factor controls the strength of the fade:
/// - 0.0: No fade (fully transparent)
/// - 1.0: Complete fade (fully opaque black)
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Apply fade effect by outputting semi-transparent black
    // fade_factor controls how much to fade (0.0 = no fade, 1.0 = complete fade)
    let alpha = 1.0 - uniforms.fade_factor;
    return vec4<f32>(0.0, 0.0, 0.0, alpha);
}