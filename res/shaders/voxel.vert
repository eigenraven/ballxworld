#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(push_constant) uniform PushConstants {
    vec3 chunk_offset;
    int highlight_index;
} push_constants;

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec3 texcoord;
layout(location = 3) in int index;
layout(location = 4) in vec4 barycentric_color_offset;

layout(location = 0) out vec4 v_position;
layout(location = 1) out vec4 v_color;
layout(location = 2) out vec3 v_texcoord;
layout(location = 3) out int v_index;
layout(location = 4) out vec4 v_barycentric_color_offset;
layout(location = 5) out vec3 v_barycentric;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(position.xyz * 64.0 - vec3(0.5) + push_constants.chunk_offset, 1.0);

    float baryx = ((index & (1 << 17)) > 0) ? 1.0 : 0.0;
    float baryy = ((index & (1 << 18)) > 0) ? 1.0 : 0.0;
    float baryz = ((index & (1 << 19)) > 0) ? 1.0 : 0.0;

    v_position = position;
    v_color = color;
    v_texcoord = texcoord;
    v_index = index & ((1 << 17) - 1);
    v_barycentric_color_offset = barycentric_color_offset;
    v_barycentric = vec3(baryx, baryy, baryz);
}
