VERTEX_SHADER = """
#version 330 core

in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;

out vec3 frag_normal;
out vec2 frag_uv;
out vec3 frag_position;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main() {
    vec4 world_pos = model * vec4(in_position, 1.0);
    gl_Position = projection * view * world_pos;
    
    frag_position = world_pos.xyz;
    frag_normal = mat3(transpose(inverse(model))) * in_normal;
    frag_uv = in_uv;
}
"""

FRAGMENT_SHADER = """
#version 330 core

in vec3 frag_normal;
in vec2 frag_uv;
in vec3 frag_position;

out vec4 out_color;

uniform sampler2D grid_texture;
uniform vec3 light_dir;
uniform vec3 camera_pos;

void main() {
    // Sample grid state from texture
    vec4 cell_color = texture(grid_texture, frag_uv);
    
    // Basic lighting
    vec3 normal = normalize(frag_normal);
    float diffuse = max(dot(normal, -light_dir), 0.0);
    float ambient = 0.3;
    
    // Specular highlight
    vec3 view_dir = normalize(camera_pos - frag_position);
    vec3 reflect_dir = reflect(light_dir, normal);
    float specular = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0) * 0.5;
    
    // Final color
    vec3 lit_color = cell_color.rgb * (ambient + diffuse) + vec3(specular);
    out_color = vec4(lit_color, 1.0);
}
"""
