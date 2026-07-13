#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#

v_shader_triangles = """
#version 330

layout(std140) uniform CameraMatrices {
    mat4 view_mat;
    mat4 proj_mat;
};
uniform mat4 model_mat;

in vec3 vert_coord;
in vec3 vert_color;
in vec3 vert_norm;

out vec3 frag_color;
out vec3 frag_coord;
out vec3 frag_norm;

void main(){
    frag_color = vert_color;
    //frag_norm = vert_norm;
    frag_norm = mat3(transpose(inverse(model_mat))) * vert_norm;
    frag_coord = (view_mat * model_mat * vec4(vert_coord, 1.0)).xyz;
    gl_Position = proj_mat * view_mat * model_mat * vec4(vert_coord, 1.0);
}
"""
f_shader_triangles = """
#version 330

struct Light {
    vec3 position;
    //vec3 color;
    vec3 intensity;
    //vec3 specular_color;
    float ambient_coef;
    float shininess;
};

uniform Light my_light;

in vec3 frag_coord;
in vec3 frag_color;
in vec3 frag_norm;

out vec4 final_color;

void main(){
    vec3 normal = normalize(frag_norm);
    // [EN] BUG FIX (user report, with screenshot: "a luz parece errada" --
    // blotchy dark/bright patches on the same continuous ribbon surface).
    // draw_representation() in representations.py calls
    // GL.glDisable(GL.GL_CULL_FACE) for this representation, so BOTH
    // sides of a thin, twisting ribbon (helix/beta, which are only ~0.2-
    // 0.4 A thick) are visible at once from many viewing angles. Every
    // triangle's normal here was computed once, facing outward from
    // whichever side make_normals() happened to wind it -- when the
    // camera sees the OTHER (back) side of that same thin surface, the
    // stored normal is pointing roughly AWAY from the viewer/light
    // instead of towards them, so diffuse_coef below clamps near 0 and
    // that patch renders dark -- while an adjacent patch still showing
    // its front face lights up normally. Exactly the alternating light/
    // dark blotches on the same ribbon in the screenshot. Standard,
    // textbook fix for lighting an open/two-sided surface (see any
    // OpenGL/GLSL reference on "two-sided lighting"): gl_FrontFacing is
    // a GLSL built-in that's true for the winding-order-facing side of
    // the triangle currently being rasterized -- flip the normal when
    // looking at the back side, so the SAME physical surface point is
    // lit consistently regardless of which side the camera currently
    // sees.
    if (!gl_FrontFacing) {
        normal = -normal;
    }
    vec3 vert_to_light = normalize(my_light.position);
    vec3 vert_to_cam = normalize(frag_coord);
    
    // Ambient Component
    vec3 ambient = my_light.ambient_coef * frag_color * my_light.intensity;
    
    // Diffuse component
    float diffuse_coef = max(0.0, dot(normal, vert_to_light));
    vec3 diffuse = diffuse_coef * frag_color * my_light.intensity;
    
    // Specular component
    float specular_coef = 0.0;
    if (diffuse_coef > 0.0)
        specular_coef = pow(max(0.0, dot(vert_to_cam, reflect(-vert_to_light, normal))), my_light.shininess);
    vec3 specular = specular_coef * my_light.intensity;
    specular = specular * (vec3(1) - diffuse);
    
    final_color = vec4(ambient + diffuse + specular, 1.0);
}
"""
