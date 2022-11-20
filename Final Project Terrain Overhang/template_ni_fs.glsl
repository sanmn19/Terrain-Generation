#version 400

out vec4 fragcolor; //the output color for this fragment    

in vec3 normal;

uniform vec3 lightPos;

void main(void)
{
	vec3 lightDir = lightPos;// normalize(lightPos);
	fragcolor = vec4(1, 1, 1, 1) * max(0.0f, dot(normal, lightDir));// texture(diffuse_tex, tex_coord);
}