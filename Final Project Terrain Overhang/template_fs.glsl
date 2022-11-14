#version 400

out vec4 fragcolor; //the output color for this fragment    

//in vec3 lightDirect;
in vec3 normal;
in vec3 posOut;
in vec4 colorOut;
flat in vec2 posBased;
flat in vec4 colOut;

uniform vec3 lightPos;

void main(void)
{
	vec3 lightDir = lightPos;// normalize(lightPos);
	//fragcolor = colorOut;//max(0.0f, dot(normal, lightDir));// texture(diffuse_tex, tex_coord);
	if (posBased.x > 0.9) {
		fragcolor = vec4(posOut.y * colOut.r, posOut.y * colOut.g, posOut.y * colOut.b, colOut.a);
	}
	else {
		fragcolor = colOut;
	}
}