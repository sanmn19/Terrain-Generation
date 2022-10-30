#version 400            
uniform mat4 PVM;

layout(location = 0) in vec3 pos_attrib; //this variable holds the position of mesh vertices
layout(location = 1) in vec3 normal_attrib;
layout(location = 2) in vec4 color;
layout(location = 8) in mat4 model_matrix;

//out vec3 lightDirect;
out vec3 normal;
out vec3 posOut;
out vec4 colorOut;

uniform vec3 lightPos;

void main(void)
{
	vec4 worldPosition = model_matrix * vec4(pos_attrib, 1);
	vec3 worldPosition3 = vec3(worldPosition.x, worldPosition.y, worldPosition.z);
	posOut = worldPosition3;
	//vec3 lightDir = vec3(worldPosition3.x, worldPosition3.y, worldPosition3.z) - lightPos;

	vec4 normalWorld = model_matrix * vec4(normal_attrib, 0);

	normal = (vec3(normalWorld.x, normalWorld.y, normalWorld.z));
	//normal = vertexNormal_modelspace;
	//lightDirect = normalize(lightDir);
	if (color.b > 0.1 && color.r < 0.04) {
		colorOut = vec4(color.r, color.g, color.b, color.a);
	}
	else {
		colorOut = vec4(worldPosition.y * color.r, worldPosition.y * color.g, worldPosition.y * color.b, color.a);
	}

	gl_Position = PVM * model_matrix * vec4(pos_attrib, 1.0); //transform vertices and send result into pipeline
}