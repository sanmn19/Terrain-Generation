#version 400            
uniform mat4 PVM;

layout(location = 0) in vec3 pos_attrib; //this variable holds the position of mesh vertices
layout(location = 1) in vec3 normal_attrib;

//out vec3 lightDirect;
out vec3 normal;
out vec3 posOut;

uniform mat4 M;
uniform vec3 lightPos;

void main(void)
{
	vec4 worldPosition = M * vec4(pos_attrib, 1);
	vec3 worldPosition3 = vec3(worldPosition.x, worldPosition.y, worldPosition.z);
	posOut = worldPosition3;
	//vec3 lightDir = vec3(worldPosition3.x, worldPosition3.y, worldPosition3.z) - lightPos;

	vec4 normalWorld = M * vec4(normal_attrib, 0);

	normal = (vec3(normalWorld.x, normalWorld.y, normalWorld.z));
	//normal = vertexNormal_modelspace;
	//lightDirect = normalize(lightDir);

	gl_Position = PVM * vec4(pos_attrib, 1.0); //transform vertices and send result into pipeline
}