#version 400            
layout(location = 0) in vec3 pos_attrib; //this variable holds the position of mesh vertices
layout(location = 1) in vec3 normal_attrib;

//out vec3 lightDirect;
out vec3 normal;

uniform mat4 PVM;
uniform mat4 M;
uniform vec3 lightPos;

void main(void)
{
	mat4 model_matrix = M;
	vec4 worldPosition = model_matrix * vec4(pos_attrib, 1);
	vec3 worldPosition3 = vec3(worldPosition.x, worldPosition.y, worldPosition.z);
	//vec3 lightDir = vec3(worldPosition3.x, worldPosition3.y, worldPosition3.z) - lightPos;

	vec4 normalWorld = model_matrix * vec4(normal_attrib, 0);

	normal = (vec3(normalWorld.x, normalWorld.y, normalWorld.z));
	//normal = vertexNormal_modelspace;
	//lightDirect = normalize(lightDir);

	gl_Position = PVM * model_matrix * vec4(pos_attrib, 1.0); //transform vertices and send result into pipeline
}