#version 450

layout (location = 0) in vec3 inPos;
layout (location = 0) in vec4 inColor0;

layout (location = 0) out vec4 outColor0;

void main() 
{
	outColor0 = inColor0;	
	gl_Position =  vec4(inPos, 1.0);
}