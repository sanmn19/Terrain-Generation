#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <omp.h>
#include "FreeImage.h"
#include <fstream>
#include "LoadTexture.h"
#include <iostream>

#define RESTART_PRIMITIVE_CODE 0xffffffff

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

class Terrain {

	glm::vec3 position;

	glm::vec3 scale;

	std::string filePath;

	//GLfloat* vertexBufferData;
	//GLfloat* vertexNormalData;

	//unsigned int* vertexIndexData;

	GLuint vertexAndNormalbuffer;
	GLuint vertexArrayObject;

	int num_indices;
	int indexCount;
	int drawMode;

	std::vector<std::vector<int>> imageData;

	void AddVertex(std::vector <GLfloat>* a, const glm::vec3 v);
	void AppendNormal(std::vector <GLfloat>& a, const glm::vec3 n, const int& i, const int& j, const int& rows, const int& cols, std::vector <int>& weights);

	GLuint index_buffer;

public:

	//unsigned int vertexCount;
	std::vector <GLfloat> vertices;
	//std::vector <GLfloat> normals;
	std::vector <unsigned int> indexArray;
	glm::mat4 modelMatrix;

	Terrain() = default;
	Terrain(glm::vec3 position, const std::string & filePath, glm::vec3 scale);

	void render(glm::mat4 view, glm::mat4 projection, GLuint programID, float lightDir[3]);

	void setScale(glm::vec3 scale);
	void setDrawMode(int drawMode);

	glm::vec3 getScale();
	void setIndexCountToRender(int indexCount);

	void generateTerrain();

	glm::mat4 getModelMatrix();
	void generateTerrain(FIBITMAP * img);

	void SaveOBJ(std::vector<char> filename);
};

inline void Terrain::setDrawMode(int drawMode) {
	this->drawMode = drawMode;
}

inline void Terrain::setIndexCountToRender(int indexCount) {
	this->indexCount = indexCount;
	if (indexCount > num_indices) {
		std::cout << "Index Count " << indexCount << " greater than num_indices " << num_indices << std::endl;
	}
}

inline Terrain::Terrain(glm::vec3 position, const std::string & filePath, glm::vec3 scale) {
	this->position = position;
	this->filePath = filePath;
	this->scale = scale;

	//generateTerrain();
	glGenVertexArrays(1, &vertexArrayObject);
	glGenBuffers(1, &vertexAndNormalbuffer);
	glGenBuffers(1, &index_buffer);

	FreeImage_Initialise();
}

inline void Terrain::render(glm::mat4 view, glm::mat4 projection, GLuint programID, float lightDir[3])
{	
	
	//const int pos_loc = 0;
	//const int normal_loc = 1;

	//glBindAttribLocation(programID, pos_loc, "pos_attrib");
	//glBindAttribLocation(programID, tex_coord_loc, "tex_coord_attrib");
	//glBindAttribLocation(programID, normal_loc, "normal_attrib");

	/*glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertexCount, vertexBufferData, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);*/

	/*glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertexCount, vertexNormalData, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);*/

	modelMatrix = getModelMatrix();

	glm::mat4 mvp = projection * view * modelMatrix;

	int MatrixID = glGetUniformLocation(programID, "PVM");
	glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &mvp[0][0]);

	int colorID = glGetUniformLocation(programID, "color");
	if (colorID >= 0) {
		glUniform3f(colorID, 0.3f, 0.3f, 0.3f);
	}

	int lightDirID = glGetUniformLocation(programID, "lightPos");
	if (lightDirID >= 0) {
		glUniform3f(lightDirID, lightDir[0], lightDir[1], lightDir[2]);
	}

	int modelViewID = glGetUniformLocation(programID, "M");
	if (modelViewID >= 0) {
		glUniformMatrix4fv(modelViewID, 1, GL_FALSE, &modelMatrix[0][0]);
	}

	glUseProgram(programID);
	glBindVertexArray(vertexArrayObject);

	glEnable(GL_PRIMITIVE_RESTART);
	glPrimitiveRestartIndex(RESTART_PRIMITIVE_CODE);
	if (drawMode == 0)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_POINT);
	} 
	else if (drawMode == 1) 
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	}
	else if (drawMode == 2) 
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
	
	glDrawElements(GL_TRIANGLE_STRIP, indexCount, GL_UNSIGNED_INT, 0);

	glDisable(GL_PRIMITIVE_RESTART);

	//glDrawArrays(GL_TRIANGLE_STRIP, 0, vertexCount);

	glBindVertexArray(0);
}

inline glm::mat4 Terrain::getModelMatrix() {
	modelMatrix = glm::mat4(1.0f);
	//modelMatrix = glm::rotate(modelMatrix, 1.570796f, glm::vec3(1, 0, 0));
	modelMatrix = glm::translate(modelMatrix, position);
	modelMatrix = glm::scale(modelMatrix, scale);
	return modelMatrix;
}

inline void Terrain::generateTerrain(FIBITMAP * img) {
	glBindVertexArray(vertexArrayObject);
	int columnSize = FreeImage_GetWidth(img);
	int rowSize = FreeImage_GetHeight(img);
	vertices = std::vector<GLfloat>();
	std::vector<GLfloat> normals = std::vector<GLfloat>(rowSize * columnSize * 3);
	indexArray = std::vector<unsigned int>();
	std::vector<int> weights = std::vector<int>(rowSize * columnSize, 0);
	//float bitDivisor = std::numeric_limits<float>::max();// pow(2, (128 / 4)) - 1;

	num_indices = ((rowSize - 1) * (columnSize) * 2) + (rowSize - 1);
	indexCount = num_indices;

	unsigned int idx = 0;

	for (int i = 0; i < rowSize; i = i + 1) {
		FIRGBAF* columnVector = (FIRGBAF*)FreeImage_GetScanLine(img, i);
		FIRGBAF* columnVectorNext = (i == rowSize - 1)? nullptr: (FIRGBAF*)FreeImage_GetScanLine(img, i + 1);
		for (int j = 0; j < columnSize; j = j + 1) {
			//std::cout << "i j " << i << " " << j << std::endl;
			if (i != rowSize - 1) {
				indexArray.push_back(idx);
				indexArray.push_back(idx + columnSize);
				if (j == (columnSize - 1)) {
					indexArray.push_back(RESTART_PRIMITIVE_CODE);
				}
			}
			float pixelValue00 = columnVector[j].red;

			float columnSizeFloat = (float)columnSize;
			float rowSizeFloat = (float)rowSize;

			float normalizedJ = j / columnSizeFloat;
			float normalizedI = i / rowSizeFloat;
			
			glm::vec3 pixel00VertexPosition = glm::vec3(normalizedI, pixelValue00, normalizedJ);
			AddVertex(&vertices, pixel00VertexPosition);

			if (columnVectorNext != nullptr) {

				float pixelValue01 = columnVector[j + 1].red;

				float pixelValue10 = columnVectorNext[j].red;

				//float pixelValue11 = columnVectorNext[j + 1].red;
				/*std::cout << "Pixel Value is  00 " << pixelValue00 << " 01 " << pixelValue01 << " 10 "
					<< pixelValue10 << " 11 "
					<< pixelValue11 << " "
					<< std::endl;*/

					//Generate vertices for these 4 pixels and connect them by triangles.
				float normalizedJPlusOne = (j + 1) / columnSizeFloat;
				float normalizedIPlusOne = (i + 1) / rowSizeFloat;

				glm::vec3 pixel01VertexPosition = glm::vec3(normalizedI, pixelValue01, normalizedJPlusOne);
				glm::vec3 pixel10VertexPosition = glm::vec3(normalizedIPlusOne, pixelValue10, normalizedJ);
				//glm::vec3 pixel11VertexPosition = glm::vec3(normalizedIPlusOne, pixelValue11, normalizedJPlusOne);

				glm::vec3 p = glm::normalize(pixel01VertexPosition - pixel00VertexPosition);
				glm::vec3 q = glm::normalize(pixel10VertexPosition - pixel00VertexPosition);

				glm::vec3 normal = glm::normalize(glm::cross(p, q));

				//Add First Triangle

				//AddVertex(&vertices, pixel11VertexPosition);
				//AddVertex(&vertices, pixel10VertexPosition);
				AppendNormal(normals, normal, i, j, rowSize, columnSize, weights);
				AppendNormal(normals, normal, i + 1, j, rowSize, columnSize, weights);
				AppendNormal(normals, normal, i, j + 1, rowSize, columnSize, weights);
				//AddVertex(&normals, normal);
				//AddVertex(&normals, normal);
				//AddVertex(&normals, normal);

				if (j != 0) {
					float normalizedJMinusOne = (j - 1) / columnSizeFloat;

					float pixelValue1Minus1 = columnVectorNext[j - 1].red;
					glm::vec3 pixel1Minus1VertexPosition = glm::vec3(normalizedIPlusOne, pixelValue1Minus1, normalizedJMinusOne);

					glm::vec3 q2 = glm::normalize(pixel1Minus1VertexPosition - pixel00VertexPosition);

					glm::vec3 normal2 = glm::normalize(glm::cross(q, q2));
					AppendNormal(normals, normal2, i, j, rowSize, columnSize, weights);
					AppendNormal(normals, normal2, i + 1, j, rowSize, columnSize, weights);
					AppendNormal(normals, normal2, i + 1, j - 1, rowSize, columnSize, weights);
				}
			}


			//Add First Triangle
			//AddVertex(&vertices, pixel00VertexPosition);
			//AddVertex(&vertices, pixel01VertexPosition);
			//AddVertex(&vertices, pixel11VertexPosition);

			//AddVertex(&normals, normal2);
			//AddVertex(&normals, normal2);
			//AddVertex(&normals, normal2);

			idx++;
		}
	}

	//replace with actual png terrain generation data by adding to vertices vector data

	std::vector<GLfloat> finalData = std::vector<GLfloat>();

	for (int i = 0; i < vertices.size(); i += 3) {
		finalData.push_back(vertices[i]);
		finalData.push_back(vertices[i + 1]);
		finalData.push_back(vertices[i + 2]);

		finalData.push_back(normals[i]);
		finalData.push_back(normals[i + 1]);
		finalData.push_back(normals[i + 2]);
	}

	//vertexBufferData = &vertices[0];
	//vertexIndexData = &indexArray[0];
	//vertexCount = vertices.size();

	glBindBuffer(GL_ARRAY_BUFFER, vertexAndNormalbuffer); //Specify the buffer where vertex attribute data is stored.
	//Upload from main memory to gpu memory.
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * finalData.size(), finalData.data(), GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer); //GL_ELEMENT_ARRAY_BUFFER is the target for buffers containing indices
	//Upload from main memory to gpu memory.
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * indexArray.size(), indexArray.data(), GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	//Tell opengl how to get the attribute values out of the vbo (stride and offset).
	const int stride = (3 + 3) * sizeof(float);
	glVertexAttribPointer(0, 3, GL_FLOAT, false, stride, BUFFER_OFFSET(0));
	glVertexAttribPointer(1, 3, GL_FLOAT, false, stride, BUFFER_OFFSET((3) * sizeof(float)));

	glBindVertexArray(0); //unbind the vao
}

inline void Terrain::generateTerrain() {
	/*cv::Mat img = cv::imread(this->filePath, cv::ImreadModes::IMREAD_GRAYSCALE);//It returns a matrix object

	if (img.empty()) {                                    // if unable to open image
		std::cout << "error: image not read from file\n\n";        // show error message on command line
		return;
	}*/

	FIBITMAP* tempImg = FreeImage_Load(FreeImage_GetFileType(this->filePath.c_str(), 0), this->filePath.c_str());
	FIBITMAP* img = FreeImage_ConvertToRGBAF(tempImg);
	FreeImage_Unload(tempImg);

	/*std::vector<std::vector<int>> rowValues(img.rows);
	for (int x = 0; x < img.rows; x++) {
		std::vector<int> columnValues(img.cols);
		for (int y = 0; y < img.cols; y++) {
			int pixelValue = (int)img.at<uchar>(x, y);
			columnValues[y] = pixelValue;
			//std::cout << "Pixel Value is  " << pixelValue << std::endl;// [0] << " " << pixelValue[1] << " " << pixelValue[2] << std::endl;
		}
		rowValues[x] = columnValues;
	}*/

	generateTerrain(img);
}

inline void Terrain::AppendNormal(std::vector <GLfloat> & a, const glm::vec3 n, const int & i, const int & j, const int & rows, const int & cols, std::vector <int> & weights) {
	int weightsIndex = (i * cols) + j;
	int index = (i * cols * 3) + (j * 3);
	glm::vec3 currNormal = glm::vec3(a[index], a[index + 1], a[index + 2]);
	currNormal = glm::normalize(currNormal + n);
	//float xAvg = ((a[index] * weights[weightsIndex]) + n.x) / (weights[weightsIndex] + 1);
	//float yAvg = ((a[index + 1] * weights[weightsIndex]) + n.y) / (weights[weightsIndex] + 1);
	//float zAvg = ((a[index + 2] * weights[weightsIndex]) + n.z) / (weights[weightsIndex] + 1);
	weights[weightsIndex] += 1;
	a[index] = currNormal.x;
	a[index + 1] = currNormal.y;
	a[index + 2] = currNormal.z;
	//a->push_back(n.x);
	//a->push_back(n.y);
	//a->push_back(n.z);
}

inline void Terrain::AddVertex(std::vector <GLfloat> * a, const glm::vec3 v)
{
	a->push_back(v.x);
	a->push_back(v.y);
	a->push_back(v.z);
}

inline void Terrain::setScale(glm::vec3 scale) {
	this->scale = scale;
}

inline glm::vec3 Terrain::getScale() {
	return this->scale;
}

/* Code from Rules TB Surface*/
inline void Terrain::SaveOBJ(std::vector<char> filename) {
	modelMatrix = glm::mat4(1.0f);
	modelMatrix = glm::scale(modelMatrix, scale);
	std::string fileNameNew = std::string(filename.data()) + ".obj";
	//std::vector<char> fileNameNew = std::vector<char>(filename);
	//fileNameNew.a('.');// , 'o', 'b', 'j');
	//fileNameNew.push_back('o');
	std::ofstream myfile;
	myfile.open(fileNameNew.data());

	myfile << "# Generated by Bedrich Benes bbenes@purdue.edu\n";
	myfile << "# vertices\n";
	for (unsigned int i = 0; i < vertices.size(); i = i + 9) {

		glm::vec4 point1 = glm::vec4{ vertices.at(i + 2), vertices.at(i + 1), vertices.at(i), 1 };
		glm::vec4 pointTransformed1 = modelMatrix * point1;

		glm::vec4 point2 = glm::vec4{ vertices.at(i + 5), vertices.at(i + 4), vertices.at(i + 3), 1 };
		glm::vec4 pointTransformed2 = modelMatrix * point2;

		glm::vec4 point3 = glm::vec4{ vertices.at(i + 8), vertices.at(i + 7), vertices.at(i + 6), 1 };
		glm::vec4 pointTransformed3 = modelMatrix * point3;

		myfile << "v " << pointTransformed1[0] << " " << pointTransformed1[1] << " " << pointTransformed1[2] << "\n";
		myfile << "v " << pointTransformed2[0] << " " << pointTransformed2[1] << " " << pointTransformed2[2] << "\n";
		myfile << "v " << pointTransformed3[0] << " " << pointTransformed3[1] << " " << pointTransformed3[2] << "\n";

		/*myfile << "v " << v->at(i).a.GetZ() << " " << v->at(i).a.GetY() << " " << v->at(i).a.GetX() << "\n";
		myfile << "v " << v->at(i).b.GetZ() << " " << v->at(i).b.GetY() << " " << v->at(i).b.GetX() << "\n";
		myfile << "v " << v->at(i).c.GetZ() << " " << v->at(i).c.GetY() << " " << v->at(i).c.GetX() << "\n";*/
	}
	int counter = 1;
	myfile << "# faces\n";
	for (unsigned int i = 0; i < vertices.size(); i = i + 9) {
		myfile << "f " << counter++;
		myfile << " " << counter++;
		myfile << " " << counter++ << " " << "\n";
	}
	myfile.close();

}
