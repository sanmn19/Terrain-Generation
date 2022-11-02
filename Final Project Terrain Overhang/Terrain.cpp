#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <omp.h>
#include "FreeImage.h"
#include <fstream>
#include "LoadTexture.h"
#include <iostream>
#include "ShaderLocs.h"
#include <cmath>

#define RESTART_PRIMITIVE_CODE 0xffffffff

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

class Terrain {

	class Voxel;
	class RainVoxel;
	class GroundVoxel;

	glm::vec3 position;

	glm::vec3 scale;
	float angleY;

	std::string filePath;
	std::string complexFilePath;

	std::vector<std::vector<std::vector<Terrain::Voxel *>>> voxelMap;

	//GLfloat* vertexBufferData;
	//GLfloat* vertexNormalData;

	//unsigned int* vertexIndexData;

	GLuint vertexAndNormalbuffer;
	GLuint vertexArrayObject;

	int num_indices;
	int indexCount;
	int drawMode = 1;
	int maxVoxelCount = 0;

	int rainVoxelCreated = 0;
	int rainVoxelEvaporated = 0;
	
	int instanceCount = 0;

	std::vector <GLfloat> boxVertices = {
		-0.5f, -0.5f, -0.5f,
		-1, -1, -1,
		0.5f, -0.5f, -0.5f,
		1, -1, -1,
		0.5f, -0.5f, 0.5f,
		1, -1, 1,
		-0.5f, -0.5f, 0.5f,
		-1, -1, 1,
		-0.5f, 0.5f, -0.5f,
		-1, 1, -1,
		0.5f, 0.5f, -0.5f,
		1, 1, -1,
		0.5f, 0.5f, 0.5f,
		1, 1, 1,
		-0.5f, 0.5f, 0.5f,
		-1, 1, 1,
	};

	std::vector <unsigned int> boxIndices = {
		0, 1, 3, 2, 7, 6, 4, 5, 0, 1, RESTART_PRIMITIVE_CODE,
		0, 4, 3, 7, RESTART_PRIMITIVE_CODE,
		1, 5, 2, 6
	};

	std::vector<std::vector<int>> imageData;

	void AddVertex(std::vector <GLfloat>* a, const glm::vec3 v);
	void AppendNormal(std::vector <GLfloat>& a, const glm::vec3 n, const int& i, const int& j, const int& rows, const int& cols, std::vector <int>& weights);
	void AppendNormal(std::vector <GLfloat>& a, const glm::vec3 n, const int& index, std::vector <int>& weights);

	GLuint index_buffer;

	FIRGBAF* multiply(FIRGBAF* pixel, float multiplier);
	FIRGBAF* add(FIRGBAF* sum, FIRGBAF* pixel1, FIRGBAF* pixel2, FIRGBAF* pixel3);

	std::vector<glm::vec3> generateIntermediateBezierPoints(int axis, glm::vec3& initialPoint, glm::vec3 & finalPoint, glm::vec2 controlPoint1, glm::vec2 controlPoint2, int pointsToBeGenerated);
	float getBezierValue(float initialPoint, float controlPoint1, float controlPoint2, float finalPoint, float t);
	int getIndexForRowColumn(const int& i, const int& j, const int& rowSize, const int& columnSize);
	void convertFloatTo4Parts(float value1, glm::vec2& controlPointNormalized1, glm::vec2& controlPointNormalized2, float heightDifference);
	void addXAxisVertices(const int& j, const int& columnSize, FIRGBAF* columnVector, glm::vec3 pixel00VertexPosition,
		glm::vec3 pixel01VertexPosition, std::vector<GLfloat>& vertices);
	std::vector<glm::vec3> getXAxisVertices(const int& j, const int& columnSize, FIRGBAF* columnVector, glm::vec3 pixel00VertexPosition, 
		glm::vec3 pixel01VertexPosition);
	std::vector<glm::vec3> getZAxisVertices(const int& j, const int& columnSize, FIRGBAF* columnVector, glm::vec3 pixel00VertexPosition,
		glm::vec3 pixel10VertexPosition);

	void generateRenderBuffers(int rowSize, int columnSize);

	FIBITMAP* originalHeightMap;
	FIBITMAP* complexImage;
	bool renderMode = 0; //0 = Voxel 1 = Mesh

	void thermalErosionCell(std::vector<Voxel*>& currentStack, std::vector<Voxel*>& neighbourStack, float maximumHeightDifference, float sumOfHeights, float maxVolume);
	void transferSediment(std::vector<Voxel*>& currentStack, std::vector<Voxel*>& neighbourStack, float maximumHeightDifference, float sumOfHeights, float maxVolume);
	void transferSedimentFromSide(std::vector<Voxel*>& currentStack, std::vector<Voxel*>& neighbourStack, float maximumHeightDifference, float sumOfHeights, float maxVolume);
	bool transferWater(std::vector<Voxel*>& currentStack, std::vector<Voxel*>& neighbourStack, float maximumHeightDifference, float sumOfHeights, float maxVolume);
	float getAngleOfTalus(float height, float verticalDifference);
	void createNewRainVoxel(RainVoxel** rainVoxel, float waterQuantity);
	int movedVoxelCount = 0;

public:
	bool isThermalErosionInProgress = false;
	bool hasThermalErosionCompleted = false;

	bool isHydraulicErosionInProgress = false;
	bool hasHydraulicErosionCompleted = false;

	bool forTraining = true;

	class Voxel {
	public:
		//glm::vec3 velocity;
		int materialId;
	};

	class GroundVoxel : public Voxel {
	public:
		double angleOfTalus;
	};

	class RainVoxel : public Voxel {
	public:
		float waterQuantity = 1;
		float sedimentFromGroundVolume = 0;
		float sedimentVolume = 0;
		std::vector<Voxel *> dissolvedGround;

		float getSedimentRatio() {
			float ratio = (sedimentVolume + sedimentFromGroundVolume) / waterQuantity;
			return ratio;
		}

		bool evaporate() {
			waterQuantity *= exp(-0.1f);
			return waterQuantity <= 0.05f;
		}

		bool absorbSedimentFromGround(float angleOfTalus = 1.0f) {
			return absorbSediment(angleOfTalus);
			/*
			float ratio = getSedimentRatio();
			if (ratio < 0.3f) {
				sedimentFromGroundVolume += 0.02f;
				if (sedimentFromGroundVolume > 1) {		//TODO limit sediment volume absorbed by water while
					sedimentFromGroundVolume -= 1;
					return true;
				}
			}

			return false;*/
		}

		bool absorbSediment(float angleOfTalus = 1.0f) {
			float ratio = getSedimentRatio();
			if (ratio < 0.8f) {
				int previousSedimentVolume = (int)sedimentVolume;
				sedimentVolume += (45 * 45) / (angleOfTalus * angleOfTalus);
				int diff = sedimentVolume - previousSedimentVolume;
				if (diff >= 1) {		//TODO limit sediment volume absorbed by water while
					return true;
				}
			}
			
			return false;
		}

		bool loseSediment() {
			int previousSedimentVolume = (int)sedimentVolume;
			sedimentVolume -= 1.0f;
			int diff = sedimentVolume - previousSedimentVolume;
			return diff >= 1;
		}

	};

	int instancesToRender;
	//unsigned int vertexCount;
	std::vector <GLfloat> vertices;
	//std::vector <GLfloat> normals;
	std::vector <unsigned int> indexArray;
	glm::mat4 modelMatrix;
	double voxelDimension;
	double voxelDimensionVertical;

	Terrain() = default;
	Terrain(bool forTraining, glm::vec3 position, const std::string & filePath, const std::string & complexFilePath, glm::vec3 scale, double voxelDimension, int renderMode);

	void render(glm::mat4 view, glm::mat4 projection, GLuint programID, float lightDir[3]);

	void setAngle(float angle);
	void setScale(glm::vec3 scale);
	void setDrawMode(int drawMode);
	void setRenderMode(int renderMode);

	glm::vec3 getScale();
	void setIndexCountToRender(int indexCount);
	void setInstancesToRender(int instanceCount);

	void exportOutput(FIBITMAP* img, const char* outputFileName);
	glm::vec4 processVoxelCell(int i , int j);

	void generateTerrain();
	void generateTerrainForFinalOutput(FIBITMAP* img);

	glm::mat4 getModelMatrix();
	void generateTerrain(FIBITMAP * img, FIBITMAP* complexImg);
	std::vector<GLuint> createModelMatrixBuffers(int rowSize, int columnSize, bool createBuffer = true);

	void SaveOBJ(std::vector<char> filename);

	void performThermalErosion(int steps);
	void performHydraulicErosion(int steps, bool addRain);

	void updateTerrain();

	std::vector<std::vector<FIRGBAF*>> convolution(FIBITMAP* img);
};

inline void Terrain::setAngle(float angle) {
	this->angleY = angle;
}

inline void Terrain::setDrawMode(int drawMode) {
	this->drawMode = drawMode;
}

inline void Terrain::setIndexCountToRender(int indexCount) {
	this->indexCount = indexCount;
	if (indexCount > num_indices) {
		std::cout << "Index Count " << indexCount << " greater than num_indices " << num_indices << std::endl;
		this->indexCount = num_indices;
	}
}

inline void Terrain::setInstancesToRender(int instanceCount) {
	this->instancesToRender = instanceCount;
	if (instancesToRender > instanceCount) {
		std::cout << "Instance Count " << instancesToRender << " greater than actual instances " << instanceCount << std::endl;
		instancesToRender = instanceCount;
	}
}

inline void Terrain::setRenderMode(int renderMode) {
	this->renderMode = renderMode;
}

inline Terrain::Terrain(bool forTraining, glm::vec3 position, const std::string & filePath, const std::string& complexFilePath, glm::vec3 scale, double voxelDimension = 1.0, int renderMode = 0) {
	this->forTraining = forTraining;
	setRenderMode(renderMode);
	this->position = position;
	this->filePath = filePath;
	this->complexFilePath = complexFilePath;
	this->scale = scale;
	this->voxelDimension = voxelDimension;
	this->voxelDimensionVertical = voxelDimension / 2;

	num_indices = boxIndices.size();
	indexCount = num_indices;

	glGenVertexArrays(1, &vertexArrayObject);
	glGenBuffers(1, &vertexAndNormalbuffer);
	glGenBuffers(1, &index_buffer);

	maxVoxelCount = 1 / voxelDimensionVertical;

	FreeImage_Initialise();
}

inline void Terrain::render(glm::mat4 view, glm::mat4 projection, GLuint programID, float lightDir[3])
{	
	modelMatrix = getModelMatrix();

	glm::mat4 mvp = projection * view;

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
	
	//glDrawElements(GL_TRIANGLE_STRIP, indexCount, GL_UNSIGNED_INT, 0);
	glDrawElementsInstanced( GL_TRIANGLE_STRIP,
		indexCount,
		GL_UNSIGNED_INT,
		0,
		instancesToRender);

	glDisable(GL_PRIMITIVE_RESTART);

	//glDrawArrays(GL_TRIANGLE_STRIP, 0, vertexCount);

	glBindVertexArray(0);
}

inline glm::mat4 Terrain::getModelMatrix() {
	modelMatrix = glm::mat4(1.0f);
	modelMatrix = glm::rotate(modelMatrix, angleY, glm::vec3(0, 1, 0));
	modelMatrix = glm::translate(modelMatrix, position);
	modelMatrix = glm::scale(modelMatrix, scale);
	return modelMatrix;
}

inline float Terrain::getBezierValue(float initialPoint, float controlPoint1, float controlPoint2, float finalPoint, float t) {
	return (pow((1 - t), 3) * initialPoint) + (3 * t * pow(1 - t, 2) * controlPoint1)
		+ (3 * pow(t, 2) * (1 - t) * controlPoint2) + (pow(t, 3) * finalPoint);
}

inline std::vector<glm::vec3> Terrain::generateIntermediateBezierPoints(int axis, glm::vec3& initialPoint, glm::vec3& finalPoint, glm::vec2 controlPointNormalized1, glm::vec2 controlPointNormalized2, int pointsToBeGenerated) {
	std::vector<glm::vec3> generatedPoints;
	generatedPoints.reserve(pointsToBeGenerated);

	float increment = (0.6f / float(pointsToBeGenerated)); //Including initial and finalPoints but based on no. of divisions 5 points = 4 divisions

	if (axis == 0) //XAxis
	{
		float controlPoint1x = (initialPoint.x * (1 - controlPointNormalized1.x)) + (finalPoint.x * controlPointNormalized1.x);
		float controlPoint1y = (initialPoint.y * (1 - controlPointNormalized1.y)) + (finalPoint.y * controlPointNormalized1.y);
		glm::vec2 controlPoint1 = glm::vec2(controlPoint1x, controlPoint1y);

		float controlPoint2x = (initialPoint.x * (1 - controlPointNormalized2.x)) + (finalPoint.x * controlPointNormalized2.x);
		float controlPoint2y = (initialPoint.y * (1 - controlPointNormalized2.y)) + (finalPoint.y * controlPointNormalized2.y);
		glm::vec2 controlPoint2 = glm::vec2(controlPoint2x, controlPoint2y);

		for (float t = 0.4f; t < 1; t += increment) {
			float x = getBezierValue(initialPoint.x, controlPoint1.x, controlPoint2.x, finalPoint.x, t);
			float y = getBezierValue(initialPoint.y, controlPoint1.y, controlPoint2.y, finalPoint.y, t);

			generatedPoints.push_back(glm::vec3(x, y, initialPoint.z));
		}
	}
	else if (axis == 1) //ZAxis
	{
		float controlPoint1z = (initialPoint.z * (1 - controlPointNormalized1.x)) + (finalPoint.z * controlPointNormalized1.x);
		float controlPoint1y = (initialPoint.y * (1 - controlPointNormalized1.y)) + (finalPoint.y * controlPointNormalized1.y);
		glm::vec2 controlPoint1 = glm::vec2(controlPoint1z, controlPoint1y);

		float controlPoint2z = (initialPoint.z * (1 - controlPointNormalized2.x)) + (finalPoint.z * controlPointNormalized2.x);
		float controlPoint2y = (initialPoint.y * (1 - controlPointNormalized2.y)) + (finalPoint.y * controlPointNormalized2.y);
		glm::vec2 controlPoint2 = glm::vec2(controlPoint2z, controlPoint2y);

		for (float t = 0.4f; t < 1 ; t += increment) {
			float z = getBezierValue(initialPoint.z, controlPoint1.x, controlPoint2.x, finalPoint.z, t);
			float y = getBezierValue(initialPoint.y, controlPoint1.y, controlPoint2.y, finalPoint.y, t);

			generatedPoints.push_back(glm::vec3(initialPoint.x, y, z));
		}
	}
	
	return generatedPoints;
}

inline void Terrain::convertFloatTo4Parts(float value1, glm::vec2 & controlPointNormalized1, glm::vec2 & controlPointNormalized2, float heightDifference) {
	if (heightDifference < -0.5) {
		controlPointNormalized1 = glm::vec2(2, 0);
		controlPointNormalized2 = glm::vec2(2, 0);
	}
	else {
		controlPointNormalized1 = glm::vec2(0, 0);
		controlPointNormalized2 = glm::vec2(0, 0);
	}
}

inline int Terrain::getIndexForRowColumn(const int & i, const int & j, const int & rowSize, const int & columnSize) {
	return (i * columnSize * 7) - (i * 3) + ((i != rowSize - 1) ? j * 7: j * 4);
}

inline void Terrain::addXAxisVertices(const int & j, const int & columnSize, FIRGBAF* columnVector, glm::vec3 pixel00VertexPosition, 
										glm::vec3 pixel01VertexPosition, std::vector<GLfloat> & vertices) {
	std::vector<glm::vec3> xAxisBezierPoints = getXAxisVertices(j, columnSize, columnVector, pixel00VertexPosition, pixel01VertexPosition);
		
	for (int n = 0; n < xAxisBezierPoints.size(); n++) {
		AddVertex(&vertices, xAxisBezierPoints[n]);
	}
}

inline std::vector<glm::vec3> Terrain::getXAxisVertices(const int& j, const int& columnSize, FIRGBAF* columnVector, glm::vec3 pixel00VertexPosition,
	glm::vec3 pixel01VertexPosition) {
	glm::vec2 controlPointNormalized1;
	glm::vec2 controlPointNormalized2;
	convertFloatTo4Parts(columnVector[j].blue, controlPointNormalized1, controlPointNormalized2, pixel00VertexPosition.y - pixel01VertexPosition.y);

	return generateIntermediateBezierPoints(1,
		pixel00VertexPosition, pixel01VertexPosition,
		controlPointNormalized1, controlPointNormalized2, 3);
}

inline std::vector<glm::vec3> Terrain::getZAxisVertices(const int& j, const int& columnSize, FIRGBAF* columnVector, glm::vec3 pixel00VertexPosition,
	glm::vec3 pixel10VertexPosition) {
	glm::vec2 controlPointNormalized1;
	glm::vec2 controlPointNormalized2;
	convertFloatTo4Parts(columnVector[j].green, controlPointNormalized1, controlPointNormalized2, pixel00VertexPosition.y - pixel10VertexPosition.y);

	return generateIntermediateBezierPoints(0,
		pixel00VertexPosition, pixel10VertexPosition,
		controlPointNormalized1, controlPointNormalized2, 3);
}

inline FIRGBAF * Terrain::multiply(FIRGBAF* pixel, float multiplier) {
	FIRGBAF* newPixel = new FIRGBAF();
	newPixel->alpha = 0;
	newPixel->red = pixel->red * multiplier;
	newPixel->green = pixel->green * multiplier;
	newPixel->blue = pixel->blue * multiplier;

	return newPixel;
}

inline FIRGBAF* Terrain::add(FIRGBAF * sum, FIRGBAF* pixel1, FIRGBAF* pixel2, FIRGBAF* pixel3) {
	sum->red += pixel1->red + pixel2->red + pixel3->red;
	sum->green += pixel1->green + pixel2->green + pixel3->green;
	sum->blue += pixel1->blue + pixel2->blue + pixel3->blue;
	sum->alpha += pixel1->alpha + pixel2->alpha + pixel3->alpha;

	return sum;
}

inline std::vector<std::vector<FIRGBAF *>> Terrain::convolution(FIBITMAP* img)
{
	int columnSize = FreeImage_GetWidth(img);
	int rowSize = FreeImage_GetHeight(img);

	std::vector<std::vector<FIRGBAF *>> result;

	for (int i = 0; i < rowSize; i = i + 1) {
		
		//std::cout << "i " << i << std::endl;
		FIRGBAF* columnVector = (FIRGBAF*)FreeImage_GetScanLine(img, i);
		FIRGBAF* columnMinusOneVector = nullptr;
		if (i != 0) {
			columnMinusOneVector = (FIRGBAF*)FreeImage_GetScanLine(img, i - 1);
		}
		else {
			columnMinusOneVector = columnVector;
		}
		 
		FIRGBAF* columnPlusOneVector = nullptr;
		if (i != rowSize - 1) {
			columnPlusOneVector = (FIRGBAF*)FreeImage_GetScanLine(img, i + 1);
		}
		else {
			columnPlusOneVector = columnVector;
		}

		std::vector<FIRGBAF *> columns;

		for (int j = 0; j < columnSize; j = j + 1) {
			FIRGBAF* value = new FIRGBAF();
			value->red = 0;
			value->green = 0;
			value->blue = 0;
			value->alpha = 0;

			if (j == 0) {
				value = add(value, multiply(&columnMinusOneVector[0], 0.0947416f), multiply(&columnMinusOneVector[j], 0.118318f), multiply(&columnMinusOneVector[j + 1], 0.0947416f));
				value = add(value, multiply(&columnVector[0], 0.118318f), multiply(&columnVector[j], 0.147761f), multiply(&columnVector[j + 1], 0.118318f));
				value = add(value, multiply(&columnPlusOneVector[0], 0.0947416f), multiply(&columnPlusOneVector[j], 0.118318f), multiply(&columnPlusOneVector[j + 1], 0.0947416f));
			}
			else if (j == columnSize - 1) {
				value = add(value, multiply(&columnMinusOneVector[j - 1], 0.0947416f), multiply(&columnMinusOneVector[j], 0.118318f), multiply(&columnMinusOneVector[columnSize - 1], 0.0947416f));
				value = add(value, multiply(&columnVector[j - 1], 0.118318f), multiply(&columnVector[j], 0.147761f), multiply(&columnVector[columnSize - 1], 0.118318f));
				value = add(value, multiply(&columnPlusOneVector[j - 1], 0.0947416f), multiply(&columnPlusOneVector[j], 0.118318f), multiply(&columnPlusOneVector[columnSize - 1], 0.0947416f));
			}
			else {
				value = add(value, multiply(&columnMinusOneVector[j - 1], 0.0947416f), multiply(&columnMinusOneVector[j], 0.118318f), multiply(&columnMinusOneVector[j + 1], 0.0947416f));
				value = add(value, multiply(&columnVector[j - 1], 0.118318f), multiply(&columnVector[j], 0.147761f), multiply(&columnVector[j + 1], 0.118318f));
				value = add(value, multiply(&columnPlusOneVector[j - 1], 0.0947416f), multiply(&columnPlusOneVector[j], 0.118318f), multiply(&columnPlusOneVector[j + 1], 0.0947416f));
			}

			if (value->red != 0) {
				value->red += 0.35f;
			}

			if (value->green != 0) {
				value->green += 0.35f;
			}

			if (value->blue != 0) {
				value->blue += 0.35f;
			}

			if (value->alpha != 0) {
				value->alpha += 0.35f;
			}
			
			columns.push_back(value);
		}

		result.push_back(columns);
	}

	return result;
}

inline std::vector<GLuint> Terrain::createModelMatrixBuffers(int rowSize, int columnSize, bool createBuffer)
{
	std::vector<glm::mat4>* matrix_data_solid = new std::vector<glm::mat4>();
	//std::vector<glm::mat4>* matrix_data_eroded = new std::vector<glm::mat4>();
	std::vector<glm::vec4>* colors = new std::vector<glm::vec4>();

	for (int i = 0; i < rowSize; i++)
	{
		for (int j = 0; j < columnSize; j++)
		{
			float height = 0;
			for (int k = 0; k < voxelMap[i][j].size(); k++)
			{
				
				if (voxelMap[i][j][k]->materialId == 0) {
					glm::vec3 tran = glm::vec3(i * voxelDimension, height * voxelDimensionVertical, j * voxelDimension);
					glm::mat4 trans = glm::translate(glm::mat4(1.f), tran);
					height++;
					trans = glm::scale(trans, glm::vec3(voxelDimension, voxelDimensionVertical, voxelDimension));
					matrix_data_solid->push_back(trans);
					colors->push_back(glm::vec4(1, 1, 1, 1));
				}
				else if (voxelMap[i][j][k]->materialId == 1) {
					glm::vec3 tran = glm::vec3(i * voxelDimension, height * voxelDimensionVertical, j * voxelDimension);
					glm::mat4 trans = glm::translate(glm::mat4(1.f), tran);
					height++;
					trans = glm::scale(trans, glm::vec3(voxelDimension, voxelDimensionVertical, voxelDimension));
					matrix_data_solid->push_back(trans);
					//matrix_data_eroded->push_back(trans);
					colors->push_back(glm::vec4(0, 0, 0, 1));
				}
				else if (voxelMap[i][j][k]->materialId == 3) {
					RainVoxel* rainVoxel = (RainVoxel*)voxelMap[i][j][k];
					glm::vec3 tran = glm::vec3(i * voxelDimension, (height + (rainVoxel->waterQuantity / 2)) * voxelDimensionVertical, j * voxelDimension);
					glm::mat4 trans = glm::translate(glm::mat4(1.f), tran);
					height += rainVoxel->waterQuantity;
					trans = glm::scale(trans, glm::vec3(voxelDimension, rainVoxel->waterQuantity * voxelDimensionVertical, voxelDimension));
					matrix_data_solid->push_back(trans);
					if (rainVoxel->dissolvedGround.size() != 0) {
						colors->push_back(glm::vec4(0, 0, 1, 1));
					}
					else {
						colors->push_back(glm::vec4(0, 0, 0.15f, 1));
					}
				}
				
			}
		}
	}

	std::vector<GLuint> model_matrix_buffers;

	if (createBuffer) {
		GLuint solidMaterialBuffer;
		glGenBuffers(1, &solidMaterialBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, solidMaterialBuffer);
		glBufferData(GL_ARRAY_BUFFER, matrix_data_solid->size() * sizeof(glm::mat4), matrix_data_solid->data(), GL_DYNAMIC_DRAW);
		model_matrix_buffers.push_back(solidMaterialBuffer);

		GLuint colorData;
		glGenBuffers(1, &colorData);
		glBindBuffer(GL_ARRAY_BUFFER, colorData);
		glBufferData(GL_ARRAY_BUFFER, colors->size() * sizeof(glm::vec4), colors->data(), GL_DYNAMIC_DRAW);
		model_matrix_buffers.push_back(colorData);

		/*GLuint erodedMaterialBuffer;
		glGenBuffers(1, &erodedMaterialBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, erodedMaterialBuffer);
		glBufferData(GL_ARRAY_BUFFER, matrix_data_eroded->size() * sizeof(glm::mat4), matrix_data_eroded->data(), GL_DYNAMIC_DRAW);
		model_matrix_buffers.push_back(erodedMaterialBuffer);*/
	}

	delete colors;
	delete matrix_data_solid;

	return model_matrix_buffers;
}

inline void Terrain::generateRenderBuffers(int rowSize, int columnSize) {
	if (renderMode == 0) {	//Voxel
		glBindVertexArray(vertexArrayObject);
		std::vector<GLuint> instanceMatrices = createModelMatrixBuffers(rowSize, columnSize, true);
		
		glBindBuffer(GL_ARRAY_BUFFER, instanceMatrices[0]);
		// bounding model matrix to shader
		for (int i = 0; i < 4; i++)
		{
			glVertexAttribPointer(AttribLoc::matPosInstance + i,
				4, GL_FLOAT, GL_FALSE,
				sizeof(glm::mat4),
				(void*)(sizeof(glm::vec4) * i));
			glEnableVertexAttribArray(AttribLoc::matPosInstance + i);
			glVertexAttribDivisor(AttribLoc::matPosInstance + i, 1);
		}

		glBindBuffer(GL_ARRAY_BUFFER, instanceMatrices[1]);
		glVertexAttribPointer(AttribLoc::colorsInstance,
			4, GL_FLOAT, GL_FALSE,
			sizeof(glm::vec4),
			(void*)0);
		glEnableVertexAttribArray(AttribLoc::colorsInstance);
		glVertexAttribDivisor(AttribLoc::colorsInstance, 1);

		//replace with actual png terrain generation data by adding to vertices vector data

		glBindBuffer(GL_ARRAY_BUFFER, vertexAndNormalbuffer); //Specify the buffer where vertex attribute data is stored.
		//Upload from main memory to gpu memory.
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * boxVertices.size(), boxVertices.data(), GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer); //GL_ELEMENT_ARRAY_BUFFER is the target for buffers containing indices
		//Upload from main memory to gpu memory.
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * boxIndices.size(), boxIndices.data(), GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);

		//Tell opengl how to get the attribute values out of the vbo (stride and offset).
		const int stride = (3 + 3) * sizeof(float);
		glVertexAttribPointer(0, 3, GL_FLOAT, false, stride, BUFFER_OFFSET(0));
		glVertexAttribPointer(1, 3, GL_FLOAT, false, stride, BUFFER_OFFSET((3) * sizeof(float)));

		glBindVertexArray(0); //unbind the vao
	}
	else if (renderMode == 1) {
		//should I use marching cubes algo? I think yes TODO
	}
}

inline void Terrain::generateTerrain(FIBITMAP * img, FIBITMAP* complexImg) {
	instanceCount = 0;
	
	int columnSize = FreeImage_GetWidth(img);
	int rowSize = FreeImage_GetHeight(img);
	vertices = std::vector<GLfloat>();
	std::vector<GLfloat> normals = std::vector<GLfloat>( (((rowSize - 1) * 4) + 1) * (((columnSize - 1) * 4) + 1) * 3 );
	indexArray = std::vector<unsigned int>();
	std::vector<int> weights = std::vector<int>(rowSize * columnSize, 0);
	//float bitDivisor = std::numeric_limits<float>::max();// pow(2, (128 / 4)) - 1;

	unsigned int idx = 0;

	float columnSizeFloat = (float)columnSize;
	float rowSizeFloat = (float)rowSize;

	std::vector<std::vector<FIRGBAF *>> gaussianBlurredMatrix = convolution(complexImg);

	//voxelMap.reserve(rowSize);
	

	for (int i = 0; i < rowSize; i = i + 1) {
		//std::cout << "i " << i << std::endl;
		FIRGBAF* columnVector = (FIRGBAF*)FreeImage_GetScanLine(img, i);
		//FIRGBAF* complexColumnVector = (FIRGBAF*)FreeImage_GetScanLine(complexImg, i);

		std::vector<std::vector<Voxel*>> columnVoxels;

		for (int j = 0; j < columnSize; j = j + 1) {
			//std::cout << "i j " << i << " " << j << std::endl;
			float height = columnVector[j].red;
			float voxelCount = height / voxelDimensionVertical;

			std::vector<Voxel*> rowColumnVoxels;

			for (int k = 0; k < voxelCount; k++) {
				Voxel* newVoxel = new GroundVoxel();
				
				rowColumnVoxels.push_back(newVoxel);
				instanceCount++;
			}

			columnVoxels.push_back(rowColumnVoxels);

			//std::cout << "Adding Vertex " << i << " " << j << std::endl;
		}

		voxelMap.push_back(columnVoxels);
	}

	instancesToRender = instanceCount;

	float maxHeightOfTerrain = 0;	//Maybe used later

	for (int i = 0; i < rowSize; i = i + 1) {
		//std::cout << "i " << i << std::endl;

		for (int j = 0; j < columnSize; j = j + 1) {
			float overHangValue = gaussianBlurredMatrix[i][j]->red;
			float caveValue = gaussianBlurredMatrix[i][j]->green;

			float looseMaterialHeight = 0;
			float looseMaterialStart = -1;
			int complexFeatureFoundCode = 0;

			float height = voxelMap[i][j].size() * voxelDimensionVertical;

			if (overHangValue > 0) {	//Overhangs override caves
				looseMaterialHeight = (overHangValue) * 0.7f * height;
				looseMaterialStart = (overHangValue) * 0.15f * height;
				complexFeatureFoundCode = 1;
			}
			else {
				if (caveValue > 0) {
					looseMaterialHeight = (caveValue) * 0.3f; //Static height of cave
					looseMaterialStart = (caveValue) * 0.15f * height;  //Is at a relative height
					complexFeatureFoundCode = 2;
				}
				idx++;
			}

			int startVoxel = int(looseMaterialStart / voxelDimensionVertical);
			int endVoxel = int((looseMaterialStart + looseMaterialHeight) / voxelDimensionVertical);

			for (int k = 0; k < voxelMap[i][j].size(); k++) {
				if (k >= startVoxel && k <= endVoxel)
				{
					voxelMap[i][j][k]->materialId = 1;
					((GroundVoxel *)voxelMap[i][j][k])->angleOfTalus = 30;
					//Assign other properties like angle of talus
				}
				else
				{
					voxelMap[i][j][k]->materialId = 0;
					((GroundVoxel *)voxelMap[i][j][k])->angleOfTalus = 45;
				}
			}

		}
	}

	generateRenderBuffers(rowSize, columnSize);
}

inline void Terrain::generateTerrainForFinalOutput(FIBITMAP* img) {
	instanceCount = 0;
	int columnSize = FreeImage_GetWidth(img);
	int rowSize = FreeImage_GetHeight(img);

	for (int i = 0; i < rowSize; i = i + 1) {
		//std::cout << "i " << i << std::endl;
		FIRGBAF* columnVector = (FIRGBAF*)FreeImage_GetScanLine(img, i);

		std::vector<std::vector<Voxel*>> columnVoxels;

		for (int j = 0; j < columnSize; j = j + 1) {
			//std::cout << "i j " << i << " " << j << std::endl;
			float baseRockHeight = columnVector[j].red;
			float airLayerHeight = columnVector[j].green;
			float overhangHeight = columnVector[j].blue;
			float baseVoxelCount = (baseRockHeight) / voxelDimensionVertical;
			float airVoxelCount = (airLayerHeight) / voxelDimensionVertical;
			float overhangCount = (overhangHeight) / voxelDimensionVertical;

			std::vector<Voxel*> rowColumnVoxels;

			for (int k = 0; k < baseVoxelCount; k++) {
				Voxel* newVoxel = new Voxel();
				newVoxel->materialId = 0;

				rowColumnVoxels.push_back(newVoxel);
				instanceCount++;
			}

			for (int k = 0; k < airVoxelCount; k++) {
				Voxel* newVoxel = new Voxel();
				newVoxel->materialId = 2;

				rowColumnVoxels.push_back(newVoxel);
				instanceCount++;
			}

			for (int k = 0; k < overhangCount; k++) {
				Voxel* newVoxel = new Voxel();
				newVoxel->materialId = 0;

				rowColumnVoxels.push_back(newVoxel);
				instanceCount++;
			}

			columnVoxels.push_back(rowColumnVoxels);

			//std::cout << "Adding Vertex " << i << " " << j << std::endl;
		}

		voxelMap.push_back(columnVoxels);
	}

	instancesToRender = instanceCount;

	generateRenderBuffers(rowSize, columnSize);
}

inline void Terrain::generateTerrain() {
	/*cv::Mat img = cv::imread(this->filePath, cv::ImreadModes::IMREAD_GRAYSCALE);//It returns a matrix object

	if (img.empty()) {                                    // if unable to open image
		std::cout << "error: image not read from file\n\n";        // show error message on command line
		return;
	}*/

	if (forTraining) {
		FIBITMAP* tempImg = FreeImage_Load(FreeImage_GetFileType(this->filePath.c_str(), 0), this->filePath.c_str());
		FIBITMAP* img = FreeImage_ConvertToRGBAF(tempImg);
		//FreeImage_Unload(tempImg);
		this->originalHeightMap = img;

		FIBITMAP* tempImg2 = FreeImage_Load(FreeImage_GetFileType(this->complexFilePath.c_str(), 0), this->complexFilePath.c_str());
		FIBITMAP* img2 = FreeImage_ConvertToRGBAF(tempImg2);
		this->complexImage = img2;

		//FreeImage_Unload(tempImg2);

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

		generateTerrain(img, img2);
	}
	else {
		FIBITMAP* tempImg = FreeImage_Load(FreeImage_GetFileType(this->filePath.c_str(), 0), this->filePath.c_str());
		FIBITMAP* img = FreeImage_ConvertToRGBAF(tempImg);
		//FreeImage_Unload(tempImg);
		this->originalHeightMap = img;

		generateTerrainForFinalOutput(img);
	}
}

inline void Terrain::updateTerrain() {
	generateRenderBuffers(voxelMap.size(), voxelMap[0].size());
}

inline void Terrain::thermalErosionCell(std::vector<Voxel*>& currentStack, std::vector<Voxel*>& neighbourStack, float maximumHeightDifference, float sumOfHeights, float maxVolume) {
	float currentStackHeight = currentStack.size() * voxelDimensionVertical;
	float neighbourStackHeight = neighbourStack.size() * voxelDimensionVertical;

	float heightDifference = (currentStackHeight - neighbourStackHeight);

	if (heightDifference > 0) {
		float angleOfTalus = getAngleOfTalus(heightDifference, voxelDimension);

		bool allVoxelTransferred = false;
		int index = currentStack.size() - 1;

		float volumeProportion = maxVolume * (heightDifference / sumOfHeights);
		int voxels = volumeProportion / voxelDimensionVertical;

		while (!allVoxelTransferred && index >= 0 && voxels > 0) {
			if (angleOfTalus > ((GroundVoxel *)currentStack[index])->angleOfTalus) {
				
				//transfer 1 voxel every loop measuring angle of talus against top layer
				movedVoxelCount++;
				neighbourStack.push_back(currentStack[index]);
				currentStack.erase(currentStack.end() - 1);
				//TODO modify the material to have a velocity or change angle of talus to give the effect of accelerated movement downward

				voxels--;

				if (voxels <= 0) {
					allVoxelTransferred = true;
				}
			}
			else {
				allVoxelTransferred = true;	
			}
			index--;
		}
	}
	
}

inline void Terrain::transferSediment(std::vector<Voxel*>& currentStack, std::vector<Voxel*>& neighbourStack, float maximumHeightDifference, float sumOfHeights, float maxVolume) {
	RainVoxel* currentRainVoxel = (RainVoxel*)currentStack[currentStack.size() - 1];

	float currentStackHeight = ((currentStack.size() - 1) * voxelDimensionVertical) + (currentRainVoxel->waterQuantity * voxelDimensionVertical);
	float neighbourStackHeight = 0;

	if (neighbourStack.size() != 0 && neighbourStack[neighbourStack.size() - 1]->materialId == 3) {
		RainVoxel* neighbourRainVoxel = (RainVoxel*)neighbourStack[neighbourStack.size() - 1];
		neighbourStackHeight = ((neighbourStack.size() - 1) * voxelDimensionVertical) + (neighbourRainVoxel->waterQuantity * voxelDimensionVertical);
	}
	else {
		neighbourStackHeight = (neighbourStack.size()) * voxelDimensionVertical;
	}

	float heightDifference = (currentStackHeight - neighbourStackHeight);

	if (heightDifference > 0) {
		float volumeProportion = maxVolume * (heightDifference / sumOfHeights);
		int voxels = volumeProportion / voxelDimensionVertical;

		RainVoxel* neighbourVoxel = (RainVoxel *) neighbourStack[neighbourStack.size() - 1];
		std::vector<Voxel *>& currentDissolvedGround = currentRainVoxel->dissolvedGround;
		std::vector<Voxel*>& neighbourDissolvedGround = neighbourVoxel->dissolvedGround;

		while (voxels > 0 && currentDissolvedGround.size() > 0) {
			GroundVoxel* groundVoxel = (GroundVoxel *) currentDissolvedGround[currentDissolvedGround.size() - 1];
			bool neighbourSediment = neighbourVoxel->absorbSediment(groundVoxel->angleOfTalus);

			if (neighbourSediment) {
				bool currentSediment = currentRainVoxel->loseSediment();
				neighbourDissolvedGround.push_back(groundVoxel);
				currentDissolvedGround.erase(currentDissolvedGround.end() - 1);
			}

			voxels--;
		}
	}
}

/*
Algo for sideward and voxel erosion

Water is seeded from the top only

Air voxel have a height param as well like rain voxels

For every cell, traverse for each rain voxel,
	1. Do bottom sediment absorption in loop until ground voxel is found. Only first ground voxel should be absorbed based on the return value from absorb function
		i. If rain voxel is below(because all of the ground in between got absorbed), merge rain voxels with sediment levels added
		ii. If air voxel is found below, delete air voxel(that should auto adjust the height)
	2. Transfer sediment to rain voxels lower in level(from top) in contact with current voxel - 
		i. Determine contact rain voxels - 
			a. Determine height range for current rain voxel(with different height)
			b. for each neighbour rain voxel
				I. Determine height range for neighbour rain voxel
				II. if neighbour rain voxel start is within height range of current rain voxel(> base height and < top height), then it is in contact 
		ii. For each contact rain voxel, transfer sediment based on weighted average of height difference
	3. Transfer water to rain/air voxels lower in level and unobstructed
		i.  To determine highest air voxel in contact with current loop start from bottom of the neighbour voxel (only for air since just rain voxel indicates there is no space)
			a. loop until top of neighbour is higher than bottom of current
		ii. Distribute based on weighted height differences
			a. For each air voxel, if rain voxel is below the air voxel, then add to water quantity of rain voxel and negate the same from airVoxel.
			b. else create a new rain voxel below air voxel and do the same as a.
			c. if not enough space distribute only what's available.
	4. Do sideways sediment absorption
		i. Determine contact ground voxel in neighbour by using same method as 2 i.
		ii. If absorb function returns true, then absorb the (lowest or highest or random?) ground voxel.
		iii. If above is air voxel, then loop downward until ground or rain is found.
			I. If air voxel is found, delete it and add the height of it to the height variable. Dont forget to add height of 1(of ground voxel).
		iv. if above is rain, then loop downward until ground or rain is found,
			I. if ground is found, then bring rain voxel above it and create air voxel above it the height traversed during the loop,
			II. if rain voxel is found, then add the rain voxel values and create an air voxel equal to height traversed
			III. if air voxel is found, delete air voxel or keep it for better memory management and reuse it.
		vii. if above and below are ground voxels, then set the ground voxel as air voxel with height 1.
	5. Evaporation
		i. If ground voxel above it(determined by looping above), then deposit particles dissolved until 33% of height is reached with the ground voxel above it.
		ii. If no ground voxel, deposit all dissolved particles.

*/

inline void Terrain::transferSedimentFromSide(std::vector<Voxel*>& currentStack, std::vector<Voxel*>& neighbourStack, float maximumHeightDifference, float sumOfHeights, float maxVolume) {
	int height = 0;
	for (int k = 0; k < currentStack.size(); k++) {		//traverse currentstack for water 
		if (currentStack[k]->materialId == 3) {
			RainVoxel* currentRainVoxel = (RainVoxel*)currentStack[k];
			
			height += currentRainVoxel->waterQuantity;
		}
		else {
			height++;
		}
	}

	RainVoxel* currentRainVoxel = (RainVoxel*)currentStack[currentStack.size() - 1];

	float currentStackNonWaterHeight = (currentStack.size() - 1) * voxelDimensionVertical;
	float neighbourStackNonWaterHeight = 0;

	float currentStackHeight = currentStackNonWaterHeight + (currentRainVoxel->waterQuantity * voxelDimensionVertical);
	float neighbourStackHeight = 0;

	if (neighbourStack.size() != 0 && neighbourStack[neighbourStack.size() - 1]->materialId == 3) {
		RainVoxel* neighbourRainVoxel = (RainVoxel*)neighbourStack[neighbourStack.size() - 1];
		neighbourStackNonWaterHeight = (neighbourStack.size() - 1) * voxelDimensionVertical;
		neighbourStackHeight = neighbourStackNonWaterHeight + (neighbourRainVoxel->waterQuantity * voxelDimensionVertical);
	}
	else {
		neighbourStackNonWaterHeight = (neighbourStack.size()) * voxelDimensionVertical;
		neighbourStackHeight = neighbourStackNonWaterHeight;
	}

	if (currentStackNonWaterHeight > neighbourStackNonWaterHeight || currentStackNonWaterHeight > neighbourStackHeight) {
		return;
	}

	float heightDifference = (currentStackHeight - neighbourStackHeight);

	if (heightDifference > 0) {
		float volumeProportion = maxVolume * (heightDifference / sumOfHeights);
		int voxels = volumeProportion / voxelDimensionVertical;

		RainVoxel* neighbourVoxel = (RainVoxel*)neighbourStack[neighbourStack.size() - 1];
		std::vector<Voxel*>& currentDissolvedGround = currentRainVoxel->dissolvedGround;
		std::vector<Voxel*>& neighbourDissolvedGround = neighbourVoxel->dissolvedGround;

		while (voxels > 0 && currentDissolvedGround.size() > 0) {
			GroundVoxel* groundVoxel = (GroundVoxel*)currentDissolvedGround[currentDissolvedGround.size() - 1];
			bool neighbourSediment = neighbourVoxel->absorbSediment(groundVoxel->angleOfTalus);

			if (neighbourSediment) {
				bool currentSediment = currentRainVoxel->loseSediment();
				neighbourDissolvedGround.push_back(currentDissolvedGround[currentDissolvedGround.size() - 1]);
				currentDissolvedGround.erase(currentDissolvedGround.end() - 1);
			}

			voxels--;
		}
	}
}

inline void Terrain::createNewRainVoxel(RainVoxel ** rainVoxel, float waterQuantity = 1) {
	*rainVoxel = new RainVoxel();
	(*rainVoxel)->materialId = 3; 
	(*rainVoxel)->sedimentVolume = 0;
	(*rainVoxel)->waterQuantity = waterQuantity;
	rainVoxelCreated++;
}

inline bool Terrain::transferWater(std::vector<Voxel*>& currentStack, std::vector<Voxel*>& neighbourStack, float maximumHeightDifference, float sumOfHeights, float maxVolume) {
	RainVoxel * currentRainVoxel = (RainVoxel *)currentStack[currentStack.size() - 1];

	float currentStackHeight = ((currentStack.size() - 1) * voxelDimensionVertical) + (currentRainVoxel->waterQuantity * voxelDimensionVertical);
	float neighbourStackHeight = 0;

	if (neighbourStack.size() != 0 && neighbourStack[neighbourStack.size() - 1]->materialId == 3) {
		RainVoxel * neighbourRainVoxel = (RainVoxel *) neighbourStack[neighbourStack.size() - 1];
		neighbourStackHeight = ((neighbourStack.size() - 1) * voxelDimensionVertical) + (neighbourRainVoxel->waterQuantity * voxelDimensionVertical);
	}
	else {
		neighbourStackHeight = (neighbourStack.size()) * voxelDimensionVertical;
	}

	float heightDifference = (currentStackHeight - neighbourStackHeight);

	if (heightDifference > 0) {
		float volumeProportion = maxVolume * (heightDifference / sumOfHeights);
		float voxels = volumeProportion / voxelDimensionVertical;

		RainVoxel* neighbourTopRainVoxel = nullptr;

		if (neighbourStack.size() != 0 && neighbourStack[neighbourStack.size() - 1]->materialId == 3) {
			neighbourTopRainVoxel = (RainVoxel*)neighbourStack[neighbourStack.size() - 1];
		}

		if(neighbourTopRainVoxel == nullptr) {
			createNewRainVoxel(&neighbourTopRainVoxel, 0);
			neighbourStack.push_back(neighbourTopRainVoxel);
		}

		neighbourTopRainVoxel->waterQuantity = neighbourTopRainVoxel->waterQuantity + voxels;
		
		if (currentRainVoxel->waterQuantity < voxels) {
			for (int dg = 0; dg < currentRainVoxel->dissolvedGround.size(); dg++) {
				neighbourTopRainVoxel->dissolvedGround.push_back(currentRainVoxel->dissolvedGround[dg]);
			}
			delete currentRainVoxel;
			currentStack.erase(currentStack.end() - 1);
			return true;
		}
		else {
			currentRainVoxel->waterQuantity -= voxels;
		}
	}

	return false;
}

inline float Terrain::getAngleOfTalus(float height, float verticalDifference) {
	float aTanAngleOfTalus = atan(height / verticalDifference);
	float angleOfTalus = aTanAngleOfTalus * (180 / 3.14159f);

	return angleOfTalus;
}

inline void Terrain::performThermalErosion(int steps = 10) {
	movedVoxelCount = 0;
	if (!isThermalErosionInProgress) {
		isThermalErosionInProgress = true;
		for (int s = 0; s < steps; s++) {
			std::cout << "Step " << s << " in progress" << std::endl;
			for (int i = 0; i < voxelMap.size(); i++) {
				//std::cout << "i " << i << std::endl;
				for (int j = 0; j < voxelMap[i].size(); j++) {
					//for (int k = 0; k < voxelMap[i][j].size(); k++) {
					int xStart = 0;
					int xEnd = 3;
					int yStart = 0;
					int yEnd = 3;
					if (i == 0) {
						xStart = 1;
					}
					else if (i == voxelMap.size() - 1) {
						xEnd = 2;
					}

					if (j == 0) {
						yStart = 1;
					}
					else if (j == voxelMap[i].size() - 1) {
						yEnd = 2;
					}

					float maximumHeightDifference = 0;
					float sumOfHeights = 0;

					for (int x = xStart; x < xEnd; x++) {
						for (int y = yStart; y < yEnd; y++) {
							if (x == 1 && y == 1) {
								continue;
							}

							std::vector<Voxel*> & neighbourStack = voxelMap[i + (x - 1)][j + (y - 1)];
							std::vector<Voxel*> & currentStack = voxelMap[i][j];

							float currentStackHeight = currentStack.size() * voxelDimensionVertical;
							float neighbourStackHeight = neighbourStack.size() * voxelDimensionVertical;

							float heightDifference = (currentStackHeight - neighbourStackHeight);

							if (heightDifference > 0) {
								if (heightDifference > maximumHeightDifference) {
									maximumHeightDifference = heightDifference;
								}

								float angleOfTalus = getAngleOfTalus(heightDifference, voxelDimension);

								if (angleOfTalus > ((GroundVoxel *) currentStack[currentStack.size() - 1])->angleOfTalus) {
									sumOfHeights += heightDifference;
								}
							}
						}
					}

					if (sumOfHeights != 0) {
						float maxVolume = maximumHeightDifference / 2;	//TODO what is the parameter in aH/2 defined by Benes

						for (int x = xStart; x < xEnd; x++) {
							for (int y = yStart; y < yEnd; y++) {
								if (x == 1 && y == 1) {
									continue;
								}
								std::vector<Voxel*> & neighbourStack = voxelMap[i + (x - 1)][j + (y - 1)];
								std::vector<Voxel*> & currentStack = voxelMap[i][j];
								thermalErosionCell(currentStack, neighbourStack, maximumHeightDifference, sumOfHeights, maxVolume);
							}
						}
					}
					//}
				}
			}
		}

		std::cout << "Total voxels moved " << movedVoxelCount<<std::endl;
		isThermalErosionInProgress = false;
		hasThermalErosionCompleted = true;
	}
}

inline void Terrain::performHydraulicErosion(int steps = 100, bool addRain = false) {
	movedVoxelCount = 0;
	rainVoxelCreated = 0;
	rainVoxelEvaporated = 0;
	int rateOfRain = 3;
	if (!isHydraulicErosionInProgress) {
		isHydraulicErosionInProgress = true;
		for (int s = 0; s < steps; s++) {
			std::cout << "Step " << s << " in progress" << std::endl;

			for (int i = 0; i < voxelMap.size(); i++) {
				//std::cout << "i " << i << std::endl;
				for (int j = 0; j < voxelMap[i].size(); j++) {
					
					if (voxelMap[i][j].size() != 0 && voxelMap[i][j][voxelMap[i][j].size() - 1]->materialId == 3) {	//no rain voxels on top
						
						float sedimentMaximumHeightDifference = 0;
						float sedimentSumOfHeights = 0;

						int xStart = 0;
						int xEnd = 3;
						int yStart = 0;
						int yEnd = 3;
						if (i == 0) {
							xStart = 1;
						}
						else if (i == voxelMap.size() - 1) {
							xEnd = 2;
						}

						if (j == 0) {
							yStart = 1;
						}
						else if (j == voxelMap[i].size() - 1) {
							yEnd = 2;
						}
						
						RainVoxel* currentRainVoxel = (RainVoxel*)voxelMap[i][j][voxelMap[i][j].size() - 1];
						
						for (int repetition = 0; repetition < 20; repetition++) {
							if (currentRainVoxel != nullptr) {
								//Sediment redistribution
								for (int x = xStart; x < xEnd; x++) {
									for (int y = yStart; y < yEnd; y++) {
										if (x == 1 && y == 1) {
											continue;
										}

										std::vector<Voxel*>& neighbourStack = voxelMap[i + (x - 1)][j + (y - 1)];
										std::vector<Voxel*>& currentStack = voxelMap[i][j];

										float currentStackHeight = ((currentStack.size() - 1) * voxelDimensionVertical) + (currentRainVoxel->waterQuantity * voxelDimensionVertical);

										float neighbourStackHeight = 0;

										if (neighbourStack.size() != 0 && neighbourStack[neighbourStack.size() - 1]->materialId == 3) {
											RainVoxel* neighbourRainVoxel = (RainVoxel*)neighbourStack[neighbourStack.size() - 1];
											neighbourStackHeight = ((neighbourStack.size() - 1) * voxelDimensionVertical) + (neighbourRainVoxel->waterQuantity * voxelDimensionVertical); //Experimenting with just ground levels
										}
										else {
											continue;
										}

										float heightDifference = (currentStackHeight - neighbourStackHeight);

										if (heightDifference > 0) {
											if (currentRainVoxel->dissolvedGround.size() != 0 && neighbourStack.size() != 0 && neighbourStack[neighbourStack.size() - 1]->materialId == 3) {
												if (heightDifference > sedimentMaximumHeightDifference) {
													sedimentMaximumHeightDifference = heightDifference;
												}

												sedimentSumOfHeights += heightDifference;
											}
										}
									}
								}

								float sedimentMaxVolume = sedimentMaximumHeightDifference;

								if (sedimentSumOfHeights > 0) {
									for (int x = xStart; x < xEnd; x++) {
										for (int y = yStart; y < yEnd; y++) {
											if (x == 1 && y == 1) {
												continue;
											}
											std::vector<Voxel*>& neighbourStack = voxelMap[i + (x - 1)][j + (y - 1)];
											std::vector<Voxel*>& currentStack = voxelMap[i][j];

											if (neighbourStack.size() != 0 && neighbourStack[neighbourStack.size() - 1]->materialId == 3) {
												transferSediment(currentStack, neighbourStack, sedimentMaximumHeightDifference, sedimentSumOfHeights, sedimentMaxVolume);
											}
										}
									}
								}

								//Water transport

								float maximumHeightDifference = 0;
								float sumOfHeights = 0;

								for (int x = xStart; x < xEnd; x++) {
									for (int y = yStart; y < yEnd; y++) {
										if (x == 1 && y == 1) {
											continue;
										}

										std::vector<Voxel*>& neighbourStack = voxelMap[i + (x - 1)][j + (y - 1)];
										std::vector<Voxel*>& currentStack = voxelMap[i][j];

										float currentStackHeight = ((currentStack.size() - 1) * voxelDimensionVertical) + (currentRainVoxel->waterQuantity * voxelDimensionVertical);

										float neighbourStackHeight = 0;

										if (neighbourStack.size() != 0 && neighbourStack[neighbourStack.size() - 1]->materialId == 3) {
											RainVoxel* neighbourRainVoxel = (RainVoxel*)neighbourStack[neighbourStack.size() - 1];
											neighbourStackHeight = ((neighbourStack.size() - 1) * voxelDimensionVertical) + (neighbourRainVoxel->waterQuantity * voxelDimensionVertical); //Experimenting with just ground levels
										}
										else {
											neighbourStackHeight = (neighbourStack.size()) * voxelDimensionVertical;
										}

										float heightDifference = (currentStackHeight - neighbourStackHeight);

										if (heightDifference > 0) {
											if (heightDifference > maximumHeightDifference) {
												maximumHeightDifference = heightDifference;
											}

											sumOfHeights += heightDifference;
										}
									}
								}

								float maxVolume = maximumHeightDifference;
								//std::cout << "processing " << i << " " << j << std::endl;

								if (sumOfHeights > 0) {
									for (int x = xStart; x < xEnd; x++) {
										for (int y = yStart; y < yEnd; y++) {
											if (x == 1 && y == 1) {
												continue;
											}
											std::vector<Voxel*>& neighbourStack = voxelMap[i + (x - 1)][j + (y - 1)];
											std::vector<Voxel*>& currentStack = voxelMap[i][j];

											if (currentStack.size() != 0 && currentStack[currentStack.size() - 1]->materialId == 3) {
												if (transferWater(currentStack, neighbourStack, maximumHeightDifference, sumOfHeights, maxVolume)) {
													currentRainVoxel = nullptr;
												}
											}
										}
									}
								}

								if (currentRainVoxel != nullptr) {
									//Sediment absorption from ground
									if (voxelMap[i][j].size() > 1) {
										Voxel* topNonWaterVoxel = voxelMap[i][j][voxelMap[i][j].size() - 2];
										GroundVoxel* groundVoxel = (GroundVoxel*) topNonWaterVoxel;
										if (currentRainVoxel->absorbSedimentFromGround(groundVoxel->angleOfTalus)) {
											currentRainVoxel->dissolvedGround.push_back(topNonWaterVoxel);
											voxelMap[i][j].erase(voxelMap[i][j].end() - 2);
										}
									}

									//Sediment absorption from sideways

									for (int x = xStart; x < xEnd; x++) {
										for (int y = yStart; y < yEnd; y++) {
											if (x == 1 && y == 1) {
												continue;
											}
											std::vector<Voxel*>& neighbourStack = voxelMap[i + (x - 1)][j + (y - 1)];
											std::vector<Voxel*>& currentStack = voxelMap[i][j];

											if (neighbourStack.size() != 0) {
												transferSedimentFromSide(currentStack, neighbourStack, sedimentMaximumHeightDifference, sedimentSumOfHeights, sedimentMaxVolume);
											}
										}
									}
								}
							}
						}

						if (currentRainVoxel != nullptr) {
							//evaporation -  Should be last since we delete currentRainVoxel
							if (currentRainVoxel->evaporate()) {
								rainVoxelEvaporated++;
								if (currentRainVoxel->dissolvedGround.size() != 0) {
									for (int dg = 0; dg < currentRainVoxel->dissolvedGround.size(); dg++) {
										voxelMap[i][j].insert(voxelMap[i][j].end() - 1, currentRainVoxel->dissolvedGround[dg]);
									}
								}

								delete currentRainVoxel;
								voxelMap[i][j].erase(voxelMap[i][j].end() - 1);		//TODO make sure the pointer is derefenced correctly
							}
						}
					}

					if (addRain || s < steps / 3) { //rains only for first half of steps to ensure that the sediments are transported
						int rainAtStep = std::rand() % (rateOfRain + 1);

						//Addition through Raining simulation
						if (rainAtStep != 0) {
							RainVoxel* rainVoxel = nullptr;

							if (voxelMap[i][j].size() != 0 && voxelMap[i][j][voxelMap[i][j].size() - 1]->materialId == 3) {
								rainVoxel = (RainVoxel*)voxelMap[i][j][voxelMap[i][j].size() - 1];
								rainVoxel->waterQuantity += rainAtStep;
							}
							else {
								createNewRainVoxel(&rainVoxel, rainAtStep);
								voxelMap[i][j].push_back(rainVoxel);
							}

							if (voxelMap[i][j].size() > 1) {
								Voxel* topNonWaterVoxel = voxelMap[i][j][voxelMap[i][j].size() - 2];
								GroundVoxel* groundVoxel = (GroundVoxel*)topNonWaterVoxel;
								if (rainVoxel->absorbSedimentFromGround(groundVoxel->angleOfTalus)) {
									rainVoxel->dissolvedGround.push_back(topNonWaterVoxel);
									voxelMap[i][j].erase(voxelMap[i][j].end() - 2);
								}
							}
						}
					}
				}
			}
		}

		
		//Force evaporate every rain voxel
		for (int i = 0; i < voxelMap.size(); i++) {
			//std::cout << "i " << i << std::endl;
			for (int j = 0; j < voxelMap[i].size(); j++) {

				if (voxelMap[i][j].size() != 0 && voxelMap[i][j][voxelMap[i][j].size() - 1]->materialId == 3) {

					RainVoxel* currentRainVoxel = (RainVoxel*)voxelMap[i][j][voxelMap[i][j].size() - 1];

					//evaporation -  Should be last since we delete currentRainVoxel
					rainVoxelEvaporated++;
					if (currentRainVoxel->dissolvedGround.size() != 0) {
						for (int dg = 0; dg < currentRainVoxel->dissolvedGround.size(); dg++) {
							voxelMap[i][j].insert(voxelMap[i][j].end() - 1, currentRainVoxel->dissolvedGround[dg]);
						}
					}

					delete currentRainVoxel;
					voxelMap[i][j].erase(voxelMap[i][j].end() - 1);		//TODO make sure the pointer is derefenced correctly
				}
			}
		}

		std::cout << "Total voxels moved " << movedVoxelCount << std::endl;
		std::cout << "Total Rain Voxel created" << rainVoxelCreated << std::endl;
		std::cout << "Total Rain voxel evaporated " << rainVoxelEvaporated << std::endl;
		isHydraulicErosionInProgress = false;
		hasHydraulicErosionCompleted = true;
	}

}

inline void Terrain::AppendNormal(std::vector <GLfloat> & a, const glm::vec3 n, const int & i, const int & j, const int & rows, const int & cols, std::vector <int> & weights) {
	int weightsIndex = (i * cols) + j;
	//int index = (i * cols * 3) + (j * 3);
	int index = getIndexForRowColumn(i, j, rows, cols) * 3;
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

inline void Terrain::AppendNormal(std::vector <GLfloat>& a, const glm::vec3 n, const int& index, std::vector <int>& weights) {
	//int index = (i * cols * 3) + (j * 3);
	//int index = getIndexForRowColumn(i, j, rows, cols) * 3;
	glm::vec3 currNormal = glm::vec3(a[index], a[index + 1], a[index + 2]);
	currNormal = glm::normalize(currNormal + n);
	//float xAvg = ((a[index] * weights[weightsIndex]) + n.x) / (weights[weightsIndex] + 1);
	//float yAvg = ((a[index + 1] * weights[weightsIndex]) + n.y) / (weights[weightsIndex] + 1);
	//float zAvg = ((a[index + 2] * weights[weightsIndex]) + n.z) / (weights[weightsIndex] + 1);
	//weights[weightsIndex] += 1;
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

inline glm::vec4 Terrain::processVoxelCell(int i, int j) {
	std::vector<Voxel*> cellVoxels = voxelMap[i][j];

	glm::vec4 layerHeights = glm::vec4(0, 0, 0, 0);

	int previousLayerMeasuredIndex = 0;
	int currentLayerMeasuredIndex = 0;
	int currentLayerHeight = 0;

	for (int k = 0; k < cellVoxels.size(); k++) {
		if (currentLayerMeasuredIndex == 0) {
			if (cellVoxels[k]->materialId == 2) { //when air layer is found
				previousLayerMeasuredIndex = 0;
				currentLayerMeasuredIndex = 1;
			}
			else {
				currentLayerHeight++;
			}
		}
		if (currentLayerMeasuredIndex == 1) {
			if (previousLayerMeasuredIndex == 0) {
				layerHeights[previousLayerMeasuredIndex] = (currentLayerHeight / (float)maxVoxelCount);
				previousLayerMeasuredIndex = 1;
				currentLayerHeight = 1;
			}
			else {
				if (cellVoxels[k]->materialId == 0) {	//when solid material is found again
					previousLayerMeasuredIndex = 1;
					currentLayerMeasuredIndex = 2;
				}
				else {
					currentLayerHeight++;
				}
			}
		}
		if (currentLayerMeasuredIndex == 2) {
			if (previousLayerMeasuredIndex == 1) {
				layerHeights[previousLayerMeasuredIndex] = (currentLayerHeight / (float)maxVoxelCount);
				previousLayerMeasuredIndex = 2;
				currentLayerHeight = 1;
			}
			else {
				if (cellVoxels[k]->materialId == 2) {	//when air material is found again then end it
					layerHeights[currentLayerMeasuredIndex] = (currentLayerHeight / (float)maxVoxelCount);
					previousLayerMeasuredIndex = 2;
					currentLayerMeasuredIndex = 3;
					break;
				}
				else {
					currentLayerHeight++;
				}
			}
		}
	}
	
	if (currentLayerMeasuredIndex == 0) {	//in case no air layers were found
		layerHeights[0] = (currentLayerHeight / (float)maxVoxelCount);
	}

	return layerHeights;
}

inline void Terrain::exportOutput(FIBITMAP* img, const char * outputFileName) {
	FreeImage_Initialise();

	img = FreeImage_Allocate(512, 256, 32);

	int bytespp = FreeImage_GetLine(img) / FreeImage_GetWidth(img);
	int originalBytespp = FreeImage_GetLine(originalHeightMap) / FreeImage_GetWidth(originalHeightMap);
	int complexBytespp = FreeImage_GetLine(complexImage) / FreeImage_GetWidth(complexImage);

	for (int i = 0; i < 256; i++)
	{
		/*BYTE* complexColumnVector = FreeImage_GetScanLine(complexImage, i);
		BYTE* columnVector = FreeImage_GetScanLine(originalHeightMap, i);
		*/

		FIRGBAF* columnVector = (FIRGBAF*)FreeImage_GetScanLine(originalHeightMap, i);
		FIRGBAF* complexColumnVector = (FIRGBAF*)FreeImage_GetScanLine(complexImage, i);

		BYTE* color = FreeImage_GetScanLine(img, i);

		for (int j = 0; j < 512; j++)
		{
			if (j < 256)
			{
				//write original image with heightmap and complex image blended in r,g,b channels
				color[FI_RGBA_RED] = int(columnVector[j].red * 255.0f);//columnVector[FI_RGBA_RED];
				color[FI_RGBA_GREEN] = int(complexColumnVector[j].red * 255.0f);//columnVector[FI_RGBA_RED];// ;
				color[FI_RGBA_BLUE] = int(complexColumnVector[j].green * 255.0f);//columnVector[FI_RGBA_RED];// ;
				color[FI_RGBA_ALPHA] = 255;
			}
			else
			{
				glm::vec4 layerHeights = processVoxelCell(i, j - 256);
				//Convert the output from voxel to layers
				color[FI_RGBA_RED] = int(layerHeights.r * 255);
				color[FI_RGBA_GREEN] = int(layerHeights.g * 255);
				color[FI_RGBA_BLUE] = int(layerHeights.b * 255);
				color[FI_RGBA_ALPHA] = 255;
			}

			color += bytespp;
			//columnVector += originalBytespp;
			//complexColumnVector += complexBytespp;
		}
	}

	/*
	img = FreeImage_AllocateT(FIT_RGBAF, 256, 256, 128);

	for (int i = 0; i < 256; i++)
	{
		FIRGBAF* columnVector = (FIRGBAF*)FreeImage_GetScanLine(originalHeightMap, i);
		FIRGBAF* complexColumnVector = (FIRGBAF*)FreeImage_GetScanLine(complexImage, i);
		FIRGBAF* color = (FIRGBAF*)FreeImage_GetScanLine(img, i);

		for (int j = 0; j < 256; j++)
		{
			if (j < 256)
			{
				//write original image with heightmap and complex image blended in r,g,b channels
				color[j].red = columnVector[j].red;
				color[j].green = columnVector[j].red;// complexColumnVector[j].red;
				color[j].blue = columnVector[j].red;// complexColumnVector[j].green;
				color[j].alpha = 255;
			}
			else
			{
				glm::vec4 layerHeights = processVoxelCell(i, j - 256);
				//Convert the output from voxel to layers
				color[j].red = layerHeights.r;
				color[j].green = layerHeights.g;
				color[j].blue = layerHeights.b;
				color[j].alpha = 255;
			}
		}
	}*/

	//img = FreeImage_ConvertTo32Bits(img);
	FreeImage_Save(FIF_PNG, img, outputFileName, 0);
	FreeImage_Save(FIF_PNG, originalHeightMap, "HeightMap_O_Test.png", 0);
	FreeImage_Save(FIF_PNG, complexImage, "ComplexMap_O_Test.png", 0);
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
