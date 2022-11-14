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
#include <map>

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
	bool isCellInRange(float firstBaseHeight, float firstTopHeight, float secondBaseHeight, float secondTopHeight);

	void generateRenderBuffers(int rowSize, int columnSize);

	FIBITMAP* originalHeightMap = nullptr;
	FIBITMAP* complexImage = nullptr;
	bool renderMode = 0; //0 = Voxel 1 = Mesh

	void thermalErosionCell(std::vector<Voxel*>& currentStack, std::vector<Voxel*>& neighbourStack, float maximumHeightDifference, float sumOfHeights, float maxVolume);
	void transferSediment(RainVoxel* currentRainVoxel, RainVoxel* neighbourVoxel, float maximumHeightDifference, float sumOfHeights, float maxVolume, float heightDifference);
	void transferSedimentFromSide(std::vector<Voxel*>& neighbourStack, GroundVoxel* neighbourGroundVoxel, int& neighbourIndex, float heightFromBase, float absorbedAmount, float neighbourBaseHeight, float neighbourTopHeight);
	bool transferWater(RainVoxel* currentRainVoxel, std::vector<Voxel*>& currentStack, int currentIndex, float currentStackTopHeight, std::vector<Voxel*>& neighbourStack, int & neighbourIndex, int & neighbourStackSize, float minNeighbourStackTopHeight, float maximumHeightDifference, float sumOfHeights, float maxVolume, float heightDifference, int & stackSize, bool createNewRainVoxel);
	float getAngleOfTalus(float height, float verticalDifference);
	void createNewRainVoxel(RainVoxel** rainVoxel, float waterQuantity);
	void createNewAirVoxel(Voxel** airVoxel, float height); 
	float determineBaseHeight(std::vector<Voxel*> stack, int index);
	void alertMaterialId(Voxel* voxel);

	void evaporate(RainVoxel* currentRainVoxel, int& currentIndex, int& stackSize, int i, int j, bool forceEvaporate);

	int movedVoxelCount = 0;
	bool sidewaysErosion = true;
	bool evaporateEnabled = true;
	bool waterFlowEnabled = true;
	bool absorbWaterFromGround = true;

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
		float height = 1;
	};

	class GroundVoxel : public Voxel {
	public:
		double angleOfTalus;

		GroundVoxel(float height = 1) {
			this->height = height;
		}
	};

	class RainVoxel : public Voxel {
	public:
		std::vector<GroundVoxel *> sedimentAccumulated; //indices are material ids

		float getSedimentRatio() {
			float ratio = getSedimentVolume();

			if (height != 0) {
				ratio /= height;
			}

			return ratio;
		}

		float getAbsorptionVolume() {
			float volume = getSedimentVolume();

			float idealVolume = height * 0.8f;

			float difference = idealVolume - volume;

			return difference;
		}

		float getSedimentVolume() {
			float sedimentVolume = 0;
			for (int i = 0; i < sedimentAccumulated.size(); i++) {
				sedimentVolume += sedimentAccumulated[i]->height;
			}

			return sedimentVolume;
		}

		bool evaporate(float & sedimentVolumeToLeave, float & evaporatedHeight) {
			float previousHeight = height;
			height *= exp(-0.002f);
			evaporatedHeight = previousHeight - height;
			if (height < 0.05f) {
				sedimentVolumeToLeave = getSedimentVolume();
				return true;	//return full volume
			}

			float sedimentToBeLeftBehind = getAbsorptionVolume();

			if (sedimentToBeLeftBehind <= 0) {
				sedimentVolumeToLeave = -sedimentToBeLeftBehind;
			}
			else {
				sedimentVolumeToLeave = 0;
			}

			return false;
		}

		float absorbSedimentFromGround(GroundVoxel* absorbedVoxel, float materialHeight = -1) {
			if (materialHeight == -1) {
				materialHeight = absorbedVoxel->height;
			}
			float ratio = getSedimentRatio();
			
			if (ratio < 0.8f) {
				GroundVoxel* groundVoxel = nullptr;

				bool newGroundCreated = false;

				if (sedimentAccumulated.size() > 0) {
					groundVoxel = sedimentAccumulated[0];
				}
				else {
					groundVoxel = new GroundVoxel();
					groundVoxel->materialId = absorbedVoxel->materialId;
					groundVoxel->height = 0;
					groundVoxel->angleOfTalus = absorbedVoxel->angleOfTalus;
					newGroundCreated = true;
				}

				if (groundVoxel->materialId != absorbedVoxel->materialId) {
					groundVoxel = new GroundVoxel();
					groundVoxel->height = 0;
					groundVoxel->materialId = absorbedVoxel->materialId;
					groundVoxel->angleOfTalus = absorbedVoxel->angleOfTalus;
					newGroundCreated = true;
				}

				float absorbableVolume = getAbsorptionVolume();

				if (absorbableVolume >= 0) {
					materialHeight = std::min(absorbableVolume, materialHeight);
					materialHeight = std::min(absorbedVoxel->height, materialHeight);

					groundVoxel->height += materialHeight;

					if (newGroundCreated) {
						sedimentAccumulated.insert(sedimentAccumulated.begin(), groundVoxel);
					}

					if (sedimentAccumulated.size() > 100) {
						//std::cout << "Sediment Accum " << sedimentAccumulated.size() << std::endl;
					}

					return materialHeight;
				}
			}

			return 0;
		}

		float absorbSedimentFromSide(GroundVoxel* absorbedVoxel, float heightAsRatio, float materialHeight = -1) {
			if (materialHeight == -1) {
				materialHeight = absorbedVoxel->height;
			}
			float ratio = getSedimentRatio();

			if (ratio < 0.8f) {
				GroundVoxel* groundVoxel = nullptr;

				bool newGroundCreated = false;

				int indexFromRatio = heightAsRatio * (sedimentAccumulated.size() - 1);
					
				if (sedimentAccumulated.size() > 0 && indexFromRatio >= 0 && indexFromRatio < sedimentAccumulated.size() && sedimentAccumulated[indexFromRatio]->materialId == absorbedVoxel->materialId) {
					groundVoxel = sedimentAccumulated[indexFromRatio];
				} else if (sedimentAccumulated.size() > 0 && indexFromRatio >= 0 && (indexFromRatio + 1) < sedimentAccumulated.size() && sedimentAccumulated[indexFromRatio + 1]->materialId == absorbedVoxel->materialId) {
					groundVoxel = sedimentAccumulated[indexFromRatio + 1];
				}
				else if (sedimentAccumulated.size() > 0 && indexFromRatio >= 0 && (indexFromRatio - 1) >= 0 && (indexFromRatio - 1 < sedimentAccumulated.size()) && sedimentAccumulated[indexFromRatio - 1]->materialId == absorbedVoxel->materialId) {
					groundVoxel = sedimentAccumulated[indexFromRatio - 1];
				}
				else {
					groundVoxel = new GroundVoxel();
					groundVoxel->materialId = absorbedVoxel->materialId;
					groundVoxel->height = 0;
					groundVoxel->angleOfTalus = absorbedVoxel->angleOfTalus;
					newGroundCreated = true;
				}

				float absorbableVolume = getAbsorptionVolume();

				if (absorbableVolume >= 0) {
					materialHeight = std::min(absorbableVolume, materialHeight);
					materialHeight = std::min(absorbedVoxel->height, materialHeight);

					groundVoxel->height += materialHeight;

					if (newGroundCreated) {
						sedimentAccumulated.insert(sedimentAccumulated.begin(), groundVoxel);
					}

					if (sedimentAccumulated.size() > 100) {
						//std::cout << "Sediment Accum " << sedimentAccumulated.size() << std::endl;
					}

					return materialHeight;
				}
			}

			return 0;
		}

		bool absorbSediment(GroundVoxel * absorbedVoxel, float materialHeight = -1) {
			if (materialHeight == -1) {
				materialHeight = absorbedVoxel->height;
			}
			float ratio = getSedimentRatio();
			if (ratio < 0.8f) {
				GroundVoxel* groundVoxel = nullptr;

				bool newGroundCreated = false;
				
				if (sedimentAccumulated.size() > 0) {
					groundVoxel = sedimentAccumulated[sedimentAccumulated.size() - 1];
				}
				else {
					groundVoxel = new GroundVoxel();
					groundVoxel->materialId = absorbedVoxel->materialId;
					groundVoxel->height = 0;
					groundVoxel->angleOfTalus = absorbedVoxel->angleOfTalus;
					newGroundCreated = true;
				}

				if (groundVoxel->materialId != absorbedVoxel->materialId) {
					groundVoxel = new GroundVoxel();
					groundVoxel->height = 0;
					groundVoxel->materialId = absorbedVoxel->materialId;
					groundVoxel->angleOfTalus = absorbedVoxel->angleOfTalus;
					newGroundCreated = true;
				}

				groundVoxel->height += materialHeight;

				if (newGroundCreated) {
					sedimentAccumulated.push_back(groundVoxel);
				}

				if (sedimentAccumulated.size() > 100) {
					//std::cout << "Sediment Accum " << sedimentAccumulated.size() << std::endl;
				}

				return true;
			}
			
			return false;
		}

		std::vector<GroundVoxel *> loseSediment(float materialHeight) {
			std::vector<GroundVoxel*> lostSediments;
			while (sedimentAccumulated.size() > 0 && materialHeight > 0) {
				if (sedimentAccumulated[0]->height > materialHeight) {
					sedimentAccumulated[0]->height -= materialHeight;
					GroundVoxel* newPartialGroundVoxel = new GroundVoxel();
					newPartialGroundVoxel->angleOfTalus = sedimentAccumulated[0]->angleOfTalus;
					newPartialGroundVoxel->materialId = sedimentAccumulated[0]->materialId;
					newPartialGroundVoxel->height = materialHeight;
					lostSediments.push_back(newPartialGroundVoxel);
					break;
				}
				else {
					materialHeight -= sedimentAccumulated[0]->height;
					lostSediments.push_back(sedimentAccumulated[0]);
					sedimentAccumulated.erase(sedimentAccumulated.begin());
				}
			}

			return lostSediments;
		}

	};

	int selectedI = 0;
	int selectedJ = 0;

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
	void setSidewaysErosion(bool sidewaysErosion);
	void setEvaporation(bool evaporate);
	void setWaterFlow(bool waterFlow);
	void setAbsorbWaterFromGround(bool absorbWaterFromGround);

	glm::vec3 getScale();
	void setIndexCountToRender(int indexCount);
	void setInstancesToRender(int instanceCount);

	float getAbsorbability(GroundVoxel* groundVoxel);

	void exportOutput(FIBITMAP* img, const char* outputFileName);
	glm::vec4 processVoxelCell(int i , int j);

	void generateTerrain();
	void generateTerrainForFinalOutput(FIBITMAP* img);

	glm::mat4 getModelMatrix();
	void generateTerrain(FIBITMAP * img, FIBITMAP* complexImg);
	std::vector<GLuint> createModelMatrixBuffers(int rowSize, int columnSize, bool createBuffer = true);

	void SaveOBJ(std::vector<char> filename);

	void performThermalErosion(int steps);
	void performHydraulicErosion(int steps, bool addRain, int rateOfRain, bool forceEvaporate);

	void updateTerrain();

	std::vector<std::vector<FIRGBAF*>> convolution(FIBITMAP* img);
};

inline void Terrain::setAngle(float angle) {
	this->angleY = angle;
}

inline void Terrain::setDrawMode(int drawMode) {
	this->drawMode = drawMode;
}

inline void Terrain::setSidewaysErosion(bool sidewaysErosion) {
	this->sidewaysErosion = sidewaysErosion;
}

inline void Terrain::setEvaporation(bool evaporate) {
	this->evaporateEnabled = evaporate;
}

inline void Terrain::setWaterFlow(bool waterFlow) {
	this->waterFlowEnabled = waterFlow;
}

inline void Terrain::setAbsorbWaterFromGround(bool absorbWaterFromGround) {
	this->absorbWaterFromGround = absorbWaterFromGround;
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
	this->voxelDimensionVertical = voxelDimension;

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
				if (voxelMap[i][j][k]->materialId != 2) {
					Voxel* voxel = voxelMap[i][j][k];
					glm::vec3 tran = glm::vec3(i * voxelDimension, (height + (voxel->height / 2)) * voxelDimensionVertical, j * voxelDimension);
					glm::mat4 trans = glm::translate(glm::mat4(1.f), tran);
					height += voxel->height;
					trans = glm::scale(trans, glm::vec3(voxelDimension, voxel->height * voxelDimensionVertical, voxelDimension));
					matrix_data_solid->push_back(trans);
					if (i == selectedI || j == selectedJ) {
						colors->push_back(glm::vec4(0, 1, 0, 1));
					}
					else {
						if (voxel->materialId == 3) {
							RainVoxel* rainVoxel = (RainVoxel*)voxel;
							//float sedimentVolume = rainVoxel->getSedimentVolume();
							//float col = 1.0f - (sedimentVolume /2.0f);
							float col = rainVoxel->getSedimentRatio();
							colors->push_back(glm::vec4(0, 0, col, 1));
						}
						else if (voxel->materialId == 0) {
							colors->push_back(glm::vec4(1, 1, 1, 1));
						}
						else if (voxel->materialId == 1) {
							colors->push_back(glm::vec4(1, 0, 0, 1));
						}
						else {
							std::cout << "something is wrong material id wrong " << voxel->materialId << std::endl;
						}
					}
				}
				else if (voxelMap[i][j][k]->materialId == 2) {
					Voxel* airVoxel = voxelMap[i][j][k];
					height += airVoxel->height;
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

	float columnSizeFloat = (float)columnSize;
	float rowSizeFloat = (float)rowSize;

	std::vector<std::vector<FIRGBAF *>> gaussianBlurredMatrix = convolution(complexImg);

	//voxelMap.reserve(rowSize);
	
	float maxHeightOfTerrain = 0;	//Maybe used later

	for (int i = 0; i < rowSize; i = i + 1) {
		//std::cout << "i " << i << std::endl;
		FIRGBAF* columnVector = (FIRGBAF*)FreeImage_GetScanLine(img, i);

		std::vector<std::vector<Voxel*>> columnVoxels;

		for (int j = 0; j < columnSize; j = j + 1) {
			std::vector<Voxel*> rowColumnVoxels;

			float overHangValue = gaussianBlurredMatrix[i][j]->red;
			float caveValue = gaussianBlurredMatrix[i][j]->green;

			float looseMaterialHeight = 0;
			float looseMaterialStart = -1;
			int complexFeatureFoundCode = 0;

			float height = columnVector[j].red;

			if (overHangValue > 0 && height > 0) {	//Overhangs override caves
				if (overHangValue > 1) {
					overHangValue = 1;
				}
				looseMaterialHeight = (overHangValue) * 0.7f * height;
				looseMaterialStart = (overHangValue) * 0.15f * height;
				complexFeatureFoundCode = 1;
				
				GroundVoxel* newBaseVoxel = new GroundVoxel();
				newBaseVoxel->height = looseMaterialStart / voxelDimensionVertical;
				newBaseVoxel->materialId = 0;
				newBaseVoxel->angleOfTalus = 45;
				rowColumnVoxels.push_back(newBaseVoxel);

				GroundVoxel* newOverHangVoxel = new GroundVoxel();
				newOverHangVoxel->height = looseMaterialHeight / voxelDimensionVertical;
				newOverHangVoxel->materialId = 1;
				newOverHangVoxel->angleOfTalus = 30;
				rowColumnVoxels.push_back(newOverHangVoxel);

				GroundVoxel* newTopVoxel = new GroundVoxel();
				newTopVoxel->height = abs(height - (looseMaterialStart + looseMaterialHeight)) / voxelDimensionVertical;
				newTopVoxel->materialId = 0;
				newTopVoxel->angleOfTalus = 45;
				rowColumnVoxels.push_back(newTopVoxel);

				instanceCount += 3;
			}
			else {
				if (caveValue > 0 && height > 0) {
					if (caveValue > 1) {
						caveValue = 1;
					}
					looseMaterialHeight = (caveValue) * 0.3f; //Static height of cave
					looseMaterialStart = (caveValue) * 0.15f * height;  //Is at a relative height
					complexFeatureFoundCode = 2;

					GroundVoxel* newBaseVoxel = new GroundVoxel();
					newBaseVoxel->height = looseMaterialStart / voxelDimensionVertical;
					newBaseVoxel->materialId = 0;
					newBaseVoxel->angleOfTalus = 45;
					rowColumnVoxels.push_back(newBaseVoxel);

					GroundVoxel* newOverHangVoxel = new GroundVoxel();
					newOverHangVoxel->height = looseMaterialHeight / voxelDimensionVertical;
					newOverHangVoxel->materialId = 1;
					newOverHangVoxel->angleOfTalus = 30;
					rowColumnVoxels.push_back(newOverHangVoxel);

					GroundVoxel* newTopVoxel = new GroundVoxel();
					newTopVoxel->height = abs(height - (looseMaterialStart + looseMaterialHeight)) / voxelDimensionVertical;
					newTopVoxel->materialId = 0;
					newTopVoxel->angleOfTalus = 45;
					rowColumnVoxels.push_back(newTopVoxel);

					instanceCount += 3;
				}
				else if(height > 0) {
					GroundVoxel* newBaseVoxel = new GroundVoxel();
					newBaseVoxel->height = height / voxelDimensionVertical;
					newBaseVoxel->materialId = 0;
					newBaseVoxel->angleOfTalus = 45;
					rowColumnVoxels.push_back(newBaseVoxel);

					instanceCount += 1;
				}
			}

			columnVoxels.push_back(rowColumnVoxels);
		}

		voxelMap.push_back(columnVoxels);
	}

	instancesToRender = instanceCount;

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

	for (int i = 0; i < voxelMap.size(); i++) {
		for (int j = 0; j < voxelMap[i].size(); j++) {
			for (int k = 0; k < voxelMap[i][j].size(); k++) {
				if (voxelMap[i][j][k]->materialId == 0 || voxelMap[i][j][k]->materialId == 1) {
					GroundVoxel* groundVoxel = (GroundVoxel*)voxelMap[i][j][k];
					delete groundVoxel;
				}
				else if (voxelMap[i][j][k]->materialId == 2) {
					Voxel* airVoxel = voxelMap[i][j][k];
					delete airVoxel;
				}
				else if (voxelMap[i][j][k]->materialId == 3) {
					RainVoxel* rainVoxel = (RainVoxel*)voxelMap[i][j][k];

					for (int dg = 0; dg < rainVoxel->sedimentAccumulated.size(); dg++) {
						GroundVoxel* groundVoxel = (GroundVoxel*)rainVoxel->sedimentAccumulated[dg];

						delete groundVoxel;
					}

					rainVoxel->sedimentAccumulated.clear();

					delete rainVoxel;
				}
			}
		}
	}

	voxelMap.clear();

	if (this->originalHeightMap != nullptr) {
		FreeImage_Unload(this->originalHeightMap);
	}

	if (forTraining) {
		if (this->complexImage != nullptr) {
			FreeImage_Unload(this->complexImage);
		}
		FIBITMAP* tempImg = FreeImage_Load(FreeImage_GetFileType(this->filePath.c_str(), 0), this->filePath.c_str());
		FIBITMAP* img = FreeImage_ConvertToRGBAF(tempImg);
		FreeImage_Unload(tempImg);
		this->originalHeightMap = img;

		FIBITMAP* tempImg2 = FreeImage_Load(FreeImage_GetFileType(this->complexFilePath.c_str(), 0), this->complexFilePath.c_str());
		FIBITMAP* img2 = FreeImage_ConvertToRGBAF(tempImg2);
		this->complexImage = img2;

		FreeImage_Unload(tempImg2);

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
	float currentStackHeight = determineBaseHeight(currentStack, currentStack.size());
	float neighbourStackHeight = determineBaseHeight(neighbourStack, neighbourStack.size());

	float heightDifference = (currentStackHeight - neighbourStackHeight);

	if (heightDifference > 0) {
		float angleOfTalus = getAngleOfTalus(heightDifference, 1);// voxelDimension);

		bool allVoxelTransferred = false;

		float volumeProportion = maxVolume * (heightDifference / sumOfHeights);
		float voxels = volumeProportion;// / voxelDimensionVertical;

		GroundVoxel* neighbourVoxel = nullptr;

		if (neighbourStack.size() > 0 && (neighbourStack[neighbourStack.size() - 1]->materialId == 0 || neighbourStack[neighbourStack.size() - 1]->materialId == 1)) {
			neighbourVoxel = (GroundVoxel*)neighbourStack[neighbourStack.size() - 1];
		}
		else {
			neighbourVoxel = new GroundVoxel();
			neighbourVoxel->height = 0;
			neighbourVoxel->materialId = currentStack[currentStack.size() - 1]->materialId;
			neighbourStack.push_back(neighbourVoxel);
			instanceCount++;
			instancesToRender = instanceCount;
		}

		while (!allVoxelTransferred && currentStack.size() > 0 && voxels > 0) {
			if (angleOfTalus > ((GroundVoxel *)currentStack[currentStack.size() - 1])->angleOfTalus) {
				
				//transfer 1 voxel every loop measuring angle of talus against top layer
				movedVoxelCount++;
				
				//TODO modify the material to have a velocity or change angle of talus to give the effect of accelerated movement downward

				if (neighbourVoxel->materialId != currentStack[currentStack.size() - 1]->materialId) {
					neighbourVoxel = new GroundVoxel();
					neighbourVoxel->height = 0;
					neighbourVoxel->materialId = currentStack[currentStack.size() - 1]->materialId;
					neighbourStack.push_back(neighbourVoxel);
					instanceCount++;
					instancesToRender = instanceCount;
				}

				float finalVolumeToBeTransferred = 0;

				if (currentStack[currentStack.size() - 1]->height < voxels) {
					voxels -= currentStack[currentStack.size() - 1]->height;
					finalVolumeToBeTransferred = currentStack[currentStack.size() - 1]->height;
					currentStack.erase(currentStack.end() - 1);
				}
				else {
					currentStack[currentStack.size() - 1]->height -= voxels;
					finalVolumeToBeTransferred = voxels;
					voxels = 0;
				}

				neighbourVoxel->height += finalVolumeToBeTransferred;

				if (voxels <= 0) {
					allVoxelTransferred = true;
				}
			}
			else {
				allVoxelTransferred = true;	
			}
		}
	}
	
}

inline void Terrain::transferSediment(RainVoxel* currentRainVoxel, RainVoxel* neighbourRainVoxel, float maximumHeightDifference, float sumOfHeights, float maxVolume, float heightDifference) {
	float volumeToGive = currentRainVoxel->getSedimentVolume() * (heightDifference / sumOfHeights);
	float volumeAbsorbable = neighbourRainVoxel->getAbsorptionVolume();
	if (volumeAbsorbable > 0) {
		float finalVolume = std::min(volumeToGive, volumeAbsorbable);

		std::vector<GroundVoxel*> lostSediments = currentRainVoxel->loseSediment(finalVolume);

		while (lostSediments.size() > 0) {
			if (!neighbourRainVoxel->absorbSediment(lostSediments[0]))
			{
				lostSediments.erase(lostSediments.begin());
				break;
			}
			else
			{
				lostSediments.erase(lostSediments.begin());
			}
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


inline void Terrain::transferSedimentFromSide(std::vector<Voxel*>& neighbourStack, GroundVoxel* neighbourGroundVoxel, int & neighbourIndex, float heightFromBase, float absorbedAmount, float neighbourBaseHeight, float neighbourTopHeight) {
	if (neighbourTopHeight > (heightFromBase + neighbourBaseHeight)) {
		//split into 2 pieces

		//create an air voxel in between the voxels. Make sure the above ground voxel 
		//starts from the heightFromBase + 0.001 to ensure it's at a higher altitude the next time

		//assign lower piece as neighbourGroundVoxel(or keep the existing one)

		float upperVoxelHeight = neighbourTopHeight - heightFromBase - neighbourBaseHeight;
		float lowerVoxelHeight = heightFromBase;

		GroundVoxel* upperGroundVoxel = new GroundVoxel();
		upperGroundVoxel->height = upperVoxelHeight;
		upperGroundVoxel->angleOfTalus = neighbourGroundVoxel->angleOfTalus;
		upperGroundVoxel->materialId = neighbourGroundVoxel->materialId;

		if (upperGroundVoxel->materialId < 0 || upperGroundVoxel->materialId > 3) {
			std::cout << "Transfer of illegal material " << std::endl;
		}

		neighbourStack.insert(neighbourStack.begin() + neighbourIndex + 1, upperGroundVoxel);

		neighbourGroundVoxel->height = lowerVoxelHeight;
	}

	float height = 0; //air height to create later
	int neighbourStackSize = neighbourStack.size();

	if (absorbedAmount >= heightFromBase) {		//if no more material present
		//delete the neighbour voxel

		//do the same logic as voxel where you take the above voxel and loop down

		height = heightFromBase;

		neighbourStack.erase(neighbourStack.begin() + neighbourIndex);
		neighbourStackSize = neighbourStack.size();
		neighbourIndex--;
	}
	else {
		//negate from height of neighbourVoxel

		height = absorbedAmount;

		neighbourGroundVoxel->height -= absorbedAmount;

		//take the above voxel and merge with extra space = heightFromBase - absorbedAmount
	}
	
	if (neighbourIndex < neighbourStackSize - 1 && neighbourIndex >= 0) {
		if (neighbourStack[neighbourIndex + 1]->materialId == 2) {	//Air voxel above
			for (int k = neighbourIndex; k >= 0; k--) {
				if (neighbourStack[k]->materialId == 0 || neighbourStack[k]->materialId == 1 || neighbourStack[k]->materialId == 3) {
					break;
				}

				Voxel* airVoxel = neighbourStack[k];
				height += airVoxel->height;
				delete airVoxel;
				neighbourStack.erase(neighbourStack.begin() + k);
				neighbourStackSize--;
				neighbourIndex--;
			}

			Voxel* topAirVoxel = neighbourStack[neighbourIndex + 1];
			topAirVoxel->height += height;
		}
		else if (neighbourStack[neighbourIndex + 1]->materialId == 3) {	//Rain Voxel above
			RainVoxel* aboveRainVoxel = (RainVoxel*)neighbourStack[neighbourIndex + 1];
			int k = 0;
			for (k = neighbourIndex; k >= 0; k--) {
				if (neighbourStack[k]->materialId == 3) {
					RainVoxel* neighbourRainVoxel = (RainVoxel*)neighbourStack[k];
					neighbourRainVoxel->height += aboveRainVoxel->height;
					for (int dg = 0; dg < aboveRainVoxel->sedimentAccumulated.size(); dg++) {
						neighbourRainVoxel->sedimentAccumulated.push_back(aboveRainVoxel->sedimentAccumulated[dg]);
					}

					delete aboveRainVoxel;
					neighbourStack.erase(neighbourStack.begin() + neighbourIndex + 1);
					neighbourStackSize--;
					k--;//To make sure the air is inserted at the right location

					break;
				}
				else if (neighbourStack[k]->materialId == 1 || neighbourStack[k]->materialId == 0) {
					break;
				}
				else if (neighbourStack[k]->materialId == 2) {
					Voxel* airVoxel = neighbourStack[k];
					height += airVoxel->height;
					delete airVoxel;
					neighbourStack.erase(neighbourStack.begin() + k);
					neighbourStackSize--;
					neighbourIndex--; //because of air voxel removal pushing indices lower
				}
			}

			if (neighbourIndex + 1 < neighbourStack.size() - 1 && k + 2 <= neighbourStack.size() && k >= 0) {
				Voxel* airVoxel = nullptr;
				if (k + 2 < neighbourStack.size() && neighbourStack[k + 2]->materialId == 2) {
					airVoxel = neighbourStack[k + 2];
					airVoxel->height += height;
				}
				else {
					createNewAirVoxel(&airVoxel, height);
					neighbourStack.insert(neighbourStack.begin() + k + 2, airVoxel);
					neighbourStackSize++;
				}
			}
		}
		else if (neighbourStack[neighbourIndex + 1]->materialId == 0 || neighbourStack[neighbourIndex + 1]->materialId == 1) {	// above voxel is ground
			int k = 0;
			for (k = neighbourIndex; k >= 0; k--) {
				if (neighbourStack[k]->materialId == 3 || neighbourStack[k]->materialId == 1 || neighbourStack[k]->materialId == 0) {
					break;
				}
				else if (neighbourStack[k]->materialId == 2) {
					Voxel* airVoxel = neighbourStack[k];
					height += airVoxel->height;
					delete airVoxel;
					neighbourStack.erase(neighbourStack.begin() + k);
					neighbourStackSize--;
					neighbourIndex--; //because of air voxel removal pushing indices lower
				}
			}

			if (neighbourIndex + 1 < neighbourStack.size() - 1 && k + 1 <= neighbourStack.size() && k >= 0) {
				Voxel* airVoxel = nullptr;
				createNewAirVoxel(&airVoxel, height);
				neighbourStackSize++;

				neighbourStack.insert(neighbourStack.begin() + k + 1, airVoxel);
			}
		}
	}
}

inline float Terrain::getAbsorbability(GroundVoxel * groundVoxel) {
	return 20.0f / (groundVoxel->angleOfTalus);
}

inline void Terrain::createNewRainVoxel(RainVoxel ** rainVoxel, float height = 1) {
	*rainVoxel = new RainVoxel();
	(*rainVoxel)->materialId = 3; 
	(*rainVoxel)->height = height;
	rainVoxelCreated++;
	instanceCount++;
	instancesToRender = instanceCount;
}

inline void Terrain::createNewAirVoxel(Voxel** airVoxel, float height = 1) {
	*airVoxel = new Voxel();
	(*airVoxel)->materialId = 2;
	(*airVoxel)->height = height;
	instanceCount++;
	instancesToRender = instanceCount;
}

inline bool Terrain::transferWater(RainVoxel * currentRainVoxel, std::vector<Voxel*>& currentStack, int currentIndex, float currentStackTopHeight, std::vector<Voxel*>& neighbourStack, int& neighbourIndex, int& neighbourStackSize, float minNeighStackTopHeight, float maximumHeightDifference, float sumOfHeights, float maxVolume, float heightDifference, int & stackSize, bool createNewRain = false) {
	//float volumeProportion = maxVolume * (heightDifference / sumOfHeights);
	float volumeProportion = ((currentStackTopHeight - minNeighStackTopHeight) / 4) * (heightDifference / sumOfHeights);
	//float volumeProportion = (currentRainVoxel->height - neighbourStack[neighbourIndex]->height) * (heightDifference / sumOfHeights);
	//float volumeProportion = (currentRainVoxel->height) / 2 * (heightDifference / sumOfHeights);
	//float voxels = volumeProportion / voxelDimensionVertical;

	RainVoxel* neighbourTopRainVoxel = nullptr;

	float maxVoxels = volumeProportion;

	if (neighbourStack[neighbourIndex]->materialId == 2) {
		Voxel* topAirVoxel = neighbourStack[neighbourIndex];

		if (neighbourIndex > 0 && neighbourStack[neighbourIndex - 1]->materialId == 3) {
			neighbourTopRainVoxel = (RainVoxel*)neighbourStack[neighbourIndex - 1];
		}

		int newRainInsertOffset = 0;

		if (neighbourTopRainVoxel == nullptr) {
			createNewRainVoxel(&neighbourTopRainVoxel, 0);
			neighbourStack.insert(neighbourStack.begin() + neighbourIndex, neighbourTopRainVoxel);
			newRainInsertOffset = 1;
			neighbourIndex++;
			neighbourStackSize++;
		}

		if (topAirVoxel->height <= maxVoxels) {
			maxVoxels = topAirVoxel->height;
			alertMaterialId(neighbourStack[neighbourIndex + newRainInsertOffset]);
			alertMaterialId(neighbourStack[neighbourIndex]);
			delete topAirVoxel;
			//alertMaterialId(neighbourStack[neighbourIndex]);
			//alertMaterialId(neighbourStack[neighbourIndex + newRainInsertOffset]);
			neighbourStack.erase(neighbourStack.begin() + neighbourIndex);
			neighbourStackSize--;
			neighbourIndex--;
		}
		else if (topAirVoxel->height > maxVoxels) {
			topAirVoxel->height = topAirVoxel->height - maxVoxels;
		}
	}
	else {
		if (createNewRain) {
			createNewRainVoxel(&neighbourTopRainVoxel, 0);
			neighbourStack.insert(neighbourStack.begin() + neighbourIndex + 1, neighbourTopRainVoxel);
			neighbourIndex++;
			neighbourStackSize++;
		}
		
		neighbourTopRainVoxel = (RainVoxel*)neighbourStack[neighbourIndex];
		alertMaterialId(neighbourStack[neighbourIndex]);
	}

	alertMaterialId(neighbourStack[neighbourIndex]);

	neighbourTopRainVoxel->height = neighbourTopRainVoxel->height + maxVoxels;
	//if (neighbourTopRainVoxel->height > 103) {
	//	std::cout << "caught the prick" << std::endl;
	//}

	//std::cout << "Transfer Water "<< currentRainVoxel->waterQuantity<< " max voxels "<<maxVoxels<<" current dissolved ground size "<<currentRainVoxel->dissolvedGround.size() << std::endl;

	bool deleteRainVoxel = false;
	float airHeight = 0;

	float waterTransferRatio = maxVoxels / currentRainVoxel->height;

	if (currentRainVoxel->height < maxVoxels) {
		airHeight = currentRainVoxel->height;
		deleteRainVoxel = true;
	}
	else {
		airHeight = maxVoxels;
		currentRainVoxel->height -= maxVoxels;
	}

	float absorbableVolume = currentRainVoxel->getSedimentVolume() * waterTransferRatio;
	float finalVolume = std::min(neighbourTopRainVoxel->getAbsorptionVolume(), absorbableVolume);
	std::vector<GroundVoxel*> lostVoxels;
	if (finalVolume > 0) {
		lostVoxels = currentRainVoxel->loseSediment(finalVolume);
		while (lostVoxels.size() > 0) {
			alertMaterialId(lostVoxels[0]);
			if (!neighbourTopRainVoxel->absorbSediment(lostVoxels[0])) {
				lostVoxels.erase(lostVoxels.begin());
				break;
			}
			lostVoxels.erase(lostVoxels.begin());
		}
	}

	if (deleteRainVoxel) {
		//If left over lostVoxel is present then deposit at the spot and negate that from air height
		for (int dg = 0; dg < lostVoxels.size(); dg++) {
			alertMaterialId(lostVoxels[dg]);
			if (currentIndex > 0) {
				if (currentStack[currentIndex - 1]->materialId == 0 || currentStack[currentIndex - 1]->materialId == 1) {
					alertMaterialId(currentStack[currentIndex - 1]);
					if (currentStack[currentIndex - 1]->materialId == lostVoxels[dg]->materialId) {
						currentStack[currentIndex - 1]->height += lostVoxels[dg]->height;
					}
					else {
						currentStack.insert(currentStack.begin() + currentIndex, lostVoxels[dg]);
						currentIndex++;
						stackSize++;
					}
					alertMaterialId(currentStack[currentIndex]);
				}
				else {
					std::cout << " shouldn't happen in evaporate" << std::endl;
				}
			}
			else {
				currentStack.insert(currentStack.begin() + currentIndex, lostVoxels[dg]);
				currentIndex++;
				stackSize++;
				
			}

			airHeight -= lostVoxels[dg]->height;
			if (airHeight < 0) {
				break;
			}
		}
	}

	//delete &lostVoxels; TODO use that vector swap thingy

	//std::cout << "Adjust air above" << std::endl;

	//Insert/Adjust air above in currentStack
	if (airHeight > 0 && currentIndex < currentStack.size() - 1) {
		if (currentStack[currentIndex + 1]->materialId == 2) {
			Voxel* currentAirVoxel = currentStack[currentIndex + 1];

			currentAirVoxel->height += airHeight;
		}
		else {
			Voxel* newAirVoxel = nullptr;
			createNewAirVoxel(&newAirVoxel, airHeight);

			currentStack.insert(currentStack.begin() + currentIndex + 1, newAirVoxel);
			stackSize++;
		}
	}

	if (deleteRainVoxel) {
		delete currentRainVoxel;
		currentStack.erase(currentStack.begin() + currentIndex);
		currentIndex--;
		stackSize--;
		if (currentIndex > 0) alertMaterialId(currentStack[currentIndex]);
		if(currentIndex > 0) alertMaterialId(currentStack[currentIndex - 1]);
	}

	alertMaterialId(neighbourStack[neighbourIndex]);

	return deleteRainVoxel;
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

							float currentStackHeight = determineBaseHeight(currentStack, currentStack.size());
							float neighbourStackHeight = determineBaseHeight(neighbourStack, neighbourStack.size());

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

inline float Terrain::determineBaseHeight(std::vector<Voxel *> stack, int index) {
	if (index > stack.size() || index < 0) {
		return -1;
	}

	float height = 0;

	for (int k = 0; k < index; k++) {
		Voxel* voxel = stack[k];
		height += voxel->height;
	}

	return height;
}

inline bool Terrain::isCellInRange(float firstBaseHeight, float firstTopHeight, float secondBaseHeight, float secondTopHeight) {
	return (secondBaseHeight > firstBaseHeight && secondBaseHeight < firstTopHeight)
		|| (secondTopHeight > firstBaseHeight && secondTopHeight < firstTopHeight) 
		|| (secondTopHeight > firstTopHeight && secondBaseHeight < firstBaseHeight);
}

inline void Terrain::alertMaterialId(Voxel* voxel) {
	if(voxel == nullptr){
		std::cout << "voxel is null";
	}
	else {
		if (voxel->materialId < 0 || voxel->materialId > 3) {
			std::cout << "voxel material id out of range" << std::endl;
		}
	}
}

inline void Terrain::performHydraulicErosion(int steps = 100, bool addRain = false, int rateOfRain = 3, bool forceEvaporate = true) {
	movedVoxelCount = 0;
	rainVoxelCreated = 0;
	rainVoxelEvaporated = 0;
	if (!isHydraulicErosionInProgress) {
		isHydraulicErosionInProgress = true;
		for (int s = 0; s < steps; s++) {
			std::cout << "Step " << s << " in progress" << std::endl;

			if (s == 1) {
				int myBlah = 2;
			}

			for (int i = 0; i < voxelMap.size(); i++) {
				for (int j = 0; j < voxelMap[i].size(); j++) {
					if ((i == 127 && j==23) || (i==127 && j==22) || (i==126&& j==22) ||(i==126 && j==23) ||(i==126 && j==24)) {
						std::cout << "I'm here" << std::endl;
					}
					//if (s == 3) {
					//	std::cout << "Step " << s << " i " << i<<" j "<<j << " in progress" << std::endl;
					//}
					int stackSize = voxelMap[i][j].size();
					for (int k = 0; k < stackSize; k++) {
						alertMaterialId(voxelMap[i][j][k]);
						if (voxelMap[i][j][k]->materialId == 3) {
							//std::cout << "i " << i << " j " << j<< " k "<<k<<" stackSize "<<stackSize << " repetition " << repetition << std::endl;
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

							RainVoxel* currentRainVoxel = (RainVoxel*)voxelMap[i][j][k];
							std::vector<Voxel*>& currentStack = voxelMap[i][j];
							float currentStackBaseHeight = determineBaseHeight(currentStack, k);
							float currentStackTopHeight = (currentStackBaseHeight + currentRainVoxel->height);

							/*if (repetition == 0 && k == currentStack.size() - 1) {
								std::cout << "i " << i << " j " << j << " top height " << currentStackTopHeight << " base height " << currentStackBaseHeight << std::endl;
							}*/

							//if ((i == 141 && j == 230)) {
							//	std::cout << "we're here" << std::endl;
							//}


							//for (int rep = 0; rep < 10; rep++) {
							//Sediment redistribution
								if (currentRainVoxel != nullptr) {
									for (int x = xStart; x < xEnd; x++) {
										for (int y = yStart; y < yEnd; y++) {
											if (x == 1 && y == 1) {
												continue;
											}

											if (currentRainVoxel->sedimentAccumulated.size() != 0) {
												std::vector<Voxel*>& neighbourStack = voxelMap[i + (x - 1)][j + (y - 1)];

												for (int n = 0; n < neighbourStack.size(); n++) {
													if (neighbourStack[n]->materialId == 3) {
														RainVoxel* neighbourRainVoxel = (RainVoxel*)neighbourStack[n];
														float neighbourStackBaseHeight = determineBaseHeight(neighbourStack, n);
														float neighbourStackTopHeight = (neighbourStackBaseHeight + neighbourRainVoxel->height);

														if (isCellInRange(currentStackBaseHeight, currentStackTopHeight, neighbourStackBaseHeight, neighbourStackTopHeight)) {
															float heightDifference = (currentStackBaseHeight - neighbourStackBaseHeight);

															if (heightDifference > 0) {
																if (heightDifference > sedimentMaximumHeightDifference) {
																	sedimentMaximumHeightDifference = heightDifference;
																}

																sedimentSumOfHeights += heightDifference;
															}
															else {
																heightDifference = (currentRainVoxel->getSedimentVolume() - neighbourRainVoxel->getSedimentVolume());

																if (heightDifference > 0) {
																	if (heightDifference > sedimentMaximumHeightDifference) {
																		sedimentMaximumHeightDifference = heightDifference;
																	}

																	sedimentSumOfHeights += heightDifference;
																}
															}
															/*else {
																heightDifference = (currentStackTopHeight - neighbourStackBaseHeight);

																if (heightDifference > 0) {
																	if (heightDifference > sedimentMaximumHeightDifference) {
																		sedimentMaximumHeightDifference = heightDifference;
																	}

																	sedimentSumOfHeights += heightDifference;
																}
															}*/
														}
													}

													alertMaterialId(neighbourStack[n]);
												}
											}
										}
									}

									//std::cout << " Sediment Redist " << std::endl;

									float sedimentMaxVolume = sedimentMaximumHeightDifference;

									if (sedimentSumOfHeights > 0) {
										for (int x = xStart; x < xEnd; x++) {
											for (int y = yStart; y < yEnd; y++) {
												if (x == 1 && y == 1) {
													continue;
												}
												std::vector<Voxel*>& neighbourStack = voxelMap[i + (x - 1)][j + (y - 1)];

												for (int n = 0; n < neighbourStack.size(); n++) {
													if (neighbourStack[n]->materialId == 3) {
														RainVoxel* neighbourRainVoxel = (RainVoxel*)neighbourStack[n];
														float neighbourStackBaseHeight = determineBaseHeight(neighbourStack, n);
														float neighbourStackTopHeight = (neighbourStackBaseHeight + neighbourRainVoxel->height);

														if (isCellInRange(currentStackBaseHeight, currentStackTopHeight, neighbourStackBaseHeight, neighbourStackTopHeight)) {
															float heightDifference = (currentStackBaseHeight - neighbourStackBaseHeight);

															if (heightDifference > 0) {
																transferSediment(currentRainVoxel, neighbourRainVoxel, sedimentMaximumHeightDifference, sedimentSumOfHeights, sedimentMaxVolume, heightDifference);
															}
															else {
																heightDifference = (currentRainVoxel->getSedimentVolume() - neighbourRainVoxel->getSedimentVolume());

																if (heightDifference > 0) {
																	transferSediment(currentRainVoxel, neighbourRainVoxel, sedimentMaximumHeightDifference, sedimentSumOfHeights, sedimentMaxVolume, heightDifference);
																}
															}
															/*else {
																heightDifference = (currentStackTopHeight - neighbourStackBaseHeight);

																if (heightDifference > 0) {
																	transferSediment(currentRainVoxel, neighbourRainVoxel, sedimentMaximumHeightDifference, sedimentSumOfHeights, sedimentMaxVolume, heightDifference);
																}
															}*/
														}
													}
												}
											}
										}
									}
								}

							//Water transport
							
								if (waterFlowEnabled) {

									float maximumHeightDifference = 0;
									float sumOfHeights = 0;
									float minNeighbourStackTopHeight = 0;

									for (int x = xStart; x < xEnd; x++) {
										for (int y = yStart; y < yEnd; y++) {
											if (x == 1 && y == 1) {
												continue;
											}

											std::vector<Voxel*>& neighbourStack = voxelMap[i + (x - 1)][j + (y - 1)];

											//if ((i + x - 1) == 141 && (j + y - 1) == 230) {
											//	std::cout << "found the we're here neighbour" << std::endl;
											//}

											for (int n = 0; n < neighbourStack.size(); n++) {
												if (neighbourStack[n]->materialId == 2) {
													Voxel* neighbourAirVoxel = neighbourStack[n];
													float neighbourStackBaseHeight = determineBaseHeight(neighbourStack, n);
													float neighbourStackTopHeight = (neighbourStackBaseHeight + neighbourAirVoxel->height);

													if (isCellInRange(currentStackBaseHeight, currentStackTopHeight, neighbourStackBaseHeight, neighbourStackTopHeight)) {
														float heightDifference = (currentStackTopHeight - neighbourStackBaseHeight);

														if (heightDifference > 0) {
															if (heightDifference > maximumHeightDifference) {
																maximumHeightDifference = heightDifference;
																minNeighbourStackTopHeight = neighbourStackBaseHeight;
															}

															sumOfHeights += (heightDifference);
														}
													}
												}
												else if (n == neighbourStack.size() - 1) {
													Voxel* peakVoxel = nullptr;
													peakVoxel = neighbourStack[n];

													float neighbourStackBaseHeight = determineBaseHeight(neighbourStack, n);
													float neighbourStackTopHeight = (neighbourStackBaseHeight + peakVoxel->height);

													float heightDifference = (currentStackTopHeight - neighbourStackTopHeight);

													if (heightDifference > 0) {
														if (heightDifference > maximumHeightDifference) {
															maximumHeightDifference = heightDifference;
															minNeighbourStackTopHeight = neighbourStackTopHeight;
														}

														sumOfHeights += (heightDifference);
													}
												}
												alertMaterialId(neighbourStack[n]);

											}
										}
									}

									if (minNeighbourStackTopHeight < currentStackBaseHeight) {
										minNeighbourStackTopHeight = currentStackBaseHeight;
									}

									float maxVolume = maximumHeightDifference;
									//std::cout << "processing " << i << " " << j << std::endl;

									//std::cout << "Water Flow " << std::endl;

									if (sumOfHeights > 0) {
										for (int x = xStart; x < xEnd; x++) {
											for (int y = yStart; y < yEnd; y++) {
												if (x == 1 && y == 1) {
													continue;
												}
												std::vector<Voxel*>& neighbourStack = voxelMap[i + (x - 1)][j + (y - 1)];

												int neighbourStackSize = neighbourStack.size();

												for (int n = 0; n < neighbourStackSize; n++) {
													if (neighbourStack[n]->materialId == 2) {
														//std::cout << "Material air" << std::endl;
														Voxel* neighbourAirVoxel = neighbourStack[n];
														float neighbourStackBaseHeight = determineBaseHeight(neighbourStack, n);
														float neighbourStackTopHeight = (neighbourStackBaseHeight + neighbourAirVoxel->height);

														if (isCellInRange(currentStackBaseHeight, currentStackTopHeight, neighbourStackBaseHeight, neighbourStackTopHeight)) {
															float heightDifference = (currentStackTopHeight - neighbourStackBaseHeight);

															if (heightDifference > 0 && currentRainVoxel != nullptr) {
																//std::cout << "Material air water transport" << std::endl;
																if (transferWater(currentRainVoxel, currentStack, k, currentStackTopHeight, neighbourStack, n, neighbourStackSize, minNeighbourStackTopHeight, maximumHeightDifference, sumOfHeights, maxVolume, heightDifference, stackSize)) {
																	currentRainVoxel = nullptr;
																	break;
																}
															}
														}
													}
													else if (n == neighbourStack.size() - 1) {
														Voxel* peakVoxel = nullptr;
														bool createNewRainVoxel = false;
														if (neighbourStack[n]->materialId == 0 || neighbourStack[n]->materialId == 1) {
															createNewRainVoxel = true;
														}
														peakVoxel = neighbourStack[n];
														//std::cout << "Material rain" << std::endl;
														float neighbourStackBaseHeight = determineBaseHeight(neighbourStack, n);
														float neighbourStackTopHeight = (neighbourStackBaseHeight + peakVoxel->height);

														float heightDifference = (currentStackTopHeight - neighbourStackTopHeight);

														if (heightDifference > 0 && currentRainVoxel != nullptr) {
															//std::cout << "Material rain water transport" << std::endl;
															/*if (heightDifference > 0.05f) {
																std::cout << "height diff greater than 0.1"<<std::endl;
															}*/

															if (transferWater(currentRainVoxel, currentStack, k, currentStackTopHeight, neighbourStack, n, neighbourStackSize, minNeighbourStackTopHeight, maximumHeightDifference, sumOfHeights, maxVolume, heightDifference, stackSize, createNewRainVoxel)) {
																currentRainVoxel = nullptr;
																break;
															}
														}

													}
													alertMaterialId(neighbourStack[n]);
												}

												if (currentRainVoxel == nullptr) {
													break;
												}
											}

											if (currentRainVoxel == nullptr) {
												break;
											}
										}
									}
								}
							//}

							//std::cout << " Sediment absorb " << std::endl;

							if (absorbWaterFromGround) {
								//Sediment absorption from ground
								if (currentRainVoxel != nullptr) {
									float height = 0;
									while (currentStack.size() > 1 && k > 0) {
										if (currentStack[k - 1]->materialId == 3) {
											//Merge 2 rain voxels below each other.
											RainVoxel* currentBelowRainVoxel = (RainVoxel*)currentStack[k - 1];
											currentBelowRainVoxel->height += currentRainVoxel->height;
											for (int dg = 0; dg < currentRainVoxel->sedimentAccumulated.size(); dg++) {
												currentBelowRainVoxel->sedimentAccumulated.push_back(currentRainVoxel->sedimentAccumulated[dg]);
											}
											currentStack.erase(currentStack.begin() + k);
											stackSize--;
											k--;
											delete currentRainVoxel;
											break;
										}
										else if (currentStack[k - 1]->materialId == 2) {
											Voxel* belowAirVoxel = currentStack[k - 1];
											height += belowAirVoxel->height;
											currentStack.erase(currentStack.begin() + k - 1);
											stackSize--;
											k--;
											delete belowAirVoxel;
										}
										else if (currentStack[k - 1]->materialId == 0 || currentStack[k - 1]->materialId == 1) {
											Voxel* topNonWaterVoxel = currentStack[k - 1];
											GroundVoxel* groundVoxel = (GroundVoxel*)topNonWaterVoxel;
											float absorbedSediment = currentRainVoxel->absorbSedimentFromGround(groundVoxel, getAbsorbability(groundVoxel));
											groundVoxel->height -= absorbedSediment;
											height += absorbedSediment;

											if (groundVoxel->height <= 0) {
												currentStack.erase(currentStack.begin() + k - 1);
												delete topNonWaterVoxel;
												k--;
												stackSize--;
											}
											else {
												break;
											}
										}
										else {
											std::cout << "Invalid material Id " << currentStack[k - 1]->materialId << " i "<< i<<" j "<<j << std::endl;
										}
									}

									if (k < currentStack.size() - 1 && height != 0) {
										if (currentStack[k + 1]->materialId == 2) {
											Voxel* airVoxel = currentStack[k + 1];
											airVoxel->height += height;
										}
										else {
											Voxel* airVoxel = nullptr;
											createNewAirVoxel(&airVoxel, height);

											currentStack.insert(currentStack.begin() + k + 1, airVoxel);
											stackSize++;
										}
									}
								}
							}
						}
					}

					for (int k = voxelMap[i][j].size() - 1; k >= 0 ; k--) {
						alertMaterialId(voxelMap[i][j][k]);
						if (voxelMap[i][j][k]->materialId == 3) {
							RainVoxel* currentRainVoxel = (RainVoxel *)voxelMap[i][j][k];
							
							if (currentRainVoxel != nullptr) {
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

								std::vector<Voxel*>& currentStack = voxelMap[i][j];
								float currentStackBaseHeight = determineBaseHeight(currentStack, k);
								float currentStackTopHeight = (currentStackBaseHeight + currentRainVoxel->height);

								if (sidewaysErosion) {
									//Sediment absorption from sideways
									for (int x = xStart; x < xEnd; x++) {
										for (int y = yStart; y < yEnd; y++) {
											if (x == 1 && y == 1) {
												continue;
											}
											std::vector<Voxel*>& neighbourStack = voxelMap[i + (x - 1)][j + (y - 1)];

											for (int n = neighbourStack.size() - 1; n > 0; n--) {
												if (neighbourStack[n]->materialId == 1)  {// || neighbourStack[n]->materialId == 0) {
													GroundVoxel* neighbourGroundVoxel = (GroundVoxel*)neighbourStack[n];
													float neighbourStackBaseHeight = determineBaseHeight(neighbourStack, n);
													float neighbourStackTopHeight = (neighbourStackBaseHeight + neighbourGroundVoxel->height);

													if (isCellInRange(currentStackBaseHeight, currentStackTopHeight, neighbourStackBaseHeight, neighbourStackTopHeight)) {
														float maxBaseHeight = std::max(neighbourStackBaseHeight, currentStackBaseHeight);
														float minTopHeight = std::min(neighbourStackTopHeight, currentStackTopHeight);

														float heightAsRatio = (minTopHeight - currentStackBaseHeight) / currentRainVoxel->height;

														float heightFromNeighbourBase = minTopHeight - neighbourStackBaseHeight;

														float absorbedSediment = currentRainVoxel->absorbSedimentFromSide(neighbourGroundVoxel, heightAsRatio, getAbsorbability(neighbourGroundVoxel));
														//float absorbedSediment = getAbsorbability(neighbourGroundVoxel);
														if (absorbedSediment > 0) {
															transferSedimentFromSide(neighbourStack, neighbourGroundVoxel, n, heightFromNeighbourBase, absorbedSediment, neighbourStackBaseHeight, neighbourStackTopHeight);
														}
													}
												}
												alertMaterialId(neighbourStack[n]);
											}
										}
									}
								}
							}

							if (evaporateEnabled) {
								//evaporation -  Should be last since we delete currentRainVoxel
								evaporate(currentRainVoxel, k, stackSize, i, j, false);
								alertMaterialId(voxelMap[i][j][k]);
							}
						}
					}

					if (addRain) {// || s < steps / 3) { //rains only for first half of steps to ensure that the sediments are transported
						int rainAtStep = rateOfRain;// std::rand() % (rateOfRain + 1);

						//Addition through Raining simulation
						if (rainAtStep != 0) {
							RainVoxel* rainVoxel = nullptr;

							if (voxelMap[i][j].size() != 0 && voxelMap[i][j][voxelMap[i][j].size() - 1]->materialId == 3) {
								rainVoxel = (RainVoxel*)voxelMap[i][j][voxelMap[i][j].size() - 1];
								rainVoxel->height += rainAtStep;
							}
							else {
								createNewRainVoxel(&rainVoxel, rainAtStep);
								voxelMap[i][j].push_back(rainVoxel);
							}

							if (voxelMap[i][j].size() > 1) {
								Voxel* topNonWaterVoxel = voxelMap[i][j][voxelMap[i][j].size() - 2];
								GroundVoxel* groundVoxel = (GroundVoxel*)topNonWaterVoxel;
								float absorbedSediment = rainVoxel->absorbSedimentFromGround(groundVoxel, getAbsorbability(groundVoxel));
								groundVoxel->height -= absorbedSediment;

								if (groundVoxel->height <= 0) {
									voxelMap[i][j].erase(voxelMap[i][j].end() - 2);
									delete groundVoxel;
									stackSize--;
								}
							}
						}
					}
				}
			}
		}

		if (forceEvaporate) {
			//Force evaporate every rain voxel
			for (int i = 0; i < voxelMap.size(); i++) {
				//std::cout << "i " << i << std::endl;
				for (int j = 0; j < voxelMap[i].size(); j++) {
					int stackSize = voxelMap[i][j].size();
					for (int k = 0; k < stackSize; k++) {
							if (voxelMap[i][j].size() != 0 && voxelMap[i][j][k]->materialId == 3) {
							RainVoxel* currentRainVoxel = (RainVoxel*)voxelMap[i][j][k];

							evaporate(currentRainVoxel, k, stackSize, i, j, true);
						}
					}
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

inline void Terrain::evaporate(RainVoxel* currentRainVoxel, int & currentIndex, int & stackSize, int i, int j, bool forceEvaporate = false) {

	float sedimentsHeightToBeDeposited = 0;
	float maxSedimentVolume = currentRainVoxel->getSedimentVolume();
	bool fullyEvaporated = false;
	float evaporatedHeight = 0;

	if (forceEvaporate) {
		sedimentsHeightToBeDeposited = maxSedimentVolume;
	}
	else {
		fullyEvaporated = currentRainVoxel->evaporate(sedimentsHeightToBeDeposited, evaporatedHeight);
	}
	
	if (sedimentsHeightToBeDeposited > 0) {
		float maxSedimentDeposit = sedimentsHeightToBeDeposited;
		float availableHeight = -1;
		Voxel* aboveAirVoxel = nullptr;
			
		//Create air voxel above only if its not top rain voxel
		if (currentIndex < voxelMap[i][j].size() - 1) {
			if (forceEvaporate) {
				availableHeight = currentRainVoxel->height;
			}
			else {
				availableHeight = evaporatedHeight;
			}

			if (voxelMap[i][j][currentIndex + 1]->materialId == 2) {
				aboveAirVoxel = voxelMap[i][j][currentIndex + 1];
				availableHeight += aboveAirVoxel->height;
			}

			/*
			if (availableHeight > 0.3f * sedimentsHeightToBeDeposited) {
				if (sedimentsHeightToBeDeposited == maxSedimentVolume) {
					maxSedimentDeposit = 0.3f * sedimentsHeightToBeDeposited;
				}
				else {
					maxSedimentDeposit = sedimentsHeightToBeDeposited;
				}
			}
			else {
				maxSedimentDeposit = availableHeight;
			}
			*/
			maxSedimentDeposit = 0;

			if (availableHeight > maxSedimentDeposit) {
				float remainingHeight = availableHeight - maxSedimentDeposit;

				if (aboveAirVoxel == nullptr) {
					createNewAirVoxel(&aboveAirVoxel, remainingHeight);
					voxelMap[i][j].insert(voxelMap[i][j].begin() + currentIndex + 1, aboveAirVoxel);
				}
				else {
					aboveAirVoxel->height = remainingHeight;
				}
			}
		}

		std::vector<GroundVoxel*> lostSediments = currentRainVoxel->loseSediment(maxSedimentDeposit);

		for (int dg = 0; dg < lostSediments.size(); dg++) {
			if (currentIndex > 0) {
				if (voxelMap[i][j][currentIndex - 1]->materialId == 0 || voxelMap[i][j][currentIndex - 1]->materialId == 1) {
					if (voxelMap[i][j][currentIndex - 1]->materialId == lostSediments[dg]->materialId) {
						voxelMap[i][j][currentIndex - 1]->height += lostSediments[dg]->height;
					}
					else {
						voxelMap[i][j].insert(voxelMap[i][j].begin() + currentIndex, lostSediments[dg]);
						currentIndex++;
						stackSize++;
					}
				}
				else {
					std::cout << "i " << i << "j " << j << " shouldn't happen in evaporate" << std::endl;
				}
			}
			else {
				voxelMap[i][j].insert(voxelMap[i][j].begin() + currentIndex, lostSediments[dg]);
				currentIndex++;
				stackSize++;
			}
		}

		//delete[] & lostSediments; TODO use that vector swap thingy instead
	}

	if (fullyEvaporated  || forceEvaporate) {
		rainVoxelEvaporated++;
		delete currentRainVoxel;
		voxelMap[i][j].erase(voxelMap[i][j].begin() + currentIndex);
		stackSize--;
		currentIndex--;
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
