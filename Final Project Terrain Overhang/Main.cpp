#include <windows.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>

#include "InitShader.h"    //Functions for loading shaders from text files
#include "LoadMesh.h"      //Functions for creating OpenGL buffers from mesh files
#include "LoadTexture.h"   //Functions for creating OpenGL textures from image files
#include "VideoMux.h"      //Functions for saving videos
#include "Terrain.cpp"
#include "DebugCallback.h"

const int init_window_width = 1024;
const int init_window_height = 1024;
const char* const window_title = "Template";

static const std::string vertex_shader("template_vs.glsl");
static const std::string fragment_shader("template_fs.glsl");
GLuint shader_program = -1;

static const std::string texture_name = "HeightMap2.png";
static const std::string complex_features_map_name = "Complex.png";

//GLuint texture_id = -1; //Texture map for mesh
//MeshData mesh_data;

int drawMode = 2;
int indexCount = 20;
int instanceToRender = 0;
float angle = -2.841f;
float scale[3] = { 1, 1, 1 };
float aspect = 1.0f;
float lightDir[3];
int renderMode = 0;
bool updateErosion = false;

float x = 0;
float y = 3.697;
float z = 5;

float c1 = 0;
float c2 = 0.84;
float c3 = 0;

bool recording = false;

Terrain * terrain = nullptr;

//For an explanation of this program's structure see https://www.glfw.org/docs/3.3/quick.html 

void draw_gui(GLFWwindow* window)
{
   //Begin ImGui Frame
   ImGui_ImplOpenGL3_NewFrame();
   ImGui_ImplGlfw_NewFrame();
   ImGui::NewFrame();

   //Draw Gui
   ImGui::Begin("Debug window");                       
   if (ImGui::Button("Quit"))                          
   {
      glfwSetWindowShouldClose(window, GLFW_TRUE);
   }    

   const int filename_len = 256;
   static char video_filename[filename_len] = "capture.mp4";

   ImGui::InputText("Video filename", video_filename, filename_len);
   ImGui::SameLine();
   if (recording == false)
   {
      if (ImGui::Button("Start Recording"))
      {
         int w, h;
         glfwGetFramebufferSize(window, &w, &h);
         recording = true;
         start_encoding(video_filename, w, h); //Uses ffmpeg
      }
      
   }
   else
   {
      if (ImGui::Button("Stop Recording"))
      {
         recording = false;
         finish_encoding(); //Uses ffmpeg
      }
   }

   ImGui::SliderFloat3("Scale", &scale[0], 0.01f, 100.0f);
   ImGui::SliderFloat3("Light Dir", &lightDir[0], -1, 1);
   ImGui::SliderInt("Draw Mode", &drawMode, 0, 3);
   if (ImGui::InputInt("Indices", &indexCount, 1, 200000)) {
       if (terrain != nullptr) {
           terrain->setIndexCountToRender(indexCount);
       }
   }
   
   if (ImGui::InputInt("Instance To Render", &instanceToRender, 1, 1000000)) {
       if (terrain != nullptr) {
           terrain->setInstancesToRender(instanceToRender);
       }
   }

   if (ImGui::Button("Process Terrains for training")) {
       terrain = new Terrain(true, glm::vec3(0.0f, 0.0f, 0.0f), texture_name, complex_features_map_name, glm::vec3(scale[0], scale[1], scale[2]), 0.02f, 0);
       terrain->generateTerrain();
       instanceToRender = terrain->instancesToRender;
   }

   if (ImGui::Button("Process Terrains for final output")) {
       terrain = new Terrain(false, glm::vec3(0.0f, 0.0f, 0.0f), texture_name, complex_features_map_name, glm::vec3(scale[0], scale[1], scale[2]), 0.02f, renderMode);
       terrain->generateTerrain();
       instanceToRender = terrain->instancesToRender;
   }

   if (terrain != nullptr) {
       const char * completionString = (terrain->hasThermalErosionCompleted ? "terrain rendering done" : "terrain rendering in progress");
       ImGui::Text(completionString);
       if (ImGui::Checkbox("Update Mesh After Erosion", &updateErosion)) {
           if (updateErosion) {
               terrain->updateTerrain();
           }
       }
       if (ImGui::Button("Thermal Erosion")) {
           terrain->performThermalErosion(1);
           if (updateErosion) {
               terrain->updateTerrain();
           }
       }
       
       if (terrain->forTraining) {
           if (ImGui::Button("Export Image")) {
                FIBITMAP* exportImage = nullptr;
                terrain->exportOutput(exportImage, "outputImage.png");
           }
       }
       else {
           if (ImGui::SliderInt("Render Mode", &renderMode, 0, 1)) {
               terrain->setRenderMode(renderMode);
           }

           if (ImGui::Button("Export as OBJ")) {
                
           }
       }
   }

   ImGui::SliderFloat("View angle", &angle, -glm::pi<float>(), +glm::pi<float>());
   //ImGui::SliderFloat("Scale", &scale, 0.01f, +100.0f);

   ImGui::SliderFloat("X", &x, -100, 100);
   ImGui::SliderFloat("Y", &y, -100, 100);
   ImGui::SliderFloat("Z", &z, -100, 100);

   ImGui::SliderFloat("C1", &c1, -100, 100);
   ImGui::SliderFloat("C2", &c2, -100, 100);
   ImGui::SliderFloat("C3", &c3, -100, 100);

   ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
   ImGui::End();

   static bool show_test = false;
   ImGui::ShowDemoWindow(&show_test);

   //End ImGui Frame
   ImGui::Render();
   ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

// This function gets called every time the scene gets redisplayed
void display(GLFWwindow* window)
{
   //Clear the screen to the color previously specified in the glClearColor(...) call.
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
   
   glm::mat4 M = glm::rotate(angle, glm::vec3(0.0f, 1.0f, 0.0f))*glm::scale(glm::vec3(scale[0], scale[1], scale[2]));
   glm::mat4 V = glm::lookAt(glm::vec3(x, y, z), glm::vec3(c1, c2, c3), glm::vec3(0.0f, 1.0f, 0.0f));
   glm::mat4 P = glm::perspective(glm::pi<float>()/4.0f, aspect, 0.1f, 1000.0f);

   //glUseProgram(shader_program);

   //glActiveTexture(GL_TEXTURE0);
   //glBindTexture(GL_TEXTURE_2D, texture_id);
   //int tex_loc = glGetUniformLocation(shader_program, "diffuse_tex");
   //glUniform1i(tex_loc, 0); // we bound our texture to texture unit 0


   //Get location for shader uniform variable
   glm::mat4 PVM = P*V*M;
   int PVM_loc = glGetUniformLocation(shader_program, "PVM");
   glUniformMatrix4fv(PVM_loc, 1, false, glm::value_ptr(PVM));

   if (terrain != nullptr) {
       terrain->setAngle(angle);
       terrain->setDrawMode(drawMode);
       terrain->setIndexCountToRender(indexCount);
       terrain->render(V, P, shader_program, lightDir);
       terrain->setScale(glm::vec3(scale[0], scale[1], scale[2]));
   }

   draw_gui(window);

   if (recording == true)
   {
      glFinish();
      glReadBuffer(GL_BACK);
      int w, h;
      glfwGetFramebufferSize(window, &w, &h);
      read_frame_to_encode(&rgb, &pixels, w, h);
      encode_frame(rgb);
   }

   /* Swap front and back buffers */
   glfwSwapBuffers(window);
}

void idle()
{
   float time_sec = static_cast<float>(glfwGetTime());

   //Pass time_sec value to the shaders
   int time_loc = glGetUniformLocation(shader_program, "time");
   glUniform1f(time_loc, time_sec);
}

void reload_shader()
{
   GLuint new_shader = InitShader(vertex_shader.c_str(), fragment_shader.c_str());

   if (new_shader == -1) // loading failed
   {
      glClearColor(1.0f, 0.0f, 1.0f, 0.0f); //change clear color if shader can't be compiled
   }
   else
   {
      //glClearColor(0.35f, 0.35f, 0.35f, 0.0f);
      glClearColor(1.0f, 0.35f, 0.35f, 0.0f);

      if (shader_program != -1)
      {
         glDeleteProgram(shader_program);
      }
      shader_program = new_shader;
   }
}

//This function gets called when a key is pressed
void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods)
{
   std::cout << "key : " << key << ", " << char(key) << ", scancode: " << scancode << ", action: " << action << ", mods: " << mods << std::endl;

   if(action == GLFW_PRESS)
   {
      switch(key)
      {
         case 'r':
         case 'R':
            reload_shader();     
         break;

         case GLFW_KEY_ESCAPE:
            glfwSetWindowShouldClose(window, GLFW_TRUE);
         break;     
      }
   }
}

//This function gets called when the mouse moves over the window.
void mouse_cursor(GLFWwindow* window, double x, double y)
{
    //std::cout << "cursor pos: " << x << ", " << y << std::endl;
}

//This function gets called when a mouse button is pressed.
void mouse_button(GLFWwindow* window, int button, int action, int mods)
{
    //std::cout << "button : "<< button << ", action: " << action << ", mods: " << mods << std::endl;
}

void resize(GLFWwindow* window, int width, int height)
{
    //Set viewport to cover entire framebuffer
    glViewport(0, 0, width, height);
    //Set aspect ratio used in view matrix calculation
    aspect = float(width) / float(height);
}

//Initialize OpenGL state. This function only gets called once.
void initOpenGL()
{
   glewInit();

   //Print out information about the OpenGL version supported by the graphics driver.	
   std::cout << "Vendor: "       << glGetString(GL_VENDOR)                    << std::endl;
   std::cout << "Renderer: "     << glGetString(GL_RENDERER)                  << std::endl;
   std::cout << "Version: "      << glGetString(GL_VERSION)                   << std::endl;
   std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION)  << std::endl;
   glEnable(GL_DEPTH_TEST);

    #ifdef _DEBUG
       RegisterCallback();
    #endif

   reload_shader();
}

//C++ programs start executing in the main() function.
int main(int argc, char **argv)
{
   GLFWwindow* window;

   /* Initialize the library */
   if (!glfwInit())
   {
      return -1;
   }

#ifdef _DEBUG
   glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
#endif

   /* Create a windowed mode window and its OpenGL context */
   window = glfwCreateWindow(init_window_width, init_window_height, window_title, NULL, NULL);
   if (!window)
   {
      glfwTerminate();
      return -1;
   }

   //Register callback functions with glfw. 
   glfwSetKeyCallback(window, keyboard);
   glfwSetCursorPosCallback(window, mouse_cursor);
   glfwSetMouseButtonCallback(window, mouse_button);
   glfwSetFramebufferSizeCallback(window, resize);

   /* Make the window's context current */
   glfwMakeContextCurrent(window);

   initOpenGL();
   
   //Init ImGui
   IMGUI_CHECKVERSION();
   ImGui::CreateContext();
   ImGui_ImplGlfw_InitForOpenGL(window, true);
   ImGui_ImplOpenGL3_Init("#version 400");

   /* Loop until the user closes the window */
   while (!glfwWindowShouldClose(window))
   {
      idle();
      display(window);

      /* Poll for and process events */
      glfwPollEvents();
   }

    // Cleanup ImGui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

   glfwTerminate();
   return 0;
}