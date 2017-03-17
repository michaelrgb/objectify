#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <libgen.h>

#include "internal.h"

#include <cuda_gl_interop.h>

GLuint framePBO, framePBO_tex;
cudaGraphicsResource_t framePBO_CUDA;
float *reorderedCudaPtr;

GLuint boxVBO;
GLuint boxVAO;
GLuint boxShaderProg, texShaderProg;
GLint viewportXY[2];

void make_row_contiguous(float* dst, const uint8_t* src,
                         int width, int height, int channels);
void onFrameBegin();
void onFrameEnd();

void loadVBO(const Box& b)
{
  float points[] = {
    b.left(), b.bottom(),
    b.right(), b.bottom(),
    b.right(), b.top(),
    b.left(), b.top(),
  };
  glBindBuffer(GL_ARRAY_BUFFER, boxVBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void drawDetection(Detection det)
{
    loadVBO(det.b);

    GLboolean old_depth = glIsEnabled(GL_DEPTH_TEST);
    if(old_depth) glDisable(GL_DEPTH_TEST);
    GLint curr_program; glGetIntegerv(GL_CURRENT_PROGRAM, &curr_program);
    glUseProgram(boxShaderProg);
    glUniform3fv(glGetUniformLocation(boxShaderProg, "in_colour"), 1, det.rgb);

    glLineWidth(det.lineWidth ? det.lineWidth : 1);
    GLint curr_vao; glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &curr_vao);
    glBindVertexArray(boxVAO);
    glDrawArrays(GL_LINE_LOOP, 0, 4);
    glBindVertexArray(curr_vao);
    glUseProgram(curr_program);
    if(old_depth) glEnable(GL_DEPTH_TEST);

    GLenum err = glGetError(); assert(!err);
}

void drawTexture()
{
    Box b;
    b.w = b.h = 1.f;
    loadVBO(b);

    GLenum err = glGetError(); assert(!err);

    glUseProgram(texShaderProg);
    int loc = glGetUniformLocation(texShaderProg, "tex");
    glUniform1i(loc, 0);

    loc = glGetUniformLocation(texShaderProg, "viewportXY");
    glUniform2iv(loc, 1, viewportXY);

    glBindVertexArray(boxVAO);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    glBindVertexArray(0);

    glUseProgram(0);

    err = glGetError(); assert(!err);
}

void dumpImage(const uint8_t* bytes, size_t n)
{
  FILE* f = fopen("/tmp/imageDump.data", "wb");
  fwrite(bytes, 1, n, f);
  fclose(f);
}

void dumpImage(const float* floats, size_t n)
{
  uint8_t* bytes = new uint8_t[n];
  for(size_t i = 0; i < n; i++)
    bytes[i] = (uint8_t)(floats[i] * 255.f + 0.5f);
  dumpImage(bytes, n);
  delete[] bytes;
}

template<typename T>
void dumpCudaImage(const T* cudaData, size_t n)
{
  T* copyFromCuda = new T[n];
  cudaError_t err = cudaMemcpy(copyFromCuda, cudaData,
                               n * sizeof(T), cudaMemcpyDeviceToHost);
  dumpImage(copyFromCuda, n);
  delete[] copyFromCuda;
}

void initBoxDraw()
{
    GLint curr_vao; glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &curr_vao);
    glGenVertexArrays(1, &boxVAO);
    glBindVertexArray(boxVAO);
    glEnableVertexAttribArray(0);

    glGenBuffers(1, &boxVBO);
    glBindBuffer(GL_ARRAY_BUFFER, boxVBO);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindVertexArray(curr_vao);

    const char *vertex_shader =
        "#version 330\n"
        "layout (location = 0) in vec2 vp;"
        "void main() {"
        "  gl_Position = vec4(vp, 0.0, 1.0);"
        "}";

    const char *fragment_shader =
        "#version 330\n"
        "uniform vec3 in_colour;"
        "out vec4 out_colour;"
        "void main() {"
        "  out_colour = vec4(in_colour, 1.0);"
        "}";

    const char *tex_fragment_shader =
        "#version 330\n"
        "uniform sampler2D tex;"
        "uniform ivec2 viewportXY;"
        "out vec4 out_colour;"
        "void main() {"
        "  vec2 coord = vec2(gl_FragCoord.x / viewportXY.x, 1.f - gl_FragCoord.y / viewportXY.y);"
        "  out_colour = texture(tex, coord);"
        //"  out_colour = vec4(coord.x, coord.y, 0.f, 1.f);"
        "}";

    char infoLog[10000];
    GLsizei length;

    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertex_shader, NULL);
    glCompileShader(vs);
    glGetShaderInfoLog(vs, sizeof(infoLog), &length, infoLog);
    assert(!length);

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragment_shader, NULL);
    glCompileShader(fs);
    glGetShaderInfoLog(vs, sizeof(infoLog), &length, infoLog);
    assert(!length);

    boxShaderProg = glCreateProgram();
    glAttachShader(boxShaderProg, fs);
    glAttachShader(boxShaderProg, vs);
    glLinkProgram(boxShaderProg);
    glGetProgramInfoLog(boxShaderProg, sizeof(infoLog), &length, infoLog);
    assert(!length);

    fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &tex_fragment_shader, NULL);
    glCompileShader(fs);
    glGetShaderInfoLog(vs, sizeof(infoLog), &length, infoLog);
    assert(!length);

    texShaderProg = glCreateProgram();
    glAttachShader(texShaderProg, fs);
    glAttachShader(texShaderProg, vs);
    glLinkProgram(texShaderProg);
    glGetProgramInfoLog(texShaderProg, sizeof(infoLog), &length, infoLog);
    assert(!length);
}

void initInterop()
{
  // PBO to read pixels from FBO
  glGenBuffers(1, &framePBO);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, framePBO);
  glBufferData(GL_PIXEL_PACK_BUFFER, DETECT_BYTES, nullptr, GL_DYNAMIC_READ);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);

  glGenTextures(1, &framePBO_tex);
  glBindTexture(GL_TEXTURE_2D, framePBO_tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  // CUDA Interop
  cudaError_t cudaErr = cudaGraphicsGLRegisterBuffer(&framePBO_CUDA, framePBO,
                                                     cudaGraphicsMapFlagsReadOnly);
  assert(!cudaErr);

  cudaMalloc((void**)&reorderedCudaPtr, DETECT_BYTES * sizeof(float));

  initBoxDraw();
}

extern "C"
{
int mainDarknet(int argc, char **argv);
void *detect_in_thread(void *ptr);
}

void initDarknet()
{
    // Initialize darknet
    const char *argv[] = {
        "darknet",
        "detector",
        "demo",
        "cfg/coco.data",
        "cfg/yolov3.cfg",
        "yolov3.weights",
        "-thresh", "0.3"
    };

    // Change to Darknet directory whilst we load the configs & weights.
    char* old_cwd = get_current_dir_name();
    char* libDir = getenv("LD_LIBRARY_PATH");
    char path[10000];
    strcpy(path, libDir);
    strcat(path, "/darknet");
    int ret = chdir(path);
    mainDarknet(ARRAY_SIZE(argv), const_cast<char **>(argv));
    ret = chdir(old_cwd);
    free(old_cwd);
}

void doDetection()
{
    if(!(getLockState() & NUM_LOCK))
        return;

    // Map the PBO into CUDA and run object detection.
    cudaError_t cudaErr = cudaGraphicsMapResources(1, &framePBO_CUDA, 0);
    assert(!cudaErr);

    uint8_t *framePBO_mapped;
    size_t nMappedBytes;
    cudaErr = cudaGraphicsResourceGetMappedPointer((void **)&framePBO_mapped,
                                                 &nMappedBytes, framePBO_CUDA);
    assert(!cudaErr);
    //dumpCudaImage(framePBO_mapped, DETECT_BYTES);

    make_row_contiguous(reorderedCudaPtr, framePBO_mapped, DETECT_WIDTH, DETECT_HEIGHT, DETECT_CHANNELS);
    //dumpCudaImage(reorderedCudaPtr, DETECT_BYTES);
    onFrameBegin();
    detect_in_thread(reorderedCudaPtr);
    onFrameEnd();

    cudaErr = cudaGraphicsUnmapResources(1, &framePBO_CUDA, 0);
    assert(!cudaErr);
}
