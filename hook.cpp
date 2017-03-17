#include <dlfcn.h>

#include "internal.h"

typedef void (*PFNGLXSWAPBUFFERSPROC)(Display *dpy, GLXDrawable drawable);
PFNGLXSWAPBUFFERSPROC glXSwapBuffers_real;
PFNGLXGETPROCADDRESSPROC glXGetProcAddress_real;

GLuint copyFBO, copyRBO;

void initHook()
{
  glXSwapBuffers_real =
      (PFNGLXSWAPBUFFERSPROC)dlsym(RTLD_NEXT, "glXSwapBuffers");

  // FBO to blit from app's framebuffer
  glGenRenderbuffers(1, &copyRBO);
  glBindRenderbuffer(GL_RENDERBUFFER, copyRBO);
  glRenderbufferStorage(GL_RENDERBUFFER,
                        GL_RGB,
                        DETECT_WIDTH,
                        DETECT_HEIGHT);
  glGenFramebuffers(1, &copyFBO);
  glBindFramebuffer(GL_FRAMEBUFFER, copyFBO);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                            GL_COLOR_ATTACHMENT0,
                            GL_RENDERBUFFER,
                            copyRBO);

  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  assert(status == GL_FRAMEBUFFER_COMPLETE);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

extern "C"
{
#define EXPORT __attribute__((visibility("default")))
EXPORT void glXSwapBuffers(Display *dpy, GLXDrawable drawable)
{
    if (!glXSwapBuffers_real)
    {
        initHook();
        initInterop();
        initDarknet();
    }

    GLint viewport[4] = {0};
    glGetIntegerv(GL_VIEWPORT, viewport);
    viewportXY[0] = viewport[2];
    viewportXY[1] = viewport[3];
    // Blit from the default FBO to our resized FBO, and flip upside down.
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, copyFBO);
    bool flipY = true;
    glBlitFramebuffer(0, flipY ? viewportXY[1] : 0, viewportXY[0], flipY ? 0 : viewportXY[1],
                      0, 0, DETECT_WIDTH, DETECT_HEIGHT,
                      GL_COLOR_BUFFER_BIT, GL_LINEAR);
    GLenum err = glGetError(); assert(!err);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

    // Read pixels from our FBO into the PBO.
    glBindBuffer(GL_PIXEL_PACK_BUFFER, framePBO);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, copyFBO);
    glReadPixels(0, 0, DETECT_WIDTH, DETECT_HEIGHT, GL_BGR, GL_UNSIGNED_BYTE, nullptr);
    err = glGetError(); assert(!err);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
#if 0
    uint8_t* bytes = new uint8_t[DETECT_BYTES];
    glReadPixels(0, 0, DETECT_WIDTH, DETECT_HEIGHT, GL_BGR, GL_UNSIGNED_BYTE, bytes);
    dumpImage(bytes, DETECT_BYTES);
#endif
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

    doDetection();

    glXSwapBuffers_real(dpy, drawable);
}

EXPORT __GLXextFuncPtr glXGetProcAddress(const GLubyte *f)
{
  if(glXGetProcAddress_real == NULL)
    glXGetProcAddress_real = (PFNGLXGETPROCADDRESSPROC)dlsym(RTLD_NEXT, "glXGetProcAddress");

  __GLXextFuncPtr realFunc = glXGetProcAddress_real(f);
  const char* func = (const char*)f;

  if(!strcmp(func, "glXSwapBuffers"))
    return (__GLXextFuncPtr)glXSwapBuffers;

  return realFunc;
}
EXPORT __GLXextFuncPtr glXGetProcAddressARB(const GLubyte *f)
{
  return glXGetProcAddress(f);
}

void *dlopenThis, *dlopenGL;

__attribute__((visibility("default"))) void *dlopen(const char *filename, int flag)
{
  //printf("dlopen: %s\n", filename); fflush(stdout);

  typedef void *(*DLOPENPROC)(const char *, int);
  static DLOPENPROC realdlopen;
  if(!realdlopen)
    realdlopen = (DLOPENPROC)dlsym(RTLD_NEXT, "dlopen");

  void *ret = realdlopen(filename, flag);

  if(!dlopenThis &&
     filename && strstr(filename, "libGL.so"))
  {
    dlopenGL = ret;
    dlopenThis = ret = realdlopen("libglhook.so", flag);
  }

  return ret;
}

void* _dl_sym(void *, const char *, void *);
EXPORT void *dlsym(void *__restrict handle, const char *__restrict name)
{
    //printf("dlsym: %s\n", name); fflush(stdout);

    if(!strcmp(name, "glXGetProcAddressARB"))
        return (void*)glXGetProcAddress;
    if(!strcmp(name, "dlsym"))
        return (void*)dlsym;
    if(!strcmp(name, "glcuR0d4nX")) // NVidia's libGL.so.1 looks this up, but not used.
        return (void*)0xf00;

    if(handle == dlopenThis)
        handle = dlopenGL;

    void* ret = _dl_sym(handle, name, (void*)dlsym);
    return ret;
}

} // extern "C"
