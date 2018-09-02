#include <assert.h>
#include <math.h>

#include <X11/Xlib.h>
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glx.h>
#include <cuda_runtime_api.h>

#include "darknet/src/box.h"

#define DETECT_WIDTH 608  // From Darknet
#define DETECT_HEIGHT 608
#define DETECT_CHANNELS 3
#define DETECT_BYTES (DETECT_WIDTH * DETECT_HEIGHT * DETECT_CHANNELS)

#define ARRAY_SIZE(a) (sizeof((a)) / sizeof((a)[0]))

struct Box : box
{
    Box() { memset(this, 0, sizeof(Box)); }
    float left() const { return x-w; }
    float right() const { return x+w; }
    float bottom() const { return y-h; }
    float top() const { return y+h; }
    float area() const { return 4*w*h; }

    Box overlap(const Box& other) const
    {
        float l = fmax(left(), other.left());
        float r = fmin(right(), other.right());
        float b = fmax(bottom(), other.bottom());
        float t = fmin(top(), other.top());

        Box ret;
        ret.w = (r - l)/2.;
        ret.h = (t - b)/2.;
        ret.x = l + ret.w;
        ret.y = b + ret.h;
        return ret;
    }
};

#include "objectify.h"

extern GLuint framePBO, framePBO_tex;
extern GLint viewportXY[2];

void initInterop();
void initDarknet();
void doDetection();

void loadVBO(const Box& b);
void drawTexture();

#define CAPS_LOCK 1
#define NUM_LOCK 2
unsigned int getLockState();
