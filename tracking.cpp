#include <string.h>
#include <sstream>
#include <vector>
#include <algorithm>
#include <X11/XKBlib.h>
#include "internal.h"

using namespace std;

const float COLOR_FOCUS[] = {1.f, 1.f, 1.f};

void drawDetection(Detection det);

vector<Detection> detections;
Detection currFocus;

unsigned int getLockState()
{
    Display *dpy = XOpenDisplay(0);
    unsigned int n;
    XkbGetIndicatorState(dpy, XkbUseCoreKbd, &n);
    XCloseDisplay(dpy);
    return n;
}

void mouseMoveX11(float x, float y)
{
  float rate = 500.f;

  stringstream ss;
  ss << "xdotool mousemove_relative -- "
     << int(x * rate) << " "
     << int(y * -rate);

  // Turn off LD_PRELOAD before calling xdotool
  unsetenv("LD_PRELOAD");
  int ret = system(ss.str().c_str());
}

void onFrameBegin()
{
  detections.clear();
}
void onFrameEnd()
{
  fflush(stdout); // To read Darknet's detection printfs.

  Detection bestNewFocus;
  float bestArea = 0.;

  for(const Detection& det: detections)
  {
    // Calc area intersection
    Box overlap = currFocus.b.overlap(det.b);
    float area = overlap.area();
    if(area > bestArea)
    {
      bestArea = area;
      bestNewFocus = det;
    }
  }

  static int waitFrames = 0;
  int MAX_WAIT_FRAMES = 5;

  if(bestArea > currFocus.b.area() * 0.1f)
  {
    waitFrames = 0;
    currFocus = bestNewFocus;
    currFocus.lineWidth = 1.f;
    memcpy(currFocus.rgb, COLOR_FOCUS, sizeof(COLOR_FOCUS));

    if(getLockState() & CAPS_LOCK)
        mouseMoveX11(currFocus.b.x, currFocus.b.y);
  }

  if(++waitFrames >= MAX_WAIT_FRAMES)
  {
    // Give up waiting and switch to new focus
    if(detections.size())
    {
      waitFrames = 0;
      currFocus = detections[0];
    }
  }
  else
  {
    drawDetection(currFocus);
  }

  detections.clear();
}

extern "C"
{
void addDetection(Detection det)
{
    box& b = det.b;
    b.x = 2.f*b.x - 1.f;
    b.y = -2.f*b.y + 1.f;

    const char* classes[] = {"person", "dog"};
    if(std::find_if(std::begin(classes), std::end(classes),
        [&det](const char* s){return !strcmp(s, det.className);} ) == std::end(classes))
        return;

    drawDetection(det);

    detections.push_back(det);
}
}
