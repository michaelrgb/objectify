#include <errno.h>
#include <stdarg.h>

#include "internal.h"

#define SCREEN_WIDTH DETECT_WIDTH
#define SCREEN_HEIGHT DETECT_HEIGHT

typedef GLXContext (*GLXCREATECONTEXTATTRIBSARBPROC)(Display *, GLXFBConfig,
                                                     GLXContext, Bool,
                                                     const int *);

static void errorExit(const char* file, int line, bool add_errno, const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);

  char buf[10000];
  vsnprintf(buf, sizeof(buf), fmt, args);

  fprintf(stderr, "%s(%i): %s", file, line, buf);
  if(add_errno)
    fprintf(stderr, ", errno %d, %s", errno, strerror(errno));
  fprintf(stderr, "\n");
  exit(EXIT_FAILURE);
}
#define EXIT_ERRNO(...) errorExit(__FILE__, __LINE__, true, __VA_ARGS__)
#define EXIT_ERR(...) errorExit(__FILE__, __LINE__, false, __VA_ARGS__)

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libswscale/swscale.h>
#include <libavdevice/avdevice.h>
}
#include <iostream>

AVFormatContext* format_ctx;
AVCodecContext* codec_ctx;
int video_stream_index;
AVStream* stream;
AVFormatContext* output_ctx;
AVFrame* picture_decoded;
AVFrame* picture_rgb;
SwsContext *img_convert_ctx;
uint8_t* picture_buffer, *picture_buffer_2;

void libav_readframe()
{
  AVPacket packet;
  av_init_packet(&packet);

  if(av_read_frame(format_ctx, &packet) >= 0)
  {
    if (packet.stream_index == video_stream_index)
    {
      //packet is video
      if (stream == NULL)
      {
        //create stream in file
        stream = avformat_new_stream(output_ctx,
                format_ctx->streams[video_stream_index]->codec->codec);
        avcodec_copy_context(stream->codec,
                format_ctx->streams[video_stream_index]->codec);
        stream->sample_aspect_ratio =
                format_ctx->streams[video_stream_index]->codec->sample_aspect_ratio;
      }
      packet.stream_index = stream->id;
      int check = 0;
#if 1
      int result = avcodec_decode_video2(codec_ctx, picture_decoded, &check, &packet);

      sws_scale(img_convert_ctx,
                picture_decoded->data, picture_decoded->linesize, 0, codec_ctx->height,
                picture_rgb->data, picture_rgb->linesize);
#else
      // Decode to the desired size in 1 step
      int result = avcodec_decode_video2(codec_ctx, picture_rgb, &check, &packet);
#endif
      std::cout << "Bytes decoded " << result << " check " << check << std::endl;

      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, framePBO);
      glBufferData(GL_PIXEL_UNPACK_BUFFER, DETECT_BYTES, picture_rgb->data[0], GL_DYNAMIC_DRAW);
      GLenum err = glGetError(); assert(!err);

      glBindTexture(GL_TEXTURE_2D, framePBO_tex);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, DETECT_WIDTH, DETECT_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
      err = glGetError(); assert(!err);

      drawTexture();

      err = glGetError(); assert(!err);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

      doDetection();
    }
  }
  av_free_packet(&packet);
}

bool libav_init()
{
  av_register_all();
  avdevice_register_all();
  avformat_network_init();

  format_ctx = avformat_alloc_context();
  AVInputFormat *fmt = av_find_input_format("video4linux2,v4l2");

  if(avformat_open_input(&format_ctx, "/dev/video0", fmt, NULL) != 0)
    return EXIT_FAILURE;

  if(avformat_find_stream_info(format_ctx, NULL) < 0)
    return EXIT_FAILURE;

  //search video stream
  for (int i = 0; i < format_ctx->nb_streams; i++) {
    if (format_ctx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO)
      video_stream_index = i;
  }

  //open output file
  output_ctx = avformat_alloc_context();

  av_read_play(format_ctx);

  AVCodec *codec = NULL;
  codec = avcodec_find_decoder(format_ctx->video_codec_id);
  //codec = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
  if(!codec)
    exit(1);

  codec_ctx = avcodec_alloc_context3(codec);
  avcodec_get_context_defaults3(codec_ctx, codec);
  avcodec_copy_context(codec_ctx, format_ctx->streams[video_stream_index]->codec);

  if (avcodec_open2(codec_ctx, codec, NULL) < 0)
    exit(1);

  img_convert_ctx = sws_getContext(codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt,
                                   DETECT_WIDTH, DETECT_HEIGHT, AV_PIX_FMT_RGB24,
                                   SWS_BICUBIC, NULL, NULL, NULL);

  int size = avpicture_get_size(AV_PIX_FMT_YUV420P, codec_ctx->width, codec_ctx->height);
  picture_buffer = (uint8_t*) (av_malloc(size));
  picture_decoded = av_frame_alloc();
  picture_rgb = av_frame_alloc();
  int size_resized = avpicture_get_size(AV_PIX_FMT_RGB24, DETECT_WIDTH, DETECT_HEIGHT);
  picture_buffer_2 = (uint8_t*) (av_malloc(size_resized));
  avpicture_fill((AVPicture *) picture_decoded, picture_buffer, AV_PIX_FMT_YUV420P,
                 codec_ctx->width, codec_ctx->height);
  avpicture_fill((AVPicture *) picture_rgb, picture_buffer_2, AV_PIX_FMT_RGB24,
                 DETECT_WIDTH, DETECT_HEIGHT);
}

void libav_term()
{
  av_free(picture_decoded);
  av_free(picture_rgb);
  av_free(picture_buffer);
  av_free(picture_buffer_2);

  av_read_pause(format_ctx);
  avio_close(output_ctx->pb);
  avformat_free_context(output_ctx);
}

int main(int argc, char **argv)
{
  libav_init();

  Display *dpy = XOpenDisplay(0);
  int nelements;
  GLXFBConfig *fbc = glXChooseFBConfig(dpy, DefaultScreen(dpy), 0, &nelements);

  static int attributeList[] = {GLX_RGBA,       GLX_DOUBLEBUFFER,
                                GLX_RED_SIZE,   1,
                                GLX_GREEN_SIZE, 1,
                                GLX_BLUE_SIZE,  1,
                                None};
  XVisualInfo *vi = glXChooseVisual(dpy, DefaultScreen(dpy), attributeList);

  XSetWindowAttributes swa;
  swa.colormap =
      XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, AllocNone);
  swa.border_pixel = 0;
  swa.event_mask = StructureNotifyMask | ExposureMask;
  Window win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 0, 0, SCREEN_WIDTH, SCREEN_HEIGHT,
                             0, vi->depth, InputOutput, vi->visual,
                             CWBorderPixel | CWColormap | CWEventMask, &swa);
  XMapWindow(dpy, win);

  viewportXY[0] = SCREEN_WIDTH;
  viewportXY[1] = SCREEN_HEIGHT;

  Atom WM_DELETE_WINDOW = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
  XSetWMProtocols(dpy, win, &WM_DELETE_WINDOW, 1);

  GLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB =
      (GLXCREATECONTEXTATTRIBSARBPROC)glXGetProcAddress(
          (const GLubyte *)"glXCreateContextAttribsARB");

  int attribs[] = {GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
                   GLX_CONTEXT_MINOR_VERSION_ARB, 0,
#if 0
                   GLX_CONTEXT_PROFILE_MASK_ARB,  GLX_CONTEXT_CORE_PROFILE_BIT_ARB,
                   GLX_CONTEXT_FLAGS_ARB,         GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
#else
                   // Compatibility context needed for glLineWidth
                   GLX_CONTEXT_PROFILE_MASK_ARB,  GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB,
#endif
                   0};
  GLXContext ctx = glXCreateContextAttribsARB(dpy, *fbc, 0, true, attribs);
  glXMakeCurrent(dpy, win, ctx);

  initInterop();
  initDarknet();

  for(;;)
  {
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    libav_readframe();

    glXSwapBuffers(dpy, win);

    if(XPending(dpy) > 0)
    {
      XEvent e;
      XNextEvent(dpy, &e);
      if(e.type == ClientMessage && e.xclient.data.l[0] == WM_DELETE_WINDOW)
        break;
      else if(e.type == Expose)
      {
        XWindowAttributes gwa;
        XGetWindowAttributes(dpy, win, &gwa);
        viewportXY[0] = gwa.width;
        viewportXY[1] = gwa.height;
        glViewport(0, 0, viewportXY[0], viewportXY[1]);
      }
    }
  }

  libav_term();

  ctx = glXGetCurrentContext();
  glXDestroyContext(dpy, ctx);
}
