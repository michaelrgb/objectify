#pragma once

#ifdef __cplusplus
extern "C" {
#else
typedef box Box;
#endif

typedef struct
{
    Box b;
    const char* className;
    float prob;
    float lineWidth;
    float rgb[3];
} Detection;

void addDetection(Detection det);

#ifdef __cplusplus
} //extern "C"
#endif
