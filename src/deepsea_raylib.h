#ifndef DS_RAYLIB_H
#define DS_RAYLIB_H
#include "deepsea.h"
#include <raylib.h>

DS_PixelsBW DS_RAYLIB_load_pixels_bw_from_image(Image *const img,
                                                const int height,
                                                const int width,
                                                const int cropping_padding,
                                                const Color background_color);

#endif // DS_RAYLIB_H
