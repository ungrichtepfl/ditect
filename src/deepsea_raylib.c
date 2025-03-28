#include "deepsea_raylib.h"

static void convert_img_to_pixel_structure(Image *const img, const int height,
                                           const int width) {
  ImageColorGrayscale(img);
  // NOTE: The following is needed such that the data is in the same order as
  // when a PNG is loaded
  ImageFlipHorizontal(img);
  ImageRotate(img, 180);

  const int resize_height = height > 0 ? height : img->height;
  const int resize_width = width > 0 ? width : img->width;

  ImageResize(img, resize_width,
              resize_height); // NOTE: Resize at the end,
                              // otherhise there will be
                              // some weird issue with the
                              // image placing
}

static void crop_image_to_drawing_area(Image *const img, const int padding) {

  Color *const colors = LoadImageColors(*img);

  Rectangle bounds = {
      .x = img->width, .y = img->height, .width = 0, .height = 0};
  // Find the smallest rectangle that contains all non-white pixels
  for (int y = 0; y < img->height; ++y) {
    for (int x = 0; x < img->width; ++x) {
      const Color color = colors[y * img->width + x];
      if (color.r > 0 || color.g > 0 ||
          color.b > 0) { // NOTE: Assuming black background
        if (x < bounds.x)
          bounds.x = x;
        if (y < bounds.y)
          bounds.y = y;
        if (x > bounds.width)
          bounds.width = x; // NOTE: At this moment width is a coordinate
        if (y > bounds.height)
          bounds.height = y; // NOTE: At this moment height is a coordinate
      }
    }
  }

  if (bounds.width > bounds.x && bounds.height > bounds.y) {
    bounds.width = bounds.width - bounds.x +
                   1; // NOTE: Convert from coordinate to actual width
    bounds.height = bounds.height - bounds.y +
                    1; // NOTE: Convert from coordinate to actual height

    bounds.x = (bounds.x > padding) ? (bounds.x - padding) : 0;
    bounds.y = (bounds.y > padding) ? (bounds.y - padding) : 0;
    bounds.width = ((bounds.x + bounds.width + padding) < img->width)
                       ? (bounds.width + 2 * padding)
                       : (img->width - bounds.x);
    bounds.height = ((bounds.y + bounds.height + padding) < img->height)
                        ? (bounds.height + 2 * padding)
                        : (img->height - bounds.y);

    ImageCrop(img, bounds);
  }
  UnloadImageColors(colors);
}

DS_PixelsBW DS_RAYLIB_load_pixels_bw_from_image(Image *const img,
                                                const int height,
                                                const int width,
                                                const int cropping_padding) {
  DS_PixelsBW pixels = {0};
  pixels.data = DS_MALLOC(sizeof(*pixels.data) * img->height * img->width);
  DS_ASSERT(pixels.data, "Could not load pixels. Out of memory.");

  Image converted_image = ImageCopy(*img);
  if (cropping_padding >= 0)
    crop_image_to_drawing_area(&converted_image, cropping_padding);
  convert_img_to_pixel_structure(&converted_image, height, width);

  pixels.width = converted_image.width;
  pixels.height = converted_image.height;
  Color *const colors = LoadImageColors(converted_image);
  for (int i = 0; i < converted_image.height * converted_image.width; ++i)
    pixels.data[i] =
        (DS_FLOAT)colors[i]
            .r // NOTE: On an grayscale image all rgb values are the same
        / 255.f;

  UnloadImageColors(colors);
  UnloadImage(converted_image);

  return pixels;
}
