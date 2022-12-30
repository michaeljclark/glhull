#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int fpng_encode_image_to_file(const char* pFilename, const void* pImage, uint32_t w, uint32_t h, uint32_t num_chans, uint32_t flags);

#ifdef __cplusplus
}
#endif
