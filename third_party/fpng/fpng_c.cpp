#include "fpng_c.h"
#include "fpng.h"

int fpng_encode_image_to_file(const char* pFilename, const void* pImage, uint32_t w, uint32_t h, uint32_t num_chans, uint32_t flags)
{
	return fpng::fpng_encode_image_to_file(pFilename, pImage, w, h, num_chans, flags);
}
