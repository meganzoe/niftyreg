#include "_reg_tools.h"
#include "_reg_maths_eigen.h"

/// @brief HERE
void GetDiscretisedValue_LCCA(nifti_image *controlPointGridImage,
                              float *discretisedValue,
                              int discretise_radius,
                              int discretise_step,
                              nifti_image *referenceImagePointer,
                              nifti_image *warpedFloatingImagePointer,
                              int *referenceMaskPointer);
