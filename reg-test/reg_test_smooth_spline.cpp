#include "_reg_ReadWriteImage.h"
#include "_reg_localTrans.h"

#define EPS 0.001

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <inputImage> <lambda> <expectedImage>\n", argv[0]);
        return EXIT_FAILURE;
    }

    char *inputImageName = argv[1];
    float lambda = atof(argv[2]);
    char *expectedFileName = argv[3];

    // Read the input image
    nifti_image *inputImage = reg_io_ReadImageFile(inputImageName);
    if (inputImage == NULL) {
        reg_print_msg_error("The input reference image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(inputImage);

    // Apply the smoothing in place
    reg_spline_Smooth(inputImage,lambda);

    // Read the expected image
    nifti_image *expectedFile = reg_io_ReadImageFile(expectedFileName);
    if (expectedFile == NULL) {
        reg_print_msg_error("The expected result image could not be read");
        return EXIT_FAILURE;
    }
    reg_tools_changeDatatype<float>(expectedFile);

    // Compute the difference between the computed and expected deformation fields
    nifti_image *diff_file = nifti_copy_nim_info(expectedFile);
    diff_file->data = (void *) malloc(diff_file->nvox*diff_file->nbyper);
    reg_tools_substractImageToImage(expectedFile, inputImage, diff_file);
    reg_tools_abs_image(diff_file);
    double max_difference = reg_tools_getMaxValue(diff_file, -1);


    reg_io_WriteImageFile(diff_file, "diff_file.nii.gz");
    reg_io_WriteImageFile(inputImage, "out_file.nii.gz");

    nifti_image_free(inputImage);
    nifti_image_free(expectedFile);

    if (max_difference > EPS){
        fprintf(stderr, "reg_test_convolution error too large: %g (>%g)\n",
                max_difference, EPS);
        reg_io_WriteImageFile(diff_file, "diff_file.nii.gz");
        nifti_image_free(diff_file);
        return EXIT_FAILURE;
    }
#ifndef NDEBUG
    fprintf(stdout, "reg_test_smooth_spline ok: %g (<%g)\n",
            max_difference, EPS);
#endif
    nifti_image_free(diff_file);

    return EXIT_SUCCESS;
}

