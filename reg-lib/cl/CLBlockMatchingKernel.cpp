#include "CLBlockMatchingKernel.h"
#include "config.h"

CLBlockMatchingKernel::CLBlockMatchingKernel(Content *conIn, std::string name) :
        BlockMatchingKernel(name) {
    //populate the CLContent object ptr
    con = static_cast<ClContent*>(conIn);

    //path to kernel file
    const char* niftyreg_install_dir = getenv("NIFTYREG_INSTALL_DIR");
    std::string clInstallPath;
    if(niftyreg_install_dir!=NULL){
        char opencl_kernel_path[255];
        sprintf(opencl_kernel_path, "%s/include/cl/", niftyreg_install_dir);
        clInstallPath = opencl_kernel_path;
    }
    else clInstallPath = CL_KERNELS_PATH;
    std::string clKernel("blockMatchingKernel.cl");

    //get opencl context params
    sContext = &CLContextSingletton::Instance();
    clContext = sContext->getContext();
    commandQueue = sContext->getCommandQueue();
    program = sContext->CreateProgram((clInstallPath + clKernel).c_str());

    // Create OpenCL kernel
    cl_int errNum;
    kernel = clCreateKernel(program, "blockMatchingKernel", &errNum);
    sContext->checkErrNum(errNum, "Error setting bm kernel.");

    //get cl ptrs
    clActiveBlock = con->getActiveBlockClmem();
    clReferenceImageArray = con->getReferenceImageArrayClmem();
    clWarpedImageArray = con->getWarpedImageClmem();
    clWarpedPosition = con->getWarpedPositionClmem();
    clReferencePosition = con->getReferencePositionClmem();
    clMask = con->getMaskClmem();
    clReferenceMat = con->getRefMatClmem();

    //get cpu ptrs
    reference = con->Content::getCurrentReference();
    params = con->Content::getBlockMatchingParams();

}
/* *************************************************************** */
void CLBlockMatchingKernel::calculate() {
    // Copy some required parameters over to the device
    unsigned int *definedBlock_h = (unsigned int*) malloc(sizeof(unsigned int));
    *definedBlock_h = 0;
    cl_int errNum;
    cl_mem definedBlock = clCreateBuffer(this->clContext, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, sizeof(unsigned int), definedBlock_h, &errNum);
    this->sContext->checkErrNum(errNum, "CLBlockMatchingKernel::calculate failed to allocate memory (definedBlock): ");

    const unsigned int blockRange = params->voxelCaptureRange%4?params->voxelCaptureRange/4+1:params->voxelCaptureRange/4;
    const unsigned int stepSize = params->stepSize;

    const unsigned int numBlocks = blockRange * 2 + 1;
    const unsigned int sMemSize = numBlocks*numBlocks*numBlocks*64;

    cl_uint3 imageSize = {{(cl_uint)this->reference->nx,
                                  (cl_uint)this->reference->ny,
                                  (cl_uint)this->reference->nz,
                                 (cl_uint)0 }};

    errNum = clSetKernelArg(kernel, 0, sMemSize * sizeof(cl_float), NULL);
    sContext->checkErrNum(errNum, "Error setting shared memory.");
    errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), &this->clWarpedImageArray);
    sContext->checkErrNum(errNum, "Error setting resultImageArray.");
    errNum |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &this->clReferenceImageArray);
    sContext->checkErrNum(errNum, "Error setting targetImageArray.");
    errNum |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &this->clWarpedPosition);
    sContext->checkErrNum(errNum, "Error setting resultPosition.");
    errNum |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &this->clReferencePosition);
    sContext->checkErrNum(errNum, "Error setting targetPosition.");
    errNum |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &this->clActiveBlock);
    sContext->checkErrNum(errNum, "Error setting mask.");
    errNum |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &this->clMask);
    sContext->checkErrNum(errNum, "Error setting mask.");
    errNum |= clSetKernelArg(kernel, 7, sizeof(cl_mem), &this->clReferenceMat);
    sContext->checkErrNum(errNum, "Error setting targetMatrix_xyz.");
    errNum |= clSetKernelArg(kernel, 8, sizeof(cl_mem), &definedBlock);
    sContext->checkErrNum(errNum, "Error setting definedBlock.");
    errNum |= clSetKernelArg(kernel, 9, sizeof(cl_uint3), &imageSize);
    sContext->checkErrNum(errNum, "Error setting image size.");
    errNum |= clSetKernelArg(kernel, 10, sizeof(cl_uint), &blockRange);
    sContext->checkErrNum(errNum, "Error setting blockRange.");
    errNum |= clSetKernelArg(kernel, 11, sizeof(cl_uint), &stepSize);
    sContext->checkErrNum(errNum, "Error setting step size.");

    const size_t globalWorkSize[3] = { (size_t)params->blockNumber[0] * 4,
                                                  (size_t)params->blockNumber[1] * 4,
                                                  (size_t)params->blockNumber[2] * 4 };
    const size_t localWorkSize[3] = { 4, 4, 4 };

    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    sContext->checkErrNum(errNum, "Error queuing blockmatching kernel for execution: ");
    clFinish(commandQueue);

    errNum = clEnqueueReadBuffer(this->commandQueue, definedBlock, CL_TRUE, 0, sizeof(unsigned int), definedBlock_h, 0, NULL, NULL);
    sContext->checkErrNum(errNum, "Error reading  var after for execution: ");
    params->definedActiveBlock = *definedBlock_h;

    free(definedBlock_h);
    clReleaseMemObject(definedBlock);
}
/* *************************************************************** */
CLBlockMatchingKernel::~CLBlockMatchingKernel() {
    if (kernel != 0)
        clReleaseKernel(kernel);
    if (program != 0)
        clReleaseProgram(program);
}