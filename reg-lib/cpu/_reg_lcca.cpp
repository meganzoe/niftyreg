#include "_reg_lcca.h"

/* *************************************************************** */
template <class DTYPE>
void GetDiscretisedValueLCCA_core3D(nifti_image *controlPointGridImage,
                                    float *discretisedValue,
                                    int discretise_radius,
                                    int discretise_step,
                                    nifti_image *refImage,
                                    nifti_image *warImage,
                                    int *mask)
{
    if(refImage->nt != warImage->nt) {
        reg_print_msg_error("The reference image and the warped image must have the same number of time points");
        reg_print_msg_error("TODO: allow different number of time points");
        reg_exit();
    }
    int cpx, cpy, cpz, t, x, y, z, a, b, c, blockIndex, blockIndex_t, voxIndex, voxIndex_t, discretisedIndex;
    int label_1D_number = (discretise_radius / discretise_step) * 2 + 1;
    int label_2D_number = label_1D_number*label_1D_number;
    int label_nD_number = label_2D_number*label_1D_number;
    //output matrix = discretisedValue (first dimension displacement label, second dim. control point)
    float gridVox[3], imageVox[3];
    float currentValue;
    // Define the transformation matrices
    mat44 *grid_vox2mm = &controlPointGridImage->qto_xyz;
    if(controlPointGridImage->sform_code>0)
        grid_vox2mm = &controlPointGridImage->sto_xyz;
    mat44 *image_mm2vox = &refImage->qto_ijk;
    if(refImage->sform_code>0)
        image_mm2vox = &refImage->sto_ijk;
    mat44 grid2img_vox = reg_mat44_mul(image_mm2vox, grid_vox2mm);

    // Compute the block size
    int blockSize[3]={
        (int)reg_ceil(controlPointGridImage->dx / refImage->dx),
        (int)reg_ceil(controlPointGridImage->dy / refImage->dy),
        (int)reg_ceil(controlPointGridImage->dz / refImage->dz),
    };
    int voxelBlockNumber = blockSize[0] * blockSize[1] * blockSize[2] * refImage->nt;
    int voxelBlockNumber1Channel = blockSize[0] * blockSize[1] * blockSize[2];
    int currentControlPoint = 0;

    // Allocate some dynamic memory
    float* refBlockValue = (float*) calloc(voxelBlockNumber, sizeof(float));
    float* warBlockValue = (float*) calloc(voxelBlockNumber, sizeof(float));
    int* refBlockValueMask = (int*) calloc(voxelBlockNumber, sizeof(int));
    int* warBlockValueMask = (int*) calloc(voxelBlockNumber, sizeof(int));
    // Calculate the mean channel by channel
    int* ref_numberOfRealVoxelInTheBlock = (int*) calloc(refImage->nt, sizeof(int));
    int* war_numberOfRealVoxelInTheBlock = (int*) calloc(warImage->nt, sizeof(int));
    float* ref_mean = (float*) calloc(refImage->nt, sizeof(float));
    float* war_mean = (float*) calloc(warImage->nt, sizeof(float));

    // Pointers to the input image
    size_t voxelNumber = (size_t)refImage->nx*
                         refImage->ny*refImage->nz;
    DTYPE *refImgPtr = static_cast<DTYPE *>(refImage->data);
    DTYPE *warImgPtr = static_cast<DTYPE *>(warImage->data);

    // Create a padded version of the warped image to avoid doundary condition check
    int warPaddedOffset [3] = {
        discretise_radius + blockSize[0],
        discretise_radius + blockSize[1],
        discretise_radius + blockSize[2],
    };
    int warPaddedDim[4] = {
        warImage->nx + 2 * warPaddedOffset[0] + blockSize[0],
        warImage->ny + 2 * warPaddedOffset[1] + blockSize[1],
        warImage->nz + 2 * warPaddedOffset[2] + blockSize[2],
        warImage->nt
    };
    size_t warPaddedVoxelNumber = (size_t)warPaddedDim[0] *
                                  warPaddedDim[1] * warPaddedDim[2];
    DTYPE *paddedWarImgPtr = (DTYPE *)calloc(warPaddedVoxelNumber*warPaddedDim[3], sizeof(DTYPE));
    voxIndex=0;
    voxIndex_t=0;
    for(t=0; t<warImage->nt; ++t){
        for(z=warPaddedOffset[2]; z<warPaddedDim[2]-warPaddedOffset[2]-blockSize[2]; ++z){
            for(y=warPaddedOffset[1]; y<warPaddedDim[1]-warPaddedOffset[1]-blockSize[1]; ++y){
                voxIndex= t * warPaddedVoxelNumber + (z*warPaddedDim[1]+y)*warPaddedDim[0]+warPaddedOffset[0];
                for(x=warPaddedOffset[0]; x<warPaddedDim[0]-warPaddedOffset[0]-blockSize[0]; ++x){
                    paddedWarImgPtr[voxIndex]=warImgPtr[voxIndex_t];
                    ++voxIndex;
                    ++voxIndex_t;
                }
            }
        }
    }
    // Loop over all control points
    for(cpz=1; cpz<controlPointGridImage->nz-1; ++cpz){//1
        gridVox[2] = cpz;
        for(cpy=1; cpy<controlPointGridImage->ny-1; ++cpy){
            gridVox[1] = cpy;
            currentControlPoint=(cpz*controlPointGridImage->ny+cpy)*controlPointGridImage->nx+1;
            for(cpx=1; cpx<controlPointGridImage->nx-1; ++cpx){
                gridVox[0] = cpx;
                // Compute the corresponding image voxel position
                reg_mat44_mul(&grid2img_vox, gridVox, imageVox);
                imageVox[0]=reg_round(imageVox[0]);
                imageVox[1]=reg_round(imageVox[1]);
                imageVox[2]=reg_round(imageVox[2]);

                // Extract the block in the reference image
                //Let's re-initialize the values
                for(int bI=0;bI<voxelBlockNumber;bI++) {
                    refBlockValue[bI]=0;
                    refBlockValueMask[bI]=0;
                }
                for(int mI=0;mI<refImage->nt;mI++) {
                    ref_mean[mI]=0;
                    ref_numberOfRealVoxelInTheBlock[mI] = 0;
                }
                blockIndex = 0;
                for(z=imageVox[2]-blockSize[2]/2; z<imageVox[2]+blockSize[2]/2; ++z){
                    for(y=imageVox[1]-blockSize[1]/2; y<imageVox[1]+blockSize[1]/2; ++y){
                        for(x=imageVox[0]-blockSize[0]/2; x<imageVox[0]+blockSize[0]/2; ++x){
                            if(x>-1 && x<refImage->nx && y>-1 && y<refImage->ny && z>-1 && z<refImage->nz) {
                                voxIndex = (z*refImage->ny+y)*refImage->nx+x;
                                if(mask[voxIndex]>-1){
                                    for(t=0; t<refImage->nt; ++t){
                                        voxIndex_t = t*voxelNumber + voxIndex;
                                        blockIndex_t = t*voxelBlockNumber1Channel + blockIndex;
                                        if(refImgPtr[voxIndex_t] == refImgPtr[voxIndex_t]) {
                                            std::cout<<"refImgPtr[voxIndex_t]="<<refImgPtr[voxIndex_t]<<std::endl;
                                            refBlockValue[blockIndex_t] = refImgPtr[voxIndex_t];
                                            refBlockValueMask[blockIndex_t] = 1;
                                            ref_mean[t] += refBlockValue[blockIndex_t];
                                            ref_numberOfRealVoxelInTheBlock[t]++;
                                        }
                                    } //t
                                } // if mask
                            } // if inside image
                            blockIndex++;
                        } // x
                    } // y
                } // z
                //DEBUG
                std::cout<<"refBlockValue[0]="<<refBlockValue[0]<<std::endl;
                //DEBUG
                //mean refBlock
                for(t=0; t<refImage->nt; ++t){
                    ref_mean[t] /= ref_numberOfRealVoxelInTheBlock[t];
                    std::cout<<"ref_mean[t]="<<ref_mean[t]<<std::endl;
                    std::cout<<"ref_numberOfRealVoxelInTheBlock[t]="<<ref_numberOfRealVoxelInTheBlock[t]<<std::endl;
                }

                // Loop over the discretised value
                DTYPE warpedValue;
                int paddedImageVox[3] = {
                    imageVox[0]+warPaddedOffset[0],
                    imageVox[1]+warPaddedOffset[1],
                    imageVox[2]+warPaddedOffset[2]
                };
                int cc;

                for(cc=0; cc<label_1D_number; ++cc){
                    discretisedIndex = cc * label_2D_number;
                    c = paddedImageVox[2]-discretise_radius + cc*discretise_step;
                    for(b=paddedImageVox[1]-discretise_radius; b<=paddedImageVox[1]+discretise_radius; b+=discretise_step){
                        for(a=paddedImageVox[0]-discretise_radius; a<=paddedImageVox[0]+discretise_radius; a+=discretise_step){
                            blockIndex = 0;
                            //Let's re-initialize the values
                            for(int bI=0;bI<voxelBlockNumber;bI++) {
                                warBlockValue[bI]=0;
                                warBlockValueMask[bI]=0;
                            }
                            for(int mI=0;mI<refImage->nt;mI++) {
                                war_mean[mI]=0;
                                war_numberOfRealVoxelInTheBlock[mI]=0;
                            }

                            for(z=c-blockSize[2]/2; z<c+blockSize[2]/2; ++z){
                                for(y=b-blockSize[1]/2; y<b+blockSize[1]/2; ++y){
                                    for(x=a-blockSize[0]/2; x<a+blockSize[0]/2; ++x){
                                        voxIndex = (z*warPaddedDim[1]+y)*warPaddedDim[0]+x;
                                        for(t=0; t<warPaddedDim[3]; ++t){
                                            voxIndex_t = t*warPaddedVoxelNumber + voxIndex;
                                            blockIndex_t = t*voxelBlockNumber1Channel + blockIndex;
                                            warpedValue = paddedWarImgPtr[voxIndex_t];
                                            if(warpedValue==warpedValue){
                                                warBlockValue[blockIndex_t] = warpedValue;
                                                warBlockValueMask[blockIndex_t] = 1;
                                                war_mean[t] += warpedValue;
                                                war_numberOfRealVoxelInTheBlock[t]++;
                                            }
                                        }
                                        blockIndex++;
                                    } // x
                                } // y
                            } // z
                            for(t=0; t<warImage->nt; ++t){
                                war_mean[t] /= war_numberOfRealVoxelInTheBlock[t];
                                std::cout<<"war_mean[t]="<<war_mean[t]<<std::endl;
                            }
                            // HERE GOES THE COMPUTATION OF LCCA BETWEEN BOTH BLOCKS (cf. Mattias publication)
                            //Allocate the 2D matrices
                            //Exx, Exy, Eyy
                            float **Exx = reg_matrix2DAllocateAndInitToZero<float>(refImage->nt, refImage->nt);
                            float **Exy = reg_matrix2DAllocateAndInitToZero<float>(refImage->nt, warImage->nt);
                            float **Eyy = reg_matrix2DAllocateAndInitToZero<float>(warImage->nt, warImage->nt);
                            float **D   = reg_matrix2DAllocate<float>(warImage->nt, refImage->nt);
                            //Loop again over the blocks
                            //Because we assume that we have the same number of time points, 1 loop is enough :)
                            int nbExx = 0;
                            int nbExy = 0;
                            int nbEyy = 0;
                            for (int m=0;m<refImage->nt;m++) {
                                //Loop over the channel
                                for(int n=0; n<warImage->nt; n++){
                                    nbExx = 0;
                                    nbExy = 0;
                                    nbEyy = 0;
                                    for(int bI=0;bI<voxelBlockNumber1Channel;bI++) {
                                        if(refBlockValueMask[bI+m*voxelBlockNumber1Channel] > 0 && refBlockValueMask[bI+n*voxelBlockNumber1Channel] > 0) {
                                            Exx[m][n]+=(refBlockValue[bI+m*voxelBlockNumber1Channel]-ref_mean[m])*(refBlockValue[bI+n*voxelBlockNumber1Channel]-ref_mean[n]);
                                            nbExx++;
                                        }
                                        if(refBlockValueMask[bI+m*voxelBlockNumber1Channel] > 0 && warBlockValueMask[bI+n*voxelBlockNumber1Channel] > 0) {
                                            Exy[m][n]+=(refBlockValue[bI+m*voxelBlockNumber1Channel]-ref_mean[m])*(warBlockValue[bI+n*voxelBlockNumber1Channel]-war_mean[n]);
                                            nbExy++;
                                        }
                                        if(warBlockValueMask[bI+m*voxelBlockNumber1Channel] > 0 && warBlockValueMask[bI+n*voxelBlockNumber1Channel] > 0) {
                                            Eyy[m][n]+=(warBlockValue[bI+m*voxelBlockNumber1Channel]-war_mean[m])*(warBlockValue[bI+n*voxelBlockNumber1Channel]-war_mean[n]);
                                            nbEyy++;
                                        }
                                    }
                                    Exx[m][n]/=nbExx;
                                    Exy[m][n]/=nbExy;
                                    Eyy[m][n]/=nbEyy;
                                }
                            }
                            //add eps to the diagononal to avoid computational problem when computing the inverse
                            for(int m=0;m<refImage->nt;m++) {
                                Exx[m][m]+=std::numeric_limits<float>::epsilon();
                                Eyy[m][m]+=std::numeric_limits<float>::epsilon();
                            }
                            //DEBUG
                            std::cout<<"Exx="<<std::endl;
                            for (int m=0;m<refImage->nt;m++) {
                                //Loop over the channel
                                for(int n=0; n<refImage->nt; n++){
                                    std::cout<<Exx[m][n]<<" ";
                                }
                                std::cout<<std::endl;
                            }
                            std::cout<<"Exy="<<std::endl;
                            for (int m=0;m<refImage->nt;m++) {
                                //Loop over the channel
                                for(int n=0; n<refImage->nt; n++){
                                    std::cout<<Exy[m][n]<<" ";
                                }
                                std::cout<<std::endl;
                            }
                            std::cout<<"Eyy="<<std::endl;
                            for (int m=0;m<refImage->nt;m++) {
                                //Loop over the channel
                                for(int n=0; n<refImage->nt; n++){
                                    std::cout<<Eyy[m][n]<<" ";
                                }
                                std::cout<<std::endl;
                            }
                            //DEBUG
                            //MATRIX DONE
                            //Let's check if everything is OK
                            float detExx = reg_matrix2DDet(Exx,refImage->nt,refImage->nt);
                            float detEyy = reg_matrix2DDet(Eyy,warImage->nt,warImage->nt);
                            //if (abs(detExx) > std::numeric_limits<float>::epsilon() && abs(detEyy) > std::numeric_limits<float>::epsilon()) {
                                //Let's calculate D now: !!! be carrefull at the names !!!!
                                reg_matrix2DTranspose(Exy, refImage->nt, warImage->nt, D);
                                std::cout<<"ExyT="<<std::endl;
                                for (int m=0;m<refImage->nt;m++) {
                                    //Loop over the channel
                                    for(int n=0; n<refImage->nt; n++){
                                        std::cout<<D[m][n]<<" ";
                                    }
                                    std::cout<<std::endl;
                                }
                                reg_matNN_inv(Exx,refImage->nt,Exx);
                                std::cout<<"Exx-1="<<std::endl;
                                for (int m=0;m<refImage->nt;m++) {
                                    //Loop over the channel
                                    for(int n=0; n<refImage->nt; n++){
                                        std::cout<<Exx[m][n]<<" ";
                                    }
                                    std::cout<<std::endl;
                                }
                                reg_matNN_inv(Eyy,warImage->nt,Eyy);
                                std::cout<<"Eyy-1="<<std::endl;
                                for (int m=0;m<refImage->nt;m++) {
                                    //Loop over the channel
                                    for(int n=0; n<refImage->nt; n++){
                                        std::cout<<Eyy[m][n]<<" ";
                                    }
                                    std::cout<<std::endl;
                                }
                                reg_matrix2DMultiply(Exx,refImage->nt, refImage->nt,Exy, refImage->nt, warImage->nt, Exy, false);
                                std::cout<<"Exx Exy="<<std::endl;
                                for (int m=0;m<refImage->nt;m++) {
                                    //Loop over the channel
                                    for(int n=0; n<refImage->nt; n++){
                                        std::cout<<Exy[m][n]<<" ";
                                    }
                                    std::cout<<std::endl;
                                }
                                reg_matrix2DMultiply(Exy,refImage->nt, refImage->nt,Eyy, refImage->nt, warImage->nt, Exy, false);
                                std::cout<<"Exy Eyy="<<std::endl;
                                for (int m=0;m<refImage->nt;m++) {
                                    //Loop over the channel
                                    for(int n=0; n<refImage->nt; n++){
                                        std::cout<<Exy[m][n]<<" ";
                                    }
                                    std::cout<<std::endl;
                                }
                                reg_matrix2DMultiply(Exy,refImage->nt, refImage->nt,D, refImage->nt, warImage->nt, D, false);
                                std::cout<<"D="<<std::endl;
                                for (int m=0;m<refImage->nt;m++) {
                                    //Loop over the channel
                                    for(int n=0; n<refImage->nt; n++){
                                        std::cout<<D[m][n]<<" ";
                                    }
                                    std::cout<<std::endl;
                                }
                                //Calculate the trace:
                                currentValue = 0;
                                for (int idTrace=0;idTrace<refImage->nt;idTrace++) {
                                    currentValue+=D[idTrace][idTrace];
                                }
                                currentValue=1-currentValue/static_cast<float>(refImage->nt);
                            //} else { //we are at the border of the image
                            //    currentValue = 1.0;//let's put something
                            //}
                            // END HERE
                            //DEBUG
                            //std::cout<<"currentValue="<<currentValue<<std::endl;
                            //DEBUG
                            discretisedValue[currentControlPoint * label_nD_number + discretisedIndex] =
                                    //currentValue / static_cast<float>(voxelBlockNumber);
                                    currentValue;//Normalisation - maybe not this time, it is already normalized.
                            ++discretisedIndex;

                            //Mr Propre
                            reg_matrix2DDeallocate(refImage->nt, Exx);
                            reg_matrix2DDeallocate(refImage->nt, Exy);
                            reg_matrix2DDeallocate(warImage->nt, Eyy);
                            reg_matrix2DDeallocate(warImage->nt,D);
                        } // a
                    } // b
                } // cc
                ++currentControlPoint;
            } // cpx
        } // cpy
    } // cpz
    //Mr Propre
    free(ref_mean);
    free(war_mean);
    free(paddedWarImgPtr);
}
/* *************************************************************** */
template <class DTYPE>
void GetDiscretisedValueLCCA_core2D(nifti_image *controlPointGridImage,
                                    float *discretisedValue,
                                    int discretise_radius,
                                    int discretise_step,
                                    nifti_image *refImage,
                                    nifti_image *warImage,
                                    int *mask)
{
    reg_print_fct_warn("GetDiscretisedValueLCCA_core2D");
    reg_print_msg_warn("No yet implemented");
    reg_exit();
}
/* *************************************************************** */
void GetDiscretisedValue_LCCA(nifti_image *controlPointGridImage,
                              float *discretisedValue,
                              int discretise_radius,
                              int discretise_step,
                              nifti_image *referenceImagePointer,
                              nifti_image *warpedFloatingImagePointer,
                              int *referenceMaskPointer)
{
    if(referenceImagePointer->nz > 1) {
        switch(referenceImagePointer->datatype)
        {
        case NIFTI_TYPE_FLOAT32:
            GetDiscretisedValueLCCA_core3D<float>
                    (controlPointGridImage,
                     discretisedValue,
                     discretise_radius,
                     discretise_step,
                     referenceImagePointer,
                     warpedFloatingImagePointer,
                     referenceMaskPointer
                     );
            break;
        case NIFTI_TYPE_FLOAT64:
            GetDiscretisedValueLCCA_core3D<double>
                    (controlPointGridImage,
                     discretisedValue,
                     discretise_radius,
                     discretise_step,
                     referenceImagePointer,
                     warpedFloatingImagePointer,
                     referenceMaskPointer
                     );
            break;
        default:
            reg_print_fct_error("reg_ssd::GetDiscretisedValue");
            reg_print_msg_error("Unsupported datatype");
            reg_exit();
        }
    } else {
        switch(referenceImagePointer->datatype)
        {
        case NIFTI_TYPE_FLOAT32:
            GetDiscretisedValueLCCA_core2D<float>
                    (controlPointGridImage,
                     discretisedValue,
                     discretise_radius,
                     discretise_step,
                     referenceImagePointer,
                     warpedFloatingImagePointer,
                     referenceMaskPointer
                     );
            break;
        case NIFTI_TYPE_FLOAT64:
            GetDiscretisedValueLCCA_core2D<double>
                    (controlPointGridImage,
                     discretisedValue,
                     discretise_radius,
                     discretise_step,
                     referenceImagePointer,
                     warpedFloatingImagePointer,
                     referenceMaskPointer
                     );
            break;
        default:
            reg_print_fct_error("reg_ssd::GetDiscretisedValue");
            reg_print_msg_error("Unsupported datatype");
            reg_exit();
        }
    }
}
