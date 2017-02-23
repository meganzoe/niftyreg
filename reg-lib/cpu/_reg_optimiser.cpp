/** @file _reg_optimiser.cpp
 * @author Marc Modat
 * @date 20/07/2012
 */

#ifndef _REG_OPTIMISER_CPP
#define _REG_OPTIMISER_CPP

#include "_reg_optimiser.h"

/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_optimiser<T>::reg_optimiser()
{
   this->dofNumber=0;
   this->dofNumber_b=0;
   this->ndim=3;
   this->optimiseX=true;
   this->optimiseY=true;
   this->optimiseZ=true;
   this->currentDOF=NULL;
   this->currentDOF_b=NULL;
   this->bestDOF=NULL;
   this->bestDOF_b=NULL;
   this->backward=false;
   this->gradient=NULL;
   this->currentIterationNumber=0;
   this->currentObjFunctionValue=0.0;
   this->maxIterationNumber=0.0;
   this->bestObjFunctionValue=0.0;
   this->objFunc=NULL;
   this->gradient_b=NULL;

#ifndef NDEBUG
   reg_print_msg_debug("reg_optimiser<T>::reg_optimiser() called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_optimiser<T>::~reg_optimiser()
{
   if(this->bestDOF!=NULL)
      free(this->bestDOF);
   this->bestDOF=NULL;
   if(this->bestDOF_b!=NULL)
      free(this->bestDOF_b);
   this->bestDOF_b=NULL;
#ifndef NDEBUG
   reg_print_msg_debug("reg_optimiser<T>::~reg_optimiser() called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_optimiser<T>::Initialise(size_t nvox,
                                  int dim,
                                  bool optX,
                                  bool optY,
                                  bool optZ,
                                  size_t maxit,
                                  size_t start,
                                  InterfaceOptimiser *obj,
                                  T *cppData,
                                  T *gradData,
                                  size_t nvox_b,
                                  T *cppData_b,
                                  T *gradData_b
                                 )
{
   this->dofNumber=nvox;
   this->ndim=dim;
   this->optimiseX=optX;
   this->optimiseY=optY;
   this->optimiseZ=optZ;
   this->maxIterationNumber=maxit;
   this->currentIterationNumber=start;
   this->currentDOF=cppData;
   if(this->bestDOF!=NULL) free(this->bestDOF);
   this->bestDOF=(T *)malloc(this->dofNumber*sizeof(T));
   memcpy(this->bestDOF,this->currentDOF,this->dofNumber*sizeof(T));
   if( gradData!=NULL)
      this->gradient=gradData;

   if(nvox_b>0)
      this->dofNumber_b=nvox_b;
   if(cppData_b!=NULL)
   {
      this->currentDOF_b=cppData_b;
      this->backward=true;
      if(this->bestDOF_b!=NULL) free(this->bestDOF_b);
      this->bestDOF_b=(T *)malloc(this->dofNumber_b*sizeof(T));
      memcpy(this->bestDOF_b,this->currentDOF_b,this->dofNumber_b*sizeof(T));
   }
   if(gradData_b!=NULL)
      this->gradient_b=gradData_b;

   this->objFunc=obj;
   this->bestObjFunctionValue = this->currentObjFunctionValue =
                                   this->objFunc->GetObjectiveFunctionValue();

#ifndef NDEBUG
   reg_print_msg_debug("reg_optimiser<T>::Initialise called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_optimiser<T>::RestoreBestDOF()
{
   // restore forward transformation
   memcpy(this->currentDOF,this->bestDOF,this->dofNumber*sizeof(T));
   // restore backward transformation if required
   if(this->currentDOF_b!=NULL && this->bestDOF_b!=NULL && this->dofNumber_b>0)
      memcpy(this->currentDOF_b,this->bestDOF_b,this->dofNumber_b*sizeof(T));
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_optimiser<T>::StoreCurrentDOF()
{
   // save forward transformation
   memcpy(this->bestDOF,this->currentDOF,this->dofNumber*sizeof(T));
   // save backward transformation if required
   if(this->currentDOF_b!=NULL && this->bestDOF_b!=NULL && this->dofNumber_b>0)
      memcpy(this->bestDOF_b,this->currentDOF_b,this->dofNumber_b*sizeof(T));
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_optimiser<T>::Perturbation(float length)
{
   // initialise the randomiser
   srand(time(NULL));
   // Reset the number of iteration
   this->currentIterationNumber=0;
   // Create some perturbation for degree of freedom
   for(size_t i=0; i<this->dofNumber; ++i)
   {
      this->currentDOF[i]=this->bestDOF[i] + length * (float)(rand() - RAND_MAX/2) / ((float)RAND_MAX/2.0f);
   }
   if(this->backward==true)
   {
      for(size_t i=0; i<this->dofNumber_b; ++i)
      {
         this->currentDOF_b[i]=this->bestDOF_b[i] + length * (float)(rand() % 2001 - 1000) / 1000.f;
      }
   }
   this->StoreCurrentDOF();
   this->currentObjFunctionValue=this->bestObjFunctionValue=
                                    this->objFunc->GetObjectiveFunctionValue();
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_optimiser<T>::Optimise(T maxLength,
                                T smallLength,
                                T &startLength)
{
   size_t lineIteration=0;
   float addedLength=0;
   float currentLength=startLength;

   // Start performing the line search
   while(currentLength>smallLength &&
         lineIteration<12 &&
         this->currentIterationNumber<this->maxIterationNumber)
   {

      // Compute the gradient normalisation value
      float normValue = -currentLength;

      this->objFunc->UpdateParameters(normValue);

      // Compute the new value
      this->currentObjFunctionValue=this->objFunc->GetObjectiveFunctionValue();

      // Check if the update lead to an improvement of the objective function
      if(this->currentObjFunctionValue > this->bestObjFunctionValue)
      {
#ifndef NDEBUG
         char text[255];
         sprintf(text, "[%i] objective function: %g | Increment %g | ACCEPTED",
                 (int)this->currentIterationNumber,
                 this->currentObjFunctionValue,
                 currentLength);
         reg_print_msg_debug(text);
#endif
         // Improvement - Save the new objective function value
         this->objFunc->UpdateBestObjFunctionValue();
         this->bestObjFunctionValue=this->currentObjFunctionValue;
         // Update the total added length
         addedLength += currentLength;
         // Increase the step size
         currentLength *= 1.1f;
         currentLength = (currentLength<maxLength)?currentLength:maxLength;
         // Save the current deformation parametrisation
         this->StoreCurrentDOF();
      }
      else
      {
#ifndef NDEBUG
         char text[255];
         sprintf(text, "[%i] objective function: %g | Increment %g | REJECTED",
                 (int)this->currentIterationNumber,
                 this->currentObjFunctionValue,
                 currentLength);
         reg_print_msg_debug(text);
#endif
         // No improvement - Decrease the step size
         currentLength*=0.5;
      }
      this->IncrementCurrentIterationNumber();
      ++lineIteration;
   }
   // update the current size for the next iteration
   startLength=addedLength;
   // Restore the last best deformation parametrisation
   this->RestoreBestDOF();
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_optimiser<T>::reg_test_optimiser()
{
   this->objFunc->UpdateParameters(1.f);
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_conjugateGradient<T>::reg_conjugateGradient()
   :reg_optimiser<T>::reg_optimiser()
{
   this->array1=NULL;
   this->array2=NULL;
   this->array1_b=NULL;
   this->array2_b=NULL;

#ifndef NDEBUG
   reg_print_msg_debug("reg_conjugateGradient<T>::reg_conjugateGradient() called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_conjugateGradient<T>::~reg_conjugateGradient()
{
   if(this->array1!=NULL)
      free(this->array1);
   this->array1=NULL;

   if(this->array2!=NULL)
      free(this->array2);
   this->array2=NULL;

   if(this->array1_b!=NULL)
      free(this->array1_b);
   this->array1_b=NULL;

   if(this->array2_b!=NULL)
      free(this->array2_b);
   this->array2_b=NULL;

#ifndef NDEBUG
   reg_print_msg_debug("reg_conjugateGradient<T>::~reg_conjugateGradient() called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_conjugateGradient<T>::Initialise(size_t nvox,
                                          int dim,
                                          bool optX,
                                          bool optY,
                                          bool optZ,
                                          size_t maxit,
                                          size_t start,
                                          InterfaceOptimiser *o,
                                          T *cppData,
                                          T *gradData,
                                          size_t nvox_b,
                                          T *cppData_b,
                                          T *gradData_b
                                          )
{
   reg_optimiser<T>::Initialise(nvox,
                                dim,
                                optX,
                                optY,
                                optZ,
                                maxit,
                                start,
                                o,
                                cppData,
                                gradData,
                                nvox_b,
                                cppData_b,
                                gradData_b
                               );
   this->firstcall=true;
   if(this->array1!=NULL) free(this->array1);
   if(this->array2!=NULL) free(this->array2);
   this->array1=(T *)malloc(this->dofNumber*sizeof(T));
   this->array2=(T *)malloc(this->dofNumber*sizeof(T));

   if(cppData_b!=NULL && gradData_b!=NULL && nvox_b>0)
   {
      if(this->array1_b!=NULL) free(this->array1_b);
      if(this->array2_b!=NULL) free(this->array2_b);
      this->array1_b=(T *)malloc(this->dofNumber_b*sizeof(T));
      this->array2_b=(T *)malloc(this->dofNumber_b*sizeof(T));
   }

#ifndef NDEBUG
   reg_print_msg_debug("reg_conjugateGradient<T>::Initialise called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_conjugateGradient<T>::UpdateGradientValues()
{

#ifdef WIN32
   long i;
   long num = (long)this->dofNumber;
   long num_b = (long)this->dofNumber_b;
#else
   size_t i;
   size_t num = (size_t)this->dofNumber;
   size_t num_b = (size_t)this->dofNumber_b;
#endif

   T *gradientPtr = this->gradient;
   T *array1Ptr = this->array1;
   T *array2Ptr = this->array2;

   T *gradientPtr_b = this->gradient_b;
   T *array1Ptr_b = this->array1_b;
   T *array2Ptr_b = this->array2_b;

   if(this->firstcall==true)
   {
#ifndef NDEBUG
      reg_print_msg_debug("Conjugate gradient initialisation");
#endif
      // first conjugate gradient iteration
#if defined (_OPENMP)
      #pragma omp parallel for default(none) \
      shared(num,array1Ptr,array2Ptr,gradientPtr) \
      private(i)
#endif
      for(i=0; i<num; i++)
      {
         array2Ptr[i] = array1Ptr[i] = - gradientPtr[i];
      }
      if(this->dofNumber_b>0)
      {
#if defined (_OPENMP)
         #pragma omp parallel for default(none) \
         shared(num_b,array1Ptr_b,array2Ptr_b,gradientPtr_b) \
         private(i)
#endif
         for(i=0; i<num_b; i++)
         {
            array2Ptr_b[i] = array1Ptr_b[i] = - gradientPtr_b[i];
         }
      }
      this->firstcall=false;
   }
   else
   {
#ifndef NDEBUG
      reg_print_msg_debug("Conjugate gradient update");
#endif
      double dgg=0.0, gg=0.0;
#if defined (_OPENMP)
      #pragma omp parallel for default(none) \
      shared(num,array1Ptr,array2Ptr,gradientPtr) \
      private(i) \
reduction(+:gg) \
reduction(+:dgg)
#endif
      for(i=0; i<num; i++)
      {
         gg += array2Ptr[i] * array1Ptr[i];
         dgg += (gradientPtr[i] + array1Ptr[i]) * gradientPtr[i];
      }
      double gam = dgg/gg;

      if(this->dofNumber_b>0)
      {
         double dgg_b=0.0, gg_b=0.0;
#if defined (_OPENMP)
         #pragma omp parallel for default(none) \
         shared(num_b,array1Ptr_b,array2Ptr_b,gradientPtr_b) \
         private(i) \
reduction(+:gg_b) \
reduction(+:dgg_b)
#endif
         for(i=0; i<num_b; i++)
         {
            gg_b += array2Ptr_b[i] * array1Ptr_b[i];
            dgg_b += (gradientPtr_b[i] + array1Ptr_b[i]) * gradientPtr_b[i];
         }
         gam = (dgg+dgg_b)/(gg+gg_b);
      }
#if defined (_OPENMP)
      #pragma omp parallel for default(none) \
      shared(num,array1Ptr,array2Ptr,gradientPtr,gam) \
      private(i)
#endif
      for(i=0; i<num; i++)
      {
         array1Ptr[i] = - gradientPtr[i];
         array2Ptr[i] = (array1Ptr[i] + gam * array2Ptr[i]);
         gradientPtr[i] = - array2Ptr[i];
      }
      if(this->dofNumber_b>0)
      {
#if defined (_OPENMP)
         #pragma omp parallel for default(none) \
         shared(num_b,array1Ptr_b,array2Ptr_b,gradientPtr_b,gam) \
         private(i)
#endif
         for(i=0; i<num_b; i++)
         {
            array1Ptr_b[i] = - gradientPtr_b[i];
            array2Ptr_b[i] = (array1Ptr_b[i] + gam * array2Ptr_b[i]);
            gradientPtr_b[i] = - array2Ptr_b[i];
         }
      }
   }
   return;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_conjugateGradient<T>::Optimise(T maxLength,
                                        T smallLength,
                                        T &startLength)
{
   this->UpdateGradientValues();
   reg_optimiser<T>::Optimise(maxLength,
                              smallLength,
                              startLength);
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_conjugateGradient<T>::Perturbation(float length)
{
   reg_optimiser<T>::Perturbation(length);
   this->firstcall=true;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_conjugateGradient<T>::reg_test_optimiser()
{
   this->UpdateGradientValues();
   reg_optimiser<T>::reg_test_optimiser();
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_ForwardBackwardSplit<T>::reg_ForwardBackwardSplit()
   :reg_optimiser<T>::reg_optimiser()
{

  this->tau = 20.f;     // step size
  this->alpha = 1.f;    // acceleration parameter
  this->previousBestDOF=NULL;
  this->previousBestDOF_b=NULL;
  this->previousSmoothedDOF=NULL;
  this->previousSmoothedDOF_b=NULL;
#ifndef NDEBUG
   reg_print_msg_debug("reg_ForwardBackwardSplit<T>::reg_ForwardBackwardSplit() called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_ForwardBackwardSplit<T>::~reg_ForwardBackwardSplit()
{
  if(this->previousBestDOF!=NULL) free(this->previousBestDOF);
  if(this->previousSmoothedDOF!=NULL) free(this->previousSmoothedDOF);
  if(this->previousBestDOF_b!=NULL) free(this->previousBestDOF_b);
  if(this->previousSmoothedDOF_b!=NULL) free(this->previousSmoothedDOF_b);
#ifndef NDEBUG
   reg_print_msg_debug("reg_ForwardBackwardSplit<T>::~reg_ForwardBackwardSplit() called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_ForwardBackwardSplit<T>::Initialise(size_t nvox,
                                             int dim,
                                             bool optX,
                                             bool optY,
                                             bool optZ,
                                             size_t maxit,
                                             size_t start,
                                             InterfaceOptimiser *o,
                                             T *cppData,
                                             T *gradData,
                                             size_t nvox_b,
                                             T *cppData_b,
                                             T *gradData_b)
{
  reg_optimiser<T>::Initialise(nvox,
                               dim,
                               optX,
                               optY,
                               optZ,
                               maxit,
                               start,
                               o,
                               cppData,
                               gradData,
                               nvox_b,
                               cppData_b,
                               gradData_b
                              );
  if(this->previousBestDOF!=NULL) free(this->previousBestDOF);
  this->previousBestDOF=(T *)malloc(this->dofNumber*sizeof(T));
  memcpy(this->previousBestDOF,this->currentDOF,this->dofNumber*sizeof(T));

  if(this->previousSmoothedDOF!=NULL) free(this->previousSmoothedDOF);
  this->previousSmoothedDOF=(T *)malloc(this->dofNumber*sizeof(T));
  memcpy(this->previousSmoothedDOF,this->currentDOF,this->dofNumber*sizeof(T));

  if(cppData_b!=NULL && gradData_b!=NULL && nvox_b>0)
  {
     if(this->previousBestDOF_b!=NULL) free(this->previousBestDOF_b);
     this->previousBestDOF_b=(T *)malloc(this->dofNumber_b*sizeof(T));
     memcpy(this->previousBestDOF_b,this->currentDOF_b,this->dofNumber_b*sizeof(T));

     if(this->previousSmoothedDOF_b!=NULL) free(this->previousSmoothedDOF_b);
     this->previousSmoothedDOF_b=(T *)malloc(this->dofNumber_b*sizeof(T));
     memcpy(this->previousSmoothedDOF_b,this->currentDOF_b,this->dofNumber_b*sizeof(T));
  }
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_ForwardBackwardSplit<T>::Optimise(T maxLength,
                                           T smallLength,
                                           T &startLength)
{
  // Allocate some required variables and pointers
  size_t i;
  T *currentSmoothedDOF = (T *)malloc(this->dofNumber*sizeof(T));
  T *currentSmoothedDOF_b = NULL;
  if(this->dofNumber_b){
    currentSmoothedDOF_b = (T *)malloc(this->dofNumber*sizeof(T));
  }

  // Start performing the line search
  if(this->currentIterationNumber>this->maxIterationNumber-1){
    startLength = 0;
    return;
  }

  this->previousCost.push_back(this->bestObjFunctionValue);
  if(this->previousCost.size()>50)
    this->previousCost.erase(this->previousCost.begin());

  // Non-monotone backtracking step
  double sum = 0.f;

  float alpha_kp1;
  float alpha_ratio;

  // acceleration parameter
  alpha_kp1 = 0.5f + 0.5f * sqrtf(1.f + 4.f * reg_pow2(this->alpha));
  alpha_ratio = (this->alpha - 1.f) / alpha_kp1;
  
  while(1){
    // Forward step
    this->objFunc->UpdateParameters(-this->tau); // this->currentDOF = this->bestDOF - tau * grad
    
    // Proximal step
    this->objFunc->CubicSplineSmoothTransformation(this->tau); // this->currentDOF <- B3(this->currentDOF)
    
    memcpy(currentSmoothedDOF, this->currentDOF, this->dofNumber*sizeof(T));
    if(this->dofNumber_b){
      memcpy(currentSmoothedDOF_b, this->currentDOF_b, this->dofNumber*sizeof(T));
    }
    
    // prediction step
    for(i=0; i<this->dofNumber; ++i){
      this->currentDOF[i] += alpha_ratio * (this->currentDOF[i] - this->bestDOF[i]);
    }
    for(i=0; i<this->dofNumber_b; ++i){
      this->currentDOF_b[i] += alpha_ratio * (this->currentDOF_b[i] - this->bestDOF_b[i]);
    }

    // Compute the objective function
    this->currentObjFunctionValue = this->objFunc->GetObjectiveFunctionValue();
    this->IncrementCurrentIterationNumber();
        
    sum = -*min_element(this->previousCost.begin(), this->previousCost.end());
    for(i=0; i<this->dofNumber;++i){
      sum += (this->currentDOF[i] - this->bestDOF[i]) * (-this->gradient[i]);
      sum += reg_pow2(this->currentDOF[i] - this->bestDOF[i]) * 0.5f / this->tau;
    }
    for(i=0; i<this->dofNumber_b;++i){
      sum += (this->currentDOF_b[i] - this->bestDOF_b[i]) * (-this->gradient_b[i]);
      sum += reg_pow2(this->currentDOF_b[i] - this->bestDOF_b[i]) * 0.5f / this->tau;
    }

    std::cout << "f(u_kp1) = " << this->currentObjFunctionValue
      << " >=? RHS = " << sum << ", \tstep size = " << this->tau << std::endl;
    if(this->currentObjFunctionValue >= sum){
      // std::cout << "done" << std::endl;
      break;
    }
    this->tau /= 1.3f;
  }

  // Check for oscillation and restart alpha if needed
  sum = 0;
  for(size_t i=0; i<this->dofNumber; ++i){
    sum += (this->currentDOF[i] - currentSmoothedDOF[i]) *
        (currentSmoothedDOF[i] - this->previousSmoothedDOF[i]) ;
  }
  for(size_t i=0; i<this->dofNumber_b; ++i){
    sum += (this->currentDOF_b[i] - currentSmoothedDOF_b[i]) *
        (currentSmoothedDOF_b[i] - this->previousSmoothedDOF_b[i]) ;
  }
  if(sum > std::numeric_limits<T>::epsilon())
    this->alpha=1.f;
  else this->alpha = alpha_kp1;

  // Prepare for next iterate
  memcpy(this->previousBestDOF, this->bestDOF, this->dofNumber*sizeof(T));
  memcpy(this->previousSmoothedDOF, currentSmoothedDOF, this->dofNumber*sizeof(T));


  if(this->dofNumber_b){
    memcpy(this->previousBestDOF_b, this->bestDOF_b, this->dofNumber_b*sizeof(T));
  }

  // Implements memcpy(this->bestDOF, this->currentDOF, this->dofNumber*sizeof(T));
  this->StoreCurrentDOF();

  // Check for convergence and store the current DOF
  double maxIncrement=0;
  for(size_t i=0; i<this->dofNumber; ++i){
    T currentVal = abs(this->bestDOF[i]-this->previousBestDOF[i]);
    maxIncrement = currentVal>maxIncrement?currentVal:maxIncrement;
  }
  for(size_t i=0; i<this->dofNumber_b; ++i){
    T currentVal = abs(this->bestDOF_b[i]-this->previousBestDOF_b[i]);
    maxIncrement = currentVal>maxIncrement?currentVal:maxIncrement;
  }

  // We might want to use that for testing
  this->currentObjFunctionValue=this->objFunc->GetObjectiveFunctionValue();
  this->objFunc->UpdateBestObjFunctionValue();
  this->bestObjFunctionValue=this->currentObjFunctionValue;

  // if(maxIncrement<smallLength)
  //   startLength = 0;
  // else 
  // startLength = maxIncrement;

#ifndef NDEBUG
  reg_print_msg_debug("reg_ForwardBackwardSplit<T>::~reg_ForwardBackwardSplit() called");
#endif
  free(currentSmoothedDOF);
  free(currentSmoothedDOF_b);
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
const T* reg_ForwardBackwardSplit<T>::GetResidual(T *afterForward, T *afterProximal) const
{

    T *residual;
    T *grad = this->gradient;   // grad f(afterProximal)

    for(size_t i=0; i<this->dofNumber; ++i){
      residual[i] += -grad[i] + this->tau*(afterForward[i] - afterProximal[i]);
    }
  return residual;
}
/* *************************************************************** */
/* *************************************************************** */
// Compute relative residual as in Goldstein2014, equation (42)
template <class T>
const float reg_ForwardBackwardSplit<T>::GetRelativeResidual(T *afterForward, T *afterProximal) const
{

    const T *residual = this->GetResidual(afterForward, afterProximal);
    T *grad = this->gradient;
    float tmp;
    
    float normResidual = 0.f;
    float normGradient = 0.f;
    float normFBSSteps = 0.f;

    for(size_t i=0; i<this->dofNumber; ++i){
      normResidual += residual[i] * residual[i];
      normGradient += grad[i] * grad[i];

      tmp = afterForward[i] - afterProximal[i];
      normFBSSteps += tmp * tmp;
    }

    normResidual = sqrtf(normResidual);
    normGradient = sqrtf(normGradient);
    normFBSSteps = sqrtf(normFBSSteps) / this->tau;

    printf("\tnormResidual = %.2f\n", normResidual);
    printf("\tnormGradient = %.2f\n", normGradient);
    printf("\tnormFBSSteps = %.2f\n", normFBSSteps);
    
    // max(normGradient, normFBSSteps)
    tmp = (normGradient<normFBSSteps)?normFBSSteps:normGradient;

    // Compute relative residual
    tmp = normResidual/(tmp + std::numeric_limits<T>::epsilon());

  return tmp;
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_ForwardBackwardSplitIpiano<T>::reg_ForwardBackwardSplitIpiano()
   :reg_optimiser<T>::reg_optimiser()
{
  this->tau = 2000.f;
  this->previousBestDOF=NULL;
  this->previousBestDOF_b=NULL;

  this->beta = 0.9f;  // inertial weight
  this->eta = 1.2f;    // factor to adaptively decrease step size
  this->c = 1.05f;     // factor to adaptively increase step size


#ifndef NDEBUG
   reg_print_msg_debug("reg_ForwardBackwardSplitIpiano<T>::reg_ForwardBackwardSplitIpiano() called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
reg_ForwardBackwardSplitIpiano<T>::~reg_ForwardBackwardSplitIpiano()
{
  if(this->previousBestDOF!=NULL) free(this->previousBestDOF);
  if(this->previousBestDOF_b!=NULL) free(this->previousBestDOF_b);
#ifndef NDEBUG
   reg_print_msg_debug("reg_ForwardBackwardSplitIpiano<T>::~reg_ForwardBackwardSplitIpiano() called");
#endif
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_ForwardBackwardSplitIpiano<T>::Initialise(size_t nvox,
                                             int dim,
                                             bool optX,
                                             bool optY,
                                             bool optZ,
                                             size_t maxit,
                                             size_t start,
                                             InterfaceOptimiser *o,
                                             T *cppData,
                                             T *gradData,
                                             size_t nvox_b,
                                             T *cppData_b,
                                             T *gradData_b)
{
  reg_optimiser<T>::Initialise(nvox,
                               dim,
                               optX,
                               optY,
                               optZ,
                               maxit,
                               start,
                               o,
                               cppData,
                               gradData,
                               nvox_b,
                               cppData_b,
                               gradData_b
                              );
  if(this->previousBestDOF!=NULL) free(this->previousBestDOF);
  this->previousBestDOF=(T *)malloc(this->dofNumber*sizeof(T));
  memcpy(this->previousBestDOF,this->currentDOF,this->dofNumber*sizeof(T));

  if(cppData_b!=NULL && gradData_b!=NULL && nvox_b>0)
  {
     if(this->previousBestDOF_b!=NULL) free(this->previousBestDOF_b);
     this->previousBestDOF_b=(T *)malloc(this->dofNumber_b*sizeof(T));
     memcpy(this->previousBestDOF_b,this->currentDOF_b,this->dofNumber_b*sizeof(T));
  }
}
/* *************************************************************** */
/* *************************************************************** */
template <class T>
void reg_ForwardBackwardSplitIpiano<T>::Optimise(T maxLength,
                                           T smallLength,
                                           T &startLength)
{
  // Allocate some required variables and pointers
  size_t i;
  size_t dofNumber;

  // Start performing the line search
  if(this->currentIterationNumber>this->maxIterationNumber-1){
    startLength = 0;
    return;
  }

  const float previousBestCost = this->bestObjFunctionValue;

  // Monotone backtracking step
  double sum = 0.f;
  int while_counter=1;
    
  // Set initial Lipschitz constant estimate
  double lipschitzConstant = 1.99f * (1.f - this->beta) / this->tau;
  while(1){
    // Update step size based on current Lipschitz constant estimate
    this->tau = 1.99f * (1.f - this->beta) / lipschitzConstant;
    
    // Forward step: Part 1 -- gradient descent
    this->objFunc->UpdateParameters(-this->tau); // this->currentDOF = this->bestDOF - tau * grad
    // Forward step: Part 2 -- adding inertia term
    for (i = 0; i < dofNumber; ++i)
    {
      this->currentDOF[i] += this->beta * (this->bestDOF[i] - this->previousBestDOF[i]);
    }

    // Proximal step
    this->objFunc->CubicSplineSmoothTransformation(this->tau); // this->currentDOF <- B3(this->currentDOF)
    
    // Compute the objective function of currentDOF
    this->currentObjFunctionValue = this->objFunc->GetObjectiveFunctionValue();

    // Increment iteration
    this->IncrementCurrentIterationNumber();

    // Compute comparison value for backtracking
    double sum = -previousBestCost;
    for(i=0; i<this->dofNumber; ++i){
      sum += (this->currentDOF[i] - this->bestDOF[i]) * (-this->gradient[i]);
      sum += reg_pow2(this->currentDOF[i] - this->bestDOF[i]) * 0.5f * lipschitzConstant;
    }

    printf("Iteration %2d: ", while_counter);
    printf("f(u_kp1) = %g >=? RHS = %g\n", this->currentObjFunctionValue, sum);
    printf("\tLipschitz constant = %g", lipschitzConstant);
    printf("\tstep size = %g\n", this->tau);    
    if(this->currentObjFunctionValue >= sum){
      std::cout << "done" << std::endl;
      break;
    }
    while_counter++;

    // Increase step size
    lipschitzConstant *= this->eta;
  }

  memcpy(this->previousBestDOF, this->bestDOF, this->dofNumber*sizeof(T));
  memcpy(this->bestDOF, this->currentDOF, this->dofNumber*sizeof(T));
  // if(this->dofNumber_b>0)
  //   memcpy(this->previousSmoothedDOF_b, this->currentDOF_b, this->dofNumber_b*sizeof(T));

  // Check for convergence and store the current DOF
  double maxIncrement=0;
  for(size_t i=0; i<this->dofNumber; ++i){
    T currentVal = (this->bestDOF[i]-this->previousBestDOF[i]);
    maxIncrement = currentVal>maxIncrement?currentVal:maxIncrement;
  }
  // for(size_t i=0; i<this->dofNumber_b; ++i){
  //   T currentVal = (this->bestDOF_b[i]-this->previousDOF_b[i]);
  //   maxIncrement = currentVal>maxIncrement?currentVal:maxIncrement;
  //   if(this->dofNumber_b>0)
  //     memcpy(this->previousDOF_b, this->bestDOF_b, this->dofNumber_b*sizeof(T));
  // }

  // Increase step size
  lipschitzConstant /= this->c;
  // Update step size based on current Lipschitz constant estimate
  this->tau = 1.99f * (1.f - this->beta) / lipschitzConstant;

  // We might want to use that for testing
  this->currentObjFunctionValue = this->objFunc->GetObjectiveFunctionValue();
  this->objFunc->UpdateBestObjFunctionValue();
  this->bestObjFunctionValue = this->currentObjFunctionValue;

  // if(maxIncrement<smallLength){
  // if(maxIncrement<smallLength/10.){
  //   startLength = 0;
  // }
  // else startLength = maxIncrement;

#ifndef NDEBUG
  reg_print_msg_debug("reg_ForwardBackwardSplitIpiano<T>::~reg_ForwardBackwardSplitIpiano() called");
#endif
}

/* *************************************************************** */
/* *************************************************************** */
#endif // _REG_OPTIMISER_CPP
