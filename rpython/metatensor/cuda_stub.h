/* Stand-in for <cuda.h> when the CUDA toolkit is absent (e.g. CI): every
   driver call reports CUDA_ERROR_NO_DEVICE, so rt_cuda_available() is 0 and
   everything runs on the CPU evaluator. */
#ifndef RTENSOR_CUDA_STUB_H
#define RTENSOR_CUDA_STUB_H
typedef int CUresult, CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUevent_st *CUevent;
#define CUDA_SUCCESS 0
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_NO_DEVICE 100
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES 8
#define CU_EVENT_DISABLE_TIMING 2
#define RT_CU_NO_DEVICE(...) CUDA_ERROR_NO_DEVICE
#define cuInit RT_CU_NO_DEVICE
#define cuDeviceGet RT_CU_NO_DEVICE
#define cuDevicePrimaryCtxRetain RT_CU_NO_DEVICE
#define cuCtxSetCurrent RT_CU_NO_DEVICE
#define cuCtxSynchronize RT_CU_NO_DEVICE
#define cuGetErrorName RT_CU_NO_DEVICE
#define cuGetErrorString RT_CU_NO_DEVICE
#define cuMemAlloc RT_CU_NO_DEVICE
#define cuMemFree RT_CU_NO_DEVICE
#define cuMemAllocHost RT_CU_NO_DEVICE
#define cuMemFreeHost RT_CU_NO_DEVICE
#define cuMemGetInfo RT_CU_NO_DEVICE
#define cuMemsetD8 RT_CU_NO_DEVICE
#define cuMemsetD32 RT_CU_NO_DEVICE
#define cuMemcpyHtoD RT_CU_NO_DEVICE
#define cuMemcpyDtoH RT_CU_NO_DEVICE
#define cuMemcpyDtoD_v2 RT_CU_NO_DEVICE
#define cuMemcpyHtoDAsync RT_CU_NO_DEVICE
#define cuModuleLoadData RT_CU_NO_DEVICE
#define cuModuleGetFunction RT_CU_NO_DEVICE
#define cuFuncSetAttribute RT_CU_NO_DEVICE
#define cuLaunchKernel RT_CU_NO_DEVICE
#define cuEventCreate RT_CU_NO_DEVICE
#define cuEventRecord RT_CU_NO_DEVICE
#define cuEventSynchronize RT_CU_NO_DEVICE
#endif
