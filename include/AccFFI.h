#include "HsFFI.h"
#include <cuda.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef HsStablePtr AccHandle;
typedef HsStablePtr AccProgram;

typedef struct {
  int*    shape;
  void**  adata;
} InputArray;

typedef struct {
  int*    shape;
  void**  adata;
  HsStablePtr stable_ptr;
} OutputArray;

extern HsStablePtr accelerateCreate(HsInt32 device, HsPtr ctx);
extern void accelerateDestroy(AccHandle hndl);
extern void runProgram(AccHandle hndl, AccProgram p, InputArray* in, OutputArray* out);
extern void freeOutput(OutputArray* out);
extern void freeProgram(AccProgram a1);

AccHandle accelerateInit(int argc, char** argv, int device, CUcontext ctx) {
  hs_init(&argc, &argv);
  return accelerateCreate(device, ctx);
}

#ifdef __cplusplus
}
#endif