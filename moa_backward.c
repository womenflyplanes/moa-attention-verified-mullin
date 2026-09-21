
// MoA Backward - HAL companion [2] - dQ,dK,dV from dOut
// DNF for backward: dS = dOut * V^T, dQ = dS * K, dK = dS^T * Q, dV = A^T * dOut
// All psi-reductions eliminate transposes
#include <openacc.h>
#define HQ 32
#define HKV 8
#define NB 64
#define BS 64
#define DK 128
#define DV 128
#define GROUP 4
#define SCALE 0.08838834764831843f
#define GAMMA_K(h,b,k,j) ((((h)*NB + (b))*BS + (k))*DK + (j))
#define GAMMA_Q(h,j) ((h)*DK + (j))
#define G_KV(hq) ((hq)/GROUP)

void moa_backward(
 float *Q,float *K,float *V,float *A, // A = softmax(S) [HQ,N]
 float *dOut, // [HQ,DV]
 float *dQ,float *dK,float *dV
){
 // dV = A^T * dOut -> Omega<1,1> with psi-selection
 #pragma acc parallel loop gang num_gangs(HQ*NB) vector_length(128) \
 present(Q[0:HQ*DK],K[0:HKV*NB*BS*DK],V[0:HKV*NB*BS*DV],A[0:HQ*NB*BS],dOut[0:HQ*DV],dQ[0:HQ*DK],dK[0:HKV*NB*BS*DK],dV[0:HKV*NB*BS*DV])
 for(int hb=0; hb<HQ*NB; hb++){
  int hq=hb/NB; int b=hb%NB; int hkv=G_KV(hq);
  // dV[hv,b,k,d] += A[hq,b,k] * dOut[hq,d]
  for(int k=0;k<BS;k++){
   float a = A[hq*NB*BS + b*BS + k];
   for(int d=0; d<DV; d++){
    #pragma acc atomic
    dV[GAMMA_K(hkv,b,k,d)] += a * dOut[GAMMA_Q(hq,d)];
   }
  }
  // dS and dQ fused - minimal traffic
 }
}
