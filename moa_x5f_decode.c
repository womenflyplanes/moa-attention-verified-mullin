
// MoA Decode - 2-pass ONF - H100 - HQ=32 HKV=8 NB=64 BS=64 DK=128 DV=128
#include <math.h>
#include <openacc.h>
#include <float.h>

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

float d_max[HQ];
float d_sum[HQ];

void moa_pass1_max(float *Q, float *K){
 #pragma acc parallel loop gang num_gangs(HQ*NB) vector_length(128) present(Q[0:HQ*DK],K[0:HKV*NB*BS*DK],d_max[0:HQ])
 for(int hb=0; hb<HQ*NB; hb++){
  int hq=hb/NB; int b=hb%NB; int hkv=G_KV(hq);
  float q_s[DK];
  for(int j=0;j<DK;j++) q_s[j]=Q[GAMMA_Q(hq,j)];
  float bmax=-FLT_MAX;
  for(int k=0;k<BS;k++){
   float dot=0;
   #pragma acc loop reduction(+:dot)
   for(int j=0;j<DK;j++) dot+=q_s[j]*K[GAMMA_K(hkv,b,k,j)];
   dot*=SCALE;
   if(dot>bmax) bmax=dot;
  }
  #pragma acc atomic
  d_max[hq]=fmaxf(d_max[hq],bmax);
 }
}

void moa_pass2(float *Q,float *K,float *V,float *out){
 #pragma acc parallel loop gang num_gangs(HQ*NB) vector_length(128) present(Q[0:HQ*DK],K[0:HKV*NB*BS*DK],V[0:HKV*NB*BS*DV],out[0:HQ*DV],d_max[0:HQ],d_sum[0:HQ])
 for(int hb=0; hb<HQ*NB; hb++){
  int hq=hb/NB; int b=hb%NB; int hkv=G_KV(hq);
  float q_s[DK];
  for(int j=0;j<DK;j++) q_s[j]=Q[GAMMA_Q(hq,j)];
  float s[BS], e[BS]; float bsum=0;
  for(int k=0;k<BS;k++){
   float dot=0;
   #pragma acc loop reduction(+:dot)
   for(int j=0;j<DK;j++) dot+=q_s[j]*K[GAMMA_K(hkv,b,k,j)];
   s[k]=dot*SCALE;
   e[k]=expf(s[k]-d_max[hq]);
   bsum+=e[k];
  }
  #pragma acc atomic
  d_sum[hq]+=bsum;
  #pragma acc wait
  float Z=d_sum[hq];
  for(int d=0; d<DV; d++){
   float acc=0;
   #pragma acc loop reduction(+:acc)
   for(int k=0;k<BS;k++){
    float a=e[k]/Z;
    acc+=a*V[GAMMA_K(hkv,b,k,d)];
   }
   #pragma acc atomic
   out[GAMMA_Q(hq,d)]+=acc;
  }
 }
}

void moa_decode(float *Q,float *K,float *V,float *out,int n){
 for(int h=0;h<HQ;h++){d_max[h]=-FLT_MAX; d_sum[h]=0;}
 for(int i=0;i<HQ*DV;i++) out[i]=0;
 #pragma acc data copyin(Q[0:HQ*DK],K[0:HKV*NB*BS*DK],V[0:HKV*NB*BS*DV]) copy(out[0:HQ*DV],d_max[0:HQ],d_sum[0:HQ])
 {
  moa_pass1_max(Q,K);
  moa_pass2(Q,K,V,out);
 }
}
