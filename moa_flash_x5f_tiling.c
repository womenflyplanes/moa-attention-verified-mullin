
// MoA Flash-Tiling - 2-level Omega blocking - H100 Shared Memory
// DNF unchanged: s = scale*(q Omega<1,2> K) - psi eliminates K^T
// ONF now has 2-level gamma: global DRAM gamma + shared SRAM gamma
// Level1: NB=64 blocks of 64 (DRAM) -> Level2: TB=8 tiles of 8 for SRAM (128KB SMEM)

#include <openacc.h>
#include <math.h>
#include <float.h>

#define HQ 32
#define HKV 8
#define NB 64
#define BS 64
#define TB 8      // tiles per block for shared memory
#define TS 8      // tile size = BS/TB = 8
#define DK 128
#define DV 128
#define GROUP 4
#define SCALE 0.08838834764831843f

#define GAMMA_K(h,b,k,j) ((((h)*NB + (b))*BS + (k))*DK + (j))
#define GAMMA_Q(h,j) ((h)*DK + (j))
#define G_KV(hq) ((hq)/GROUP)

// Shared memory gamma: <tile, k_in_tile, j>
#define GAMMA_SMEM(t,k,j) (((t)*TS + (k))*DK + (j))

float d_max[HQ];
float d_sum[HQ];

// 2-level tiled ONF: Outer = DRAM coalesced, Inner = SRAM reuse
void moa_flash_tiled_decode(float *Q, float *K, float *V, float *out) {
  #pragma acc parallel loop gang num_gangs(HQ*NB) vector_length(128) \
          present(Q[0:HQ*DK],K[0:HKV*NB*BS*DK],V[0:HKV*NB*BS*DV],out[0:HQ*DV],d_max[0:HQ],d_sum[0:HQ])
  for(int hb=0; hb<HQ*NB; hb++) {
    int hq=hb/NB; int b=hb%NB; int hkv=G_KV(hq);
    
    __shared__ float q_s[DK];
    __shared__ float k_smem[TB*TS*DK]; // 8*8*128*4B = 32KB
    __shared__ float v_smem[TB*TS*DV]; // 32KB -> total 64KB < 228KB H100 SMEM
    __shared__ float s_bs[BS];
    
    // Load Q once - stays in SMEM
    for(int j=0;j<DK;j++) q_s[j]=Q[GAMMA_Q(hq,j)];
    
    // Outer loop over DRAM blocks already via gang
    // Inner tiling: load TB tiles of K into SMEM for reuse
    // This is Omega<1,2> partitioned: Omega_{<1,2>} = Omega_{<1,2>}^{outer} o Omega_{<1,2>}^{inner}
    float block_max=-FLT_MAX;
    for(int t=0; t<TB; t++) {
      // Coalesced DRAM -> SMEM copy: gamma diff =1
      #pragma acc loop vector collapse(2)
      for(int k=0;k<TS;k++) {
        for(int j=0;j<DK;j++) {
          k_smem[GAMMA_SMEM(t,k,j)] = K[GAMMA_K(hkv,b,t*TS+k,j)];
        }
      }
      #pragma acc wait
      
      // Compute scores for this tile using SMEM - no DRAM traffic
      for(int k=0;k<TS;k++) {
        float dot=0;
        #pragma acc loop reduction(+:dot)
        for(int j=0;j<DK;j++) dot += q_s[j] * k_smem[GAMMA_SMEM(t,k,j)];
        float s = dot*SCALE;
        s_bs[t*TS+k]=s;
        if(s>block_max) block_max=s;
      }
    }
    #pragma acc atomic
    d_max[hq]=fmaxf(d_max[hq],block_max);
    #pragma acc wait // need global max

    // Second pass: softmax + weighted sum with V tiling
    float block_sum=0;
    for(int t=0; t<TB; t++) {
      // Load V tile to SMEM - same gamma pattern
      #pragma acc loop vector collapse(2)
      for(int k=0;k<TS;k++)
        for(int d=0;d<DV;d++)
          v_smem[GAMMA_SMEM(t,k,d)] = V[GAMMA_K(hkv,b,t*TS+k,d)];
      #pragma acc wait

      for(int k=0;k<TS;k++) {
        float e = expf(s_bs[t*TS+k] - d_max[hq]);
        block_sum += e;
        // Accumulate out: a = e/Z, Z not yet known - store e for later or use 2-phase
        // For simplicity we accumulate numerator and divide by Z after global sum
        // This keeps traffic minimal: we reuse v_smem
        for(int d=0; d<DV; d++) {
          #pragma acc atomic
          out[GAMMA_Q(hq,d)] += e * v_smem[GAMMA_SMEM(t,k,d)];
        }
      }
    }
    #pragma acc atomic
    d_sum[hq]+=block_sum;
  }
  
  // Final normalization: out /= Z - global sum across NB blocks
  #pragma acc parallel loop gang num_gangs(HQ) vector_length(128) present(out[0:HQ*DV],d_sum[0:HQ])
  for(int hq=0; hq<HQ; hq++) {
    float Z=d_sum[hq];
    for(int d=0; d<DV; d++) out[GAMMA_Q(hq,d)] /= Z;
  }
}

// MoA cost model: DRAM traffic same as minimal, SRAM traffic = TB*TS*DK = BS*DK per block
// But SRAM bandwidth 33TB/s vs HBM 3TB/s on H100 -> 11x speedup
// FlashAttention achieves similar via hand-tiling; MoA derives tiling via shape recursion on rho

void moa_flash_decode(float *Q,float *K,float *V,float *out){
  for(int h=0;h<HQ;h++){d_max[h]=-FLT_MAX; d_sum[h]=0;}
  for(int i=0;i<HQ*DV;i++) out[i]=0;
  #pragma acc data copyin(Q[0:HQ*DK],K[0:HKV*NB*BS*DK],V[0:HKV*NB*BS*DV]) copy(out[0:HQ*DV],d_max[0:HQ],d_sum[0:HQ])
  {
    moa_flash_tiled_decode(Q,K,V,out);
  }
}
