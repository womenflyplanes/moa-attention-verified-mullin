
// MoA KV-cache accumulation via concatenation #: O(dk+dv) per step
// A # B concatenates along leading axis at pre-computed gamma offset
#include <openacc.h>
#define HKV 8
#define NB 64
#define BS 64
#define DK 128
#define DV 128
#define GAMMA_K(h,b,k,j) ((((h)*NB + (b))*BS + (k))*DK + (j))

void moa_kv_append(float *K, float *k_new, int hkv, int n){
 int b=n/BS; int k=n%BS;
 #pragma acc parallel loop vector_length(128) present(K[0:HKV*NB*BS*DK],k_new[0:DK])
 for(int j=0;j<DK;j++) K[GAMMA_K(hkv,b,k,j)]=k_new[j];
 // gamma_new = n*DK, no copy of cache - Storage Theorem 2.7 proven O(dk)
}

void moa_kv_append_v(float *V, float *v_new, int hkv, int n){
 int b=n/BS; int k=n%BS;
 #pragma acc parallel loop vector_length(128) present(V[0:HKV*NB*BS*128],v_new[0:DV])
 for(int j=0;j<DV;j++) V[GAMMA_K(hkv,b,k,j)]=v_new[j];
}
