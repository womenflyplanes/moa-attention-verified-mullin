
// moa_cost_model.c - H100 Occupancy + Cost Functions
// Based on: New Mathematics for Computer Performance: Array Algebra and Cost Functions
// MoA cost = f(shape, gamma, machine array)

#include <stdio.h>

typedef struct {
  int sm_count;
  int max_threads_per_sm;
  int max_blocks_per_sm;
  int shared_mem_per_sm; // bytes
  int shared_mem_per_block_max;
  int regs_per_sm;
  float hbm_bw_GBs; // GB/s
  float smem_bw_TBs; // TB/s
  float fp32_tflops;
} H100Spec;

H100Spec h100 = {
  .sm_count = 144,
  .max_threads_per_sm = 2048,
  .max_blocks_per_sm = 32,
  .shared_mem_per_sm = 228*1024, // 228KB per SM H100
  .shared_mem_per_block_max = 228*1024,
  .regs_per_sm = 65536,
  .hbm_bw_GBs = 3000, // 3 TB/s HBM3
  .smem_bw_TBs = 33, // ~33 TB/s aggregate shared mem
  .fp32_tflops = 66.9
};

typedef struct {
  int HQ, HKV, NB, BS, TB, TS, DK, DV;
  int smem_bytes_per_block; // K smem + V smem + q_s + s_bs
  int threads_per_block;
  int regs_per_thread;
  int blocks;
} MoAKernel;

MoAKernel kernel_flash = {
  .HQ=32, .HKV=8, .NB=64, .BS=64, .TB=8, .TS=8, .DK=128, .DV=128,
  .smem_bytes_per_block = (8*8*128*4)*2 + 128*4 + 64*4, // 32KB+32KB+512+256 = 66048
  .threads_per_block = 128,
  .regs_per_thread = 64,
  .blocks = 32*64 // HQ*NB = 2048
};

void occupancy_model(MoAKernel k, H100Spec m) {
  int max_blocks_by_smem = m.shared_mem_per_sm / k.smem_bytes_per_block;
  int max_blocks_by_threads = m.max_threads_per_sm / k.threads_per_block;
  int max_blocks_by_regs = m.regs_per_sm / k.regs_per_thread;
  
  int blocks_per_sm = max_blocks_by_smem;
  if(max_blocks_by_threads < blocks_per_sm) blocks_per_sm = max_blocks_by_threads;
  if(max_blocks_by_regs < blocks_per_sm) blocks_per_sm = max_blocks_by_regs;
  if(m.max_blocks_per_sm < blocks_per_sm) blocks_per_sm = m.max_blocks_per_sm;
  
  float occupancy = (float)(blocks_per_sm * k.threads_per_block) / m.max_threads_per_sm * 100;
  
  printf("=== MoA Flash-Tiled Kernel Occupancy H100 ===\n");
  printf("Kernel: HQ=%d HKV=%d NB=%d BS=%d TB=%d TS=%d DK=%d\n",k.HQ,k.HKV,k.NB,k.BS,k.TB,k.TS,k.DK);
  printf("SMEM/block: %d bytes (%.1f KB) - K_smem %d + V_smem %d\n",k.smem_bytes_per_block, k.smem_bytes_per_block/1024.0, k.TB*k.TS*k.DK*4, k.TB*k.TS*k.DV*4);
  printf("Threads/block: %d, Regs/thread: %d\n",k.threads_per_block,k.regs_per_thread);
  printf("Limits per SM:\n");
  printf("  by SMEM: %d blocks\n",max_blocks_by_smem);
  printf("  by threads: %d blocks\n",max_blocks_by_threads);
  printf("  by regs: %d blocks\n",max_blocks_by_regs);
  printf("  -> Achievable: %d blocks/SM\n",blocks_per_sm);
  printf("  Occupancy: %.1f%% (%d threads active / %d max)\n",occupancy, blocks_per_sm*k.threads_per_block, m.max_threads_per_sm);
  printf("  Total blocks: %d, SMs: %d, Waves: %.2f\n",k.blocks, m.sm_count, (float)k.blocks/(m.sm_count*blocks_per_sm));
  
  // Cost functions: T = T_DRAM + T_SMEM + T_COMP
  // MoA gamma gives exact byte counts
  long long n = 4096;
  long long dram_bytes = (k.DK + n*k.DK + n*k.DV + k.DV)*4; // per h_q group
  long long dram_bytes_total = dram_bytes * (k.HQ/k.HKV); // GQA reduction
  float t_dram_ms = (dram_bytes_total / (1024.0*1024*1024)) / m.hbm_bw_GBs * 1000;
  
  long long smem_bytes = (long long)k.blocks * k.smem_bytes_per_block;
  float t_smem_ms = (smem_bytes / (1024.0*1024*1024*1024.0)) / m.smem_bw_TBs * 1000;
  
  long long flops = (long long)k.HQ * n * k.DK * 2 + (long long)k.HQ * n * k.DV * 2; // QK + AV
  float t_comp_ms = (flops / 1e12) / m.fp32_tflops * 1000;
  
  printf("\n=== MoA Cost Model (Storage Theorem 2.7) ===\n");
  printf("DRAM: %lld bytes total (minimal), t_DRAM=%.3f ms @ %.0f GB/s\n",dram_bytes_total, t_dram_ms, m.hbm_bw_GBs);
  printf("SMEM: %lld bytes moved, t_SMEM=%.3f ms @ %.0f TB/s\n",smem_bytes, t_smem_ms, m.smem_bw_TBs);
  printf("COMP: %lld FLOPs, t_COMP=%.3f ms @ %.1f TFLOPS\n",flops, t_comp_ms, m.fp32_tflops);
  printf("T_total ~ max(t_DRAM, t_COMP) overlapped = %.3f ms (SMEM hidden)\n", fmaxf(t_dram_ms,t_comp_ms));
  printf("\nMoA insight: More shape-aware ONF -> larger SMEM tiles -> less DRAM, higher occupancy tradeoff\n");
  printf("At 64KB/block: %d blocks/SM, at 128KB/block: %d blocks/SM, at 228KB: %d block/SM\n",
    m.shared_mem_per_sm/66048, m.shared_mem_per_sm/131072, m.shared_mem_per_sm/228000);
}

int main(){ occupancy_model(kernel_flash, h100); return 0; }
