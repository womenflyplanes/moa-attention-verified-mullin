# MoA Attention Verified - Memory-Optimal Transformer Kernels

Formal verification of attention achieving theoretical memory lower bounds, with real-hardware validation across two HPC clusters.

By Lenore Mullin & Peilun Ju

## 5-Part Paper Series

1. Paper I - Foundation: MoA formalism & lower bounds
2. Paper II - Fused Kernels: Fused attention implementation  
3. Paper III - CPU Verification: x86/ARM validation
4. Paper IV - GPU Verification: GPU kernels & atomics
5. Paper V - Real Hardware Validation: Submitted Sep 2026 - arXiv submit/8132497 [ON HOLD for moderation], 
      - Title: Validating Memory-Optimal Transformer Kernels on Real Hardware: From Formal Derivation to Measured Performance Across Two HPC Clusters

How to cite Paper V while on hold:
Mullin, L. & Ju, P. (2026). Validating Memory-Optimal Transformer Kernels on Real Hardware. Submitted to arXiv (submit/8132497, on hold). HAL:05734881. Paper V of MoA series.

## Software - Paper V

This repository contains the complete software package for Paper V (2.4MB, 2026-09-26):

paper_V_software/
  attention/     - core kernels
  experiments/   - benchmarks CPU+GPU
  verification/  - verification code
  benchmarking/  - run_cpu_sweep.sh
  paper/         - PDF + tex + plots

## Cross-links
- HAL: https://hal.science/hal-05734881
- arXiv V: submit/8132497 (pending)
- GitHub: https://github.com/womenflyplanes/moa-attention-verified-mullin

MIT License. All Paper V experiments reproducible from this repo.
