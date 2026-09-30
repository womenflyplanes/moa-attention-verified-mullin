---
title: "Validating Memory-Optimal Transformer Kernels on Real HPC Hardware"
author: "Lenore Mullin"
tags: ["MoA", "Transformers", "HPC", "ACCESS", "Performance"]
---

## TL;DR

We derived memory-optimal cost functions for every piece of a transformer block — forward, backward, fused, decode, full block — verified to machine precision in PyTorch, but never tested on real hardware. Until now.

We validated all of them on real ACCESS HPC machines: Purdue Anvil and NCSA Delta, CPU and GPU, using MoA's shape vocabulary (ρ, ψ, ι) to both derive the kernel and diagnose when prediction diverges from measurement.

Three takeaways engineers will care about.

## 1. We Found and Fixed a Real GPU Regression — 2.5x Speedup

Fusing forward + backward is proven to avoid materializing an O(n²) intermediate. On GPU it initially ran *slower* than the naive version.

Root cause: atomic-memory contention. Profiling showed 2.0000x more atomic instructions than a related kernel — to four decimal places.

Fix: Restructured the ONF (machine-specific realization) only — DNF (formal spec) stayed fixed and verified. Result: up to **2.5x real speedup**. No re-derivation needed.

**Lesson:** Hardware optimization = targeted rewrite of ONF alone.

## 2. Same Derivation, Wildly Different Real Costs

Identical formal derivation produced completely different costs on different topologies:

* 535x NUMA-locality penalty on one cluster
* <3x oversubscription cost on another

Optimal deployment is a function of the target machine's own array structure, not a fixed property of the algorithm.

## 3. A Genuine Anomaly We Only Partially Resolved

Identical denotational computations:
* Faster in C than Fortran on CPU
* Faster in Fortran than C on GPU

We narrowed it to one dominant kernel and one stall mechanism, but compiler-level cause remains open. Honest reporting.

## What We Built — Now on GitHub

All programs that implement and test these kernels on the ACCESS suite are now open:

**GitHub:** https://github.com/womenflyplanes/moa-attention-verified-mullin

Includes:
- Forward / backward / fused / decode / full block ONFs
- PyTorch autograd verification (machine precision)
- Anvil + Delta CPU/GPU measurement harnesses

## Full Paper

For derivations, cost functions, and full measurement tables:

**arXiv:2609.33916** — https://arxiv.org/abs/2609.33916
Validating Memory-Optimal Transformer Kernels on Real Hardware

This methodology is a candidate for scaling AI systems onto evolving hardware without re-deriving correctness from scratch.

---
