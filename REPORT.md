# Parallel PageRank — Report

## Title
Parallel PageRank with MPI (Python)

## Abstract
(approx. 120–200 words) — Summarize the goal: implement and benchmark PageRank using MPI in Python. Mention the dataset type (synthetic or real), evaluation metrics, and key results (scaling behavior).

## Introduction
- Motivation: PageRank as core algorithm for ranking web pages.
- Objectives: implement distributed PageRank, measure scaling, analyze performance.

## Approach / Methodology
- Algorithm: power iteration with damping factor d.
- Parallelization strategy: distribute source nodes among MPI ranks; each rank produces contributions to global vector; use `MPI.Allreduce` to combine.
- Data structures: adjacency lists (outgoing lists), handling dangling nodes.

## Implementation details
- Language & libs: Python, mpi4py, numpy.
- Files: list of files and responsibility.
- Convergence criterion: L1 norm < tol.
- Optimizations used or possible: sparse matrices, partitioned IO, asynchronous communication.

## Experiments
- Hardware: machine/cluster specs (CPU, RAM).
- Datasets: describe graph sizes (N, M), synthetic generation parameters.
- Metrics: runtime, iterations to converge, per-iteration time, speedup.

## Results
- Present tables and plots (time vs processes, speedup vs ideal).
- Discuss the trend — where benefits saturate, overheads from Allreduce.

## Discussion
- Bottlenecks: large Allreduce on dense vectors; memory for dense vectors; IO cost.
- Improvements: sparse communication, graph partitioning, hybrid MPI+OpenMP in C++.

## Conclusion
- Summarize successes and next steps.

## Appendix
- Commands used to run experiments and reproduce plots.
