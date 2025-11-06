#!/usr/bin/env python3
"""
Parallel PageRank using mpi4py (dense Allreduce approach).
Usage:
  mpiexec -n 4 python3 src/pagerank_mpi.py --edges data/edges.txt --tol 1e-6 --maxit 100 --d 0.85 --verbose
"""
import argparse
import time
from mpi4py import MPI
import numpy as np
import os, csv

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--edges', type=str, required=True, help='Edge list file (u v) per line, 0-indexed nodes')
    p.add_argument('--tol', type=float, default=1e-6, help='Convergence tolerance (L1)')
    p.add_argument('--maxit', type=int, default=100, help='Max iterations')
    p.add_argument('--d', type=float, default=0.85, help='Damping factor')
    p.add_argument('--verbose', action='store_true', help='Print per-iteration info from rank 0')
    return p.parse_args()

def read_edge_list(path):
    edges = []
    max_node = -1
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) < 2: continue
            try:
                u = int(parts[0]); v = int(parts[1])
            except ValueError:
                continue
            edges.append((u, v))
            if u > max_node: max_node = u
            if v > max_node: max_node = v
    N = max_node + 1
    return edges, N

def build_outgoing(edges, N):
    outgoing = [[] for _ in range(N)]
    outdeg = np.zeros(N, dtype=np.int64)
    for u, v in edges:
        outgoing[u].append(v)
        outdeg[u] += 1
    return outgoing, outdeg

def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    t0 = MPI.Wtime()

    if rank == 0 and args.verbose:
        print(f"[rank 0] Starting PageRank with {size} processes")

    edges, N = read_edge_list(args.edges)
    nodes_per_rank = (N + size - 1) // size
    start = rank * nodes_per_rank
    end = min(start + nodes_per_rank, N)
    local_nodes = list(range(start, end))
    if rank == 0 and args.verbose:
        print(f"[rank 0] Graph size N={N}, edges={len(edges)}, nodes_per_rank~{nodes_per_rank}")

    outgoing, outdeg = build_outgoing(edges, N)
    local_outgoing = {u: outgoing[u] for u in local_nodes if outdeg[u] > 0}
    local_outdeg = outdeg[start:end]

    d = float(args.d)
    r = np.full(N, 1.0 / N, dtype=np.float64)
    teleport = (1.0 - d) / N
    tol = float(args.tol)
    maxit = int(args.maxit)

    if rank == 0 and args.verbose:
        print("[rank 0] Starting iterations...")

    comm.Barrier()
    iter_times = []
    it = 0
    for it in range(1, maxit + 1):
        it_start = MPI.Wtime()
        local_contrib = np.zeros(N, dtype=np.float64)

        local_dangling_mass = 0.0
        for u in local_nodes:
            ru = r[u]
            if outdeg[u] == 0:
                local_dangling_mass += ru
            else:
                share = ru / outdeg[u]
                for v in outgoing[u]:
                    local_contrib[v] += share

        comm.Allreduce(MPI.IN_PLACE, local_contrib, op=MPI.SUM)
        total_dangling_mass = comm.allreduce(local_dangling_mass, op=MPI.SUM)
        dangling_term = total_dangling_mass / N
        new_r = teleport + d * (local_contrib + dangling_term)

        diff = np.linalg.norm(new_r - r, ord=1)
        diff_all = comm.allreduce(diff, op=MPI.SUM)
        r[:] = new_r
        it_time = MPI.Wtime() - it_start
        iter_times.append(it_time)

        if rank == 0 and args.verbose:
            print(f"it {it:3d} diff={diff_all:.6e} time={it_time:.4f}s")
        converged_flag = 1 if diff_all < tol else 0
        converged_flag = comm.bcast(converged_flag, root=0)
        if converged_flag == 1:
            if rank == 0 and args.verbose:
                print(f"[rank 0] Converged at iteration {it} with diff={diff_all:.3e}")
            break

    total_time = MPI.Wtime() - t0
    total_iter_time = sum(iter_times)
    times = comm.gather(total_iter_time, root=0)
    if rank == 0:
        print("=== Summary (Python MPI) ===")
        print(f"Processes: {size}, Iterations run: {it}, Total time: {total_time:.4f}s")
        idx_sorted = np.argsort(-r)
        print("Top ranks (node:rank):")
        for i in range(min(10, N)):
            node = idx_sorted[i]
            print(f"{node}: {r[node]:.6e}")
        os.makedirs("experiments", exist_ok=True)
        out_csv = os.path.join("experiments", f"python_results_n{N}_p{size}.csv")
        with open(out_csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(["processes", "nodes", "edges", "iterations", "total_time"])
            w.writerow([size, N, len(edges), it, total_time])
        if args.verbose:
            print(f"[rank 0] Saved experiment summary to {out_csv}")

    comm.Barrier()
    if rank == 0 and args.verbose:
        print(f"[rank 0] Done. Wall time: {MPI.Wtime() - t0:.4f}s")

if __name__ == '__main__':
    main()
