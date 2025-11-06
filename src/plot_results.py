#!/usr/bin/env python3
"""
Plot single-line experiments (processes vs total_time) from CSVs produced by implementations.
Usage:
  python3 src/plot_results.py --pattern "experiments/*results_*.csv" --out experiments/plot.png
"""
import glob, argparse, csv, matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pattern', type=str, default='experiments/*results_*.csv')
    p.add_argument('--out', type=str, default='experiments/plot.png')
    return p.parse_args()

def main():
    args = parse_args()
    files = glob.glob(args.pattern)
    data = []
    for fn in files:
        with open(fn) as f:
            r = csv.reader(f)
            header = next(r, None)
            row = next(r, None)
            if row:
                processes = int(row[0]); nodes = int(row[1]); edges = int(row[2]); iters = int(row[3]); total_time = float(row[4])
                data.append((processes, total_time, nodes, edges, iters, fn))
    if not data:
        print("No files found for pattern", args.pattern); return
    data.sort()
    procs = [d[0] for d in data]; times = [d[1] for d in data]
    plt.figure(figsize=(6,4))
    plt.plot(procs, times, marker='o')
    plt.xlabel('Processes'); plt.ylabel('Total time (s)')
    plt.title('Scaling')
    plt.grid(True); plt.savefig(args.out, dpi=150)
    print("Saved plot to", args.out)

if __name__ == '__main__':
    main()
