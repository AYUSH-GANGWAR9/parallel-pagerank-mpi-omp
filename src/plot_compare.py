#!/usr/bin/env python3
"""
Compare python and C++ experiment CSVs and make side-by-side plots:
- Time vs Processes
- Speedup vs Processes (normalized to P=1)
Usage:
python3 src/plot_compare.py --python_pattern "experiments/python_*.csv" --cpp_pattern "experiments/cpp_*.csv" --outdir experiments/plots
"""
import glob, argparse, csv, os
import matplotlib.pyplot as plt

def read_csvs(pattern):
    files = glob.glob(pattern)
    data = {}
    for fn in files:
        with open(fn) as f:
            r = csv.reader(f)
            header = next(r, None)
            row = next(r, None)
            if row:
                p = int(row[0]); nodes = int(row[1]); edges = int(row[2]); iters = int(row[3]); total_time = float(row[4])
                data[p] = {'time': total_time, 'nodes': nodes, 'edges': edges, 'iters': iters, 'file': fn}
    return data

def plot_time(procs, py_times, cpp_times, out):
    plt.figure(figsize=(6,4))
    plt.plot(procs, py_times, marker='o', label='Python MPI')
    plt.plot(procs, cpp_times, marker='o', label='C++ MPI+OpenMP')
    plt.xlabel('Processes'); plt.ylabel('Total time (s)')
    plt.title('Time vs Processes')
    plt.legend(); plt.grid(True); plt.savefig(out, dpi=150)
    print("Saved", out)

def plot_speedup(procs, base_py, py_times, base_cpp, cpp_times, out):
    py_speed = [base_py / t if t>0 else 0.0 for t in py_times]
    cpp_speed = [base_cpp / t if t>0 else 0.0 for t in cpp_times]
    plt.figure(figsize=(6,4))
    plt.plot(procs, py_speed, marker='o', label='Python speedup')
    plt.plot(procs, cpp_speed, marker='o', label='C++ speedup')
    plt.xlabel('Processes'); plt.ylabel('Speedup (relative to P=1)')
    plt.title('Speedup vs Processes')
    plt.legend(); plt.grid(True); plt.savefig(out, dpi=150)
    print("Saved", out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--python_pattern', type=str, default='experiments/python_results_*.csv')
    p.add_argument('--cpp_pattern', type=str, default='experiments/cpp_results_*.csv')
    p.add_argument('--outdir', type=str, default='experiments/plots')
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    py = read_csvs(args.python_pattern)
    cpp = read_csvs(args.cpp_pattern)
    procs = sorted(set(list(py.keys()) + list(cpp.keys())))
    procs = sorted(procs)
    py_times = [py[p]['time'] if p in py else None for p in procs]
    cpp_times = [cpp[p]['time'] if p in cpp else None for p in procs]

    # replace None with large number for plotting
    py_plot = [t if t is not None else float('nan') for t in py_times]
    cpp_plot = [t if t is not None else float('nan') for t in cpp_times]

    # base times at P=1
    base_py = py[1]['time'] if 1 in py else py_plot[0]
    base_cpp = cpp[1]['time'] if 1 in cpp else cpp_plot[0]

    plot_time(procs, py_plot, cpp_plot, os.path.join(args.outdir, 'time_vs_procs.png'))
    plot_speedup(procs, base_py, py_plot, base_cpp, cpp_plot, os.path.join(args.outdir, 'speedup_vs_procs.png'))

if __name__ == '__main__':
    main()
