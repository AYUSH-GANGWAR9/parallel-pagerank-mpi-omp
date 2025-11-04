# Parallel PageRank (MPI, Python)

## Requirements
- Python 3.8+
- mpi4py (`pip install mpi4py`)
- numpy (`pip install numpy`)
- matplotlib (for plotting) (`pip install matplotlib`)
Optional:
- scipy (`pip install scipy`) for sparse matrices (not required)

To run with mpiexec (example with 4 processes):

1. Generate a graph:
   python src/generate_graph.py --n 5000 --m 30000 --out data/edges.txt

2. Run PageRank:
   mpiexec -n 4 python src/pagerank_mpi.py --edges data/edges.txt --tol 1e-6 --maxit 200 --d 0.85 --verbose

This will create an experiments/results_n{N}_p{P}.csv file with summary info.

3. Run multiple experiments (varying process counts) and collect `experiments/*.csv` then:
   python src/plot_results.py --pattern "experiments/results_n*_p*.csv"

## Notes on scaling and improvements
- Current implementation broadcasts/sums a dense vector of length N each iteration (via Allreduce). Good for small->medium graphs (N up to ~100k depending on memory).
- For larger graphs:
  - Partition the graph on disk; read only a chunk per process.
  - Use sparse partitioning and only exchange messages for nonzero contributions (MPI point-to-point or Alltoallv).
  - Consider hybrid MPI+OpenMP (C/C++) for top performance.

## Experiment suggestions
- Strong scaling: fix N & edges, measure time with p = 1,2,4,8,16
- Weak scaling: fix edges per process, increase N and p proportionally
- Measure iterations-to-converge and per-iteration time

## Commands

# generate graph
python src/generate_graph.py --n 20000 --m 100000 --out data/edges.txt

# run pagerank with 4 processes
mpiexec -n 4 python src/pagerank_mpi.py --edges data/edges.txt --tol 1e-6 --maxit 200 --d 0.85 --verbose

## Strong-scaling experiment automation (bash on cluster):

GRAPH=data/edges.txt
for P in 1 2 4 8; do
  mpiexec -n $P python src/pagerank_mpi.py --edges $GRAPH --tol 1e-6 --maxit 200 --d 0.85
done
# collect experiments/*.csv and plot
python src/plot_results.py --pattern "experiments/results_n*"

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Parallel PageRank (MPI Python & C++ MPI+OpenMP)

## Overview
This repo contains:
- Python MPI baseline (mpi4py) implementation: `src/pagerank_mpi.py`
- Optimized C++ MPI + OpenMP implementation: `src_cpp/pagerank_mpi_omp.cpp`
- Graph generators (Python + C++), plotting and experiment automation.

## Build
Install prerequisites:
```bash
sudo apt update
sudo apt install -y build-essential libopenmpi-dev openmpi-bin python3-pip
pip3 install mpi4py numpy matplotlib



Build binaries:

make
# or with cmake:
mkdir build && cd build
cmake ..
make

Quick run (single experiment)

Generate graph:

python3 src/generate_graph.py --n 20000 --m 100000 --out data/edges_n20000_m100000.txt
# or
bin/generate_graph_cpp --n 20000 --m 100000 --out data/edges_n20000_m100000.txt


Run Python MPI:

mpiexec -n 4 python3 src/pagerank_mpi.py --edges data/edges_n20000_m100000.txt --tol 1e-6 --maxit 200 --d 0.85 --verbose


Run C++ MPI+OpenMP:

export OMP_NUM_THREADS=4
mpiexec -n 4 bin/pagerank_mpi_omp --edges data/edges_n20000_m100000.txt --tol 1e-6 --maxit 200 --d 0.85 --verbose

Experiments (strong/weak)

Use automation script:

bash scripts/run_experiments.sh --mode strong --graph data/edges_n20000_m100000.txt --procs "1 2 4 8" --out experiments/
# or for weak scaling:
bash scripts/run_experiments.sh --mode weak --procs "1 2 4 8"

Plotting

After runs:

python3 src/plot_compare.py --python_pattern "experiments/python_results_*.csv" --cpp_pattern "experiments/cpp_results_*.csv" --outdir experiments/plots

