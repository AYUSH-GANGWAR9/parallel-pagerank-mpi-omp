#!/usr/bin/env bash
# scripts/run_experiments.sh
# Usage:
#  bash scripts/run_experiments.sh --mode strong --graph data/edges.txt --procs "1 2 4 8" --out experiments/
set -e
MODE="strong"
GRAPH=""
PROCS="1 2 4 8"
OUT="experiments"
PYTHON_CMD="python3 src/pagerank_mpi.py"
CPP_BIN="./bin/pagerank_mpi_omp"
MAXIT=200
D=0.85
TOL=1e-6

while [[ $# -gt 0 ]]; do
  case $1 in
    --mode) MODE="$2"; shift 2 ;;
    --graph) GRAPH="$2"; shift 2 ;;
    --procs) PROCS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --maxit) MAXIT="$2"; shift 2 ;;
    --d) D="$2"; shift 2 ;;
    --tol) TOL="$2"; shift 2 ;;
    *) echo "Unknown $1"; exit 1 ;;
  esac
done

if [[ -z "$GRAPH" ]]; then
  echo "Provide --graph path"
  exit 1
fi

mkdir -p "$OUT"
mkdir -p experiments

# helper for running one impl
run_python() {
  P=$1
  echo "Running Python MPI with P=${P}"
  mpiexec -n ${P} ${PYTHON_CMD} --edges ${GRAPH} --tol ${TOL} --maxit ${MAXIT} --d ${D}
}

run_cpp() {
  P=$1
  echo "Running C++ MPI+OpenMP with P=${P}"
  export OMP_NUM_THREADS=4
  mpiexec -n ${P} ${CPP_BIN} --edges ${GRAPH} --tol ${TOL} --maxit ${MAXIT} --d ${D} --verbose
}

if [[ "$MODE" == "strong" ]]; then
  for P in ${PROCS}; do
    run_python ${P}
    run_cpp ${P}
  done
elif [[ "$MODE" == "weak" ]]; then
  # weak: scale graph size with P; baseline N0/M0
  N0=20000; M0=100000
  for P in ${PROCS}; do
    N=$((N0 * P))
    M=$((M0 * P))
    G="data/edges_n${N}_m${M}.txt"
    echo "Generating graph N=${N} M=${M}"
    ./bin/generate_graph_cpp --n ${N} --m ${M} --out ${G} --seed 42
    run_python ${P}
    run_cpp ${P}
  done
else
  echo "Unknown mode ${MODE}"
  exit 1
fi
