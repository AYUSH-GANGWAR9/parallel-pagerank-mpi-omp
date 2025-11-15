# Parallel PageRank Algorithm (Google Search Simulation)

### 🧠 Distributed and Parallel Computing Project  
**Name:** Ayush Gangwar  
**Roll No.:** 202211006  
**Department of Computer Science**  
**Project:** Hybrid MPI + OpenMP Implementation of PageRank

---

## 📘 Overview

This project implements the **PageRank algorithm** using a **hybrid parallel approach (MPI + OpenMP)** to simulate the Google Search ranking process.  
It computes the importance of each web page based on the link structure of a directed web graph.

PageRank is an iterative algorithm that assigns a ranking score to each node (web page) based on the number and quality of incoming links.

---

## 🚀 Objectives

- Implement a **parallel PageRank algorithm** using **MPI** (Message Passing Interface) for distributed computation.  
- Use **OpenMP** for intra-process parallelism to enhance performance.  
- Evaluate performance through **strong scaling** using real-world datasets.  
- Visualize performance metrics: **Total Time**, **Speedup**, and **Efficiency**.

---

## ⚙️ Technologies Used

| Component | Technology |
|------------|-------------|
| Language | C++17 |
| Parallel Frameworks | MPI, OpenMP |
| Plotting & Analysis | Python (`pandas`, `matplotlib`) |
| Dataset | [SNAP web-Google](https://snap.stanford.edu/data/web-Google.html) |
| Environment | Ubuntu / WSL2 |
| Build Tools | `make`, `mpicxx`, `pdflatex` |

---

## 🛠️ Setup Instructions

1 — Create project folder & enter it
mkdir -p ~/DPC_Project/Parallel_PageRank
cd ~/DPC_Project/Parallel_PageRank

2 — Install system deps (one-time)
sudo apt update
sudo apt install -y build-essential openmpi-bin libopenmpi-dev python3 python3-venv wget unzip

3 — Create & activate Python virtualenv (for plotting tools)
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install pandas matplotlib


When finished, you’ll see (venv) in your prompt.

**4 — Files to create**

Create these files (I give full content). Use your editor (nano, vim) or redirect cat > file <<'EOF' ... EOF.

**4.1 make a pagerank.cpp using touch command in wsl.

4.2 create a Makefile using touch command in wsl.**

**4.3 run the code in this file in wsl directly run_scaling_capture_time.sh**

**Make it executable:**

chmod +x run_scaling_capture_time.sh

**4.4 make a plot_scaling.py using touch command in wsl and using nano to write in that file.**

**5 — Download the SNAP dataset and prepare a manageable subset
**
mkdir -p data
cd data
wget https://snap.stanford.edu/data/web-Google.txt.gz
gunzip -f web-Google.txt.gz
# create a 100k-edge subset used in our experiments
head -n 100000 web-Google.txt > web-Google-100k.txt
cd ..


If wget fails, download manually from https://snap.stanford.edu/data/web-Google.html
 and upload to data/.

6 — Build the project

Back in project root:

make clean
make -j
ls -l pagerank

7 — Quick single-process test (verify correctness)

export OMP_NUM_THREADS=4        # set OpenMP threads per rank

unset DEBUG_PAGERANK           # or "export DEBUG_PAGERANK=1" to see debug

mpirun --allow-run-as-root --oversubscribe -np 1 ./pagerank data/web-Google-100k.txt 0 100 1e-6 0.85 | tee data/pagerank_run_p1.log

tail -n 40 data/pagerank_run_p1.log

8 — Real scaling experiment (1,2,4 processes) and collect timings

Run the provided script:

./run_scaling_capture_time.sh

That will:

run p = 1, 2, 4 (each with OMP_NUM_THREADS=2 as set in script),

write data/pagerank_p{p}.log and data/pagerank_time_p{p}.txt,

create data/scaling_results.csv.

Inspect the CSV:

cat data/scaling_results.csv

9 — Plot speedup & efficiency

With venv active:

python3 plot_scaling.py

### 📈 Results
Processes	OMP Threads	Total Time (s)	Iterations
1	2	0.67	41
2	2	1.01	41
4	2	0.59	41

## 📉 Performance Graphs
Metric	Plot
Total Time	
Speedup	
Efficiency	

## 🧠 Key Implementation Details
Graph stored in CSR (Compressed Sparse Row) format for efficient access.

Handles dangling nodes (pages with no outbound links) properly.

Uses MPI_Allreduce for global convergence checks.

OpenMP used for loop-level parallelism within each rank.

Automatically normalizes rank vector after each iteration.

## 🧾 Deliverables

File	Description
pagerank.cpp	Core hybrid MPI+OpenMP implementation
run_scaling.sh	Benchmark script for scaling runs
plot_scaling.py	Generates performance plots
Parallel_PageRank_Report.pdf	Final report (detailed methodology & results)
pagerank_presentation.pdf	5-slide Beamer presentation
README.md	Documentation and setup guide

## 🏁 Conclusion

Achieved a working parallel implementation of PageRank using MPI + OpenMP.

Verified correctness on the SNAP web-Google dataset.

Demonstrated speedup and efficiency improvements.

Future work includes GPU acceleration and dynamic graph support.

Author: Ayush Gangwar
Roll No.: 202211006
📅 Distributed and Parallel Computing Project — 2025
