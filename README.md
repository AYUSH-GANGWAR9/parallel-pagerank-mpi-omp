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

### 1️⃣ Install Dependencies
```bash
sudo apt update
sudo apt install -y build-essential openmpi-bin libopenmpi-dev python3 python3-pip
pip install pandas matplotlib

### 2️⃣ Compile the Project
bash
Copy code
mpicxx -O3 -fopenmp -std=c++17 pagerank.cpp -o pagerank

##3 3️⃣ Run PageRank
bash
Copy code
mpirun -np 2 ./pagerank data/web-Google-100k.txt 0 100 1e-6 0.85

Arguments:

php-template
Copy code
<file> <N_nodes=0(auto)> <max_iter> <tolerance> <damping_factor>
Example:

bash
Copy code
mpirun -np 4 ./pagerank data/web-Google-100k.txt 0 100 1e-6 0.85

### 📊 Running Performance Analysis

To test strong scaling automatically:

bash
Copy code
bash run_scaling.sh
This script:

Runs with 1, 2, and 4 processes.

Records execution time and iterations.

Saves all logs in data/scaling_results.csv.

Then visualize using:

bash
Copy code
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
