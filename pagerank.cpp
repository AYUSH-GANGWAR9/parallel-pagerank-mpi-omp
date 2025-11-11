#include <mpi.h>
#include <omp.h>

#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <cstdlib> // getenv

using idx_t = int64_t;
using val_t = double;

struct CSR {
    std::vector<idx_t> row_ptr; // size n+1
    std::vector<idx_t> col;     // incoming neighbors
};

void build_csr_from_edgelist(idx_t n, const std::vector<std::pair<idx_t,idx_t>>& edges,
                             CSR &csr, std::vector<idx_t> &outdeg) {
    outdeg.assign(n, 0);
    for (auto &e : edges) {
        idx_t u = e.first;
        if (u>=0 && u<n) outdeg[u]++;
    }
    std::vector<idx_t> indeg(n,0);
    for (auto &e : edges) {
        idx_t v = e.second;
        if (v>=0 && v<n) indeg[v]++;
    }
    csr.row_ptr.resize(n+1);
    csr.row_ptr[0]=0;
    for (idx_t i=0;i<n;i++) csr.row_ptr[i+1]=csr.row_ptr[i]+indeg[i];
    csr.col.resize(csr.row_ptr[n]);
    std::vector<idx_t> cur = csr.row_ptr;
    for (auto &e : edges) {
        idx_t u = e.first;
        idx_t v = e.second;
        if (v<0 || v>=n || u<0 || u>=n) continue;
        csr.col[cur[v]++] = u;
    }
}

void read_edge_list(const std::string &fname, std::vector<std::pair<idx_t,idx_t>>& edges, idx_t &maxnode) {
    std::ifstream in(fname);
    if (!in.is_open()) {
        std::cerr<<"Cannot open "<<fname<<"\n";
        exit(1);
    }
    std::string line;
    edges.clear();
    maxnode = -1;
    while (std::getline(in,line)) {
        if (line.size()==0) continue;
        if (line[0]=='#') continue;
        std::istringstream iss(line);
        idx_t u,v;
        if (!(iss>>u>>v)) continue;
        if (u<0 || v<0) continue;
        edges.emplace_back(u,v);
        if (u>maxnode) maxnode=u;
        if (v>maxnode) maxnode=v;
    }
    in.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

    if (argc < 6) {
        if (rank==0) {
            std::cout<<"Usage: "<<argv[0]<<" <edge_list.txt> <N_nodes> <max_iter> <tol> <damping>\n";
            std::cout<<"If N_nodes==0, it reads max node index from file and uses max+1\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string edgefile = argv[1];
    idx_t N = atoll(argv[2]);
    int max_iter = atoi(argv[3]);
    val_t tol = atof(argv[4]);
    val_t damping = atof(argv[5]);

    // Read edge list (each rank reads full file)
    std::vector<std::pair<idx_t,idx_t>> edges;
    idx_t maxnode;
    read_edge_list(edgefile, edges, maxnode);
    if (N==0) N = maxnode+1;
    if (rank==0) {
        std::cout<<"Read "<<edges.size()<<" edges, N="<<N<<", procs="<<nprocs<<"\n";
    }

    // Build CSR and outdegree
    CSR csr;
    std::vector<idx_t> outdeg;
    build_csr_from_edgelist(N, edges, csr, outdeg);

    // Partition rows
    idx_t rows_per = (N + nprocs - 1) / nprocs;
    idx_t row_start = rank * rows_per;
    idx_t row_end = std::min(N, row_start + rows_per);
    idx_t local_n = std::max((idx_t)0, row_end - row_start);

    // Prepare Allgatherv params
    std::vector<int> counts(nprocs), displs(nprocs);
    for (int p=0;p<nprocs;p++) {
        idx_t rs = p * rows_per;
        idx_t re = std::min(N, rs + rows_per);
        counts[p] = (int)(re - rs);
        displs[p] = (int)(rs);
    }

    // Initialize PR vectors
    std::vector<val_t> pr_local(local_n, 1.0 / (val_t)N);
    std::vector<val_t> pr_all(N, 0.0);
    MPI_Allgatherv(pr_local.data(), (int)local_n, MPI_DOUBLE,
                   pr_all.data(), counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    val_t base = (1.0 - damping) / (val_t)N;
    bool debug = (getenv("DEBUG_PAGERANK") != nullptr);

    if (debug) {
        val_t s=0.0;
        for (idx_t i=0;i<N;i++) s += pr_all[i];
        std::cout<<"RANK "<<rank<<" initial pr_all sum="<<s<<" sample[0..4]=";
        for (int k=0;k<5 && k<N; ++k) std::cout<<pr_all[k]<<" ";
        std::cout<<"\n";
    }

    double total_start = MPI_Wtime();
    double comp_time = 0.0, comm_time = 0.0;
    val_t global_diff = 0.0;
    int iter;

    for (iter=0; iter<max_iter; ++iter) {
        double iter_start = MPI_Wtime();

        // ----------------------------
        // Dangling mass (compute from local piece ONLY)
        // ----------------------------
        val_t local_dang = 0.0;
        for (idx_t i = 0; i < local_n; ++i) {
            idx_t node = row_start + i;
            if (node < N && outdeg[node] == 0) local_dang += pr_local[i];
        }
        val_t global_dang = 0.0;
        MPI_Allreduce(&local_dang, &global_dang, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Compute new local PR
        std::vector<val_t> pr_new(local_n, 0.0);
        double comp_s = MPI_Wtime();
        #pragma omp parallel for schedule(dynamic,64)
        for (idx_t i = row_start; i < row_end; ++i) {
            val_t sum = 0.0;
            idx_t r0 = csr.row_ptr[i];
            idx_t r1 = csr.row_ptr[i+1];
            for (idx_t p = r0; p < r1; ++p) {
                idx_t src = csr.col[p];
                if (src >= 0 && src < N) {
                    if (outdeg[src] > 0) sum += pr_all[src] / (val_t)outdeg[src];
                }
            }
            pr_new[i - row_start] = base + damping * (sum + global_dang / (val_t)N);
        }
        comp_time += MPI_Wtime() - comp_s;

        // Local diff
        val_t local_diff = 0.0;
        for (idx_t i=0;i<local_n;i++) local_diff += std::abs(pr_new[i] - pr_local[i]);
        pr_local.swap(pr_new);

        // Gather local pieces into pr_all
        double comm_s = MPI_Wtime();
        MPI_Allgatherv(pr_local.data(), (int)local_n, MPI_DOUBLE,
                       pr_all.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        comm_time += MPI_Wtime() - comm_s;

        // Correct normalization (sum local pr_local, reduce)
        val_t local_sum = 0.0;
        for (idx_t i = 0; i < (idx_t)local_n; ++i) local_sum += pr_local[i];
        val_t global_sum = 0.0;
        MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (global_sum > 0.0 && std::abs(global_sum - 1.0) > 1e-15) {
            for (idx_t i=0;i<N;i++) pr_all[i] /= global_sum;
            for (idx_t i = row_start; i < row_end; ++i)
                pr_local[i - row_start] = pr_all[i];
        }

        // Global diff reduction
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        if (debug && iter < 5) {
            val_t s = 0.0;
            for (idx_t i=0;i<N;i++) s += pr_all[i];
            std::cout<<"RANK "<<rank<<" iter="<<iter<<" pr_all sum="<<s<<" sample[0..4]=";
            for (int k=0;k<5 && k<N; ++k) std::cout<<pr_all[k]<<" ";
            std::cout<<"\n";
        }

        if (rank==0) {
            double iter_time = MPI_Wtime() - iter_start;
            std::cout<<"Iter "<<iter<<" diff="<<global_diff<<", time(s)="<<iter_time<<"\n";
        }
        if (global_diff < tol) break;
    }

    double total_end = MPI_Wtime();
    double total_time = total_end - total_start;

    if (rank==0) {
        std::vector<std::pair<val_t, idx_t>> pairs;
        pairs.reserve(N);
        for (idx_t i=0;i<N;i++) pairs.emplace_back(pr_all[i], i);
        std::sort(pairs.begin(), pairs.end(), std::greater<>());
        std::cout<<"Top 10 PageRank (val, node):\n";
        for (int k=0;k<10 && k<(int)pairs.size();k++) {
            std::cout<<k<<": "<<pairs[k].first<<" , "<<pairs[k].second<<"\n";
        }
    }

    // Timing summary
    double max_total_time, max_comp_time, max_comm_time;
    MPI_Reduce(&total_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank==0) {
        std::cout<<"Total time (max over ranks) = "<<max_total_time<<" s\n";
        std::cout<<"Comp time (max) = "<<max_comp_time<<" s, Comm time (max) = "<<max_comm_time<<" s\n";
        std::cout<<"Iterations = "<<(iter+1)<<"\n";
    }

    MPI_Finalize();
    return 0;
}
