/* src_cpp/pagerank_mpi_omp.cpp
   Optimized C++ MPI + OpenMP PageRank (sparse send using Alltoallv).
   Build:
     mpicxx -O3 -fopenmp -std=c++17 -o bin/pagerank_mpi_omp src_cpp/pagerank_mpi_omp.cpp
   Run example:
     export OMP_NUM_THREADS=4
     mpiexec -n 4 ./bin/pagerank_mpi_omp --edges data/edges.txt --tol 1e-6 --maxit 200 --d 0.85 --verbose
*/
#include <mpi.h>
#include <omp.h>
#include <bits/stdc++.h>
using namespace std;
struct Args { string edges; double tol=1e-6; int maxit=100; double d=0.85; bool verbose=false; };
Args parse_args(int argc, char** argv) {
    Args a;
    for (int i=1;i<argc;++i) {
        string s=argv[i];
        if (s=="--edges"&&i+1<argc) a.edges=argv[++i];
        else if (s=="--tol"&&i+1<argc) a.tol=stod(argv[++i]);
        else if (s=="--maxit"&&i+1<argc) a.maxit=stoi(argv[++i]);
        else if (s=="--d"&&i+1<argc) a.d=stod(argv[++i]);
        else if (s=="--verbose") a.verbose=true;
    }
    return a;
}
long long read_N_from_file(const string &path) {
    ifstream fin(path);
    if (!fin) throw runtime_error("open error");
    string line; long long maxn=-1;
    while (getline(fin,line)) {
        if (line.empty()||line[0]=='#') continue;
        istringstream iss(line);
        long long u,v; if (!(iss>>u>>v)) continue;
        if (u>maxn) maxn=u; if (v>maxn) maxn=v;
    }
    return maxn+1;
}
long long read_edges_local(const string &path, long long start, long long end, vector<vector<int>> &local_adj, vector<int> &local_outdeg, long long &N_global) {
    N_global = read_N_from_file(path);
    long long local_n = max(0LL, end - start);
    local_adj.assign(local_n, vector<int>());
    local_outdeg.assign(local_n, 0);
    ifstream fin(path);
    if (!fin) throw runtime_error("open error second pass");
    string line;
    while (getline(fin,line)) {
        if (line.empty()||line[0]=='#') continue;
        istringstream iss(line);
        long long u,v; if (!(iss>>u>>v)) continue;
        if (u>=start && u<end) {
            long long idx = u - start;
            local_adj[idx].push_back((int)v);
            local_outdeg[idx] += 1;
        }
    }
    return local_n;
}
int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    Args args = parse_args(argc,argv);
    if (args.edges.empty()) { if (rank==0) cerr<<"Usage: --edges PATH\n"; MPI_Finalize(); return 1; }
    double t_total_start = MPI_Wtime();
    long long N_global = read_N_from_file(args.edges);
    long long nodes_per_rank = (N_global + size - 1) / size;
    long long start = rank * nodes_per_rank;
    long long end = min((long long)N_global, (rank+1) * nodes_per_rank);
    long long local_n = max(0LL, end - start);
    vector<vector<int>> local_adj; vector<int> local_outdeg;
    read_edges_local(args.edges, start, end, local_adj, local_outdeg, N_global);
    if (rank==0 && args.verbose) cout<<"Graph N="<<N_global<<", processes="<<size<<", nodes_per_rank~"<<nodes_per_rank<<"\n";
    vector<double> r_local(local_n, 1.0/(double)N_global), new_r_local(local_n,0.0);
    double teleport = (1.0 - args.d) / (double)N_global;
    vector<vector<int>> destin_rank_per_src(local_n);
    vector<vector<int>> destin_localidx_per_src(local_n);
    for (long long i=0;i<local_n;++i) {
        auto &adj = local_adj[i];
        destin_rank_per_src[i].reserve(adj.size());
        destin_localidx_per_src[i].reserve(adj.size());
        for (int v : adj) {
            int dest_rank = (int)(v / nodes_per_rank);
            long long dest_start = (long long)dest_rank * nodes_per_rank;
            int dest_local_idx = (int)(v - dest_start);
            destin_rank_per_src[i].push_back(dest_rank);
            destin_localidx_per_src[i].push_back(dest_local_idx);
        }
    }
    vector<int> send_counts(size,0), recv_counts(size,0);
    vector<vector<int>> send_dests_per_rank(size);
    vector<vector<double>> send_vals_per_rank(size);
    int it;
    double start_time = MPI_Wtime();
    for (it=1; it<=args.maxit; ++it) {
        double it_start = MPI_Wtime();
        for (int p=0;p<size;++p) { send_dests_per_rank[p].clear(); send_vals_per_rank[p].clear(); send_counts[p]=0; }
        double local_dangling_mass = 0.0;
        #pragma omp parallel
        {
            vector<vector<int>> thread_send_dests(size);
            vector<vector<double>> thread_send_vals(size);
            #pragma omp for schedule(static)
            for (long long i=0;i<local_n;++i) {
                double ru = r_local[i];
                int outdeg = local_outdeg[i];
                if (outdeg==0) {
                    #pragma omp atomic
                    local_dangling_mass += ru;
                } else {
                    double share = ru / (double)outdeg;
                    auto &ranks = destin_rank_per_src[i];
                    auto &localidx = destin_localidx_per_src[i];
                    size_t m=ranks.size();
                    for (size_t k=0;k<m;++k) {
                        int pr=ranks[k]; int desti=localidx[k];
                        thread_send_dests[pr].push_back(desti);
                        thread_send_vals[pr].push_back(share);
                    }
                }
            }
            for (int p=0;p<size;++p) {
                if (!thread_send_dests[p].empty()) {
                    #pragma omp critical
                    {
                        auto &A = send_dests_per_rank[p];
                        auto &B = send_vals_per_rank[p];
                        A.insert(A.end(), thread_send_dests[p].begin(), thread_send_dests[p].end());
                        B.insert(B.end(), thread_send_vals[p].begin(), thread_send_vals[p].end());
                    }
                }
            }
        }
        for (int p=0;p<size;++p) send_counts[p] = (int)send_dests_per_rank[p].size();
        MPI_Alltoall(send_counts.data(),1,MPI_INT, recv_counts.data(),1,MPI_INT, MPI_COMM_WORLD);
        vector<int> send_displs(size,0), recv_displs(size,0);
        int send_total=0, recv_total=0;
        for (int p=0;p<size;++p) { send_displs[p]=send_total; send_total+=send_counts[p]; }
        for (int p=0;p<size;++p) { recv_displs[p]=recv_total; recv_total+=recv_counts[p]; }
        vector<int> send_dests_flat; send_dests_flat.reserve(send_total);
        vector<double> send_vals_flat; send_vals_flat.reserve(send_total);
        for (int p=0;p<size;++p) {
            auto &A=send_dests_per_rank[p]; auto &B=send_vals_per_rank[p];
            send_dests_flat.insert(send_dests_flat.end(), A.begin(), A.end());
            send_vals_flat.insert(send_vals_flat.end(), B.begin(), B.end());
        }
        vector<int> recv_dests_flat(recv_total);
        vector<double> recv_vals_flat(recv_total);
        MPI_Alltoallv(send_dests_flat.data(), send_counts.data(), send_displs.data(), MPI_INT,
                      recv_dests_flat.data(), recv_counts.data(), recv_displs.data(), MPI_INT,
                      MPI_COMM_WORLD);
        MPI_Alltoallv(send_vals_flat.data(), send_counts.data(), send_displs.data(), MPI_DOUBLE,
                      recv_vals_flat.data(), recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                      MPI_COMM_WORLD);
        double total_dangling_mass = 0.0;
        MPI_Allreduce(&local_dangling_mass, &total_dangling_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double dangling_term = total_dangling_mass / (double)N_global;
        #pragma omp parallel for schedule(static)
        for (long long i=0;i<local_n;++i) new_r_local[i] = teleport + args.d * dangling_term;
        #pragma omp parallel for schedule(static)
        for (int k=0;k<recv_total;++k) {
            int local_idx = recv_dests_flat[k]; double val = recv_vals_flat[k];
            #pragma omp atomic
            new_r_local[local_idx] += args.d * val;
        }
        double local_diff = 0.0;
        #pragma omp parallel for reduction(+:local_diff)
        for (long long i=0;i<local_n;++i) local_diff += fabs(new_r_local[i] - r_local[i]);
        double global_diff = 0.0;
        MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        r_local.swap(new_r_local);
        double it_time = MPI_Wtime() - it_start;
        if (rank==0 && args.verbose) cout<<"it "<<it<<" diff="<<scientific<<global_diff<<" time="<<fixed<<setprecision(4)<<it_time<<"s\n";
        if (global_diff < args.tol) { if (rank==0 && args.verbose) cout<<"Converged at it="<<it<<"\n"; break; }
    }
    double total_time = MPI_Wtime() - start_time;
    int topk=10;
    vector<pair<double,long long>> local_pairs;
    for (long long i=0;i<local_n;++i) local_pairs.emplace_back(r_local[i], start + i);
    sort(local_pairs.begin(), local_pairs.end(), greater<>());
    int send_k = min((int)local_pairs.size(), topk);
    vector<double> send_vals_top(send_k); vector<long long> send_ids_top(send_k);
    for (int i=0;i<send_k;++i) { send_vals_top[i]=local_pairs[i].first; send_ids_top[i]=local_pairs[i].second; }
    vector<int> recv_counts_top(size,0);
    int my_count_top = send_k;
    MPI_Gather(&my_count_top,1,MPI_INT, recv_counts_top.data(),1,MPI_INT, 0, MPI_COMM_WORLD);
    vector<int> recv_displs_top;
    vector<double> gathered_vals;
    vector<long long> gathered_ids;
    if (rank==0) {
        recv_displs_top.resize(size);
        int total_recv=0; for (int p=0;p<size;++p) { recv_displs_top[p]=total_recv; total_recv+=recv_counts_top[p]; }
        gathered_vals.resize(total_recv); gathered_ids.resize(total_recv);
    }
    MPI_Gatherv(send_vals_top.data(), my_count_top, MPI_DOUBLE,
                rank==0 ? gathered_vals.data() : nullptr, recv_counts_top.data(), rank==0 ? recv_displs_top.data() : nullptr, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Datatype MPI_LL = MPI_LONG_LONG;
    MPI_Gatherv(send_ids_top.data(), my_count_top, MPI_LL,
                rank==0 ? gathered_ids.data() : nullptr, recv_counts_top.data(), rank==0 ? recv_displs_top.data() : nullptr, MPI_LL,
                0, MPI_COMM_WORLD);
    if (rank==0) {
        vector<pair<double,long long>> all;
        int total_recv = (int)gathered_vals.size();
        for (int i=0;i<total_recv;++i) all.emplace_back(gathered_vals[i], gathered_ids[i]);
        sort(all.begin(), all.end(), greater<>());
        cout<<"=== Summary (C++ MPI+OpenMP) ===\n";
        cout<<"Processes: "<<size<<", N="<<N_global<<", iterations run="<<it<<", total_time="<<total_time<<"s\n";
        cout<<"Top ranks (node:rank):\n";
        for (int i=0;i<min((int)all.size(), topk); ++i) cout<<all[i].second<<": "<<scientific<<all[i].first<<"\n";
        // Save summary CSV
        FILE *fh = fopen("experiments/cpp_results_tmp.csv","w");
        if (fh) {
            fprintf(fh,"processes,nodes,edges,iterations,total_time\n");
            fprintf(fh,"%d,%lld,%d,%d,%.6f\n", size, N_global, 0, it, total_time);
            fclose(fh);
            // rename with N and p
            char buf[256];
            sprintf(buf,"experiments/cpp_results_n%lld_p%d.csv", N_global, size);
            rename("experiments/cpp_results_tmp.csv", buf);
        }
    }
    double t_total_end = MPI_Wtime();
    if (rank==0 && args.verbose) cout<<"Total wall-clock time: "<<(t_total_end - t_total_start)<<"s\n";
    MPI_Finalize();
    return 0;
}
