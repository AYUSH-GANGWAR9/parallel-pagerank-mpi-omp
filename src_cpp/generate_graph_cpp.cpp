// src_cpp/generate_graph_cpp.cpp
// Simple C++ random edge-list generator
#include <bits/stdc++.h>
using namespace std;
int main(int argc, char** argv){
    long long n = 1000, m = 5000; string out = "data/edges.txt"; unsigned seed = 42;
    for (int i=1;i<argc;++i){
        string s = argv[i];
        if (s=="--n" && i+1<argc) n = atoll(argv[++i]);
        else if (s=="--m" && i+1<argc) m = atoll(argv[++i]);
        else if (s=="--out" && i+1<argc) out = argv[++i];
        else if (s=="--seed" && i+1<argc) seed = (unsigned)atoi(argv[++i]);
    }
    ios::sync_with_stdio(false);
    mt19937_64 rng(seed);
    uniform_int_distribution<long long> dist(0, n-1);
    {
        auto pos = out.find_last_of("/");
        if (pos != string::npos) {
            string cmd = "mkdir -p " + out.substr(0,pos);
            system(cmd.c_str());
        }
    }
    ofstream fout(out);
    for (long long i=0;i<m;++i){
        long long u = dist(rng);
        long long v = dist(rng);
        if (u==v){ i--; continue; }
        fout << u << " " << v << "\n";
    }
    cerr << "Wrote " << m << " edges to " << out << " (n=" << n << ").\n";
    return 0;
}
