CXX=mpicxx
CXXFLAGS=-O3 -fopenmp -std=c++17
PY=python3

all: bin/pagerank_mpi_omp bin/generate_graph_cpp

bin/pagerank_mpi_omp: src_cpp/pagerank_mpi_omp.cpp
	mkdir -p bin
	$(CXX) $(CXXFLAGS) -o $@ $<

bin/generate_graph_cpp: src_cpp/generate_graph_cpp.cpp
	mkdir -p bin
	g++ -O3 -std=c++17 -o $@ $<

.PHONY: clean
clean:
	rm -rf bin experiments/*.csv experiments/plots data/*.txt
