CXX=mpicxx
CXXFLAGS=-O3 -fopenmp -std=c++17
TARGET=pagerank

all: $(TARGET)

$(TARGET): pagerank.cpp
	$(CXX) $(CXXFLAGS) pagerank.cpp -o $(TARGET)

clean:
	rm -f $(TARGET)
