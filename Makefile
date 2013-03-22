
all: stand-alone

clean:
	rm -rf *.o
	rm -rf *.oct
	rm -rf *.mex

.PHONY: stand-alone octave-bindings matlab-bindings

# Dependencies
HEADERS = lso_cluster.hpp lso_cluster_impl.hpp loss_functions.hpp trace_file_io.cpp
CXXFLAGS = -g -Wall

# Stand alone program
stand-alone: lso_cluster
lso_cluster: lso_cluster.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ lso_cluster.cpp

# Octave bindings
octave-bindings: lso_cluster.oct
lso_cluster.oct: lso_cluster.cpp $(HEADERS)

# Matlab bindings
matlab-bindings: lso_cluster.mex
lso_cluster.mex: lso_cluster.mex.cpp $(HEADERS)
