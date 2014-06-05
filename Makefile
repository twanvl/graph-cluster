
all: stand-alone

clean:
	rm -rf *.o
	rm -rf *.oct
	rm -rf *.mex

.PHONY: stand-alone octave-bindings matlab-bindings

# Dependencies
HEADERS = trace_file_io.cpp argument_parser.hpp argument_parser_cli.hpp lso_argument_parser.hpp util.hpp sparse_matrix.hpp
LSO_HEADERS = $(HEADERS) lso_cluster.hpp lso_cluster_impl.hpp loss_functions.hpp
NMF_HEADERS = $(HEADERS) nmf_cluster.hpp
#CXXFLAGS = -O2 -Wall -g
CXXFLAGS = -g -Wall

# Stand alone program
stand-alone: lso_cluster nmf_cluster
lso_cluster: lso_cluster.cpp $(LSO_HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ lso_cluster.cpp
nmf_cluster: nmf_cluster.cpp $(NMF_HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ nmf_cluster.cpp

# Octave bindings
octave-bindings: lso_cluster.oct
lso_cluster.oct: lso_cluster_octave.cpp $(LSO_HEADERS)
	mkoctfile $< -o $@

# Matlab bindings
matlab-bindings: lso_cluster.mex
lso_cluster.mex: lso_cluster_mex.cpp $(LSO_HEADERS)
	mex $< -o $@

