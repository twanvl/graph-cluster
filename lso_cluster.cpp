// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#include "lso_cluster.hpp"
#include "trace_file_io.cpp"
#include "loss_functions.hpp"
#include <vector>
#include <map>
#include <iostream>

#define OCTAVE_QUIT do{}while(0)
#include "lso_cluster_impl.hpp"

using namespace std;

// -----------------------------------------------------------------------------
// File formats
// -----------------------------------------------------------------------------

struct NodeInfo {
	string id;
	string label;
	map<string, double> weight_to;
};
SparseMatrix load_graph(istream& in) {
	std::map<string, NodeInfo> graph;
	SparseMatrix out(nodes.size(), nodes.size(), nnzs);
	
}

struct ParamSourceCommandline : ParamSource {
};


// -----------------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------------

void print_usage(char const* argv0) {
}

int main(int argc, char const** argv) {
	// Parse command line
	for (int i = 1 ; i < argc ; ++i) {
		
	}
	return 0;
}
