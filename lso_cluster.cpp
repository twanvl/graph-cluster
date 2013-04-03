// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#include "lso_cluster.hpp"
#include "argument_parser.hpp"
#include <vector>
#include <map>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <boost/lexical_cast.hpp>

#define OCTAVE_QUIT do{}while(0)
#include "lso_cluster_impl.hpp"

using namespace std;
using namespace lso_cluster;
using boost::lexical_cast;

// -----------------------------------------------------------------------------
// File formats
// -----------------------------------------------------------------------------

template <typename T>
struct LabeledGraph {
	map<T,map<T,double> > nodes;
	void add_nodes(int i) {
		for (int k = (int)nodes.size() ; k <= i ; ++k) {
			nodes[k]; // make it exist
		}
	}
	void add(T i, T j, double w) {
		add_nodes(i);
		add_nodes(j);
		if (w != 0.0) {
			nodes[i][j] = w;
			nodes[j][i] = w; // symmetric
		}
	}
	int size() const {
		return (int)nodes.size();
	}
	int nnz() const {
		size_t nnz = 0;
		for (typename map<T,map<T,double> >::const_iterator it = nodes.begin() ; it != nodes.end() ; ++it) {
			nnz += it->second.size();
		}
		return (int)nnz;
	}
	SparseMatrix to_sparse_matrix() const {
		// find unique ids
		int id = 0;
		map<T,int> ids;
		for (typename map<T,map<T,double> >::const_iterator it = nodes.begin() ; it != nodes.end() ; ++it) {
			ids[it->first] = id++;
		}
		// to graph
		SparseMatrix out(size(), size(), nnz());
		int k = 0;
		for (typename map<T,map<T,double> >::const_iterator it = nodes.begin() ; it != nodes.end() ; ++it) {
			out.cidx(ids[it->first]) = k;
			for (typename map<T,double>::const_iterator jt = it->second.begin() ; jt != it->second.end() ; ++jt) {
				out.ridx(k) = ids[jt->first];
				out.data(k) = jt->second;
				k++;
			}
		}
		out.cidx(size()) = k;
		return out;
	}
};
SparseMatrix load_graph(istream& in) {
	LabeledGraph<int> graph;
	while (in) {
		int i = -1, j = -1;
		double w = -1;
		in >> i >> j >> w;
		if (i != -1 && j != -1 && w != -1) {
			graph.add(i,j,w);
		}
	}
	return graph.to_sparse_matrix();
}
vector<clus_t> load_clustering(istream& in) {
	vector<string> labels;
	while (in) {
		string str;
		getline(in, str);
		if (!str.empty()) {
			labels.push_back(str);
		}
	}
	return clustering_from_array(&labels[0], labels.size());
}

SparseMatrix load_graph(string const& fn) {
	if (fn == "-") {
		return load_graph(cin);
	} else {
		ifstream fs(fn.c_str());
		if (!fs.good()) {
			throw std::runtime_error("Unable to open file: " + fn);
		}
		return load_graph(fs);
	}
}
vector<clus_t> load_clustering(string const& fn) {
	if (fn == "-") {
		return load_clustering(cin);
	} else {
		ifstream fs(fn.c_str());
		if (!fs.good()) {
			throw std::runtime_error("Unable to open file: " + fn);
		}
		return load_clustering(fs);
	}
}

// -----------------------------------------------------------------------------
// Command line parsing
// -----------------------------------------------------------------------------

struct ParamSourceCommandline : ParamSource {
  private:
	int argc, i;
	char const** argv;
	const char* next() {
		if (i < argc) {
			return argv[i++];
		} else {
			throw std::invalid_argument("Expected an additional argument");
		}
	}
  public:
	ParamSourceCommandline(int argc, const char** argv)
		: argc(argc), i(0), argv(argv)
	{}
	virtual bool end() {
		return i >= argc;
	}
	virtual string get_parameter_name() {
		// parameters are indicated by "--OPT"
		const char* opt = next();
		if (opt[0] == '-' && opt[1] == '-') {
			// negated boolean flags
			if (opt[2] == 'n' && opt[3] == 'o' && opt[4] == '-') {
				return opt+5;
			}
			// normal arguments
			return opt+2;
		} else {
			throw std::invalid_argument("Expected an optional parameter name ('--something')");
		}
	}
	virtual string get_string_argument(vector<double>* more_out = 0) {
		return next();
	}
	virtual int get_int_argument() {
		return lexical_cast<int>(next());
	}
	virtual bool get_bool_argument() {
		if (i >= argc || argv[i][0] == '-') {
			if (i > 0 && i+1 < argc && argv[i-1][0] == '-' && argv[i-1][0] == '-' && argv[i-1][0] == 'n' && argv[i-1][0] == 'o') {
				return false;
			} else {
				return true;
			}
		}
		char const* arg = next();
		if (strcmp(arg,"yes") == 0 || strcmp(arg,"true") == 0 || strcmp(arg,"1") == 0) return true;
		if (strcmp(arg,"no") == 0 || strcmp(arg,"false") == 0 || strcmp(arg,"0") == 0) return false;
		throw std::invalid_argument("Expected a boolean value");
	}
	virtual double get_double_argument() {
		return lexical_cast<double>(next());
	}
	virtual vector<clus_t> get_1dvec_argument() {
		return load_clustering(next());
	}
	virtual SparseMatrix get_matrix_argument() {
		return load_graph(next());
	}
};


// -----------------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------------

void print_usage(char const* argv0) {
	cout << "Usage: " << argv0 << " <FILE> [OPTIONS]" << endl;
}

int main(int argc, char const** argv) {
	if (argc == 1) {
		print_usage(argv[0]);
		return EXIT_SUCCESS;
	}
	try {
		// parse arguments, and run the clustering algorithm
		ParamSourceCommandline param_source(argc-1,argv+1);
		LsoMainFunction runner(cout);
		runner.add_all_parameters(param_source);
		runner.run();
		
		// store outputs
		cerr << "loss: " << runner.loss << endl;
		cerr << "num clusters: " << runner.num_clusters << endl;
		for (size_t i = 0 ; i < runner.clustering.size() ; ++i) {
			cout << runner.clustering[i] << endl;
		}
		
	} catch (std::exception const& e) {
		cerr << e.what() << endl;
		return EXIT_FAILURE;
	} catch (...) {
		cerr << "Unexpected error" << endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
