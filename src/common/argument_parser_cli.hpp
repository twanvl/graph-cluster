// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#ifndef HEADER_LSO_CLUSTER_ARGUMENT_PARSER_CLI
#define HEADER_LSO_CLUSTER_ARGUMENT_PARSER_CLI

#include "argument_parser.hpp"
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <map>

namespace lso_cluster {

using boost::lexical_cast;
using std::map;
using namespace std;

// -----------------------------------------------------------------------------
// File formats
// -----------------------------------------------------------------------------

template <typename T>
struct LabeledSparseMatrix : public SparseMatrix {
	const vector<T> labels;
	LabeledSparseMatrix(vector<T> const& labels, int nnz)
		: SparseMatrix(labels.size(), labels.size(), nnz)
		, labels(labels)
	{}
};

template <typename T>
struct LabeledGraph {
	map<T,map<T,double> > nodes;
	void add_nodes(T i) {
		nodes[i]; // make it exist
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
	LabeledSparseMatrix<string> to_sparse_matrix() const {
		// find unique ids
		map<T,int> ids;
		vector<string> labels;
		for (typename map<T,map<T,double> >::const_iterator it = nodes.begin() ; it != nodes.end() ; ++it) {
			ids[it->first] = (int)labels.size();
			labels.push_back(lexical_cast<string>(it->first));
		}
		// to graph
		LabeledSparseMatrix<string> out(labels, nnz());
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
template <> void LabeledGraph<int>::add_nodes(int i) {
	// make all nodes [0..i] exist
	for (int k = (int)nodes.size() ; k <= i ; ++k) {
		nodes[k]; // make it exist
	}
}

enum GraphNodeType {
	GRAPH_INT, GRAPH_STRING
};

template <typename T> LabeledSparseMatrix<string> load_graph(istream& in) {
	LabeledGraph<T> graph;
	while (in) {
		T i, j;
		double w = -1;
		in >> i >> j >> w;
		if (in) {
			graph.add(i,j,w);
		}
	}
	return graph.to_sparse_matrix();
}
LabeledSparseMatrix<string> load_graph(istream& in, GraphNodeType node_type) {
	if (node_type == GRAPH_INT) return load_graph<int>(in);
	else return load_graph<string>(in);
}
LabeledSparseMatrix<string> load_graph(string const& fn, GraphNodeType node_type) {
	if (fn == "-") {
		return load_graph(cin, node_type);
	} else {
		ifstream fs(fn.c_str());
		if (!fs.good()) {
			throw std::runtime_error("Unable to open file: " + fn);
		}
		return load_graph(fs, node_type);
	}
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

void print_clustering(ostream& out, vector<string> const& labels, vector<clus_t> const& clustering) {
	for (size_t i = 0 ; i < clustering.size() ; ++i) {
		if (i < labels.size()) out << labels[i] << "\t";
		out << clustering[i] << endl;
	}
}
void print_clustering(std::string const& fn, vector<string> const& labels, vector<clus_t> const& clustering) {
	if (fn == "-" || fn.empty()) {
		print_clustering(cout, labels, clustering);
	} else {
		ofstream fs(fn.c_str());
		if (!fs.good()) {
			throw std::runtime_error("Unable to open file: " + fn);
		}
		print_clustering(fs, labels, clustering);
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
	GraphNodeType graph_node_type;
	
	ParamSourceCommandline(int argc, const char** argv)
		: argc(argc), i(0), argv(argv), graph_node_type(GRAPH_STRING)
	{}
	virtual bool end() {
		return i >= argc;
	}
	virtual string try_get_parameter_name() {
		// parameters are indicated by "--OPT"
		if (i >= argc) return "";
		const char* opt = argv[i];
		if (opt[0] == '-' && opt[1] == '-') {
			// negated boolean flags
			if (opt[2] == 'n' && opt[3] == 'o' && opt[4] == '-') {
				return opt+5;
			}
			// normal arguments
			next();
			return opt+2;
		} else if (opt[0] == '-' && opt[1] == 'o' && opt[2] == 0) {
			next();
			return "out";
		} else if (opt[0] == '-' && opt[1] == 'q' && opt[2] == 0) {
			next();
			return "quiet";
		} else {
			return "";
		}
	}
	virtual string get_parameter_name() {
		// parameters are indicated by "--OPT"
		string param = try_get_parameter_name();
		if (param.empty()) {
			throw std::invalid_argument("Expected an optional parameter name ('--something')");
		} else {
			return param;
		}
	}
	virtual string get_string_argument(vector<double>* more_out = 0) {
		const char* s = next();
		if (i + 1 < argc) {
			// grab extra string data
			more_out->clear();
			for (int j = i; j < argc && argv[j][0] != '-'; ++j) {
				more_out->push_back(atof(argv[j]));
			}
			i += more_out->size();
		}
		return s;
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
		return get_labeled_matrix_argument();
	}
	virtual LabeledSparseMatrix<string> get_labeled_matrix_argument() {
		return load_graph(next(), graph_node_type);
	}
};

// -----------------------------------------------------------------------------
}
#endif
