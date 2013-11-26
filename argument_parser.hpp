// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#ifndef HEADER_LSO_CLUSTER_ARGUMENT_PARSER
#define HEADER_LSO_CLUSTER_ARGUMENT_PARSER

#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include "sparse_matrix.hpp"
#include "util.hpp"

namespace lso_cluster {

class LossFunction;
using std::string;
using std::vector;
using boost::shared_ptr;

typedef int clus_t;

// -----------------------------------------------------------------------------
// Argument sources, can be overloaded for different interfaces
// -----------------------------------------------------------------------------

struct ArgSource {
	virtual string         get_string_argument(vector<double>* more_out = 0) = 0;
	virtual double         get_double_argument() = 0;
	virtual int            get_int_argument() {
		return get_double_argument();
	}
	virtual bool           get_bool_argument() {
		return get_int_argument();
	}
	virtual vector<clus_t> get_1dvec_argument() = 0;
	virtual SparseMatrix   get_matrix_argument() = 0;
	// try to interpret the argument as a loss function
	// doesn't consume the argument on failure
	virtual shared_ptr<LossFunction> try_get_loss_function() {
		return shared_ptr<LossFunction>();
	}
};

struct ParamSource : ArgSource {
	virtual bool end() = 0;
	virtual string get_parameter_name() {
		return get_string_argument();
	}
};

// -----------------------------------------------------------------------------
// Argument parsing, shared by all interfaces
// -----------------------------------------------------------------------------

void normalize_key(string& key) {
	for (string::iterator it = key.begin() ; it != key.end() ; ++it) {
		if (*it == ' ' || *it == '-') *it = '_';
	}
}

// -----------------------------------------------------------------------------
}
#endif
