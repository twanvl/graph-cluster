// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#ifndef HEADER_LSO_CLUSTER_ARGUMENT_PARSER_OCTAVE
#define HEADER_LSO_CLUSTER_ARGUMENT_PARSER_OCTAVE

#include <octave/oct.h>
#include <octave/Cell.h>
#include <octave/parse.h>
#include "argument_parser.hpp"

namespace lso_cluster {

// -----------------------------------------------------------------------------
// Argument parsing for octave
// -----------------------------------------------------------------------------

ColumnVector to_octave(vector<int> const& x) {
	ColumnVector y(x.size());
	for (size_t i = 0 ; i < x.size() ; ++i) {
		y(i) = x[i];
	}
	return y;
}

vector<int> clustering_from_octave(Matrix const& m) {
	vector<int> clus(m.nelem());
	map<double,int> first_in_clus;
	for (size_t i = 0 ; i < clus.size() ; ++i) {
		map<double,int>::const_iterator it = first_in_clus.find(m(i));
		if (it == first_in_clus.end()) {
			first_in_clus[m(i)] = i;
			clus[i] = i;
		} else {
			clus[i] = it->second;
		}
	}
	return clus;
}

struct OctaveLossFunction : LossFunction {
	octave_function* fn;
	OctaveLossFunction(octave_function* fn) : fn(fn) {}
	// we give the entire clustering to the octave function
	virtual bool want_entire_clustering() const {
		return true;
	}
	virtual double loss_entire(vector<clus_t> const& clustering) const {
		octave_value_list args;
		args(0) = to_octave(clustering);
		octave_value_list retval = feval(fn, args, 1);
		if (retval.length() < 1) {
			throw std::runtime_error("Missing result in custom loss function");
		}
		return retval(0).double_value();
	}
};

struct ParamSourceOctave : ParamSource {
  private:
	octave_value_list args;
	int i;
	const octave_value& next() {
		if (i < args.length()) {
			return args(i++);
		} else {
			throw std::invalid_argument("Expected an additional argument");
		}
	}
  public:
	ParamSourceOctave(octave_value_list const& args)
		: args(args), i(0)
	{}
	virtual bool end() {
		return i >= args.length();
	}
	virtual string get_string_argument(vector<double>* more_out = 0) {
		if (more_out && i < args.length() && args(i).is_cell()) {
			// optionally: a cell array with multiple arguments
			Cell more = next().cell_value();
			if (more.nelem() == 0) {
				throw std::invalid_argument("Use a cell {'name',args,..} for loss functions with arguments");
			}
			for (int i = 1 ; i < more.nelem() ; ++i) {
				more_out->push_back(more(i).double_value());
			}
			return more(0).string_value();
		} else {
			return next().string_value();
		}
	}
	virtual double get_double_argument() {
		return next().double_value();
	}
	virtual int get_int_argument() {
		return next().int_value();
	}
	virtual bool get_bool_argument() {
		return next().bool_value();
	}
	virtual vector<clus_t> get_1dvec_argument() {
		return clustering_from_octave(next().matrix_value());
	}
	virtual SparseMatrix get_matrix_argument() {
		return next().sparse_matrix_value();
	}
	virtual shared_ptr<LossFunction> try_get_loss_function() {
		// try to interpret the argument as a loss function
		const octave_value& val = next();
		if (val.is_function_handle() || val.is_inline_function()) {
			octave_function* fn = val.function_value();
			if (!error_state) {
				return shared_ptr<LossFunction>(new OctaveLossFunction(fn));
			}
		}
		i--; // failed to parse
		return shared_ptr<LossFunction>();
	}
};

// -----------------------------------------------------------------------------
}
#endif
