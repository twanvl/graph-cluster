// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#include <octave/oct.h>
#include <octave/Cell.h>
#include "lso_cluster.hpp"
#include "lso_cluster_impl.hpp"
#include "argument_parser.hpp"
#include <memory>

using std::vector;
using std::map;
using namespace lso_cluster;

// -----------------------------------------------------------------------------
// Implementation
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
};


DEFUN_DLD(lso_cluster,args,nargout,USAGE_INFO){
	// loosely based on findclusunities/clusunity.cpp
	if (args.length() < 1) {
		print_usage();
		return octave_value_list();
	}
	try {
		// parse arguments, and run the clustering algorithm
		ParamSourceOctave param_source(args);
		LsoMainFunction runner(octave_stdout);
		runner.add_all_parameters(param_source);
		runner.run();
		
		// return
		octave_value_list retval;
		retval(0) = to_octave(runner.clustering);
		retval(1) = runner.loss;
		retval(2) = runner.num_clusters;
		return retval;
		
	} catch (std::exception const& e) {
		error("%s",e.what());
	} catch (...) {
		error("Unexpected error");
	}
	return octave_value_list();
}

