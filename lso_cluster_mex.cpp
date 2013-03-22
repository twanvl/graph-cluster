// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#include <mex.h>

#include <memory>
#include <algorithm>
#include <ostream>
#include <vector>
#include <map>

#include "lso_cluster.hpp"
#include "argument_parser.hpp"

using namespace lso_cluster;
using namespace std;

// -----------------------------------------------------------------------------
// Interrupt handling
// -----------------------------------------------------------------------------

#if USE_LIBUT
	extern bool utIsInterruptPending();
	#define OCTAVE_QUIT \
		if (utIsInterruptPending()) throw std::runtime_error("interrupted");
#else
	#define OCTAVE_QUIT do{}while(0)
#endif

// -----------------------------------------------------------------------------
// The implementation
// -----------------------------------------------------------------------------

#include "lso_cluster_impl.hpp"

// -----------------------------------------------------------------------------
// Conversion
// -----------------------------------------------------------------------------

SparseMatrix sparse_from_mex(const mxArray* ptr) {
	if (mxIsComplex(ptr)) throw std::invalid_argument("A graph should be a real numeric matrix");
	if (mxIsSparse(ptr)) {
		if (mxIsDouble(ptr)) {
			return SparseMatrix(
				mxGetM(ptr), mxGetN(ptr), mxGetNzmax(ptr),
				(int*)mxGetJc(ptr),
				(int*)mxGetIr(ptr),
				(double*)mxGetPr(ptr));
		} else if (mxIsLogical(ptr)) {
			return SparseMatrix(mxGetM(ptr), mxGetN(ptr), mxGetNzmax(ptr), (int*)mxGetJc(ptr), (int*)mxGetIr(ptr), (mxLogical*)mxGetPr(ptr));
		}
	} else {
		if (mxIsDouble(ptr)) {
			return SparseMatrix::from_dense(mxGetM(ptr), mxGetN(ptr), (double*)mxGetData(ptr));
		} else if (mxIsLogical(ptr)) {
			return SparseMatrix::from_dense(mxGetM(ptr), mxGetN(ptr), (mxLogical*)mxGetData(ptr));
		}
	}
	throw std::invalid_argument("Expected a sparse array of double values");
}

vector<clus_t> clustering_from_mex(const mxArray* ptr) {
	if (mxIsSparse(ptr))  throw std::invalid_argument("A clustering should be a dense matrix"); 
	if (mxIsComplex(ptr)) throw std::invalid_argument("A clustering should be a real numeric matrix");
	if (mxGetM(ptr) != 1 && mxGetN(ptr) != 1) throw std::invalid_argument("A clustering should be a one dimensional array");
	size_t n = mxGetNumberOfElements(ptr);
	switch (mxGetClassID(ptr)) {
		case mxCHAR_CLASS:   return clustering_from_array((mxChar   const*)mxGetData(ptr), n);
		case mxINT8_CLASS:   return clustering_from_array((int8_T   const*)mxGetData(ptr), n);
		case mxINT32_CLASS:  return clustering_from_array((int32_T  const*)mxGetData(ptr), n);
		case mxDOUBLE_CLASS: return clustering_from_array((double   const*)mxGetData(ptr), n);
		case mxSINGLE_CLASS: return clustering_from_array((float    const*)mxGetData(ptr), n);
		default:;
	}
	throw std::invalid_argument("Expected a one dimensional array of numeric values");
}

double double_from_mex(const mxArray* ptr) {
	throw "ARK";
}
int int_from_mex(const mxArray* ptr) {
	throw "BORK";
}
string string_from_mex(const mxArray* ptr) {
	mexPrintf("string_from_mex");
	char* str = mxArrayToString(ptr);
	mexPrintf("string_from_mex: %s",str);
	if (str == 0) {
		throw std::invalid_argument("Expected a string");
	}
	string out(str);
	mxFree(str);
	return out;
}

mxArray* to_mex(vector<clus_t> const& x) {
	int dims[2] = {(int)x.size(),1};
	mxArray* y = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
	int32_T* y_values = (int32_T*)mxGetData(y);
	std::copy(x.begin(), x.end(), y_values);
	return y;
}

// (inefficient) wrapper for ostream that uses mexPrintf
struct MexOstreambuf : public std::streambuf {
	virtual int overflow (int c) {
		mexPrintf("%c",c);
		return 0;
	}
};


struct ParamSourceMatlab : ParamSource {
  private:
	int argc, i;
	const mxArray** argv;
	const mxArray* next() {
		if (i < argc) {
			return argv[i++];
		} else {
			throw std::invalid_argument("Expected an additional argument");
		}
	}
  public:
	ParamSourceMatlab(int argc, const mxArray** argv)
		: argc(argc), i(0), argv(argv)
	{}
	virtual bool end() {
		return i >= argc;
	}
	virtual int get_int_argument() {
		return int_from_mex(next());
	}
	virtual string get_string_argument(vector<double>* more_out = 0) {
		// optionally: a cell array with multiple arguments
		return string_from_mex(next());
	}
	virtual double get_double_argument() {
		return double_from_mex(next());
	}
	virtual vector<clus_t> get_1dvec_argument() {
		return clustering_from_mex(next());
	}
	virtual SparseMatrix get_matrix_argument() {
		return sparse_from_mex(next());
	}
};

// -----------------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------------

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
	// repeatability
	srand(100);
	
	// loosely based on findclusunities/clusunity.cpp
	if (nrhs == 0) {
		mexPrintf("%s",USAGE_INFO);
		return;
	}
	
	try {
		// parse arguments, and run the clustering algorithm
		MexOstreambuf mex_streambuf;
		ostream mex_output(&mex_streambuf);
		ParamSourceMatlab param_source(nrhs,prhs);
		LsoMainFunction runner(mex_output);
		runner.add_all_parameters(param_source);
		runner.run();
		// store outputs
		if (nlhs > 0) plhs[0] = to_mex(runner.clustering);
		if (nlhs > 1) plhs[1] = mxCreateDoubleScalar(runner.loss);
		if (nlhs > 2) plhs[2] = mxCreateDoubleScalar(runner.clustering.size());
	} catch (std::exception const& e) {
		mexErrMsgTxt(e.what());
	} catch (...) {
		mexErrMsgTxt("Unexpected error");
	}
}
