// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#ifndef HEADER_LSO_CLUSTER_ARGUMENT_PARSER_OCTAVE
#define HEADER_LSO_CLUSTER_ARGUMENT_PARSER_OCTAVE

#include "argument_parser.hpp"
#include <stdexcept>
#include <cstdio>

namespace lso_cluster {

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
// Conversion
// -----------------------------------------------------------------------------

SparseMatrix sparse_from_mex(const mxArray* ptr) {
	if (mxIsComplex(ptr)) throw std::invalid_argument("A graph should be a real numeric matrix");
	if (mxIsSparse(ptr)) {
		if (mxIsDouble(ptr)) {
			return SparseMatrix(
				mxGetM(ptr), mxGetN(ptr), mxGetNzmax(ptr),
				mxGetJc(ptr),
				mxGetIr(ptr),
				(double*)mxGetPr(ptr));
		} else if (mxIsLogical(ptr)) {
			return SparseMatrix(mxGetM(ptr), mxGetN(ptr), mxGetNzmax(ptr), mxGetJc(ptr), mxGetIr(ptr), (mxLogical*)mxGetPr(ptr));
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
	if (mxGetNumberOfElements(ptr) != 1) throw std::invalid_argument("Expected a scalar value");
	return mxGetScalar(ptr);
}
int int_from_mex(const mxArray* ptr) {
	double v = double_from_mex(ptr);
	return (int)round(v);
}
string string_from_mex(const mxArray* ptr) {
	char* str = mxArrayToString(ptr);
	if (str == 0) {
		throw std::invalid_argument("Expected a string");
	}
	string out(str);
	mxFree(str);
	return out;
}

mxArray* to_mex(vector<clus_t> const& x) {
	mwSize dims[2] = {(int)x.size(),1};
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
		const mxArray* ptr = next();
		if (mxIsCell(ptr)) {
			int n = mxGetNumberOfElements(ptr);
			if (n == 0) {
				throw std::invalid_argument("Use a cell {'name',args,..} for loss functions with arguments");
			}
			for (int i = 1 ; i < n ; ++i) {
				more_out->push_back(double_from_mex(mxGetCell(ptr,i)));
			}
			return string_from_mex(mxGetCell(ptr,0));
		} else {
			return string_from_mex(ptr);
		}
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
}
#endif
