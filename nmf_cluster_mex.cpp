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

#include "argument_parser_mex.hpp"
#include "nmf_cluster.hpp"
#include "nmf_argument_parser.hpp"

using namespace nmf_cluster;
using namespace lso_cluster;
using namespace std;

// matlab already catches exceptions for us
#ifndef CATCH_EXCEPTIONS
#define CATCH_EXCEPTIONS 0
#endif

// -----------------------------------------------------------------------------
// The implementation
// -----------------------------------------------------------------------------

#include "lso_cluster_impl.hpp"

// -----------------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------------

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
	// repeatability
	srand(100);
	
	if (nrhs == 0) {
		mexPrintf("%s",USAGE_INFO);
		return;
	}
	
	#if CATCH_EXCEPTIONS
	try {
	#endif
		// parse arguments, and run the clustering algorithm
		MexOstreambuf mex_streambuf;
		ostream mex_output(&mex_streambuf);
		ParamSourceMatlab param_source(nrhs,prhs);
		NmfMainFunction runner(mex_output);
		runner.add_all_parameters(param_source);
		runner.run();
		// store outputs
		if (nlhs > 0) plhs[0] = to_mex(runner.clustering);
		if (nlhs > 1) plhs[1] = mxCreateDoubleScalar(runner.loss);
		if (nlhs > 2) plhs[2] = mxCreateDoubleScalar(runner.clustering.rows());
	#if CATCH_EXCEPTIONS
	} catch (std::exception const& e) {
		mexErrMsgTxt(e.what());
	} catch (...) {
		mexErrMsgTxt("Unexpected error");
	}
	#endif
}
