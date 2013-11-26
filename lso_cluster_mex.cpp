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
#include "lso_argument_parser.hpp"
#include "argument_parser_mex.hpp"

using namespace lso_cluster;
using namespace std;

// matlab already catches exceptions for us
#define CATCH_EXCEPTIONS 0

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
	
	// loosely based on findclusunities/clusunity.cpp
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
		LsoMainFunction runner(mex_output);
		runner.add_all_parameters(param_source);
		runner.run();
		// store outputs
		if (nlhs > 0) plhs[0] = to_mex(runner.clustering);
		if (nlhs > 1) plhs[1] = mxCreateDoubleScalar(runner.loss);
		if (nlhs > 2) plhs[2] = mxCreateDoubleScalar(runner.num_clusters);
	#if CATCH_EXCEPTIONS
	} catch (std::exception const& e) {
		mexErrMsgTxt(e.what());
	} catch (...) {
		mexErrMsgTxt("Unexpected error");
	}
	#endif
}
