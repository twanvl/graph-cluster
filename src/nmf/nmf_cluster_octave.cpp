// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#include <octave/oct.h>
#include <octave/Cell.h>
#include <octave/parse.h>
#include "nmf_cluster.hpp"
#include "nmf_argument_parser.hpp"
#include "argument_parser_octave.hpp"
#include "nmf_cluster_impl.cpp"
#include <memory>

using std::vector;
using std::map;
using namespace lso_cluster;
using namespace nmf_cluster;

// -----------------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------------

DEFUN_DLD(nmf_cluster,args,nargout,USAGE_INFO){
	// loosely based on findclusunities/clusunity.cpp
	if (args.length() < 1) {
		print_usage();
		return octave_value_list();
	}
	try {
		// parse arguments, and run the clustering algorithm
		ParamSourceOctave param_source(args);
		NmfMainFunction runner(octave_stdout);
		runner.add_all_parameters(param_source);
		runner.run();
		
		// return
		octave_value_list retval;
		retval(0) = runner.clustering;
		retval(1) = runner.loss;
		retval(2) = runner.clustering.cols();
		return retval;
		
	} catch (std::exception const& e) {
		error("%s",e.what());
	} catch (...) {
		error("Unexpected error");
	}
	return octave_value_list();
}
