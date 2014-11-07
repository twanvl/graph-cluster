// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#include <octave/oct.h>
#include <octave/Cell.h>
#include <octave/parse.h>
#include "lso_cluster.hpp"
#include "lso_argument_parser.hpp"
#include "argument_parser_octave.hpp"
#include <memory>

using std::vector;
using std::map;
using namespace lso_cluster;

// -----------------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------------

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
