// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#include "argument_parser_cli.hpp"
#include "lso_argument_parser.hpp"
#include "lso_cluster.hpp"
#include <vector>
#include <map>
#include <stdio.h>
#include <cstring>
#include <cstdlib>

#define OCTAVE_QUIT do{}while(0)
#include "lso_cluster_impl.hpp"

using namespace std;
using namespace lso_cluster;
using boost::lexical_cast;

// -----------------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------------

struct LsoMainFunctionCommandLine : LsoMainFunction {
	// extra options
	string output_file;
	bool quiet;
	// labels of the graph
	vector<string> labels;
	
	LsoMainFunctionCommandLine()
		: LsoMainFunction(cerr)
		, quiet(false)
	{}
	
	virtual void add_parameter(string const& key, ArgSource& args) {
		if (key == "out") {
			output_file = args.get_string_argument();
		} else if (key == "quiet") {
			quiet = true;
		} else if (key == "numeric") {
			static_cast<ParamSourceCommandline&>(args).graph_node_type = GRAPH_INT;
		} else {
			LsoMainFunction::add_parameter(key,args);
		}
	}
	
	virtual void add_all_parameters(ParamSource& args) {
		ParamSourceCommandline& pargs = static_cast<ParamSourceCommandline&>(args);
		// arguments with "-" are optional parameters, they can appear before the graph filename
		while (!pargs.end()) {
			string key = pargs.try_get_parameter_name();
			if (key.empty()) break; // doesn't start with "-"
			normalize_key(key);
			add_parameter(key, pargs);
		}
		// first non "-" parameter is the graph file
		LabeledSparseMatrix<string> lgraph = pargs.get_labeled_matrix_argument();
		graph = lgraph;
		labels = lgraph.labels;
		// then come the optional parameters
		add_optional_parameters(pargs);
	}
};


void print_usage(char const* argv0) {
	cout << "Usage: " << argv0 << " <FILE> [OPTIONS]" << endl;
}

int main(int argc, char const** argv) {
	if (argc == 1) {
		print_usage(argv[0]);
		return EXIT_SUCCESS;
	}
	try {
		// parse arguments, and run the clustering algorithm
		ParamSourceCommandline param_source(argc-1,argv+1);
		LsoMainFunctionCommandLine runner;
		runner.add_all_parameters(param_source);
		runner.run();
		
		// store outputs
		if (!runner.quiet) {
			cerr << "loss: " << runner.loss << endl;
			cerr << "num clusters: " << runner.num_clusters << endl;
		}
		print_clustering(runner.output_file, runner.labels, runner.clustering);
		
	} catch (std::exception const& e) {
		cerr << e.what() << endl;
		return EXIT_FAILURE;
	} catch (...) {
		cerr << "Unexpected error" << endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
