// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#ifndef HEADER_NMF_CLUSTER_NMF_ARGUMENT_PARSER
#define HEADER_NMF_CLUSTER_NMF_ARGUMENT_PARSER

#include "nmf_cluster.hpp"
#include "argument_parser.hpp"

namespace nmf_cluster {

// -----------------------------------------------------------------------------
// Argument parsing, shared by all interfaces
// -----------------------------------------------------------------------------

LikelihoodFun parse_likelihood(string const& x) {
	if (x == "flat") return LH_FLAT;
	if (x == "Gaussian" || x == "gaussian" || x == "gauss" || x == "euclidean" || x == "l2") return LH_GAUSSIAN;
	if (x == "Poisson" || x == "poisson") return LH_POISSON;
	throw std::invalid_argument("Unknown likelihood type: " + x);
}
WeightPriorFun parse_weight_prior(string const& x) {
	if (x == "flat") return PRIOR_FLAT;
	if (x == "hn" || x == "HN" || x == "half_normal") return PRIOR_HALF_NORMAL;
	if (x == "gamma") return PRIOR_GAMMA;
	throw std::invalid_argument("Unknown weight prior: " + x);
}
SizePriorFun parse_size_prior(string const& x) {
	if (x == "flat") return SIZE_FLAT;
	if (x == "crp" || x == "CRP") return SIZE_CRP;
	throw std::invalid_argument("Unknown cluster size prior: " + x);
}
SupportPriorFun parse_support_prior(string const& x) {
	if (x == "flat") return SUPPORT_FLAT;
	if (x == "one" || x == "hard") return SUPPORT_ONE;
	if (x == "Poisson" || x == "poisson") return SUPPORT_POISSON;
	throw std::invalid_argument("Unknown node support prior: " + x);
}

struct NmfMainFunction {
	// The configuration
	SparseMatrix graph;
	NMFParams params;
	bool optimize;
	int seed;
	// The output
	NMFClustering clustering;
	double loss;
	vector<double> losses;
	
	// Defaults
	NmfMainFunction(NMFParams const& params)
		: params(params)
		, optimize(true)
		, seed(1234567)
		, loss(0)
	{}
	
	// Argument parsing
	virtual void add_all_parameters(ParamSource& args) {
		// first parameter is the graph
		graph = args.get_matrix_argument();
		// then come the optional parameters
		add_optional_parameters(args);
	}
	
	virtual void add_optional_parameters(ParamSource& args) {
		while (!args.end()) {
			string key = args.get_parameter_name();
			normalize_key(key);
			add_parameter(key, args);
		}
	}
	
	virtual void add_parameter(string const& key, ArgSource& args) {
		// parameters of the optimizer
		if (key == "verbose" || key == "verbosity") {
			params.verbosity = args.get_int_argument();
		} else if (key == "max_cluster_per_node" || key == "max_clusters_per_node" || key == "max_num_cluster_per_node" || key == "cluster_per_node") {
			params.max_cluster_per_node = args.get_int_argument();
		} else if (key == "num_iter" || key == "num_iterations") {
			params.num_iter = args.get_int_argument();
		} else if (key == "num_repeats" || key == "num_repeat" || key == "repeats") {
			params.num_repeats = args.get_int_argument();
		} else if (key == "seed") {
			seed = args.get_int_argument();
		
		} else if (key == "init" || key == "initial") {
			clustering = args.get_matrix_argument();
		} else if (key == "eval" || key == "evaluate") {
			clustering = args.get_matrix_argument();
			optimize = false;
		
		// parameters of the objective function
		} else if (key == "likelihood") {
			params.objective.likelihood = parse_likelihood(args.get_string_argument());
		} else if (key == "weight_prior" || key == "prior") {
			params.objective.weight_prior = parse_weight_prior(args.get_string_argument());
		} else if (key == "beta" || key == "weight_beta") {
			params.objective.weight_beta = args.get_double_argument();
		} else if (key == "size_prior") {
			params.objective.size_prior = parse_size_prior(args.get_string_argument());
		} else if (key == "support_prior" || key == "support") {
			params.objective.support_prior = parse_support_prior(args.get_string_argument());
		} else {
			throw std::invalid_argument("Unrecognized key: " + key);
		}
	}
	
	// Running
	
	void run() {
		// Do some validation of the arguments
		if (graph.rows() != graph.cols()) {
			throw std::invalid_argument("Matrix must be square");
		}
		if (graph.any_element_is_inf_or_nan()) {
			throw std::invalid_argument("Matrix includes Inf or NaN values");
		}
		if (graph.any_element_is_negative()) {
			throw std::invalid_argument("Matrix includes negative values");
		}
		if (!is_symmetric(graph)) {
			throw std::invalid_argument("Matrix must be symmetric");
		}
/*		if (!clustering.empty() && (int)clustering.size() != graph.rows()) {
			throw std::invalid_argument("Initial value must have same size as the matrix");
		}*/
		
		// initialize NmfOptimizer object
		srand(seed);
		NMFOptimizer optimizer(graph, params);
		if (!clustering.empty()) {
			// initial value
			optimizer.set_clustering(clustering);
		}
		
		// perform clustering
		if (optimize) {
			optimizer.run();
		}
		
		// outputs
		clustering = optimizer.get_clustering();
		loss = optimizer.get_loss();
		losses = optimizer.get_losses();
	}
};

// -----------------------------------------------------------------------------
// Usage info (matlab / octave)
// -----------------------------------------------------------------------------

#define USAGE_INFO \
	"Perform approximate symmetric Non-negative Matrix Factorization of a symmetric matrix.\n" \
	"i.e. approximate A = factor*factor'.\n" \
	"\n" \
	"Usage: \n" \
	"   [U,clus,loss,losses] = nmf_cluster(A, [varargin])\n" \
	"\n" \
	"Inputs:\n" \
	"   A:        The adjacency matrix of a graph.\n" \
	"              All entries must be non-negative and finite.\n" \
	"              A must be symmetric, i.e. A==A', this is not checked.\n" \
	"   varargin: Extra options as 'key',value pairs\n" \
	"\n" \
	"Outputs:\n" \
	"   U:        The factor matrix, it will have a row for each node, and a column for each cluster." \
	"   clus:     The index of the largest factor for each node\n" \
	"   loss:     Loss of the solution\n" \
	"   losses:   Loss after each iteration\n" \
	"\n" \
	"Extra options:\n" \
	"   'loss':             Loss function to use. (See below)\n" \
	"   'verbose':          Verbosity level, default 0\n" \
	"   'seed':             Random seed.\n" \
	"   'max_cluster_per_node': \n" \
	"                       The maximum number of clusters to which a node can be assigned.\n" \
	"   'eval':             Don't optimize, but evaluate loss on given clustering.\n" \
	"   'init':             Start from the given initial clustering.\n" \
	"\n" \
	"Loss functions:\n" \
	"   'square':  minimize  L(U) = (A - U*U')^2 + Regularizer(U)\n" \
	"\n"

// -----------------------------------------------------------------------------
}
#endif
