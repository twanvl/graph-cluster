// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#ifndef HEADER_LSO_CLUSTER
#define HEADER_LSO_CLUSTER

#define _USE_MATH_DEFINES
#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <limits>
#include <limits.h>
#include <math.h>
#include <boost/shared_ptr.hpp>
#include <boost/math/special_functions/fpclassify.hpp> // isnan / isinf

#ifndef INFINITY
#define INFINITY std::numeric_limits<double>::infinity()
#endif

#include "sparse_matrix.hpp"

namespace lso_cluster {

using namespace std;
using namespace boost::math;
using boost::shared_ptr;

// -----------------------------------------------------------------------------
// Nodes and clusters
// -----------------------------------------------------------------------------

typedef int node_t;
typedef int clus_t;

// -----------------------------------------------------------------------------
// cluster/node statistics
// -----------------------------------------------------------------------------

/// Statistics of a cluster or a single node.
/// these statistics can be used to calculate loss
struct Stats {
	double size;   ///< number of nodes in the cluster
	double degree; ///< sum of edge weights = (out)degrees
	double self;   ///< weight of self loops/within cluster edges
	/// Number of edges leaving the cluster
	inline double exit() const { return degree - self; }
	
	inline Stats()
		: size(0), degree(0), self(0)
	{}
	inline Stats(double size, double degree, double self)
		: size(size), degree(degree), self(self)
	{}
	inline Stats& operator += (Stats const& x) {
		size   += x.size;
		degree += x.degree;
		self   += x.self;
		return *this;
	}
	inline Stats& operator -= (Stats const& x) {
		size   -= x.size;
		degree -= x.degree;
		self   -= x.self;
		return *this;
	}
	inline Stats operator + (Stats const& x) const {
		return Stats(*this) += x;
	}
	inline Stats operator - (Stats const& x) const {
		return Stats(*this) -= x;
	}
};

/// Statistics for a cluster*ing*,
/// i.e. for each cluster, as well as the sum of those
class ClusteringStats : public vector<Stats> {
  public:
	Stats total; ///< Sum of stats for each cluster
};

// -----------------------------------------------------------------------------
// Loss functions
// -----------------------------------------------------------------------------

/// An array of some doubles, used for passing information from local to global loss function
struct Doubles {
  private:
	#define MAX_DOUBLES 2
	double x[MAX_DOUBLES];
  public:
	inline Doubles()                     { x[0] = 0.; x[1] = 0.; }
	inline Doubles(double x0)            { x[0] = x0; x[1] = 0.; }
	inline Doubles(double x0, double x1) { x[0] = x0; x[1] = x1; }
	inline double  operator [] (int i) const { return x[i]; }
	inline double& operator [] (int i) { return x[i]; }
	inline Doubles operator += (const Doubles& y) {
		for (int i = 0 ; i < MAX_DOUBLES ; ++i) x[i] += y[i];
		return *this;
	}
	inline Doubles operator -= (const Doubles& y) {
		for (int i = 0 ; i < MAX_DOUBLES ; ++i) x[i] -= y[i];
		return *this;
	}
};

/// A loss function
/** Loss is calculated as:
 *    loss = global(local(com1,total) + local(com2,total) + ...)
 */
struct LossFunction {
	/// Per cluster calculation of loss
	/// Requirements:
	///   * local(empty cluster) = 0
	///   * local does not depend on total.self
	virtual Doubles local(Stats const& clus, Stats const& total) const {
		return 0.;
	}
	virtual double global(Doubles const& sum_local, Stats const& total, int num_clusters) const {
		return sum_local[0];
	}
	
	/// Calculate loss for the given clustering statistics
	double loss(const ClusteringStats& stats, int num_clusters, Doubles* sum_local_out = NULL) const {
		Doubles sum_local;
		for (size_t i = 0 ; i < stats.size() ; ++i) {
			if (stats[i].size > 0) sum_local += local(stats[i], stats.total);
		}
		if (sum_local_out) *sum_local_out = sum_local;
		return global(sum_local, stats.total, num_clusters);
	}
	
	// alternatively, but slower, loss can be calculated based on an entire clustering
	virtual bool want_entire_clustering() const {
		return false;
	}
	virtual double loss_entire(vector<clus_t> const&) const {
		return 0;
	}
};

// -----------------------------------------------------------------------------
// Neighborhood accumulation
// -----------------------------------------------------------------------------

/// List of neighboring clusters of a node, and sum of edge weights to them
class Neighbors {
  private:
	vector<clus_t> clus_;   ///< list of neighboring clusters for current node
	vector<double> weight_; ///< weight == -INFINITY indicates not used
  public:
	Neighbors(int n)
		: weight_(n, -INFINITY)
	{}
	
	/// Iterate over all neighbors to which any weight was added
	inline vector<clus_t>::const_iterator begin() const {
		return clus_.begin();
	}
	inline vector<clus_t>::const_iterator end() const {
		return clus_.end();
	}
	/// Sort the neighboring clusters by index
	inline void sort() {
		std::sort(clus_.begin(), clus_.end());
	}
	
	inline size_t size() const {
		return clus_.size();
	}
    /// Get the weight to a particular cluster
    inline double weight(clus_t com) const {
		return max(0., weight_[com]);
	}
	
	/// Clear all weights
	inline void clear() {
		for (size_t idx = 0 ; idx < size() ; ++idx) {
			weight_[clus_[idx]] = -INFINITY;
		}
		clus_.clear();
	}
	/// Add weight to the link to cluster c, if c is not yet in the list of neighbors, add it.
	inline void add(clus_t c, double weight) {
		if (weight_[c] == -INFINITY) {
			clus_.push_back(c);
			weight_[c] = weight;
		} else {
			weight_[c] += weight;
		}
	}
};

// -----------------------------------------------------------------------------
// Utility functions: moving nodes between clusters
// -----------------------------------------------------------------------------

/// Initialize statistics from graph adjacency matrix, and optionally a mapping of nodes to clusters
/// mapping can be NULL.
/// sub_stats, if not NULL, is used only to determine the size of single nodes
void init_stats(ClusteringStats& stats, SparseMatrix const& a, const clus_t* mapping = NULL, const ClusteringStats* sub_stats = NULL);

/// Update cluster stats, by moving node i from cluster c1 to cluster c2
/// node_stats = indivisual node stats (duh)
/// neighbors_i = sum of weights from node i to the clusters
void update_stats_move(ClusteringStats& clus_stats, ClusteringStats const& node_stats, Neighbors const& neighbors_i, node_t i, clus_t c1, clus_t c2);

/// How would loss change after update_stats_move?
double dloss_of_move(LossFunction const& loss, ClusteringStats const& clus_stats, ClusteringStats const& node_stats, Neighbors const& neighbors_i, node_t i, clus_t c1, clus_t c2);

/// The move of node i from cluster c1 to cluster c2
struct SingleMove {
	node_t i;
	clus_t c1, c2;
	double weight_i_c1; // number of edges between i and any node in c1
	double weight_i_c2;
	double loss_after;
	Doubles sum_local_after;
};

// -----------------------------------------------------------------------------
// Utility functions: working with multiple levels
// -----------------------------------------------------------------------------

/// Get a higher level graph b, where each node in the b is a cluster in a
SparseMatrix higher_level_graph(SparseMatrix const& a, vector<clus_t> const& node_clus, size_t num_clus);

/// Transfer partitions to the higher level
void higher_level_partition(vector<int>& clus_partition, const vector<int>& node_partition, const vector<clus_t>& node_clus, size_t num_clus);

/// Given a clustering of a higher level graph, merge the assignments in this graph
void merge_from_higher_level(vector<clus_t>& node_clus, vector<clus_t> const& clus_superclus);

// -----------------------------------------------------------------------------
// Utility functions: clusterings
// -----------------------------------------------------------------------------

/// Construct a cluster by converting unique labels to [0,1,..]
template <typename T>
vector<clus_t> clustering_from_array(T const* data, size_t size) {
	vector<clus_t> clus(size);
	map<T,clus_t> first_in_clus;
	for (size_t i = 0 ; i < size ; ++i) {
		typename map<T,clus_t>::const_iterator it = first_in_clus.find(data[i]);
		if (it == first_in_clus.end()) {
			first_in_clus[data[i]] = i;
			clus[i] = i;
		} else {
			clus[i] = it->second;
		}
	}
	return clus;
}

/// Compress label assignments to the range [0..m-1], return m
/// all values in x must be 0 <= x[i] < n
size_t compress_assignments(vector<clus_t>& node_clus);

// -----------------------------------------------------------------------------
// Parameters
// -----------------------------------------------------------------------------

/// Trace of the optimizer steps
struct TraceStep {
	string description;
	vector<node_t> sub_mapping;
	vector<shared_ptr<TraceStep> > sub_steps;
	vector<clus_t> node_clus;
	double loss;
	int num_clusters;
};
typedef vector<shared_ptr<TraceStep> > TraceSteps;

struct OptimizationParams {
	bool check_invariants;
	const LossFunction* lossfun;
	int max_num_clusters;
	int min_num_clusters;
	int num_repeats;
	int num_partitions;
	int num_loss_tweak_iterations;
	bool always_consider_empty;
	bool consider_random_if_no_moves;
	bool optimize_after_higher_level;
	bool optimize_higher_level;
	bool optimize_num_clusters_with_outer_loop;
	bool optimize_globally_best_moves;
	bool use_loss_tweak;
	// output
	int verbosity;
	std::ostream& debug_out;
	TraceSteps* trace_out;
	
	OptimizationParams(std::ostream& debug_out)
		: check_invariants(false)
		, lossfun(0)
		, max_num_clusters(INT_MAX)
		, min_num_clusters(0)
		, num_repeats(1)
		, num_partitions(0)
		, num_loss_tweak_iterations(32)
		, always_consider_empty(true)
		, consider_random_if_no_moves(false)
		, optimize_after_higher_level(true)
		, optimize_higher_level(true)
		, optimize_num_clusters_with_outer_loop(true)
		, optimize_globally_best_moves(false)
		, use_loss_tweak(true)
		, verbosity(0)
		, debug_out(debug_out)
		, trace_out(0)
	{}
};

// -----------------------------------------------------------------------------
// Optimization
// -----------------------------------------------------------------------------

/// This class optimizes the clustering for a given graph
class Clustering {
  private:
	// parameters
	OptimizationParams const& params;
	TraceSteps*     trace_out;       // (optional) where to store traces
	// temporary data
	mutable Neighbors neighbors;     // neighbor information
	mutable vector<node_t> node_perm;// random permutation of the nodes
	// keeping track of empty clusters
	vector<clus_t>  empty_cluss;     // list of empty clusters
	vector<int>     clus_size;       // for each cluster: number of elements, or if empty: minus index into empty_cluss
	// statistics and loss
	ClusteringStats node_stats;      // for each node: its size, volume, within weight
	ClusteringStats clus_stats;      // for each cluster: 
	Doubles         sum_local_loss;  // sum of lossfun->local for clusters
	double          extra_loss_self; // add extra_loss_self*clus_stats.total.self to the loss, this is used for getting a certain number of clusters
	double          loss;
	// The important stuff
	const SparseMatrix& a;           // the graph
	vector<clus_t>  node_clus;       // The clustering, i.e. a mapping node -> cluster
	const Clustering* parent;        // For higher level clustering
  public:
	vector<int>     node_partition;  // Optional: only allow clusters that stay within partitions
	
	inline size_t num_nodes() const {
		return node_clus.size();
	}
	inline size_t num_clusters() const {
		return num_nodes() - empty_cluss.size();
	}
	inline double get_loss() const {
		return loss;
	}
	
	vector<clus_t> const& get_clustering();
	void set_clustering(vector<clus_t> const& clus);
	
	Clustering(const SparseMatrix& graph, const OptimizationParams& params, const Clustering* parent = NULL);
	
	/// Reset the clustering to each node being in its own cluster
	void reset_to_singletons();
	
	/// Recalculate internal data structures.
	/// Must be called after manually changing node_clus
	void recalc_internal_data();

	
	/// Fully optimize, according to the given parameters.
	/// This is the optimization function that should be called from user code.
	///
	/// Throws std::exception derived classes on error
	void optimize();
	
	/// Do optimization by moving nodes around, and when that has converged, optimize a higher level graph.
	/// Repeat this until convergence.
	/// Return true if the clustering has changed (improved).
	bool optimize_all_levels();
	
	/// Optimize by greedyliy moving single nodes around.
	/// Repeated until convergence.
	bool optimize_single_moves();
	// Go over nodes once, in a random order, and try to move to neighboring cluster
	bool optimize_single_moves_pass();
	// A single step in optimize_single_moves_pass().
	// This function is the workhorse of the optimizer.
	// All actual changes to the clustering happen here
	bool optimize_single_move_for_node(node_t i);
	
	// Perform the single globally best move
	bool optimize_best_single_move_pass();
	// Perform the single globally best move, even if the objective increases
	bool perform_best_single_forced_move();
	
	/// Cluster a higher level graph with nodes that consist of the clusters of this clustering.
	/// Then merge the clusunties on this level based on the clustering for the higher level.
	/// At the higher level, the function opt() is used
	bool optimize_higher_level(bool always_accept = false, bool (Clustering::*opt)() = &Clustering::optimize_single_moves);
	
	/// Optimize the clustering inside each cluster
	bool optimize_partition(bool always_accept = false);
	
	/// Optimize with exhaustive search.
	/// i.e. do a full exhaustive search over ALL possible clusterings
	bool optimize_exhaustive();
	
	
	/// Reduce the number of clusters to at most the number specified in params, by greedy merging of components
	void reduce_num_clusters();
	/// Reduce the number of clusters by changing extra_loss_self
	bool reduce_num_clusters_with_extra_loss();
	
	/// Find the best single move of node i
	/// if force_change==true, then don't consider the move where i stays in same cluster
	SingleMove best_single_move_for_node(node_t i, bool force_change = false) const;
	/// Find the overall best single move
	SingleMove best_single_move(bool force_change = false) const;
	/// Perform a single move
	bool perform_single_move(const SingleMove& move);
	
  private:
	// update clus_size and empty_cluss, by adding delta_size to cluster c
	void update_clus_size(int c, int delta_size);
	// calculate clustering for a loss function that wants_entire_clustering
	double loss_entire() const;
	double loss_entire(vector<clus_t> const& clus_superclus) const;
	
	// Verify that internal data makes sense
	void verify_invariants() const;
	// Verify that clusters don't cross partitions
	void verify_partition() const;
	// Verify that loss and clus_stats was updated correctly
	void verify_clus_stats() const;
	
	// Add an entry to the trace
	void trace(const string& description, const vector<shared_ptr<TraceStep> >* sub_steps = NULL, const vector<node_t>* sub_mapping = NULL);
};


// -----------------------------------------------------------------------------
}
#endif
