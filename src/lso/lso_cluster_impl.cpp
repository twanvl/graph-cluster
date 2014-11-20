// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#include "lso_cluster.hpp"

#include <ostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>
#include <stdio.h>

namespace lso_cluster {

using std::abs;

static const double epsilon = 1e-10;
static const double validate_epsilon = 1e-10;

// -----------------------------------------------------------------------------
// Utilities for debug output
// -----------------------------------------------------------------------------

template <typename T>
std::ostream& operator << (std::ostream& str, std::vector<T> const& x) {
	str << "[";
	for (size_t i = 0 ; i < x.size() ; ++i) {
		if (i > 0) str << ",";
		str << x[i];
	}
	str << "]";
	return str;
}

std::ostream& operator << (std::ostream& str, Stats const& s) {
	return str << setprecision(6) << "{" << s.self << "," << s.degree << "," << s.size << "}";
}

// -----------------------------------------------------------------------------
// Utility functions: initialize statistics
// -----------------------------------------------------------------------------

void init_stats(ClusteringStats& stats, SparseMatrix const& a, const clus_t* mapping, const ClusteringStats* sub_stats) {
	int n = a.cols();
	assert(a.rows() == a.cols());
	// clear stats
	stats.resize(n);
	std::fill(stats.begin(), stats.end(), Stats());
	stats.total = Stats();
	// accumulate edge weights
	for (int i = 0 ; i < n ; ++i) {
		int ii = mapping ? mapping[i] : i;
		double size = sub_stats ? (*sub_stats)[i].size : 1.0;
		stats[ii].size   += size;
		stats.total.size += size;
		for (int j_it = a.cidx(i) ; j_it < a.cidx(i+1) ; ++j_it) {
			int j = a.ridx(j_it);
			int jj = mapping ? mapping[j] : j;
			double w = a.data(j_it);
			
			stats[ii].degree += w;
			stats.total.degree += w;
			if (ii == jj) {
				stats[ii].self += w;
				stats.total.self += w;
			}
		}
	}
}

// -----------------------------------------------------------------------------
// Utility functions: working with multiple levels
// -----------------------------------------------------------------------------

SparseMatrix higher_level_graph(SparseMatrix const& a, vector<clus_t> const& node_clus, size_t num_clus) {
	// determine nodes for each cluster
	vector<vector<node_t> > clus_nodes(num_clus);
	for (size_t i = 0 ; i < node_clus.size() ; ++i) {
		clus_nodes[node_clus[i]].push_back(i);
	}
	// fill matrix, allocate memory as we go
	SparseMap<double> neighbors(num_clus);
	size_t nnz = num_clus;
	SparseMatrix out((int)num_clus,(int)num_clus,(int)nnz);
	out.cidx(0) = 0;
	size_t idx = 0;
	for (size_t c = 0 ; c < num_clus ; ++c) {
		// neighborsing clusters of any node in cluster c
		neighbors.clear();
		for (std::vector<node_t>::const_iterator i_it = clus_nodes[c].begin() ; i_it != clus_nodes[c].end() ; ++i_it) {
			int i = *i_it;
			for (int j_it = a.cidx(i) ; j_it < a.cidx(i+1) ; ++j_it) {
				neighbors.add(node_clus[a.ridx(j_it)], a.data(j_it));
			}
		}
		// SparseMatrix requires that the rows are sorted
		neighbors.sort();
		// add to new matrix
		for (std::vector<clus_t>::const_iterator n_it = neighbors.begin() ; n_it != neighbors.end() ; ++n_it) {
			if (idx >= nnz) {
				nnz = 2*nnz;
				out.change_capacity(nnz);
			}
			out.ridx(idx) = *n_it;
			out.data(idx) = neighbors.weight(*n_it);
			idx++;
		}
		out.cidx(c+1) = idx;
	}
	out.maybe_compress(idx);
	return out;
}

void merge_from_higher_level(vector<clus_t>& node_clus, vector<clus_t> const& clus_superclus) {
	for (size_t i = 0 ; i < node_clus.size() ; ++i) {
		node_clus[i] = clus_superclus[node_clus[i]];
	}
}

void higher_level_partition(vector<int>& clus_partition, const vector<int>& node_partition, const vector<clus_t>& node_clus, size_t num_clus) {
	if (node_partition.empty()) {
		// no parition
		clus_partition.clear();
	} else {
		// transfer partition
		clus_partition.resize(num_clus);
		for (size_t i = 0 ; i < node_clus.size() ; ++i) {
			clus_partition[node_clus[i]] = node_partition[i];
		}
	}
}

// -----------------------------------------------------------------------------
// Utility functions: exhaustive search
// -----------------------------------------------------------------------------

void first_clustering(vector<clus_t>& clus, int num_clus) {
	fill(clus.begin(), clus.end(), 0);
}

// next clustering in lexicographical ordering
// invariants: cluster ids must appear in increasing order, without gaps
// that ensures that we don't consider clusterings that are the same up to permutation of cluster ids
bool next_clustering(vector<clus_t>& clus, int num_clus) {
	// first: find for each index the highest cluster number that we are allowed to use
	assert(clus.size() <= 100);
	int max_clus[100];
	max_clus[0] = 0;
	for (size_t i = 1 ; i < clus.size() ; ++i) {
		max_clus[i] = min(num_clus-1, max(clus[i-1]+1, max_clus[i-1]));
	}
	// now find next clustering
	int i = (int)clus.size() - 1;
	for ( ; i >= 0 ; --i) {
		if (clus[i] == max_clus[i]) {
			clus[i] = 0;
		} else {
			clus[i]++;
			break;
		}
	}
	if (i == -1) return false; // wrapped around to [0,0,0,...]
	return true;
}

// -----------------------------------------------------------------------------
// Implementation: initialization stuff
// -----------------------------------------------------------------------------

Clustering::Clustering(const SparseMatrix& a, const OptimizationParams& params, const Clustering* parent)
	: params(params)
	, trace_out(params.trace_out)
	, neighbors(a.cols())
	, node_perm(a.cols())
	, clus_size(a.cols())
	, extra_loss_self(parent ? parent->extra_loss_self : 0.0)
	, a(a)
	, node_clus(a.cols())
	, parent(parent)
{
	// parameters that make no sense
	if (this->params.min_num_clusters < 0) {
		throw std::invalid_argument("min_num_clusters should be non-negative");
	}
	if (this->params.max_num_clusters <= 0) {
		throw std::invalid_argument("max_num_clusters should be positive");
	}
	// intialize node_perm and node_stats.
	// These don't depend on the clustering, so they don't have to go into recalc_internal_data()
	for (size_t i = 0 ; i < node_perm.size() ; ++i) {
		node_perm[i] = i;
	}
	if (parent) {
		node_stats = parent->clus_stats;
		node_sum_local_loss = parent->node_sum_local_loss;
	} else {
		init_stats(node_stats, a);
		// calculate sum of local for the individual nodes
		node_sum_local_loss = 0.;
		for (size_t i = 0 ; i < node_clus.size() ; ++i) {
			node_sum_local_loss += params.lossfun->local(node_stats[i], node_stats.total);
		}
	}
	// initial clustering
	reset_to_singletons();
}

void Clustering::reset_to_singletons() {
	// initialize to each node in its own cluster
	for (size_t i = 0 ; i < node_clus.size() ; ++i) {
		node_clus[i] = i;
	}
	// calculate degrees and loss
	recalc_internal_data();
}

void Clustering::recalc_internal_data() {
	if (params.check_invariants) verify_partition();
	// recalculate accumulated degrees
	init_stats(clus_stats, a, &node_clus[0], &node_stats);
	// find empty clusters
	fill(clus_size.begin(),clus_size.end(), 0);
	empty_cluss.clear();
	for (size_t i = 0 ; i < node_clus.size() ; ++i) {
		clus_size[node_clus[i]]++;
	}
	for (size_t i = 0 ; i < clus_stats.size() ; ++i) {
		if (clus_size[i] == 0) {
			clus_size[i] = -static_cast<int>(empty_cluss.size());
			empty_cluss.push_back(i);
		}
	}
	// calculate loss
	if (params.lossfun->want_entire_clustering()) {
		loss = loss_entire();
	} else {
		loss = params.lossfun->loss(clus_stats, num_clusters(), &sum_local_loss, node_sum_local_loss);
	}
	loss += extra_loss_self * clus_stats.total.self / clus_stats.total.degree;
}

vector<clus_t> const& Clustering::get_clustering() {
	compress_assignments(node_clus);
	return node_clus;
}

void Clustering::set_clustering(vector<clus_t> const& clus) {
	node_clus = clus;
	recalc_internal_data();
}

// -----------------------------------------------------------------------------
// Implementation: single moves
// -----------------------------------------------------------------------------

SingleMove Clustering::best_single_move(bool force_change) const {
	assert(node_clus.size() > 0);
	SingleMove best_move = best_single_move_for_node(0, force_change);
	for (size_t i = 1 ; i < node_clus.size() ; ++i) {
		SingleMove m2 = best_single_move_for_node(i, force_change);
		if (m2.loss_after < best_move.loss_after - epsilon || (force_change && best_move.c1 == best_move.c2)) {
			best_move = m2;
		}
	}
	return best_move;
}

SingleMove Clustering::best_single_move_for_node(node_t i, bool force_change) const {
	clus_t c1 = node_clus[i]; // current cluster of i
	// find neighboring clusters, and sum of weights to them
	// i.e. set neighbors.weight(c) = sum of edges from i to any node in c
	neighbors.clear();
	neighbors.add(c1, -node_stats[i].self); // don't count the self loops for node i twice
	for (int j_it = a.cidx(i) ; j_it < a.cidx(i+1) ; ++j_it) {
		int j = a.ridx(j_it);
		if (!node_partition.empty() && node_partition[j] != node_partition[i]) {
			// don't allow this move, i can only be clustered together with nodes from the same partition
			continue;
		}
		neighbors.add(node_clus[j], a.data(j_it));
	}
	// always also consider moving i into an empty cluster
	if (params.always_consider_empty && !empty_cluss.empty()) {
		neighbors.add(empty_cluss.back(), 0.);
	}
	// always make sure that at least one move to another cluster is considered
	// this allows different connected components to be merged
	if (params.consider_random_if_no_moves && neighbors.nnz() == 1 && num_nodes() > 1) {
		int j = rand() % (num_nodes() - 1);
		if (j>=i) j++;
		neighbors.add(j, 0.);
		if (params.verbosity >= 7) {
			params.debug_out << "      considering a random move" << endl;
		}
	}
	// default move: don't move
	SingleMove best_move;
	best_move.i  = i;
	best_move.c2 = best_move.c1 = c1;
	best_move.weight_i_c2 = best_move.weight_i_c1 = neighbors.weight(c1);
	best_move.loss_after = loss;
	best_move.sum_local_after = sum_local_loss;
	// delta loss for move to possible neighboring clusters
	for (vector<clus_t>::const_iterator n_it = neighbors.begin() ; n_it != neighbors.end() ; ++n_it) {
		int c2 = *n_it;
		if (c1 == c2) continue;
		// How would the loss change after moving i to c2?
		double weight_i_c2 = neighbors.weight(c2);
		Stats delta1 = node_stats[i]; delta1.self += 2.0 * best_move.weight_i_c1;
		Stats delta2 = node_stats[i]; delta2.self += 2.0 * weight_i_c2;
		Stats total_after = clus_stats.total - delta1 + delta2;
		int num_cluster_after = num_clusters() - (node_stats[i].size == clus_stats[c1].size ? 1 : 0) + (clus_stats[c2].size == 0 ? 1 : 0);
		Doubles sum_local_after = sum_local_loss;
		sum_local_after -= params.lossfun->local(clus_stats[c1], clus_stats.total);
		sum_local_after -= params.lossfun->local(clus_stats[c2], clus_stats.total);
		sum_local_after += params.lossfun->local(clus_stats[c1]-delta1, total_after);
		sum_local_after += params.lossfun->local(clus_stats[c2]+delta2, total_after);
		double loss_after;
		if (params.lossfun->want_entire_clustering()) {
			// temporarily change assignment to calculate loss
			const_cast<Clustering*>(this)->node_clus[i] = c2;
			loss_after = loss_entire();
			const_cast<Clustering*>(this)->node_clus[i] = c1;
		} else {
			loss_after = params.lossfun->global(sum_local_after, node_sum_local_loss, total_after, num_cluster_after);
		}
		loss_after += extra_loss_self * total_after.self / total_after.degree;
		// is it better?
		if ((loss_after < best_move.loss_after - epsilon) || (force_change && best_move.c2 == c1)) {
			best_move.c2              = c2;
			best_move.weight_i_c2     = weight_i_c2;
			best_move.loss_after      = loss_after;
			best_move.sum_local_after = sum_local_after;
		}
		if (params.verbosity >= 7) {
			params.debug_out << "      consider " << i << " from " << c1 << " to " << c2 << " loss " << setprecision(12) << loss_after << " in " << num_cluster_after << " clusters" << endl;
		}
	}
	return best_move;
}

SwapMove Clustering::best_swap_move_for_node(node_t i) const {
	clus_t c1 = node_clus[i];
	
	// count edges from i to neighboring clusters
	neighbors.clear();
	neighbors.add(c1, -node_stats[i].self); // don't count the self loops for node i
	for (int j_it = a.cidx(i) ; j_it < a.cidx(i+1) ; ++j_it) {
		int j = a.ridx(j_it);
		if (i != j && (node_partition.empty() || node_partition[j] == node_partition[i])) {
			neighbors.add(node_clus[j], a.data(j_it));
		}
	}
	
	// default best move: swap i with itself
	double weight_i_c1 = neighbors.weight(c1);
	SwapMove best_move;
	best_move.i  = best_move.j = i;
	best_move.weight_ij_c1 = best_move.weight_ij_c2 = 0;
	best_move.loss_after = loss;
	best_move.sum_local_after = sum_local_loss;
    
	// for every neighbor of i in another cluster...
	for (int j_it = a.cidx(i) ; j_it < a.cidx(i+1) ; ++j_it) {
		int j = a.ridx(j_it);
		int c2 = node_clus[j];
		if (c2 == c1) continue;
		// consider swapping i with j
		// find edges j--c1 and j--c2
		double weight_j_c1 = 0.;
		double weight_j_c2 = 0.;
		for (int k_it = a.cidx(j) ; k_it < a.cidx(j+1) ; ++k_it) {
			int k = a.ridx(k_it);
			if (k == j) {
				// don't count self-loops
			} else if (node_clus[k] == c1) {
				weight_j_c1 += a.data(k_it);
			} else if (node_clus[k] == c2) {
				weight_j_c2 += a.data(k_it);
			}
		}
		// How would the loss change after swapping clustering of i and j?
		double weight_i_c2 = neighbors.weight(c2);
		Stats delta1 = node_stats[i] - node_stats[j]; delta1.self += 2.0 * (weight_i_c1 - weight_j_c1);
		Stats delta2 = node_stats[i] - node_stats[j]; delta2.self += 2.0 * (weight_i_c2 - weight_j_c2);
		Stats total_after = clus_stats.total - delta1 + delta2;
		Doubles sum_local_after = sum_local_loss;
		sum_local_after -= params.lossfun->local(clus_stats[c1], clus_stats.total);
		sum_local_after -= params.lossfun->local(clus_stats[c2], clus_stats.total);
		sum_local_after += params.lossfun->local(clus_stats[c1]-delta1, total_after);
		sum_local_after += params.lossfun->local(clus_stats[c2]+delta2, total_after);
		double loss_after;
		if (params.lossfun->want_entire_clustering()) {
			// temporarily change assignment to calculate loss
			const_cast<Clustering*>(this)->node_clus[i] = c2;
			const_cast<Clustering*>(this)->node_clus[j] = c1;
			loss_after = loss_entire();
			const_cast<Clustering*>(this)->node_clus[i] = c1;
			const_cast<Clustering*>(this)->node_clus[j] = c2;
		} else {
			loss_after = params.lossfun->global(sum_local_after, node_sum_local_loss, total_after, num_clusters());
		}
		loss_after += extra_loss_self * total_after.self / total_after.degree;
		// is it better?
		if (loss_after < best_move.loss_after - epsilon) {
			best_move.j               = j;
			best_move.weight_ij_c1    = weight_i_c1 - weight_j_c1;
			best_move.weight_ij_c2    = weight_i_c2 - weight_j_c2;
			best_move.loss_after      = loss_after;
			best_move.sum_local_after = sum_local_after;
		}
	}
	return best_move;
}

bool Clustering::perform_single_move(const SingleMove& move) {
	if (move.c1 == move.c2) return false;
	
	// update stats
	Stats delta1 = node_stats[move.i]; delta1.self += 2.0 * move.weight_i_c1;
	Stats delta2 = node_stats[move.i]; delta2.self += 2.0 * move.weight_i_c2;
	clus_stats[move.c1] -= delta1;
	clus_stats[move.c2] += delta2;
	clus_stats.total -= delta1;
	clus_stats.total += delta2;
	// update loss
	sum_local_loss = move.sum_local_after;
	loss = move.loss_after;
	// update cluster assignment
	if(node_clus[move.i] != move.c1) throw std::logic_error("node_clus[move.i] != move.c2");
	node_clus[move.i] = move.c2;
	// update
	update_clus_size(move.c1, -1);
	update_clus_size(move.c2, +1);
	
	// recheck invariants
	if (params.check_invariants) verify_invariants();
	if (params.verbosity >= 6) {
		params.debug_out << "     moved " << move.i << " from " << move.c1;
		if (!node_partition.empty()) params.debug_out << " in " << node_partition[move.c1];
		params.debug_out << " to " << move.c2;
		if (!node_partition.empty()) params.debug_out << " in " << node_partition[move.c2];
		params.debug_out << " loss " << setprecision(12) << move.loss_after << " in " << num_clusters() << " clusters" << endl;
	}
	return true;
}

bool Clustering::perform_swap_move(const SwapMove& move) {
	if (node_clus[move.i] == node_clus[move.j]) return false;
	
	// update stats
	Stats delta1 = node_stats[move.i] - node_stats[move.j]; delta1.self += 2.0 * move.weight_ij_c1;
	Stats delta2 = node_stats[move.i] - node_stats[move.j]; delta2.self += 2.0 * move.weight_ij_c2;
	clus_stats[node_clus[move.i]] -= delta1;
	clus_stats[node_clus[move.j]] += delta2;
	clus_stats.total -= delta1;
	clus_stats.total += delta2;
	// update loss
	sum_local_loss = move.sum_local_after;
	loss = move.loss_after;
	// update cluster assignment
	swap(node_clus[move.i], node_clus[move.j]);
	
	return true;
}

// update clus_size and empty_cluss, by adding dsize to cluster c
void Clustering::update_clus_size(int c, int dsize) {
	if (clus_size[c] <= 0) {
		// no longer empty
		int c2 = empty_cluss.back();
		swap(empty_cluss[-clus_size[c]], empty_cluss.back());
		empty_cluss.pop_back();
		clus_size[c2] = clus_size[c];
		clus_size[c] = 0;
	}
	clus_size[c] += dsize;
	if (clus_size[c] <= 0) {
		// has become empty
		clus_size[c] = -static_cast<int>(empty_cluss.size());
		empty_cluss.push_back(c);
	}
}

double Clustering::loss_entire() const {
	if (parent) {
		return parent->loss_entire(node_clus);
	} else {
		return params.lossfun->loss_entire(node_clus);
	}
}
double Clustering::loss_entire(vector<clus_t> const& clus_superclus) const {
	vector<clus_t> merged_node_clus(node_clus.size());
	for (size_t i = 0 ; i < node_clus.size() ; ++i) {
		merged_node_clus[i] = clus_superclus[node_clus[i]];
	}
	if (parent) {
		return parent->loss_entire(merged_node_clus);
	} else {
		return params.lossfun->loss_entire(merged_node_clus);
	}
}

// -----------------------------------------------------------------------------
// Implementation: optimization
// -----------------------------------------------------------------------------

void Clustering::optimize() {
	vector<clus_t> node_clus_initial = node_clus;
	vector<clus_t> best_node_clus;
	double best_loss = 1e100;
	for (int rep = 0 ; rep < params.num_repeats ; ++rep) {
		if (params.verbosity >= 1) params.debug_out << "Repetition " << rep << endl;
		// reset clustering
		node_clus = node_clus_initial;
		recalc_internal_data();
		trace("initial");
		// lots of optimization
		if (params.optimize_exhaustive) {
			optimize_exhaustive();
		} else if (params.optimize_num_clusters_with_outer_loop &&
				(params.min_num_clusters > 1 || params.max_num_clusters < (int)num_nodes())) {
			reduce_num_clusters_with_extra_loss();
		} else {
			for (int it = 0 ; it < params.num_partitions ; ++it) {
				optimize_all_levels();
				optimize_partition();
			}
			optimize_all_levels();
			// enforce number of clusters?
			if (params.min_num_clusters > (int)num_clusters() || params.max_num_clusters < (int)num_clusters()) {
				reduce_num_clusters();
			}
		}
		// is it an improvement?
		if (rep == 0 || loss < best_loss) {
			best_loss = loss;
			best_node_clus = node_clus;
		}
		trace("done");
	}
	// store
	this->loss = best_loss;
	this->node_clus = best_node_clus;
}

bool Clustering::optimize_all_levels() {
	bool changes = optimize_single_moves();
	if (params.optimize_higher_level) {
		while (optimize_higher_level()) {
			changes = true;
			if (params.optimize_after_higher_level) optimize_single_moves();
		}
	}
	return changes;
}

// Optimize by greedyliy moving single nodes around.
// Repeated until convergence.
bool Clustering::optimize_single_moves() {
	bool changes = true;
	int iteration = 0;
	if (params.verbosity >= 4) {
		params.debug_out << "   Initially loss is " << setprecision(12) << loss << " in " << num_clusters() << " clusters" << endl;
	}
	while (changes) {
		double loss_before = loss;
		if (params.optimize_globally_best_moves) {
			changes = optimize_best_single_move_pass();
		} else {
			changes = optimize_single_moves_pass();
		}
		if (!changes && params.optimize_with_swap_moves) {
			changes = optimize_swap_moves_pass();
		}
		iteration++;
		if (params.verbosity >= 5) {
			params.debug_out << "    After iteration " << iteration << " loss is " << setprecision(12) << loss << " in " << num_clusters() << " clusters" << endl;
		}
		trace(changes ? "single moves" : "no single moves");
		if (changes && loss >= loss_before) {
			break; // something went wrong: we made the loss worse, probably due to numerical errors
		}
	}
	if (params.verbosity == 4) {
		params.debug_out << "   After " << iteration << " iterations, loss is " << setprecision(12) << loss << " in " << num_clusters() << " clusters" << endl;
	}
	return iteration > 1;
}

// go over nodes in a random order, and try to move to neighboring cluster
bool Clustering::optimize_single_moves_pass() {
	if (params.check_invariants) verify_invariants();
	bool changes = false;
	std::random_shuffle(node_perm.begin(), node_perm.end());
	for (vector<int>::const_iterator i_it = node_perm.begin() ; i_it != node_perm.end() ; ++i_it) {
		OCTAVE_QUIT;
		if (optimize_single_move_for_node(*i_it)) changes = true;
	}
	if (changes) {
		// it is important to recalculate the loss, otherwise we accumulate numerical errors
		recalc_internal_data();
	}
	return changes;
}

bool Clustering::optimize_single_move_for_node(int i) {
	SingleMove move = best_single_move_for_node(i);
	if (move.loss_after < loss - epsilon) {
		return perform_single_move(move);
	} else {
		return false;
	}
}

bool Clustering::optimize_best_single_move_pass() {
	SingleMove move = best_single_move(false);
	if (move.loss_after < loss - epsilon) {
		return perform_single_move(move);
	} else {
		return false;
	}
}

bool Clustering::perform_best_single_forced_move() {
	SingleMove move = best_single_move(true);
	return perform_single_move(move);
}

bool Clustering::optimize_swap_moves_pass() {
	if (params.check_invariants) verify_invariants();
	bool changes = false;
	std::random_shuffle(node_perm.begin(), node_perm.end());
	for (vector<int>::const_iterator i_it = node_perm.begin() ; i_it != node_perm.end() ; ++i_it) {
		OCTAVE_QUIT;
		if (optimize_swap_move_for_node(*i_it)) changes = true;
	}
	if (changes) {
		// it is important to recalculate the loss, otherwise we accumulate numerical errors
		recalc_internal_data();
	}
	return changes;
}

bool Clustering::optimize_swap_move_for_node(int i) {
	SwapMove move = best_swap_move_for_node(i);
	if (move.loss_after < loss - epsilon) {
		return perform_swap_move(move);
	} else {
		return false;
	}
}

bool Clustering::optimize_higher_level(bool always_accept, bool (Clustering::*opt)()) {
	// first, reduce the clusters to [0..m]
	size_t num_clus = compress_assignments(node_clus);
	recalc_internal_data();
	
	// Generate higher level Clustering object
	if (params.verbosity >= 3) params.debug_out << "  Optimize higher level: " << endl;
	SparseMatrix b = higher_level_graph(a, node_clus, num_clus);
	Clustering higher(b, params, this);
	higher_level_partition(higher.node_partition, node_partition, node_clus, num_clus);
	
	// check that loss calculation is correct
	if (abs(loss - higher.loss) > validate_epsilon) {
		//(params.check_invariants ? error : warning)("Incorrect loss on higher level: %f != %f", loss, higher.loss);
		printf("Incorrect loss on higher level: %f != %f\n", loss, higher.loss);
		if (params.check_invariants) throw mk_logic_error("Incorrect loss on higher level: %f != %f\n", loss, higher.loss);
	}
	
	// trace for higher level
	vector<shared_ptr<TraceStep> > sub_trace;
	vector<node_t> sub_mapping;
	if (trace_out) {
		sub_mapping = node_clus;
		higher.trace_out = &sub_trace;
		higher.trace("init");
	}
	
	// now cluster on a higher level
	bool changes = (higher.*opt)();
	//higher.recalc_internal_data(); // make sure loss is exact
	
	// use for this?
	if (changes && (always_accept || higher.loss < this->loss - epsilon)) {
		merge_from_higher_level(node_clus,higher.node_clus);
		recalc_internal_data();
		if (params.verbosity >= 3) params.debug_out << "  Merge from higher level: " << setprecision(12) << higher.loss << " = " << setprecision(12) << this->loss << ", an improvement of " << setprecision(12) << (higher.loss - this->loss) << endl;
		trace("higher",&sub_trace,&sub_mapping);
		return true;
	} else {
		if (params.verbosity >= 3) params.debug_out << "  Don't merge from higher level: " << setprecision(12) << higher.loss << " > " << setprecision(12) << this->loss << endl;
		return false;
	}
}

bool Clustering::optimize_partition(bool always_accept) {
	if (params.verbosity >= 2) params.debug_out << " Optimize partitions" << endl;
	double loss_before = loss;
	vector<clus_t> node_clus_before = node_clus;
	// optimize inside each partition
	this->node_partition = this->node_clus;
	reset_to_singletons();
	verify_partition();
	// sub-trace
	vector<shared_ptr<TraceStep> > sub_trace, *original_trace_out = trace_out;
	if (trace_out) {
		trace_out = &sub_trace;
		trace("init partition");
	}
	// optimize
	bool changes = optimize_all_levels();
	// done
	this->node_partition.clear();
	trace_out = original_trace_out;
	// is it an improvement?
	if (changes && (always_accept || loss < loss_before - epsilon)) {
		if (params.verbosity >= 2) params.debug_out << " Accepting optimized partitions" << endl;
		trace("partition",&sub_trace);
		return true;
	} else {
		if (params.verbosity >= 2) params.debug_out << " Rejecting optimized partitions" << endl;
		this->node_clus = node_clus_before;
		recalc_internal_data();
		return false;
	}
}

bool Clustering::optimize_exhaustive() {
	// target number of clusters
	int target_num = params.max_num_clusters;
	if (target_num <= 0) target_num = (int)node_clus.size();
	target_num = min(target_num, (int)node_clus.size());
	// best so far
	double best_loss = INFINITY;
	vector<clus_t> best_node_clus = this->node_clus;
	// exhaustive search over possible clusterings of nodes
	first_clustering(node_clus, target_num);
	while (true) {
		OCTAVE_QUIT;
		// try this clustering
		recalc_internal_data();
		if (loss < best_loss) {
			best_loss = loss;
			best_node_clus = node_clus;
		}
		if (params.verbosity >= 4) params.debug_out << "   Exhaustive: " << node_clus << " loss " << setprecision(12) << loss << endl;
		// next
		if (!next_clustering(node_clus, target_num)) break;
	}
	// use the best one
	node_clus = best_node_clus;
	recalc_internal_data();
	trace("exhaustive");
	return true;
}

// use an exhaustive search? Only if it is not too much work
bool use_exhaustive_for(int num_cluster, int target_num) {
	int num_to_consider = 1;
	int max_label = 1;
	for (int i = 0; i < num_cluster ; ++i) {
		num_to_consider *= max_label;
		max_label = min(max_label+1,target_num);
		if (num_to_consider > 1000) return false;
	}
	return true;
}

void Clustering::reduce_num_clusters() {
	// tweak loss function to reduce number of clusters
	if (params.use_loss_tweak && ((int)num_clusters() > params.max_num_clusters || (int)num_clusters() < params.min_num_clusters)) {
		//optimize_higher_level(true, &Clustering::reduce_num_clusters_with_extra_loss);
		reduce_num_clusters_with_extra_loss();
	}
	// repeatedly perform a move on higher level graph
	// since clustering starts out as singletons, this always merges two clusters
	while ((int)num_clusters() > params.max_num_clusters && !use_exhaustive_for(num_clusters(), params.max_num_clusters)) {
		if (!optimize_higher_level(true, &Clustering::perform_best_single_forced_move)) break;
	}
	// with few enough clusters, we can do an exhaustive search
	if ((int)num_clusters() > params.max_num_clusters && use_exhaustive_for(num_clusters(), params.max_num_clusters)) {
		optimize_higher_level(true, &Clustering::optimize_exhaustive);
	}
}

bool Clustering::reduce_num_clusters_with_extra_loss() {
	double lower = -M_PI_2;
	double upper = M_PI_2;
	size_t min_allowed_clusters = params.min_num_clusters;
	size_t max_allowed_clusters = params.max_num_clusters;
	
	if (params.verbosity >= 2) {
		params.debug_out << " Use extra loss to get " << params.min_num_clusters << " <= " << num_clusters() << " <= " << params.max_num_clusters << endl;
	}
	if (params.verbosity >= 2) {
		params.debug_out << " Use extra loss to get " << min_allowed_clusters << " <= " << num_clusters() << " <= " << max_allowed_clusters << endl;
	}
	
	// now do a binary search
	// mapping using tan(), so we cover the entire range from -infinity to +infinity
	// because of the random optimization, this is technically not valid, but it should work in practice.
	vector<clus_t> init_node_clus = node_clus;
	vector<clus_t> best_node_clus = node_clus;
	double best_loss = loss; // not including the extra loss
	int best_too_few_or_many = max(0,(int)num_clusters() - (int)max_allowed_clusters)
	                         + max(0,(int)min_allowed_clusters - (int)num_clusters());
	
	for (int it = 0 ; it < params.num_loss_tweak_iterations ; ++it) {
		// use tan(-pi/2...pi/2) to search over all reals
		
		// use the given extra_loss_self
		double middle = (lower + upper) * 0.5;
		extra_loss_self = tan(middle);
		node_clus = init_node_clus;
		recalc_internal_data();
		for (int it = 0 ; it < params.num_partitions ; ++it) {
			optimize_all_levels();
			optimize_partition();
		}
		optimize_all_levels();
		
		// calculate loss without the extra_loss_self change
		double actual_loss = this->loss - extra_loss_self * clus_stats.total.self / clus_stats.total.degree;
		int too_few  = max(0,(int)min_allowed_clusters - (int)num_clusters());
		int too_many = max(0,(int)num_clusters() - (int)max_allowed_clusters);
		int too_few_or_many = too_few + too_many;
		if (params.verbosity >= 2) {
			params.debug_out << " Extra loss " << setprecision(5) << tan(lower) << " < " << extra_loss_self << " < " << tan(upper) << " gives loss " << actual_loss << " in " << num_clusters() << " clusters, " << too_few << "+" << too_many << "=" << too_few_or_many << " off" << endl;
		}
		if (too_few_or_many < best_too_few_or_many || (too_few_or_many == best_too_few_or_many && actual_loss < best_loss)) {
			best_loss = actual_loss;
			best_too_few_or_many = too_few_or_many;
			best_node_clus = node_clus;
			if (params.verbosity >= 2) {
				params.debug_out << " (best so far)" << endl;
			}
		}
		if (num_clusters() > max_allowed_clusters) {
			if (params.verbosity >= 2) params.debug_out << " (too many clusters: " << num_clusters() << " > " << max_allowed_clusters << "), decrease upper bound" << endl;
			upper = middle; // need more loss contribution to bring down #of clusters
		} else if (num_clusters() < min_allowed_clusters) {
			if (params.verbosity >= 2) params.debug_out << " (too few clusters: " << num_clusters() << " < " << min_allowed_clusters << "), increase lower bound" << endl;
			lower = middle;
		} else {
			if (params.verbosity >= 2) params.debug_out << " (okay nr. of clusters)" << endl;
			// we are in the accepted region, prefer smaller (in absolute sense) loss tweaks
			if (abs(upper) > abs(lower)) {
				upper = middle; 
			} else {
				lower = middle; 
			}
		}
	}
	
	// store
	//double best_extra_loss_self = extra_loss_self;
	extra_loss_self = 0.;
	node_clus = best_node_clus;
	recalc_internal_data();
	return true;
}

// -----------------------------------------------------------------------------
// Implementation: verify invariants
// -----------------------------------------------------------------------------

void Clustering::verify_clus_stats() const {
	#define check(sub,str,ii) \
		if (abs(clus_stats sub - cs sub) > validate_epsilon) \
			throw mk_logic_error("At " str "  %f != %f (diff: %g)\n",ii,clus_stats sub, cs sub, clus_stats sub - cs sub)
	ClusteringStats cs;
	init_stats(cs, a, &node_clus[0], &node_stats);
	for (size_t i = 0 ; i < clus_stats.size() ; ++i) {
		check([i].degree,"[%d].degree",(int)i);
		check([i].self,"[%d].self",(int)i);
		check([i].size,"[%d].size",(int)i);
	}
	check(.total.degree,"%s.degree","total");
	check(.total.self,"%s.self","total");
	check(.total.size,"%s.size","total");
	Doubles new_sl;
	double new_loss;
	if (params.lossfun->want_entire_clustering()) {
		new_loss = loss_entire();
	} else {
		new_loss = params.lossfun->loss(cs, num_clusters(), &new_sl, node_sum_local_loss);
	}
	new_loss += extra_loss_self * cs.total.self / cs.total.degree;
	if (abs(loss - new_loss) > validate_epsilon) {
		throw mk_logic_error("Incorrect loss update:  %f != %f\n",loss,new_loss);
	}
	for (int i = 0 ; i < MAX_DOUBLES ; ++i) {
		if (abs(sum_local_loss[i] - new_sl[i]) > validate_epsilon) {
			throw mk_logic_error("Incorrect local loss [%d] update:  %f != %f\n",i, sum_local_loss[i], new_sl[i]);
		}
	}
	#undef check
}

// Verify that clusters don't cross partitions
void Clustering::verify_partition() const {
	if (node_partition.empty()) return;
	vector<vector<node_t> > clus_nodes(node_clus.size());
	for (size_t i = 0 ; i < node_clus.size() ; ++i) {
		clus_nodes.at(node_clus.at(i)).push_back(i);
	}
	for (size_t i = 0 ; i < clus_nodes.size() ; ++i) {
		for (size_t j = 0 ; j < clus_nodes[i].size() ; ++j) {
			if (node_partition[clus_nodes[i][0]] != node_partition[clus_nodes[i][j]]) {
				throw mk_logic_error("ERROR: cluster %d contains nodes %d in %d, and %d in %d",(int)i,
						clus_nodes[i][0], node_partition[clus_nodes[i][0]],
						clus_nodes[i][j], node_partition[clus_nodes[i][j]]);
			}
		}
	}
	#undef error
}

void Clustering::verify_invariants() const {
	verify_partition();
	verify_clus_stats();
}

// -----------------------------------------------------------------------------
// Implementation: traces
// -----------------------------------------------------------------------------

void Clustering::trace(const string& description, const vector<shared_ptr<TraceStep> >* sub_steps, const vector<node_t>* sub_mapping) {
	if (!trace_out) return;
	shared_ptr<TraceStep> trace(new TraceStep);
	trace->description = description;
	if (sub_steps) trace->sub_steps = *sub_steps;
	if (sub_mapping) trace->sub_mapping = *sub_mapping;
	trace->loss = this->loss;
	trace->num_clusters = this->num_clusters();
	trace->node_clus = this->node_clus;
	trace_out->push_back(trace);
}

// -----------------------------------------------------------------------------
}
