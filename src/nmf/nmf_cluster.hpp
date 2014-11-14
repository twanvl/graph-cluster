// -----------------------------------------------------------------------------
// Local Search Non-negative Matrix Factorization
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#ifndef HEADER_NMF_CLUSTER
#define HEADER_NMF_CLUSTER

#include "util.hpp"
#include "sparse_matrix.hpp"
#include "sparse_vector.hpp"
#include "nmf_clustering.hpp"
#include "nmf_objective_function.hpp"
#include <vector>
#include <iomanip>
#include <stdexcept>

#include <stdarg.h>
#include <stdio.h>

namespace nmf_cluster {

using namespace lso_cluster;
using namespace std;

// -----------------------------------------------------------------------------
// Objective functions
// -----------------------------------------------------------------------------

// remove a node from all clusters
struct RemoveMove {
	node_t i;
};
// add a node to a cluster (it should not already be in that cluster)
struct AddMove {
	node_t i;
	clus_t clus;
	double weight;
	
	AddMove(node_t i, clus_t clus, double weight) : i(i), clus(clus), weight(weight) {}
};

// Calculate the loss
double calculate_loss(NMFObjectiveFun const& obj, SparseMatrix const& graph, NMFClustering const& clustering);
double calculate_loss_row(NMFObjectiveFun const& obj, SparseMatrix const& graph, NMFClustering const& clustering, node_t i);

// -----------------------------------------------------------------------------
// Parameters
// -----------------------------------------------------------------------------

struct NMFParams {
	int num_iter;
	int max_cluster_per_node;
	int max_num_cluster;
	// the objective function
	NMFObjectiveFun objective;
	// output
	int verbosity;
	std::ostream& debug_out;
	
	NMFParams(std::ostream& debug_out)
		: num_iter(100)
		, max_cluster_per_node(std::numeric_limits<int>::max())
		, max_num_cluster(std::numeric_limits<int>::max())
		, verbosity(0)
		, debug_out(debug_out)
	{}
	
	void debug_printf(const char* fmt,...) const;
};

// -----------------------------------------------------------------------------
// Data structures
// -----------------------------------------------------------------------------

class NMFOptimizer {
  private:
	// inputs
	SparseMatrix graph;
	NMFParams const& params;
	
	// outputs
	NMFClustering clustering; // for each node, the clusters it is in
	double loss;
	vector<double> losses; // loss after each iteration
	
	// intermediate/auxilliary values
	//std::vector<double> clus_sum; // total membership of each cluster, invariant: clus_size[k] = sum_i{ clustering[i](k) }
	//std::vector<double> clus_sumsq;
	//vector<double> prior;
	
	// state used during computation
	
	// iterating over nodes in a random order
	mutable std::vector<node_t> node_perm;
	
	// clusters that are 'adjacent' to the current node i
	//  for each neighbor: weight of edges to it / factor
	//  n[c] = ∑{j} a[ij]/u[jc]
	//typedef SparseMap<pair<double,double> > Neighbors;
	typedef SparseMap<double> Neighbors;
	mutable Neighbors neighbors;
	
	// backup of assignments for node i
	//vector<NodeFactors> old_clus_i;
	
	void reset();
	void calculate_loss();
	void calculate_loss_debug() const;
	// optimization
	bool optimization_pass();
	bool optimize_for_node(node_t i);
//	NodeFactors best_move_of(node_t i);
	//void move_for_node(node_t i);
	void set_active_node(node_t i);
	void greedy_move(node_t i);
	void multiplicative_gradient_move();
	void accept_or_reject();
	
	void likelihood();
	
	double greedy_optimal_membership(node_t i, clus_t clus) const;
	bool simple_greedy_add_move(node_t i);
	bool simple_greedy_move(node_t i);
	double loss_after_addition(node_t i, clus_t clus, double weight);
	double loss_after_addition_slow(node_t i, clus_t clus, double weight);
	
  public:
	NMFOptimizer(SparseMatrix const& graph, NMFParams const& params);
	
	// run the optimizer
	void run();
	
	// number of nodes
	inline node_t size() const {
		return (node_t)clustering.size();
	}
	inline clus_t max_num_clus() const {
		return clustering.max_num_clus();
	}
	// return the current clustering
	NMFClustering const& get_clustering() const {
		return clustering;
	}
	// the loss
	double get_loss() const {
		return loss;
	}
	// the loss after each iteration
	const vector<double>& get_losses() const {
		return losses;
	}
};

// -----------------------------------------------------------------------------
// Debug
// -----------------------------------------------------------------------------

void NMFParams::debug_printf(const char* fmt,...) const {
	char buf[1024];
	va_list args;
	va_start(args,fmt);
	vsnprintf(buf,1024,fmt,args);
	va_end (args);
	debug_out << buf;
}

template <typename T> std::ostream& operator << (std::ostream& out, std::vector<T> const& vec) {
	out << "[";
	for (typename vector<T>::const_iterator it = vec.begin() ; it != vec.end() ; ++it) {
		if (it != vec.begin()) out << ",";
		out << *it;
	}
	return out << "]";
}
template <typename T> std::ostream& operator << (std::ostream& out, SparseMap<T> const& vec) {
	out << "{";
	for (SparseMap<pair<double,double> >::const_iterator it = vec.begin() ; it != vec.end() ; ++it) {
		if (it != vec.begin()) out << ", ";
		out << *it << ":" << vec.weight(*it);
	}
	return out << "}";
}
std::ostream& operator << (std::ostream& out, SparseMap<std::pair<double,double> > const& vec) {
	out << "{";
	for (SparseMap<pair<double,double> >::const_iterator it = vec.begin() ; it != vec.end() ; ++it) {
		if (it != vec.begin()) out << ", ";
		out << *it << ":" << vec.weight(*it).second << "/" << vec.weight(*it).first;
	}
	return out << "}";
}

// -----------------------------------------------------------------------------
// Optimization
// -----------------------------------------------------------------------------

static const double epsilon = 1e-10;

NMFOptimizer::NMFOptimizer(SparseMatrix const& graph, NMFParams const& params)
	: graph(graph)
	, params(params)
	, clustering(graph.cols(), min(params.max_num_cluster, graph.cols()))
	//, clus_sum(min(params.max_num_cluster, graph.cols()))
	//, clus_sumsq(min(params.max_num_cluster, graph.cols()))
	, node_perm(graph.cols())
	, neighbors(min(params.max_num_cluster, graph.cols()))
{
	// initialize node_perm
	for (node_t i = 0 ; i < size() ; ++i) {
		node_perm[i] = i;
	}
	// initial solution
	reset();
}

void NMFOptimizer::reset() {
	// initialize clustering
	clustering.clear();
	for (node_t i = 0 ; i < size() ; ++i) {
		clus_t k = i % max_num_clus();
		double w = 1.0;
		clustering.add(i, k, w);
	}
	// loss
	calculate_loss();
	losses.clear();
	losses.reserve(params.num_iter+1);
	losses.push_back(loss);
}

void NMFOptimizer::calculate_loss() {
	loss = nmf_cluster::calculate_loss(params.objective, graph, clustering);
}
void NMFOptimizer::calculate_loss_debug() const {
	double loss = 0.0;
	params.debug_out << "    loss = ";
	for (int i = 0 ; i < size(); ++i) {
		if (i > 0) params.debug_out << "         + ";
		double loss_i = 0.;
		for (int j = 0 ; j < size(); ++j) {
			if (j > 0) params.debug_out << " + ";
			if (i == j) {
				params.debug_out << "      ";
			} else {
				double vh = dot(clustering[i], clustering[j]);
				double v  = graph(i,j);
				params.debug_out << "(" << v << "-" << vh << ")²";
				loss_i += (v-vh)*(v-vh);
			}
		}
		params.debug_out << "  -- " << loss_i << endl;
		loss += loss_i;
	}
	params.debug_out << "         = " << loss << endl;;
}

void NMFOptimizer::run() {
	if (params.verbosity >= 1) {
		calculate_loss();
		params.debug_out << "inital loss: " << loss << endl;
		if (params.verbosity >= 4) calculate_loss_debug();
		if (params.verbosity >= 2) {
			params.debug_printf("  solution: \n");
			for (int i = 0 ; i < size() ; ++i) {
				params.debug_out << "    " << i << " " << setprecision(15) << clustering[i] << endl;
			}
		}
	}
	// passes
	for (int i = 0 ; i < params.num_iter ; ++i) {
		bool change = optimization_pass();
		if (params.verbosity >= 1) {
			params.debug_printf("iteration: %4d  loss: %f\n",i,loss);
			if (params.verbosity >= 4) calculate_loss_debug();
			if (params.verbosity >= 2) {
				params.debug_printf("  solution: \n");
				for (int i = 0 ; i < size() ; ++i) {
					params.debug_out << "    " << i << " " << setprecision(15) << clustering[i] << endl;
				}
			}
		}
		if (!change) break;
		losses.push_back(loss);
	}
}

bool NMFOptimizer::optimization_pass() {
	// shuffle
	bool change = false;
	std::random_shuffle(node_perm.begin(), node_perm.end());
	for (vector<int>::const_iterator i_it = node_perm.begin() ; i_it != node_perm.end() ; ++i_it) {
		OCTAVE_QUIT;
		//change |= optimize_for_node(*i_it);
		change |= simple_greedy_move(*i_it);
	}
	return change;
}

bool NMFOptimizer::optimize_for_node(node_t i) {
	greedy_move(i);
	return true;
}



void NMFOptimizer::greedy_move(node_t i) {
/*
	// weight to cluster c should be (something like) A/H
	// as a test case, consider (node j, edge ij, membership j c₁, membership j c₁)
	//   j  A_ij  U_j1  U_j2
	//   1  1     1     0
	//   2  2     0.3   0.6
	//   3  1     0.8   0.1
	// now with guassian loss
	//   L = ∑{j}(A_ij - ∑{k}U_ik*U_jk)^2 + ∑{k} 1/2 β U_ik^2
	//   ∂L/∂U_ik = ∑{j}U_jk*(∑{l}U_il*U_jl - A_ij) + β U_ik
	//            = ∑{j}U_jk*(U_ik - A_ij + ∑{l≠k}U_il*U_jl) + β U_ik = 0
	// assuming only one non-zero, Uil=0 for l≠optimal k
	// so
	//     ∑{j}U_jk*U_ik + β U_ik = ∑{j}U_jk*A_ij
	//  => U_ik = (∑{j}U_jk*A_ij) / (β + ∑{j}U_jk)
	// this changes L by
	//   ΔL = L'-L = ∑{j}A_ij*U_jk*(U_ik'-U_ik) + (∑{l}U_il*U_jl)*U_jk*(U_ik'-U_ik) + U_jk^2*(U_ik'^2-U_ik^2) + ½β (U_ik'² - U_ik²)
	//   L = ∑{j}A_ij^2 - 2*A_ij*(∑{k}U_ik*U_jk) + (∑{k}U_ik*U_jk)^2 + ∑{l} 1/2 β U_il^2
	if (params.verbosity >= 2) {
		params.debug_out << "  greedy moves for " << i << ", currently " << clustering[i] << endl;
	}
	
	// find adjacent clusters, i.e. clusters to which we share at least some edge weight
	// collect sum of cluster membership and sum of edge weights*cluster membership
	neighbors.clear();
	for (ColumnIterator edge(graph,i); !edge.end(); ++edge) {
		for (SparseVector::const_iterator cit = clustering[edge.row()].begin() ; cit != clustering[edge.row()].end() ; ++cit) {
			neighbors.add(cit->clus, cit->weight * edge.data());
		}
	}
	// clear neighbors
	while (!clustering[i].empty()) {
		clus_t clus = clustering[i].back().clus;
		double w = clustering[i].back().weight;
		clus_sum  [clus] -= w;
		clus_sumsq[clus] -= w*w;
		clustering[i].pop_back();
	}
	// greedy allocate neighbors
	for (int num_neigh = 0 ; num_neigh < params.max_cluster_per_node ; ++num_neigh) {
		if (params.verbosity >= 3) {
			params.debug_out << "    edges: ";
			for (ColumnIterator edge(graph,i); !edge.end(); ++edge) {
				double pr = dot(clustering[i],clustering[edge.row()]);
				params.debug_out << i << "→" << edge.row() << "=" << edge.data() << "=" << pr << "+" << clustering[edge.row()] << "  ";
			}
			params.debug_out << endl;
		}
		if (params.verbosity >= 3) {
			params.debug_out << "    clus weights: " << neighbors << endl;
		}
		double best_weight = -1;
		double best_dloss = 0;
		clus_t best_clus = -1;
		for (Neighbors::const_iterator it = neighbors.begin() ; it != neighbors.end() ; ++it) {
			clus_t clus = *it;
			double a = neighbors.weight(clus);
			double b = clus_sumsq[clus] + params.objective.weight_beta;
			double weight = a / b;
			double dloss = -4*weight*a + 2*weight*weight*b;
			if (params.verbosity >= 4) {
				params.debug_out << "      to: " << clus << ", weight " << a << "/" << b << " = " << setprecision(5) << weight << ", dloss " << setprecision(5) << dloss << endl;
			}
			//if (weight > best_weight) {
			if (dloss < best_dloss) {
				best_weight = weight;
				best_clus = clus;
				best_dloss = dloss; // can't calculate this without looping over *all* nodes in clus
			}
		}
		// did we find something
		if (best_weight <= epsilon) break;
		if (params.verbosity >= 3) {
			double loss_before = calculate_loss();
			//params.debug_out << "    consider " << i << " to " << best_clus << " with " << fixed << setprecision(5) << best_weight << " for Δ " << best_dloss << ", loss:" << calculate_loss() << endl;
			params.debug_printf("    consider %d to %d with %.1f for Δ %.1f, loss: %.1f\n", i, best_clus, best_weight, best_dloss, loss_before);
			if (params.verbosity >= 4 && false) calculate_loss_debug();
			// try new loss for different weights
			if (params.verbosity >= 5) {
				double ws[]={0,0.5*best_weight,best_weight,1.5*best_weight};
				for (int k=0;k<4;++k) {
					double a = neighbors.weight(best_clus);
					double b = (clus_sumsq[best_clus] + params.objective.weight_beta);
					double delta_loss = -4*ws[k]*a + 2*ws[k]*ws[k]*b;
					// for validation, calculate delta loss the brute force way
					SparseVector ci = clustering[i];
					ci.push_back(SparseItem(best_clus,ws[k]));
					ci.sort();
					swap(clustering[i],ci);
					double loss_after = calculate_loss();
					swap(clustering[i],ci);
					//params.debug_out << "      with " << setw(6) << ws[k] << " loss " << setw(6) << setprecision(1) << fixed << loss_after << "  Δ " << (loss_after - loss_before) << endl;
					params.debug_printf("      with %.2f, loss %.2f, Δ %.2f = %.2f\n", ws[k], loss_after, loss_after - loss_before, delta_loss);
				}
			}
		}
		// add to clustering for node i
		clustering[i].push_back(SparseItem(best_clus,best_weight));
		clus_sum[best_clus] += best_weight;
		clus_sumsq[best_clus] += best_weight*best_weight;
		// don't consider it again
		neighbors.add(best_clus, -1e100); // note: using -INFINITY would break the SparseMap
		// update neighbors
		for (ColumnIterator edge(graph,i); !edge.end(); ++edge) {
			// is this item in cluster j?
			if (edge.row() == i) {
				// TODO: self loop, what to do?
			} else {
				double clus_weight = clustering[edge.row()](best_clus);
				if (clus_weight != 0.0) {
					// I have to iterate over all edge.row()'s clusters
					// this makes this function cost O(E/V*k^2)
					// I could reduce this to O(E/V*k+k^2)
					for (SparseVector::const_iterator cit = clustering[edge.row()].begin() ; cit != clustering[edge.row()].end() ; ++cit) {
						neighbors.add(cit->clus, -cit->weight * clus_weight*best_weight);
					}
				}
			}
		}
	}
	// we need clustering to be sorted (invariant of SparseVector)
	clustering[i].sort();
	// 
	//  U = U ./ (sum(V,1) + U.^(p-1)*beta + epsilon) .* (A_over_Ah * V);
	// do some gradient iterations?
	if (params.verbosity >= 2) {
		params.debug_out << "  new solution for " << i << " " << clustering[i] << ", loss " << calculate_loss() << endl;
		if (params.verbosity >= 4) calculate_loss_debug();
	}
	*/
}

bool NMFOptimizer::simple_greedy_move(node_t i) {
	// current solution
	calculate_loss();
	double old_loss = loss;
	SparseVector old_clustering_i = clustering[i];
	// 0. prepare
	set_active_node(i);
	// 1. clear the memberships for node i
	clustering.clear_node(i);
	calculate_loss();
	// 2. greedily add to clusters
	bool change = true;
	while (change && (int)clustering[i].nnz() < params.max_cluster_per_node) {
		change = simple_greedy_add_move(i);
	}
	// if the solution is not better, then restore the old one
	//if (loss < old_loss || (loss == old_loss && clustering[i].nnz() < old_clustering_i.nnz())) {
	if (loss < old_loss - epsilon || (loss <= old_loss + epsilon && clustering[i].nnz() < old_clustering_i.nnz())) {
		return true;
	} else {
		if (params.verbosity >= 1) {
			params.debug_out << " Reject all these moves, old = " << old_loss << " <= " << loss << endl;
		}
		clustering.set(i, old_clustering_i);
		loss = old_loss;
		return false;
	}
}

double NMFOptimizer::loss_after_addition_slow(node_t i, clus_t clus, double weight) {
	// very inefficient calculation
	clustering.add(i,clus,weight);
	double new_loss = nmf_cluster::calculate_loss(params.objective, graph, clustering);
	clustering.remove(i,clus);
	return new_loss;
}
double NMFOptimizer::loss_after_addition(node_t i, clus_t clus, double weight) {
	// less inefficient calculation
	double old_row_loss = calculate_loss_row(params.objective, graph, clustering, i);
	clustering.add(i,clus,weight);
	double new_row_loss = calculate_loss_row(params.objective, graph, clustering, i);
	clustering.remove(i,clus);
	double new_loss = loss + new_row_loss - old_row_loss;
	// check (slow!)
	if (params.verbosity >= 5 && abs(loss_after_addition_slow(i,clus,weight) - new_loss) > epsilon) {
		params.debug_out << "Loss is incorrect: for " << i << " " << clus << " with " << weight << endl;
		params.debug_out << "   " << new_loss << " vs " << loss_after_addition_slow(i,clus,weight) << endl;
	}
	return new_loss;
}

void NMFOptimizer::set_active_node(node_t i) {
	
	// find adjacent clusters, i.e. clusters to which we share at least some edge weight
	// collect sum of cluster membership and sum of edge weights*cluster membership
	neighbors.clear();
	for (ColumnIterator edge(graph,i); !edge.end(); ++edge) {
		for (SparseVector::const_iterator cit = clustering[edge.row()].begin() ; cit != clustering[edge.row()].end() ; ++cit) {
			neighbors.add(cit->clus, cit->weight * edge.data());
		}
	}
}

bool NMFOptimizer::simple_greedy_add_move(node_t i) {
	if (params.verbosity >= 2) {
		params.debug_out << "  Move " << i << " to 0.." << max_num_clus() << ", initial loss: " << loss << endl;
		if (false) {
			params.debug_printf("  current solution: \n");
			for (int i = 0 ; i < size() ; ++i) {
				params.debug_out << "    " << i << " " << setprecision(15) << clustering[i] << endl;
			}
		}
	}
	SparseVector cur_clustering = clustering[i];
	double best_loss = loss;
	clus_t best_clus = -1;
	double best_weight = 0.;
	// try all neighboring clusters
	// TODO: we should only try neighboring clusters
	//for (clus_t clus = 0 ; clus < max_num_clus() ; ++clus) {
	for (Neighbors::const_iterator nit = neighbors.begin() ; nit != neighbors.end() ; ++nit) {
		clus_t clus = *nit;
		if (clustering[i].contains(clus)) continue; // i is already in this cluster
		// try with the given move
		double weight = greedy_optimal_membership(i, clus);
		if (weight <= epsilon) continue;
		double new_loss = loss_after_addition(i, clus, weight);
		// check that we indeed found a locally optimal weight
		if (params.verbosity >= 4) {
			params.debug_out << "    Consider move: " << i << " to " << clus << " with " << weight << ": " << loss << " -> " << new_loss << endl;
			if (loss_after_addition(i, clus, max(0.,weight - 1e-4)) < new_loss ||
			    loss_after_addition(i, clus, max(0.,weight + 1e-4)) < new_loss) {
				params.debug_out << "Not an optimal weight: " << i << " to " << clus << " with " << weight << endl
								 << " losses: " << loss_after_addition(i, clus, weight - 1e-4) << "  "
								 << new_loss << "  " << loss_after_addition(i, clus, weight + 1e-4) << endl;
				params.debug_printf("  current solution: \n");
				for (int i = 0 ; i < size() ; ++i) {
					params.debug_out << "    " << i << " " << setprecision(15) << clustering[i] << endl;
				}
			}
		}
		// is it better?
		if (new_loss < best_loss) {
			best_loss = new_loss;
			best_clus = clus;
			best_weight = weight;
		}
	}
	if (best_loss < loss) {
		// add to a cluster
		if (params.verbosity >= 2) {
			params.debug_out << "  Accept move: " << i << " to " << best_clus << " with " << best_weight << ": " << loss << " -> " << best_loss << endl;
		}
		loss = best_loss;
		clustering.add(i, best_clus, best_weight);
		return true;
	} else {
		if (params.verbosity >= 2) {
			params.debug_out << "  No more moves, loss is " << loss << endl;
		}
		return false;
	}
}

// optimal weight for adding node i to a certain cluster
double NMFOptimizer::greedy_optimal_membership(node_t i, clus_t clus) const {
	assert(clustering[i](clus) == 0.);
	if (params.objective.likelihood == LH_GAUSSIAN || true) {
		// Minimize
		//   L = 0.5*∑{i<j} (∑{k} U{i,k}U{j,k} - A{i,j})^2  +  0.5*beta*∑{ik}U{i,k}^2 + ..
		// with respect to U{i₀,k₀}
		//   ∂L/∂U{i₀,k₀} = ∑{j≠i₀} U{j,k₀}(∑{k} U{i₀,k}U{j,k} - A{i₀,j}) + beta*U{i₀,k₀}
		//                = ∑{j≠i₀,k} U{j,k₀}U{i₀,k}U{j,k} - ∑{j≠i₀}(U{j,k₀}A{i₀,j}) + beta*U{i₀,k₀}
		//                = ∑{j≠i₀,k≠k₀} U{j,k₀}U{i₀,k}U{j,k} - ∑{j≠i₀}(U{j,k₀}A{i₀,j}) + (beta + ∑{j≠i₀}U{j,k₀}²)*U{i₀,k₀}
		//                = 0 
		// assuming that U{i₀,k} = 0, ∑{j≠i₀,k≠k₀} U{j,k₀}U{i₀,k}U{j,k} = ∑{j≠i₀,k} U{j,k₀}U{i₀,k}U{j,k}
		// solving gives
		//   U{i₀,k₀} = (..) / (beta + ∑{j}U[j,k₀]^2)
		double a = 0.;
		double b = params.objective.weight_beta;
		// TODO: we only need to consider nodes j that are in clus
		for (node_t j = 0 ; j < size() ; ++j) {
			if (j == i) continue;
			double j_clus = clustering[j](clus);
			if (j_clus == 0.) continue; // optimization
			a += j_clus * (dot(clustering[i], clustering[j]) - graph(i,j));
			b += j_clus * j_clus;
		}
		if (b == 0.) return 0.;
		return max(0., -a / b);
	} else if (params.objective.likelihood == LH_POISSON) {
		// Minimize
		//   L = ∑{i,j}( A{i,j}*log(∑{k}U{i,k}U{j,k}) - (∑{k}U{i,k}U{j,k}) )
		//   ∂L/∂U{i₀,k₀} = ∑{j}( A{i₀,j} * U{j,k₀} / (∑{k}U{i₀,k}U{j,k}) - U{j,k₀} ) = 0
		// there is no simple solution
		throw "TODO";
	} else {
		throw std::invalid_argument("unknown likelihood");
	}
}

// -----------------------------------------------------------------------------
}
#endif
