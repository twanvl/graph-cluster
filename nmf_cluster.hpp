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
#include <vector>
#include <iomanip>
#include <stdexcept>

#include <stdarg.h>
#include <stdio.h>

namespace nmf_cluster {

using namespace lso_cluster;
using namespace std;

// An NMF clustering gives for each node the clusters of which it is a member
typedef std::vector<SparseVector> NMFClustering;

// -----------------------------------------------------------------------------
// Objective functions
// -----------------------------------------------------------------------------

enum LikelihoodFun {
	LH_FLAT,
	LH_GAUSSIAN,
	LH_POISSON,
};
enum WeightPriorFun {
	PRIOR_FLAT,
	PRIOR_HALF_NORMAL,
	PRIOR_GAMMA,
};
enum SizePriorFun { // prior on cluster size
	SIZE_FLAT, // no prior
	SIZE_CRP,
};
enum SupportPriorFun { // prior on clusters-per-node
	SUPPORT_FLAT,
	SUPPORT_ONE, // exactly one cluster per node
	SUPPORT_POISSON
};

struct NMFObjectiveFun {
	LikelihoodFun      likelihood;
	WeightPriorFun     weight_prior;
	SizePriorFun       size_prior;
	SupportPriorFun    support_prior;
	double             likelihood_beta; // inverse scale parameter of likelihood
	double             weight_beta;     // inverse scale parameter of membership prior
	double             support_lambda;
	bool               exclude_diagonal; // exclude A(i,i) from likelihood calculation
	bool               fixed_normalization; // pretend that the number of clusters is fixed, include normalization constant for all membership coefficients
	
	NMFObjectiveFun()
		: likelihood      (LH_GAUSSIAN)
		, weight_prior    (PRIOR_HALF_NORMAL)
		, size_prior      (SIZE_FLAT)
		, support_prior   (SUPPORT_FLAT)
		, likelihood_beta (1.0)
		, weight_beta     (0.0)
		, support_lambda  (0.0)
		, exclude_diagonal(true)
		, fixed_normalization(false)
	{}
};

// Calculate the loss
double calculate_loss(NMFObjectiveFun const& obj, SparseMatrix const& graph, NMFClustering const& clustering);

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
	
	// intermediate/auxilliary values
	std::vector<double> clus_sum; // total membership of each cluster, invariant: clus_size[k] = sum_i{ clustering[i](k) }
	std::vector<double> clus_sumsq;
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
	double calculate_loss() const;
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
	
  public:
	NMFOptimizer(SparseMatrix const& graph, NMFParams const& params);
	
	// run the optimizer
	void run();
	
	// number of nodes
	inline node_t size() const {
		return (node_t)clustering.size();
	}
	inline clus_t max_num_clus() const {
		return (clus_t)neighbors.max_size();
	}
	// return the current clustering
	SparseMatrix get_clustering() const;
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
// Public interface
// -----------------------------------------------------------------------------

SparseMatrix NMFOptimizer::get_clustering() const {
	// find number of non-empty clusters, and remap them to [0..]
	std::vector<int> clus_id(max_num_clus(), -1);
	size_t num_clus = 0, num_inclus = 0;
	for (NMFClustering::const_iterator it = clustering.begin() ; it != clustering.end() ; ++it) {
		for (SparseVector::const_iterator it2 = it->begin() ; it2 != it->end() ; ++it2) {
			if (clus_id[it2->clus] == -1) {
				clus_id[it2->clus] = num_clus;
				num_clus++;
			}
			num_inclus++;
		}
	}
	
	// build clustering matrix
	// transpose of the usual notation: rows=clusters
	SparseMatrix out((int)num_clus,(int)size(),(int)num_inclus);
	int k = 0;
	out.cidx(0) = k;
	for (node_t j = 0 ; j < size() ; ++j) {
		for (SparseVector::const_iterator it2 = clustering[j].begin() ; it2 != clustering[j].end() ; ++it2) {
			out.ridx(k) = clus_id[it2->clus];
			out.data(k) = it2->weight;
			k++;
		}
		out.cidx(j) = k;
	}
	return out;
}

// -----------------------------------------------------------------------------
// Optimization
// -----------------------------------------------------------------------------

static const double epsilon = 1e-10;

NMFOptimizer::NMFOptimizer(SparseMatrix const& graph, NMFParams const& params)
	: graph(graph)
	, params(params)
	, clustering(graph.cols())
	, clus_sum(min(params.max_num_cluster, graph.cols()))
	, clus_sumsq(min(params.max_num_cluster, graph.cols()))
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
	std::fill(clus_sum.begin(), clus_sum.end(), 0.);
	std::fill(clus_sumsq.begin(), clus_sumsq.end(), 0.);
	for (node_t i = 0 ; i < size() ; ++i) {
		clus_t k = i % max_num_clus();
		double w = 1.0;
		clustering[i].clear();
		clustering[i].push_back(SparseItem(k,w));
		clus_sum[k] += w;
		clus_sumsq[k] += w*w;
	}
	// loss
	calculate_loss();
}

double log_factorial(int n) {
	double lf = 0.;
	for (int i = 2 ; i < n ; ++i) {
		lf += log(i);
	}
	return lf;
}

// loss is negative log likelihood
double calculate_loss(NMFObjectiveFun const& obj, SparseMatrix const& graph, NMFClustering const& clustering, size_t max_num_clus) {
	double loss = 0.0;
	// Loss for non-edges
	/*
	if (obj.likelihood == LH_GAUSSIAN) {
		// Gaussian loss:
		//   L = ∑{i<j} 0.5*(∑{k}U(i,k)U(j,k) - A(i,j))^2
		//     = 0.5*∑{i<j}(∑{k}U(i,k)U(j,k))^2  -  ∑{i<j} A(i,j)*(∑{k}U(i,k)U(j,k))  +  0.5*∑{i<j} A(i,j)^2
		// Calculate first term:
		for (size_t i = 0 ; i < clustering.size(); ++i) {
			for (size_t j = i+1 ; j < clustering.size(); ++j) {
				double vh = dot(clustering[i], clustering[j]);
				loss += 0.5*vh*vh;
			}
		}
	} else if (obj.likelihood == LH_POISSON) {
		// Poisson:
		//     ∑{ij} ∑{k}U(i,k)U(j,k) - A(i,j)
		//   = ∑{k}(∑{i}U(i,k))(∑{j}U(j,k))
		vector<double> clus_weight(max_num_clus, 0.);
		for (size_t i = 0 ; i < clustering.size() ; ++i) {
			clus_weight += clustering[i];
		}
		for (vector<double>::const_iterator it = clus_weight.begin() ; it != clus_weight.end() ; ++it) {
			loss += *it * *it;
		}
		loss *= 0.5; // only count below diagonal
	} else {
		throw std::invalid_argument("Unknown likelihood");
	}
	// Loss for edges
	for (node_t i = 0 ; i < (int)clustering.size(); ++i) {
		for (ColumnIterator edge(graph,i); !edge.end(); ++edge) {
			if (i >= edge.row()) continue; // only below diagonal
			// predicted value, v̂ = (U*U')(i,j) = U(i,:)*U(j,:)
			// loss is 
			double vh = dot(clustering[i], clustering[edge.row()]);
			double v  = edge.data();
			if (obj.likelihood == LH_GAUSSIAN) {
				loss += 0.5*v*v - v*vh; // we already counted vh^2 above
			} else if (obj.likelihood == LH_POISSON) {
				loss += v * (log(v) - log(vh));
			}
		}
	}
	*/
	if (obj.likelihood == LH_GAUSSIAN) {
		// Gaussian loss:
		//   L = ∑{i<j} 0.5*(∑{k}U(i,k)U(j,k) - A(i,j))^2
		//     = 0.5*∑{i<j}(∑{k}U(i,k)U(j,k))^2  -  ∑{i<j} A(i,j)*(∑{k}U(i,k)U(j,k))  +  0.5*∑{i<j} A(i,j)^2
		// Calculate first term:
		for (size_t i = 0 ; i < clustering.size(); ++i) {
			for (size_t j = i+1 ; j < clustering.size(); ++j) {
				double vh = dot(clustering[i], clustering[j]);
				double v  = graph(i,j);
				loss += 0.5*(v-vh)*(v-vh);
			}
		}
	} else {
		throw "TODO";
	}
	
	// Total number of memberships (for normalization constant)
	size_t total_memberships = 0;
	if (obj.fixed_normalization) {
		total_memberships = clustering.size() * max_num_clus;
	} else {
		for (size_t i = 0 ; i < clustering.size() ; ++i) {
			total_memberships += clustering[i].nnz();
		}
	}
	// Regularization term / prior on membership coefficients
	if (obj.weight_prior == PRIOR_FLAT) {
		// nothing
	} else if (obj.weight_prior == PRIOR_HALF_NORMAL) {
		// L = ∑{ik} -log( sqrt(2*beta/pi) * exp(-0.5*beta*U(i,k)^2) )
		for (size_t i = 0 ; i < clustering.size() ; ++i) {
			loss += 0.5 * obj.weight_beta * sumsq(clustering[i]);
		}
		loss += total_memberships * sqrt(2.0 / M_PI * obj.weight_beta);
	} else {
		throw std::invalid_argument("Unknown weight prior");
	}
	
	// Cluster sizes
	vector<int> clus_size(max_num_clus, 0);
	for (size_t i = 0 ; i < clustering.size() ; ++i) {
		for (SparseVector::const_iterator it = clustering[i].begin() ; it != clustering[i].end() ; ++it) {
			clus_size[it->clus]++;
		}
	}
	for (vector<int>::const_iterator it = clus_size.begin() ; it != clus_size.end() ; ++it) {
		// the prior on size
		if (obj.size_prior == SIZE_FLAT) {
			// nothing
		} else if (obj.size_prior == SIZE_CRP) {
			loss += log_factorial(*it-1);
		} else {
			throw std::invalid_argument("Unknown size prior");
		}
	}
	
	// Clusters per node
	for (size_t i = 0 ; i < clustering.size() ; ++i) {
		if (obj.support_prior == SUPPORT_FLAT) {
			// nothing
		} else if (obj.support_prior == SUPPORT_ONE) {
			if (clustering[i].nnz() > 1) {
				loss += std::numeric_limits<double>::infinity();
			}
		} else if (obj.support_prior == SUPPORT_POISSON) {
			loss += -clustering[i].nnz() * log(obj.support_lambda) + log_factorial(clustering[i].nnz()) + obj.support_lambda;
		} else {
			throw std::invalid_argument("Unknown support prior");
		}
	}
	
	return loss;
}

double NMFOptimizer::calculate_loss() const {
	/*double loss = 0.0;
	// Loss for non-edges
	if (params.likelihood == LH_GAUSSIAN) {
		// Gaussian:
		//     ∑{ij} 0.5*(∑{k}U(i,k)U(j,k))^2
		// no fast solution
		for (int i = 0 ; i < size(); ++i) {
			for (int j = i+1 ; j < size(); ++j) {
				double vh = dot(clustering[i], clustering[j]);
				loss += vh*vh;
			}
		}
		loss *= 2;
	} else if (params.likelihood == LH_POISSON) {
		// Poisson:
		//     ∑{ij} ∑{k}U(i,k)U(j,k)
		//   = ∑{k}(∑{i}U(i,k))(∑{j}U(j,k))
		vector<double> clus_weight(max_num_clus(),0.);
		for (int i = 0 ; i < size() ; ++i) {
			clus_weight += clustering[i];
		}
		for (vector<double>::const_iterator it = clus_weight.begin() ; it != clus_weight.end() ; ++it) {
			loss += *it * *it;
		}
	}
	// Loss for edges
	for (int i = 0 ; i < size(); ++i) {
		for (ColumnIterator edge(graph,i); !edge.end(); ++edge) {
			// predicted value, v̂ = (U*U')(i,j) = U(i,:)*U(j,:)
			// loss is 
			double vh = dot(clustering[i], clustering[edge.row()]);
			double v  = edge.data();
			if (params.likelihood == LH_GAUSSIAN) {
				loss += v*v - 2*v*vh; // we already counted vh^2 above
			} else if (params.likelihood == LH_POISSON) {
				loss += v * (log(v) - log(vh));
			}
		}
	}
	// Regularization term
	for (int i = 0 ; i < size() ; ++i) {
		loss += 0.5 * params.beta * sumsq(clustering[i]);
	}
	// Prior
	return loss;
	*/
	return nmf_cluster::calculate_loss(params.objective, graph, clustering, max_num_clus());
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
		params.debug_out << "inital loss: " << calculate_loss() << endl;
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

void NMFOptimizer::set_active_node(node_t i) {
	//this->i = i;
	//old_clus_i = clustering[i];
	//old_loss_i = loss_part[i];
}


void NMFOptimizer::greedy_move(node_t i) {
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
}

// remove a node from all clusters
struct RemoveMove {
	node_t i;
};
// add a node to a cluster (it should not already be in that cluster)
struct AddMove {	
	node_t i;
	clus_t clus;
	double weight;
};

bool NMFOptimizer::simple_greedy_move(node_t i) {
	// current solution
	loss = calculate_loss();
	double old_loss = loss;
	SparseVector old_clustering_i = clustering[i];
	// 1. clear the memberships for node i
	clustering[i].clear();
	loss = calculate_loss();
	// 2. greedily add to clusters
	bool change = true;
	while (change) {
		change = simple_greedy_add_move(i);
	}
	// if the solution is not better, then restore the old one
	if (loss < old_loss || (loss == old_loss && clustering[i].nnz() < old_clustering_i.nnz())) {
		return true;
	} else {
		if (params.verbosity >= 1) {
			params.debug_out << " Reject all these moves, " << old_loss << " >= " << loss << endl;
		}
		clustering[i] = old_clustering_i;
		loss = old_loss;
		return false;
	}
}

double NMFOptimizer::loss_after_addition(node_t i, clus_t clus, double weight) {
	// very inefficient calculation
	SparseVector cur_clustering = clustering[i];
	clustering[i].push_back(SparseItem(clus,weight));
	clustering[i].sort();
	double new_loss = calculate_loss();
	clustering[i] = cur_clustering;
	return new_loss;
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
	SparseItem best_clus;
	for (clus_t clus = 0 ; clus < max_num_clus() ; ++clus) {
		if (clustering[i].contains(clus)) continue; // i is already in this cluster
		// try with the given move
		double weight = greedy_optimal_membership(i, clus);
		double new_loss = loss_after_addition(i, clus, weight);
		// check that we indeed found a locally optimal weight
		if (params.verbosity >= 4) {
			params.debug_out << "    Consider move: " << i << " to " << clus << " with " << weight << ": " << loss << " -> " << new_loss << endl;
		}
		if (loss_after_addition(i, clus, max(0.,weight - 1e-4)) < new_loss ||
		    loss_after_addition(i, clus, max(0.,weight + 1e-4)) < new_loss) {
			params.debug_out << "Not an optimal weight: " << i << " to " << clus << " with " << weight << endl
			                 << " losses: " << loss_after_addition(i, clus, weight - 1e-4) << "  "
			                 << new_loss << "  " << loss_after_addition(i, clus, weight + 1e-4) << endl;
			/*
			params.debug_printf("  current solution: \n");
			for (int i = 0 ; i < size() ; ++i) {
				params.debug_out << "    " << i << " " << setprecision(15) << clustering[i] << endl;
			}
			SparseVector cur_clustering = clustering[i];
			clustering[i].push_back(SparseItem(clus,weight));
			clustering[i].sort();
			calculate_loss_debug();
			clustering[i] = cur_clustering;*/
		}
		// is it better?
		if (new_loss < best_loss) {
			best_loss = new_loss;
			best_clus = SparseItem(clus,weight);
		}
	}
	if (best_loss < loss) {
		// add to a cluster
		if (params.verbosity >= 2) {
			params.debug_out << "  Accept move: " << i << " to " << best_clus.clus << " with " << best_clus.weight << ": " << loss << " -> " << best_loss << endl;
		}
		loss = best_loss;
		clustering[i].push_back(best_clus);
		clustering[i].sort();
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
	if (params.objective.likelihood == LH_GAUSSIAN) {
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
		throw "TODO";
	} else {
		throw std::invalid_argument("unknown likelihood");
	}
}

// -----------------------------------------------------------------------------
}
#endif
