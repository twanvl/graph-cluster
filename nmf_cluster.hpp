// -----------------------------------------------------------------------------
// Local Search Non-negative Matrix Factorization
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#include "util.hpp"
#include <vector>
#include <iomanip>

#include <stdarg.h>

namespace nmf_cluster {

using namespace lso_cluster;

// -----------------------------------------------------------------------------
// Data structures
// -----------------------------------------------------------------------------

struct SparseItem {
	clus_t clus;
	double weight;
	SparseItem() {}
	SparseItem(clus_t clus, double weight) : clus(clus), weight(weight) {}
};
inline bool operator < (SparseItem const& x, clus_t y) {
	return x.clus < y;
}
inline bool operator < (clus_t x, SparseItem const& y) {
	return x < y.clus;
}
inline bool operator < (SparseItem const& x, SparseItem const& y) {
	return x.clus < y.clus;
}

template <class ForwardIterator> bool is_sorted (ForwardIterator first, ForwardIterator last) {
  if (first==last) return true;
  ForwardIterator next = first;
  while (++next!=last) {
    if (*next<*first) return false;
    ++first;
  }
  return true;
}

struct SparseVector : public std::vector<SparseItem> {
	// lookup a value
	double operator () (clus_t i) const {
		assert(is_sorted(this->begin(),this->end()));
		const_iterator it = std::lower_bound(this->begin(),this->end(),i);
		if (it != this->end() && it->clus == i) {
			return it->weight;
		} else {
			return 0.;
		}
	}
	// re-sort by cluster id
	void sort() {
		std::sort(this->begin(), this->end());
	}
};

double dot(SparseVector const& x, SparseVector const& y) {
	/*SparseVector::const_iterator it1 = x.begin(), it2 = y.begin();
	if (it1 == x.end()) return 0.;
	if (it2 == y.end()) return 0.
	double sum = 0.;
	while (true) {
		if (it1->clus < it2->clus) {
			if (++it1 == x.end()) break;
		} else if (it1->clus > it2->clus) {
			if (++it2 == y.end()) break;
		} else {
			sum += it1->weight * it2->weight;
			if (++it1 == x.end()) break;
			if (++it2 == y.end()) break;
		}
	}
	return sum;*/
	double sum = 0.;
	for (SparseVector::const_iterator it1 = x.begin(), it2 = y.begin() ; it1 != x.end() && it2 != y.end() ; ) {
		if (it1->clus < it2->clus) {
			++it1;
		} else if (it1->clus > it2->clus) {
			++it2;
		} else {
			sum += it1->weight * it2->weight;
			++it1; ++it2;
		}
	}
	return sum;
}

double sumsq(SparseVector const& x) {
	double sum = 0.;
	for (SparseVector::const_iterator it = x.begin() ; it != x.end() ; ++it) {
		sum += it->weight * it->weight;
	}
	return sum;
}

void operator += (vector<double>& x, SparseVector const& y) {
	for (SparseVector::const_iterator it = y.begin() ; it != y.end() ; ++it) {
		x[it->clus] += it->weight;
	}
}

// -----------------------------------------------------------------------------
// Data structures
// -----------------------------------------------------------------------------

typedef std::vector<SparseVector> NMFClustering;

enum LikelihoodFun {
	LH_GAUSSIAN,
	LH_POISSON,
};
enum PriorFun {
	PRIOR_HALF_NORMAL,
	PRIOR_GAMMA,
};

struct NMFParams {
	int num_iter;
	int max_cluster_per_node;
	int max_num_cluster;
	LikelihoodFun likelihood;
	double beta;
	// output
	int verbosity;
	std::ostream& debug_out;
	
	NMFParams(std::ostream& debug_out)
		: num_iter(100)
		, max_cluster_per_node(std::numeric_limits<int>::max())
		, max_num_cluster(std::numeric_limits<int>::max())
		, likelihood(LH_GAUSSIAN)
		, beta(0.)//1e-10)
		, verbosity(0)
		, debug_out(debug_out)
	{}
	
	void debug_printf(const char* fmt,...) const;
};

class NMFOptimizer {
  private:
	// inputs
	SparseMatrix graph;
	NMFParams const& params;
	
	// outputs
	NMFClustering clustering;
	vector<double> clus_sum; // total membership of each cluster, invariant: clus_size[k] = sum_i{ clustering[i](k) }
	vector<double> clus_sumsq;
	//vector<double> prior;
	double loss;
	
	// state used during computation
	
	// iterating over nodes in a random order
	mutable vector<node_t> node_perm;
	
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

ostream& operator << (ostream& out, SparseVector const& vec) {
	out << "[";
	for (SparseVector::const_iterator it = vec.begin() ; it != vec.end() ; ++it) {
		if (it != vec.begin()) out << ", ";
		out << it->clus << ":" << it->weight;
	}
	return out << "]";
}
template <typename T> ostream& operator << (ostream& out, SparseMap<T> const& vec) {
	out << "{";
	for (SparseMap<pair<double,double> >::const_iterator it = vec.begin() ; it != vec.end() ; ++it) {
		if (it != vec.begin()) out << ", ";
		out << *it << ":" << vec.weight(*it);
	}
	return out << "}";
}
ostream& operator << (ostream& out, SparseMap<pair<double,double> > const& vec) {
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

double NMFOptimizer::calculate_loss() const {
	double loss = 0.0;
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
	return loss;
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
	}
	// passes
	for (int i = 0 ; i < params.num_iter ; ++i) {
		optimization_pass();
		if (params.verbosity >= 1) {
			params.debug_printf("iteration: %4d  loss: %f\n",i,loss);
			if (params.verbosity >= 4) calculate_loss_debug();
			if (params.verbosity >= 2) {
				params.debug_printf("  solution: \n");
				for (int i = 0 ; i < size() ; ++i) {
					params.debug_out << "    " << i << " " << clustering[i] << endl;
				}
			}
		}
	}
}

bool NMFOptimizer::optimization_pass() {
	// shuffle
	bool change = false;
	std::random_shuffle(node_perm.begin(), node_perm.end());
	for (vector<int>::const_iterator i_it = node_perm.begin() ; i_it != node_perm.end() ; ++i_it) {
		OCTAVE_QUIT;
		change |= optimize_for_node(*i_it);
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
		double best_dloss = -1;
		clus_t best_clus = -1;
		for (Neighbors::const_iterator it = neighbors.begin() ; it != neighbors.end() ; ++it) {
			clus_t clus = *it;
			double weight = neighbors.weight(clus) / (clus_sumsq[clus] + params.beta);
			if (weight > best_weight) {
				best_weight = weight;
				best_clus = clus;
				best_dloss = -2 * 0; // can't calculate this without looping over *all* nodes in clus
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
					clustering[i].push_back(SparseItem(best_clus,ws[k]));
					double loss_after = calculate_loss();
					clustering[i].pop_back();
					//params.debug_out << "      with " << setw(6) << ws[k] << " loss " << setw(6) << setprecision(1) << fixed << loss_after << "  Δ " << (loss_after - loss_before) << endl;
					params.debug_printf("      with %.2f, loss %.2f, Δ %.2f\n", ws[k], loss_after, loss_after - loss_before);
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


// -----------------------------------------------------------------------------
}
