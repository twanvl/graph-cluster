// -----------------------------------------------------------------------------
// Local Search Non-negative Matrix Factorization
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#ifndef HEADER_NMF_OBJECTIVE_FUNCTION
#define HEADER_NMF_OBJECTIVE_FUNCTION

#include "nmf_clustering.hpp"
#include <vector>

namespace nmf_cluster {

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
	PRIOR_EXPONENTIAL,
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
		: likelihood      (LH_POISSON)
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

// -----------------------------------------------------------------------------
// Utility
// -----------------------------------------------------------------------------

double log_factorial(int n) {
	double lf = 0.;
	for (int i = 2 ; i < n ; ++i) {
		lf += log(i);
	}
	return lf;
}

// -----------------------------------------------------------------------------
// Calculation
// -----------------------------------------------------------------------------

double weight_prior_term(NMFObjectiveFun const& obj, double weight) {
	if (obj.weight_prior == PRIOR_FLAT) {
		return 0.; // nothing
	} else if (obj.weight_prior == PRIOR_HALF_NORMAL) {
		// L = ∑{ik} -log( sqrt(2*beta/pi) * exp(-0.5*beta*U(i,k)^2) )
		return 0.5 * obj.weight_beta * weight * weight;
	} else {
		throw std::invalid_argument("Unknown weight prior");
	}
}

double weight_prior_partition_term(NMFObjectiveFun const& obj) {
	if (obj.weight_prior == PRIOR_FLAT) {
		return 0.; // nothing
	} else if (obj.weight_prior == PRIOR_HALF_NORMAL) {
		// L = ∑{ik} -log( sqrt(2*beta/pi) * exp(-0.5*beta*U(i,k)^2) )
		if (obj.weight_beta == 0.) {
			return 0.; // treat as a flat prior
		} else {
			return log(sqrt(2.0 / M_PI * obj.weight_beta));
		}
	} else {
		throw std::invalid_argument("Unknown weight prior");
	}
}

double size_prior_term(NMFObjectiveFun const& obj, int size) {
	if (obj.size_prior == SIZE_FLAT) {
		return 0.; // nothing
	} else if (obj.size_prior == SIZE_CRP) {
		return log_factorial(size - 1);
	} else {
		throw std::invalid_argument("Unknown size prior");
	}
}

double support_prior_term(NMFObjectiveFun const& obj, int nnz) {
	if (obj.support_prior == SUPPORT_FLAT) {
		return 0.; // nothing
	} else if (obj.support_prior == SUPPORT_ONE) {
		if (nnz > 1) {
			return std::numeric_limits<double>::infinity();
		} else {
			return 0.;
		}
	} else if (obj.support_prior == SUPPORT_POISSON) {
		return -nnz * log(obj.support_lambda) + log_factorial(nnz) + obj.support_lambda;
	} else {
		throw std::invalid_argument("Unknown support prior");
	}
}

// loss is negative log likelihood
double calculate_loss(NMFObjectiveFun const& obj, SparseMatrix const& graph, NMFClustering const& clustering) {
	double loss = 0.0;
	// Loss for non-edges
	if (obj.likelihood == LH_GAUSSIAN) {
		// Gaussian loss:
		//   L = ∑{ij} 0.5*(∑{k}U(i,k)U(j,k) - A(i,j))^2
		//     = 0.5*∑{ij}(∑{k}U(i,k)U(j,k))^2  -  ∑{ij} A(i,j)*(∑{k}U(i,k)U(j,k))  +  0.5*∑{ij} A(i,j)^2
		// note: only include the strict lower diagonal part
		for (node_t i = 0 ; i < clustering.num_nodes() ; ++i) {
			for (DenseColumnIterator edge(graph,i); edge.row() < i; ++edge) {
				double vh = dot(clustering[i], clustering[edge.row()]);
				double v  = edge.data();
				loss += 0.5*(v - vh)*(v - vh);
			}
		}
	} else if (obj.likelihood == LH_POISSON) {
		// Poisson:
		//    L = ∑{ij} -log( (∑{k} U{i,k}U{j,k})^A{i,j}/A{i,j}! * exp(-∑{k} U{i,k}U{j,k}) )
		//      = ∑{ij} -log(∑{k} U{i,k}U{j,k})*A{i,j} + log(A{i,j}!) + (∑{ij}∑{k} U{i,k}U{j,k})
		//      = ∑{ij} -log(∑{k} U{i,k}U{j,k})*A{i,j} + ∑{k}(∑{i}U{i,k})^2 + const.
		// first term is only for edges, second term is constant, third term can be decomposed
		
		// we only want to count below or above diagonal, i.e. (∑{ij,i<j}∑{k} U{i,k}U{j,k})
		// just count half, and to exclude the diagonal we need ∑{ik} U{i,k}^2, do that below
		for (clus_t k = 0; k < clustering.max_num_clus() ; ++k) {
			loss += 0.5 * clustering.clus_weight(k) * clustering.clus_weight(k);
		}
		for (node_t i = 0 ; i < clustering.num_nodes(); ++i) {
			loss -= 0.5 * sumsq(clustering[i]);
			for (ColumnIterator edge(graph,i); !edge.end(); ++edge) {
				if (edge.row() >= i) break; // only below diagonal
				double v  = edge.data();
				if (v == 0.) continue;
				double vh = dot(clustering[i], clustering[edge.row()]);
				const double epsilon = 1e-15;
				loss -= v * log(vh + epsilon);
			}
		}
	} else {
		throw std::invalid_argument("Unknown likelihood");
	}
	
	// Regularization term / prior on membership coefficients
	for (node_t i = 0 ; i < clustering.num_nodes() ; ++i) {
		for (SparseVector::const_iterator it = clustering[i].begin() ; it != clustering[i].end() ; ++it) {
			loss += weight_prior_term(obj, it->weight);
		}
	}
	loss += weight_prior_partition_term(obj) * (obj.fixed_normalization ? clustering.total_size() : clustering.nnz());
	
	// Cluster sizes
	for (clus_t k = 0 ; k < clustering.max_num_clus() ; ++k) {
		// the prior on size
		loss += size_prior_term(obj, clustering.clus_size(k));
	}
	
	// Clusters per node
	for (node_t i = 0 ; i < clustering.num_nodes() ; ++i) {
		loss += support_prior_term(obj, clustering[i].nnz());
	}
	
	return loss;
}

// Some part of the loss for a specific row
double calculate_loss_row(NMFObjectiveFun const& obj, SparseMatrix const& graph, NMFClustering const& clustering, node_t i) {
	double loss = 0.0;
	// Loss for non-edges
	if (obj.likelihood == LH_GAUSSIAN) {
		// Gaussian loss:
		//   L = ∑{i<j} 0.5*(∑{k}U(i,k)U(j,k) - A(i,j))^2
		//     = 0.5*∑{i<j}(∑{k}U(i,k)U(j,k))^2  -  ∑{i<j} A(i,j)*(∑{k}U(i,k)U(j,k))  +  0.5*∑{i<j} A(i,j)^2
		// Calculate first term:
		for (DenseColumnIterator edge(graph,i); edge.row() < i; ++edge) {
			double vh = dot(clustering[i], clustering[edge.row()]);
			double v  = edge.data();
			loss += 0.5*(v - vh)*(v - vh);
		}
	} else if (obj.likelihood == LH_POISSON) {
		// Poisson:
		//     ∑{ij} ∑{k}U(i,k)U(j,k) - A(i,j)
		//   = ∑{k}(∑{i}U(i,k))(∑{j}U(j,k))
		for (clus_t k = 0; k < clustering.max_num_clus() ; ++k) {
			loss += 0.5 * clustering.clus_weight(k) * clustering.clus_weight(k);
		}
		loss -= 0.5 * sumsq(clustering[i]);
		for (ColumnIterator edge(graph,i); !edge.end(); ++edge) {
			if (edge.row() == i) continue; // only off-diagonal
			double v  = edge.data();
			if (v == 0.) continue;
			double vh = dot(clustering[i], clustering[edge.row()]);
			const double epsilon = 1e-15;
			loss -= v * log(vh + epsilon);
		}
	} else {
		throw std::invalid_argument("Unknown likelihood");
	}
	
	// Regularization term / prior on membership coefficients
	for (SparseVector::const_iterator it = clustering[i].begin() ; it != clustering[i].end() ; ++it) {
		loss += weight_prior_term(obj, it->weight);
	}
	if (!obj.fixed_normalization) {
		loss += weight_prior_partition_term(obj) * clustering[i].nnz();
	}
	
	// Cluster sizes
	for (clus_t k = 0 ; k < clustering.max_num_clus() ; ++k) {
		// the prior on size
		loss += size_prior_term(obj, clustering.clus_size(k));
	}
	
	// Clusters per node
	loss += support_prior_term(obj, clustering[i].nnz());
	
	return loss;
}

/*
// How the loss would change if row i were cleared
double calculate_loss_clear_row(NMFObjectiveFun const& obj, SparseMatrix const& graph, NMFClustering const& clustering, node_t i) {
	// Loss for edges
	for (ColumnIterator edge(graph,i); !edge.end(); ++edge) {
		if (i == edge.row()) continue; // exclude diagonal, but do count below diagonal
		// predicted value, v̂ = (U*U')(i,j) = U(i,:)*U(j,:)
		// loss is 
		double vh = dot(clustering[i], clustering[edge.row()]);
		double v  = edge.data();
		if (obj.likelihood == LH_GAUSSIAN) {
			loss -= 0.5*v*v - v*vh; // we already counted vh^2 above
			loss += 0.5*v*v;
		} else if (obj.likelihood == LH_POISSON) {
			loss -= v * (log(v) - log(vh));
		}
	}
	
	// Regularization term / prior on membership coefficients
	for (SparseVector::const_iterator it = clustering[i].begin() ; it != clustering[i].end() ; ++it) {
		loss -= weight_prior_term(obj, it->weight);
	}
	if (!obj.fixed_normalization) {
		loss -= weight_prior_partition_term(obj) * clustering[i].nnz();
	}
	
	// Cluster sizes
	for (SparseVector::const_iterator it = clustering[i].begin() ; it != clustering[i].end() ; ++it) {
		loss -= size_prior_term(obj, clustering.clus_size(it->clus));
		loss += size_prior_term(obj, clustering.clus_size(it->clus)-1);
	}
	
	// Clusters per node
	loss -= support_prior_term(obj, clustering[i].nnz());
	loss += support_prior_term(obj, 0);
}

// How the loss would change if node i were cleared to cluster clus with weight w
*/

// -----------------------------------------------------------------------------
}
#endif
