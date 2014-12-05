// -----------------------------------------------------------------------------
// Local Search Non-negative Matrix Factorization
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#ifndef HEADER_NMF_CLUSTERING
#define HEADER_NMF_CLUSTERING

#include "util.hpp"
#include "sparse_matrix.hpp"
#include "sparse_vector.hpp"
#include <vector>
#include <iomanip>
#include <stdexcept>

namespace nmf_cluster {

using namespace lso_cluster;
using namespace std;

// -----------------------------------------------------------------------------
// Data type storing a clustering
// -----------------------------------------------------------------------------

// An NMF clustering gives for each node the clusters of which it is a member
struct NMFClustering {
  private:
	std::vector<SparseVector> clustering; // which clusters is each node in?
	std::vector<int>    cluster_size;     // size of each cluster
	std::vector<double> cluster_weight;   // total weight of each cluster
	friend struct IndirectClustering;
	friend void swap(NMFClustering&,NMFClustering&);
	
  public:
	NMFClustering() {}
	NMFClustering(int num_node, int max_num_clus)
		: clustering(num_node)
		, cluster_size  (max_num_clus, 0)
		, cluster_weight(max_num_clus, 0.)
	{}
	NMFClustering(std::vector<SparseVector> const& clustering, int max_num_clus)
		: clustering(clustering)
		, cluster_size  (max_num_clus, 0)
		, cluster_weight(max_num_clus, 0.)
	{
		recalc_internal_data();
	}
	
	// An upport bound on cluster ids
	inline clus_t max_num_clus() const {
		return (clus_t)cluster_size.size();
	}
	// Number of nodes
	inline node_t num_nodes() const {
		return (node_t)clustering.size();
	}
	inline int clus_size(clus_t k) const {
		return cluster_size[k];
	}
	inline double clus_weight(clus_t k) const {
		return cluster_weight[k];
	}
	int nnz() const {
		int nnz = 0;
		for (node_t i = 0 ; i < num_nodes() ; ++i) {
			nnz += clustering[i].nnz();
		}
		return nnz;
	}
	int total_size() const {
		return num_nodes() * max_num_clus();
	}
	int number_of_zeros() const {
		return total_size() - nnz();
	}
	bool empty() {
		return clustering.empty();
	}
	
	// iteration over the nodes
	typedef std::vector<SparseVector>::const_iterator const_iterator;
	inline const_iterator begin() const {
		return clustering.begin();
	}
	inline const_iterator end() const {
		return clustering.end();
	}
	inline SparseVector const& operator [] (node_t i) const {
		return clustering[i];
	}
	
	// Convert to a sparse matrix
	SparseMatrix to_sparse_matrix() const;
	// Convert from a sparse matrix
	NMFClustering(SparseMatrix const&);
	void operator = (SparseMatrix const&);
	
	// Convert to a hard clustering: cluster with highest membership for each node
	vector<clus_t> to_hard_clustering() const;
	clus_t best_cluster(node_t i) const;
	
	// Modification
	void add(node_t i, clus_t clus, double weight) {
		clustering[i].insert(clus, weight);
		cluster_size[clus]++;
		cluster_weight[clus] += weight;
	}
	void remove(node_t i, clus_t clus) {
		cluster_weight[clus] -= clustering[i].remove(clus);
		cluster_size[clus]--;
	}
	void clear() {
		std::fill(cluster_size.begin(),   cluster_size.end(),   0.);
		std::fill(cluster_weight.begin(), cluster_weight.end(), 0.);
		for (node_t i = 0 ; i < num_nodes() ; ++i) {
			clustering[i].clear();
		}
	}
	void clear_node(node_t i) {
		for (SparseVector::const_iterator it = clustering[i].begin() ; it != clustering[i].end() ; ++it) {
			cluster_size[it->clus]--;
			cluster_weight[it->clus] -= it->weight;
		}
		clustering[i].clear();
	}
	void set(node_t i, SparseVector const& new_clustering) {
		clear_node(i);
		for (SparseVector::const_iterator it = new_clustering.begin() ; it != new_clustering.end() ; ++it) {
			cluster_size[it->clus]++;
			cluster_weight[it->clus] += it->weight;
		}
		clustering[i] = new_clustering;
	}
  private:
	void recalc_internal_data() {
		std::fill(cluster_size.begin(),   cluster_size.end(),   0.);
		std::fill(cluster_weight.begin(), cluster_weight.end(), 0.);
		for (std::vector<SparseVector>::const_iterator clus_it = clustering.begin() ; clus_it != clustering.end() ; ++clus_it) {
			for (SparseVector::const_iterator it = clus_it->begin() ; it != clus_it->end() ; ++it) {
				assert((size_t)it->clus < cluster_size.size());
				assert((size_t)it->clus < cluster_weight.size());
				cluster_size[it->clus]++;
				cluster_weight[it->clus] += it->weight;
			}
		}
	}
};

SparseMatrix NMFClustering::to_sparse_matrix() const {
	// find number of non-empty clusters, and remap them to [0..]
	std::vector<int> clus_id(max_num_clus(), -1); // mapping cluster id to compressed cluster id
	size_t num_clus = 0;   // number of non-empty clusters
	size_t num_inclus = 0; // number of non-zeros
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
	// transpose of the usual notation, i.e. here columns=nodes, rows=clusters
	SparseMatrix out((int)num_clus, (int)clustering.size(), (int)num_inclus);
	int k = 0;
	out.cidx(0) = k;
	for (size_t j = 0 ; j < clustering.size() ; ++j) {
		for (SparseVector::const_iterator it2 = clustering[j].begin() ; it2 != clustering[j].end() ; ++it2) {
			out.ridx(k) = clus_id[it2->clus];
			out.data(k) = it2->weight;
			k++;
		}
		out.cidx(j+1) = k;
	}
	return out;
}

vector<clus_t> NMFClustering::to_hard_clustering() const {
	std::vector<clus_t> best_clus(num_nodes());
	for (node_t i = 0 ; i < num_nodes() ; ++i) {
		best_clus[i] = best_cluster(i);
	}
	compress_assignments(best_clus);
	return best_clus;
}

clus_t NMFClustering::best_cluster(node_t i) const {
	clus_t best = i;
	double max_weight = -1;
	for (SparseVector::const_iterator it = clustering[i].begin() ; it != clustering[i].end() ; ++it) {
		if (it->weight > max_weight) {
			best = it->clus;
			max_weight = it->weight;
		}
	}
	return best;
}

void NMFClustering::operator = (SparseMatrix const& mat) {
	NMFClustering that(mat);
	swap(*this,that);
}
NMFClustering::NMFClustering(SparseMatrix const& mat)
	: clustering(mat.cols())
	, cluster_size(mat.rows())
	, cluster_weight(mat.rows())
{
	for (int i = 0 ; i < mat.cols() ; ++i) {
		clustering[i] = ColumnIterator(mat, i);
	}
	recalc_internal_data();
}

void swap(NMFClustering& a, NMFClustering& b) {
	swap(a.clustering,     b.clustering);
	swap(a.cluster_size,   b.cluster_size);
	swap(a.cluster_weight, b.cluster_weight);
}

// -----------------------------------------------------------------------------
// A clustering of a clustering
// -----------------------------------------------------------------------------

// each cluster in the indirect clustering is a linear combination of clusters in the base clustering
// in other words, I=B*R
struct IndirectClustering {
	vector<vector<node_t> > nodes_in_cluster; // which base nodes are in which base cluster?
	NMFClustering const& base;
	NMFClustering refined;
	
	IndirectClustering(NMFClustering const& base, int max_num_clus)
		: base(base)
		, refined(base.max_num_clus(), max_num_clus)
	{}
	
	NMFClustering to_clustering() const {
		return NMFClustering(base.clustering * refined.clustering, refined.max_num_clus());
	}
};

// -----------------------------------------------------------------------------
}
#endif
