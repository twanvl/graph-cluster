// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#ifndef HEADER_LSO_CLUSTER_UTIL
#define HEADER_LSO_CLUSTER_UTIL

#include <vector>
#include <map>
#include <limits>
#include <algorithm>

#ifndef INFINITY
#define INFINITY std::numeric_limits<double>::infinity()
#endif

namespace lso_cluster {

using std::max;

// -----------------------------------------------------------------------------
// Nodes and clusters
// -----------------------------------------------------------------------------

typedef int node_t;
typedef int clus_t;

// -----------------------------------------------------------------------------
// Utility functions: clusterings
// -----------------------------------------------------------------------------

/// Construct a cluster by converting unique labels to [0,1,..]
template <typename T>
std::vector<clus_t> clustering_from_array(T const* data, size_t size) {
	std::vector<clus_t> clus(size);
	std::map<T,clus_t> first_in_clus;
	for (size_t i = 0 ; i < size ; ++i) {
		typename std::map<T,clus_t>::const_iterator it = first_in_clus.find(data[i]);
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
size_t compress_assignments(std::vector<clus_t>& node_clus);

// -----------------------------------------------------------------------------
// Neighborhood accumulation
// -----------------------------------------------------------------------------

template <typename T> inline T sentinel_value();
template <> inline double sentinel_value<double>() {
	return -std::numeric_limits<double>::infinity();
}

/// List of neighboring clusters of a node, and sum of edge weights to them
template <typename T>
class Neighbors {
  private:
	std::vector<clus_t> clus_; ///< list of neighboring clusters for current node
	std::vector<T> weight_; ///< weight == -INFINITY indicates not used
  public:
	Neighbors(int n)
		: weight_(n, sentinel_value<T>())
	{}
	
	/// Iterate over all neighbors to which any weight was added
	inline std::vector<clus_t>::const_iterator begin() const {
		return clus_.begin();
	}
	inline std::vector<clus_t>::const_iterator end() const {
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
    inline T weight(int c) const {
		return max(T(), (T)weight_[c]);
	}
	
	/// Clear all weights
	inline void clear() {
		for (size_t idx = 0 ; idx < size() ; ++idx) {
			weight_[clus_[idx]] = sentinel_value<T>();
		}
		clus_.clear();
	}
	/// Add weight to the link to cluster c, if c is not yet in the list of neighbors, add it.
	inline void add(clus_t c, double weight) {
		if (weight_[c] == sentinel_value<T>()) {
			clus_.push_back(c);
			weight_[c] = weight;
		} else {
			weight_[c] += weight;
		}
	}
};

// -----------------------------------------------------------------------------
}
#endif
