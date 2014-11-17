// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#ifndef HEADER_LSO_CLUSTER_UTIL
#define HEADER_LSO_CLUSTER_UTIL

#if OCTAVE
#include <octave/oct.h>
#endif
#include <vector>
#include <map>
#include <limits>
#include <algorithm>
#include <utility>
#include <memory>

#ifndef INFINITY
#define INFINITY std::numeric_limits<double>::infinity()
#endif

// -----------------------------------------------------------------------------
// C++11
// -----------------------------------------------------------------------------

#if __cplusplus < 201103L
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
namespace std {
	using boost::shared_ptr;
	using boost::make_shared;
}
#define nullptr 0
#endif

namespace lso_cluster {

using std::max;

// -----------------------------------------------------------------------------
// Octave compatibility
// -----------------------------------------------------------------------------

#ifndef OCTAVE_QUIT
#define OCTAVE_QUIT do{}while(0)
#endif

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

template <typename T> struct Sentinel;
template <> struct Sentinel<double> {
	static inline double value () {
		return -std::numeric_limits<double>::infinity();
	}
};

template <typename A, typename B> struct Sentinel<std::pair<A,B> > {
	static inline std::pair<A,B> value() {
		return std::make_pair(Sentinel<A>::value(), Sentinel<B>::value());
	}
};
template <typename A, typename B>
inline std::pair<A,B> operator + (std::pair<A,B> const& x, std::pair<A,B> const& y) {
	return std::make_pair(x.first + y.first, x.second + y.second);
}
template <typename A, typename B>
inline std::pair<A,B>& operator += (std::pair<A,B>& x, std::pair<A,B> const& y) {
	x.first += y.first;
	x.second += y.second;
	return x;
}

/// A sparse map from integers [0..n) to T
template <typename T>
class SparseMap {
  private:
	std::vector<clus_t> clus_; ///< list of non empty items
	std::vector<T> weight_; ///< value of all items, sentinel_value() indicates not used
  public:
	SparseMap(size_t n)
		: weight_(n, Sentinel<T>::value())
	{}
	
	/// Iterate over all neighbors to which any weight was added
	typedef typename std::vector<clus_t>::const_iterator const_iterator;
	inline const_iterator begin() const {
		return clus_.begin();
	}
	inline const_iterator end() const {
		return clus_.end();
	}
	/// Sort the neighboring clusters by index
	inline void sort() {
		std::sort(clus_.begin(), clus_.end());
	}
	
	// number of non-zeros
	inline size_t nnz() const {
		return clus_.size();
	}
	// maximum size of the domain
	inline size_t max_size() const {
		return weight_.size();
	}
	/// Get the weight to a particular cluster
	inline T weight(int c) const {
		return max(T(), (T)weight_[c]);
	}
	
	/// Clear all weights
	inline void clear() {
		for (std::vector<clus_t>::const_iterator it = clus_.begin() ; it != clus_.end() ; ++it) {
			weight_[*it] = Sentinel<T>::value();
		}
		clus_.clear();
	}
	/// Add weight to the link to cluster c, if c is not yet in the list of neighbors, add it.
	inline void add(clus_t c, T const& value) {
		if (weight_[c] == Sentinel<T>::value()) {
			clus_.push_back(c);
			weight_[c] = value;
		} else {
			weight_[c] += value;
		}
	}
};

// -----------------------------------------------------------------------------
}
#endif
