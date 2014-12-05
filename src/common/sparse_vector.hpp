// -----------------------------------------------------------------------------
// Local Search Non-negative Matrix Factorization
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#ifndef HEADER_SPARSE_VECTOR
#define HEADER_SPARSE_VECTOR

#include "util.hpp"
#include "sparse_matrix.hpp"
#include <vector>

namespace lso_cluster {

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

// A sparse vector is a vector of SparseItems, sorted by key
struct SparseVector : private std::vector<SparseItem> {
  private:
	SparseVector(size_t n) : std::vector<SparseItem>(n) {}
  public:
	SparseVector() {}
	using std::vector<SparseItem>::iterator;
	using std::vector<SparseItem>::const_iterator;
	using std::vector<SparseItem>::begin;
	using std::vector<SparseItem>::end;
	using std::vector<SparseItem>::clear;
	using std::vector<SparseItem>::empty;
	
	// this can invalidate the order, call sort() to fix it
	// maybe a conversion function would be nicer?
	using std::vector<SparseItem>::push_back;
	using std::vector<SparseItem>::pop_back;
	using std::vector<SparseItem>::back;
	
	// lookup a value
	inline double operator () (clus_t i) const;
	inline bool contains(clus_t i) const;
	
	// re-sort by cluster-id/key
	inline void sort() {
		std::sort(this->begin(), this->end());
	}
	
	// insert a value, it must not yet exist
	inline void insert(clus_t k, double weight);
	// remove an item, return the old weight
	inline double remove(clus_t k);
	
	inline size_t nnz() const { return size(); }
	
	// convert from column of a sparse matrix
	inline void operator = (ColumnIterator it);
	
	friend SparseVector operator + (SparseVector const&, SparseVector const&);
	friend SparseVector operator * (SparseVector const&, double);
	void addmul(double, SparseVector const&);
};

std::ostream& operator << (std::ostream& out, SparseVector const& vec) {
	out << "[";
	for (SparseVector::const_iterator it = vec.begin() ; it != vec.end() ; ++it) {
		if (it != vec.begin()) out << ", ";
		out << it->clus << ":" << it->weight;
	}
	return out << "]";
}

double SparseVector::operator () (clus_t k) const {
	assert(is_sorted(this->begin(),this->end()));
	/*if (!is_sorted(this->begin(),this->end())) {
		octave_stdout << "not sorted! " << *this << endl;
		throw "bork";
	}*/
	
	const_iterator it = std::lower_bound(this->begin(),this->end(),k);
	if (it != this->end() && it->clus == k) {
		return it->weight;
	} else {
		return 0.;
	}
}
bool SparseVector::contains(clus_t k) const {
	const_iterator it = std::lower_bound(this->begin(),this->end(),k);
	return (it != this->end() && it->clus == k);
}

void SparseVector::insert(clus_t k, double weight) {
	iterator it = std::lower_bound(this->begin(),this->end(),k);
	assert(it == this->end() || it->clus != k);
	std::vector<SparseItem>::insert(it, SparseItem(k,weight));
}
double SparseVector::remove(clus_t k) {
	iterator it = std::lower_bound(this->begin(),this->end(),k);
	assert(it != this->end() && it->clus == k);
	double weight = it->weight;
	std::vector<SparseItem>::erase(it);
	return weight;
}

double dot(SparseVector const& x, SparseVector const& y) {
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

void operator += (std::vector<double>& x, SparseVector const& y) {
	for (SparseVector::const_iterator it = y.begin() ; it != y.end() ; ++it) {
		x[it->clus] += it->weight;
	}
}

void SparseVector::operator = (ColumnIterator it) {
	clear();
	for ( ; !it.end() ; ++it) {
		push_back(SparseItem(it.row(),it.data()));
	}
}

SparseVector operator + (SparseVector const& a, SparseVector const& b) {
	SparseVector out;
	out.reserve(a.nnz()+b.nnz());
	for (SparseVector::const_iterator it1 = a.begin(), it2 = b.begin() ; it1 != a.end() && it2 != b.end() ; ) {
		if (it1->clus < it2->clus) {
			out.push_back(*it1);
			++it1;
		} else if (it1->clus > it2->clus) {
			out.push_back(*it2);
			++it2;
		} else {
			out.push_back(SparseItem(it1->clus, it1->weight + it2->weight));
			++it1; ++it2;
		}
	}
	return out;
}

SparseItem operator * (SparseItem const& x, double y) {
	return SparseItem(x.clus, x.weight * y);
}
SparseVector operator * (SparseVector const& a, double b) {
	SparseVector out(a.size());
	//std::transform(a.begin(), a.end(), out.begin(), bind2nd(multiplies<double>(),b));
	SparseVector::iterator it2 = out.begin();
	for (SparseVector::const_iterator it1 = a.begin() ; it1 != a.end() ; ++it1 ) {
		*it2 = *it1 * b;
	}
	return out;
}
SparseVector operator * (double a, SparseVector const& b) {
	return b * a;
}

// *this += w * b;
void SparseVector::addmul(double w, SparseVector const& b) {
	*this = *this + w * b;
}

// -----------------------------------------------------------------------------
// Matrix product
// -----------------------------------------------------------------------------


SparseVector operator * (SparseVector const& a, std::vector<SparseVector> const& b) {
	SparseVector out;
	for (SparseVector::const_iterator it = a.begin() ; it != a.end() ; ++it) {
		out.addmul(it->weight, b[it->clus]);
	}
	return out;
}

std::vector<SparseVector> operator * (std::vector<SparseVector> const& a, std::vector<SparseVector> const& b) {
	std::vector<SparseVector> out(a.size());
	for (size_t i = 0 ; i < a.size() ; ++i) {
		out[i] = a[i] * b;
	}
	return out;
}

// -----------------------------------------------------------------------------
}
#endif
