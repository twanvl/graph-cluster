// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#ifndef HEADER_LSO_CLUSTER_SPARSE_MATRIX
#define HEADER_LSO_CLUSTER_SPARSE_MATRIX

#include <vector>
//#include <stdio.h> //DEBUG
#include <boost/math/special_functions/fpclassify.hpp> // isnan / isinf

// -----------------------------------------------------------------------------
// Sparse arrays
// This is a simple replacement for the octave SparseArray class
// -----------------------------------------------------------------------------

#ifdef OCTAVE
	
	#include <octave/Sparse.h>
	typedef octave_idx_type SparseMatrix_index;
	
#else
	
	using namespace boost::math;
	typedef int SparseMatrix_index;
	
	class SparseMatrix {
	  private:
		int rows_;
		std::vector<int> cidx_; // column start,end indices. size: cols+1
		std::vector<int> ridx_; // row index. size: nnz
		std::vector<double> data_; // size: nnz
		struct DontDelete {
			inline void operator () (void*) {}
		};
		
	  public:
		SparseMatrix() : rows_(0) {}
		SparseMatrix(int rows_, int cols_, int nnz_)
			: rows_(rows_)
			, cidx_(cols_+1, 0)
			, ridx_(nnz_, rows_)
			, data_(nnz_, 0.0)
		{}
		template <typename CidxIt, typename RidxIt, typename DataIt>
		SparseMatrix(int rows_, int cols_, int nnz_, CidxIt cidxs, RidxIt ridxs, DataIt datas)
			: rows_(rows_)
			, cidx_(cidxs, cidxs + cols_+1)
			, ridx_(ridxs, ridxs + nnz_)
			, data_(datas, datas + nnz_)
		{}
		// initialize from a dense column-major matrix
		template <typename DataIt>
		static SparseMatrix from_dense(int rows, int cols, DataIt datas) {
			// phase 1 : count non-zeros
			int nnz = 0;
			for (int i = 0 ; i < cols * rows ; ++i) {
				if (datas[i] != 0.0) nnz++;
			}
			SparseMatrix out(rows, cols, nnz);
			int k = 0;
			for (int c = 0, i = 0 ; c < cols ; ++c) {
				out.cidx_[c] = k;
				for (int r = 0 ; r < rows ; ++r, ++i) {
					if (datas[i] != 0.0) {
						out.ridx_[k] = r;
						out.data_[k] = datas[i];
						k++;
					}
				}
			}
			out.cidx_[cols] = k;
			return out;
		}
		
		inline int rows() const {
			return rows_;
		}
		inline int cols() const {
			return (int)cidx_.size() - 1;
		}
		inline int nnz() const {
			return (int)data_.size();
		}
		
		inline int cidx(int j) const {
			return cidx_[(size_t)j];
		}
		inline int ridx(int k) const {
			return ridx_[(size_t)k];
		}
		inline double data(int k) const {
			return data_[(size_t)k];
		}
		
		inline int& cidx(int j) {
			return cidx_[(size_t)j];
		}
		inline int& ridx(int k) {
			return ridx_[(size_t)k];
		}
		inline double& data(int k) {
			return data_[(size_t)k];
		}
		
		std::vector<int>::const_iterator cidx_begin() const { return cidx_.begin(); }
		std::vector<int>::const_iterator cidx_end()   const { return cidx_.end(); }
		std::vector<int>::const_iterator ridx_begin() const { return ridx_.begin(); }
		std::vector<int>::const_iterator ridx_end()   const { return ridx_.end(); }
		std::vector<double>::const_iterator data_begin() const { return data_.begin(); }
		std::vector<double>::const_iterator data_end()   const { return data_.end(); }
		
		double operator () (int i, int j) const {
			int k0 = cidx(j), k1 = cidx(j+1);
			// find index with binary search
			while (k0 + 1 < k1) {
				int km = (k0 + k1) / 2;
				if (ridx(km) <= i) {
					k0 = km;
				} else {
					k1 = km;
				}
			}
			return ridx(k0) == i ? data(k0) : 0.0;
		}
		
		void change_capacity(int nnz_new) {
			ridx_.resize(nnz_new, rows_);
			data_.resize(nnz_new, 0.0);
		}
		inline void maybe_compress(int nnz_new) {
			change_capacity(nnz_new);
		}
		
		bool any_element_is_inf_or_nan() const {
			for (int k = 0 ; k < nnz() ; ++k) {
				if (isinf(data(k)) || isnan(data(k))) return true;
			}
			return false;
		}
		bool any_element_is_negative() const {
			for (int k = 0 ; k < nnz() ; ++k) {
				if (data(k) < 0) return true;
			}
			return false;
		}
	};
	
#endif

// -----------------------------------------------------------------------------
// Some more utility functions
// -----------------------------------------------------------------------------

// iterate over the nonzeros in a column of a sparse matrix
struct ColumnIterator {
  public:
	inline ColumnIterator(SparseMatrix const& mat, int j)
		: mat(mat), k(mat.cidx(j)), kend(mat.cidx(j+1))
	{}
	inline int row() const {
		return mat.ridx(k);
	}
	inline double data() const {
		return mat.data(k);
	}
	inline bool end() const {
		return k >= kend;
	}
	inline void operator ++() {
		k++;
	}
  private:
	SparseMatrix const& mat;
	int k, kend;
};

// iterate over a column, including zeros
struct DenseColumnIterator {
  public:
	inline DenseColumnIterator(SparseMatrix const& mat, int j)
		: mat(mat), i(0), k(mat.cidx(j))
	{}
	inline int row() const {
		return i;
	}
	inline double data() const {
		if (mat.ridx(k) == i) {
			return mat.data(k);
		} else {
			return 0.;
		}
	}
	inline bool end() const {
		return i >= mat.rows();
	}
	inline void operator ++() {
		if (mat.ridx(k) == i) {
			k++;
		}
		i++;
	}
  private:
	SparseMatrix const& mat;
	int i, k;
};

inline bool is_symmetric(SparseMatrix const& mat) {
	for (int j = 0 ; j < mat.cols() ; ++j) {
		for (ColumnIterator it(mat,j) ; !it.end() ; ++it) {
			if (mat(j,it.row()) != it.data()) return false;
		}
	}
	return true;
}

#endif
