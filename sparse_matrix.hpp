// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#ifndef HEADER_LSO_CLUSTER_SPARSE_MATRIX
#define HEADER_LSO_CLUSTER_SPARSE_MATRIX

#include <vector>
#include <stdio.h> //DEBUG

// -----------------------------------------------------------------------------
// Sparse arrays
// This is a simple replacement for the octave SparseArray class
// -----------------------------------------------------------------------------

#ifdef OCTAVE_VERSION
	
	#include <octave/Sparse.h>
	typedef octave_idx_type SparseMatrix_index;
	
#else
	
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

bool is_symmetric(SparseMatrix const& mat) {
	for (int j = 0 ; j < mat.cols() ; ++j) {
		for (int k = mat.cidx(j) ; k < mat.cidx(j+1) ; ++k) {
			int i = mat.ridx(k);
			if (mat(j,i) != mat.data(k)) return false;
		}
	}
	return true;
}

#endif
