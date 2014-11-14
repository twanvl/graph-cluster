// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#include "argument_parser_cli.hpp"
#include "nmf_cluster.hpp"

//#include "nm_cluster_impl.hpp"

using namespace std;
using namespace nmf_cluster;

// -----------------------------------------------------------------------------
// Debug
// -----------------------------------------------------------------------------

std::ostream& operator << (std::ostream& out, SparseMatrix const& mat) {
	for (int j = 0 ; j < mat.cols() ; ++j) {
		for (ColumnIterator it(mat,j) ; !it.end() ; ++it) {
			out << "(" << it.row() << "," << j << ") -> " << it.data() << endl;
		}
	}
	return out;
}

// -----------------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------------

void print_usage(char const* argv0) {
	cout << "Usage: " << argv0 << " <FILE> [OPTIONS]" << endl;
}

int main(int argc, char const** argv) {
	/*if (argc == 1) {
		print_usage(argv[0]);
		return EXIT_SUCCESS;
	}*/
	try {
		// Test case
		SparseMatrix graph;
		if (1) {
			double mat[6*6] = {0,1,1,0,0,0 , 1,0,1,0,0,0 , 1,1,0,1,0,0 , 0,0,1,0,1,1 , 0,0,0,1,0,1 , 0,0,0,1,1,0};
			graph = SparseMatrix::from_dense(6,6,mat);
		} else if (0) {
			double mat[3*3] = {1,0,0, 0,1,0, 0,0,1};
			graph = SparseMatrix::from_dense(3,3,mat);
		} else if (0) {
			srand(1234567);
			double mat[3*3] = {1,1,1, 1,1,1, 1,1,1};
			graph = SparseMatrix::from_dense(3,3,mat);
		} else if (0) {
			srand(1234567);
			const int n = 60;
			double mat[n*n];
			for (size_t i = 0 ; i < n*n ; ++i) mat[i] = 1.;
			graph = SparseMatrix::from_dense(n,n,mat);
		} else if (1) {
			// n*n matrix with m non-zeros
			srand(1234567);
			const int n = 100;
			const int m = 1000;
			double mat[n*n] = {0};
			for (int k = 0 ; k < m ; ++k) {
				int i = rand()%n, j = rand()%n;
				mat[i+j*n] = mat[j+i*n] = 1.;
			}
			graph = SparseMatrix::from_dense(n,n,mat);
		} else {
			srand(1234567);
			double mat[3*3] = {0,1,1, 1,0,1, 1,1,0};
			graph = SparseMatrix::from_dense(3,3,mat);
		}
		
		NMFParams params(cout);
		params.verbosity = 1;
		//params.num_iter = 16;
		params.num_iter = 1;
		params.objective.likelihood = LH_POISSON;
		//params.objective.support_prior = SUPPORT_ONE;
		params.objective.weight_beta = 0.1;
		//params.objective.support_prior = SUPPORT_POISSON;
		//params.objective.support_lambda = 1;
		params.max_cluster_per_node = 1;
		NMFOptimizer optimizer(graph,params);
		optimizer.run();
		// print
		cout << optimizer.get_clustering().to_sparse_matrix();
		
	} catch (std::exception const& e) {
		cerr << e.what() << endl;
		return EXIT_FAILURE;
	} catch (const char* p) {
		cerr << p << endl;
		return EXIT_FAILURE;
	} catch (...) {
		cerr << "Unexpected error" << endl;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
