#include "util.hpp"
#include <vector>
#include <stdarg.h>

namespace lso_cluster {

// -----------------------------------------------------------------------------
// Utility functions: debug output and error handling
// -----------------------------------------------------------------------------

std::logic_error mk_logic_error(char const* format, ...) {
	char buf[1024];
	va_list args;
	va_start(args, format);
	vsprintf(buf, format, args);
	va_end(args);
	return std::logic_error(buf);
}

std::logic_error mk_invalid_argument(char const* format, ...) {
	char buf[1024];
	va_list args;
	va_start(args, format);
	vsprintf(buf, format, args);
	va_end(args);
	return std::invalid_argument(buf);
}

// -----------------------------------------------------------------------------
// Utility functions: working with multiple levels
// -----------------------------------------------------------------------------

// change cluster labels to consecutive integers, i.e. [0,1,..]
size_t compress_assignments(std::vector<clus_t>& x) {
	clus_t num = 0;
	std::vector<clus_t> relabel(x.size(), 0);
	for (size_t i = 0 ; i < x.size() ; ++i) {
		if (relabel[x[i]] == 0) {
			relabel[x[i]] = ++num;
		}
	}
	for (size_t i = 0 ; i < x.size() ; ++i) {
		x[i] = relabel[x[i]] - 1;
	}
	return num;
}

// -----------------------------------------------------------------------------
}
