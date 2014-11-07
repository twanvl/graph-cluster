// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#ifndef HEADER_LSO_CLUSTER_TRACE_FILE_IO
#define HEADER_LSO_CLUSTER_TRACE_FILE_IO

#include <stdio.h>
#include <vector>

#include "lso_cluster.hpp"

namespace lso_cluster {

// -----------------------------------------------------------------------------

void write_trace(FILE* f, vector<shared_ptr<TraceStep> > const& trace);
void write_trace(FILE* f, TraceStep const& step);
void write_trace(FILE* f, vector<shared_ptr<TraceStep> > const& trace);
void write_trace_file(const string& filename, vector<shared_ptr<TraceStep> > const& trace);

// -----------------------------------------------------------------------------
}
#endif
