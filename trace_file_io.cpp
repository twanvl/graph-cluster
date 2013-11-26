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

void fprintf(FILE* f, vector<int> const& x) {
	fprintf(f,"[");
	for (size_t i = 0 ; i < x.size() ; ++i) {
		if (i > 0) fprintf(f,",");
		fprintf(f, "%d", x[i]);
	}
	fprintf(f,"]");
}

void write_trace(FILE* f, vector<shared_ptr<TraceStep> > const& trace);
void write_trace(FILE* f, TraceStep const& step) {
	fprintf(f, "{\"description\":\"%s\"", step.description.c_str());
	fprintf(f, ",\"loss\":%f", step.loss);
	fprintf(f, ",\"num_clusters\":%d", step.num_clusters);
	fprintf(f, ",\"node_clus\":");
	fprintf(f, step.node_clus);
	fprintf(f, ",\"sub_steps\":");
	write_trace(f,step.sub_steps);
	if (!step.sub_mapping.empty()) {
		fprintf(f, ",\"sub_mapping\":");
		fprintf(f, step.sub_mapping);
	}
	fprintf(f, "}");
}

void write_trace(FILE* f, vector<shared_ptr<TraceStep> > const& trace) {
	fprintf(f, "[");
	for (size_t i = 0 ; i < trace.size() ; ++i) {
		if (i > 0) fprintf(f,",\n");
		write_trace(f,*trace[i]);
	}
	fprintf(f, "]");
}

void write_trace_file(const string& filename, vector<shared_ptr<TraceStep> > const& trace) {
	FILE* f = fopen(filename.c_str(),"wt");
	fprintf(f,"var trace=");
	write_trace(f,trace);
	fprintf(f,";\n");
	fclose(f);
}

// -----------------------------------------------------------------------------
}
#endif
