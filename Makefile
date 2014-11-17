
all: stand-alone
#all: octave-bindings
#all: matlab-bindings
all: scripts

###########################################################
# Dependencies

HEADERS  = src/common/util.hpp
HEADERS += src/common/argument_parser.hpp
HEADERS += src/common/argument_parser_cli.hpp
HEADERS += src/common/argument_parser_mex.hpp
HEADERS += src/common/argument_parser_octave.hpp
HEADERS += src/common/sparse_matrix.hpp

LSO_HEADERS = $(HEADERS)
LSO_HEADERS += src/lso/lso_cluster.hpp
LSO_HEADERS += src/lso/lso_argument_parser.hpp

LSO_SOURCES = src/lso/lso_cluster_impl.cpp
LSO_SOURCES += src/lso/loss_functions.cpp
LSO_SOURCES += src/lso/trace_file_io.cpp
LSO_SOURCES += src/common/util.cpp
LSO_SOURCES_CLI = $(LSO_SOURCES) src/lso/lso_cluster_cli.cpp
LSO_SOURCES_OCT = $(LSO_SOURCES) src/lso/lso_cluster_octave.cpp
LSO_SOURCES_MEX = $(LSO_SOURCES) src/lso/lso_cluster_mex.cpp

NMF_HEADERS = $(HEADERS)
NMF_HEADERS += src/nmf/nmf_cluster.hpp
NMF_HEADERS += src/nmf/nmf_clustering.hpp
NMF_HEADERS += src/nmf/nmf_objective_function.hpp

NMF_SOURCES = src/nmf/nmf_cluster_impl.cpp
NMF_SOURCES += src/common/util.cpp
NMF_SOURCES_CLI = $(NMF_SOURCES) src/nmf/nmf_cluster_cli.cpp
NMF_SOURCES_OCT = $(NMF_SOURCES) src/nmf/nmf_cluster_octave.cpp
NMF_SOURCES_MEX = $(NMF_SOURCES) src/nmf/nmf_cluster_mex.cpp

LSO_OBJECTS_CLI = $(LSO_SOURCES_CLI:.cpp=.o)
$(LSO_OBJECTS_CLI): $(LSO_HEADERS)

NMF_OBJECTS_CLI = $(NMF_SOURCES_CLI:.cpp=.o)
$(NMF_OBJECTS_CLI): $(NMF_HEADERS)

###########################################################

CXXFLAGS = -std=c++11 -O2 -Wall -Isrc/common
CXXFLAGS_OCT = -Wall -Isrc/common -DOCTAVE
CXXFLAGS_MEX = -Wall -Isrc/common
MKOCTFILE = mkoctfile
MEX = mex

###########################################################

clean:
	rm -rf src/lso/*.o
	rm -rf src/nmf/*.o
	rm -rf lso_cluster.oct
	rm -rf lso_cluster.mex

.PHONY: all stand-alone octave-bindings matlab-bindings scripts

###########################################################

# Stand alone program
stand-alone: lso-cluster
lso-cluster: $(LSO_OBJECTS_CLI)
	$(CXX) $(CXXFLAGS) -o $@ $(LSO_OBJECTS_CLI)

stand-alone: nmf-cluster
nmf-cluster: $(NMF_OBJECTS_CLI)
	$(CXX) $(CXXFLAGS) -o $@ $(NMF_OBJECTS_CLI)

# Octave bindings
octave-bindings: lso_cluster.oct
lso_cluster.oct: $(LSO_SOURCES_OCT) $(LSO_HEADERS)
	$(MKOCTFILE) $(CXXFLAGS_OCT) -o $@ $(LSO_SOURCES_OCT)
octave-bindings: nmf_cluster.oct
nmf_cluster.oct: $(NMF_SOURCES_OCT) $(NMF_HEADERS)
	$(MKOCTFILE) $(CXXFLAGS_OCT) -o $@ $(NMF_SOURCES_OCT)

# Matlab bindings
matlab-bindings: lso_cluster.mex
lso_cluster.mex: $(LSO_SOURCES_MEX) $(LSO_HEADERS)
	$(MEX) $(CXXFLAGS) -o $@ $(LSO_SOURCES_OCT)
matlab-bindings: nmf_cluster.mex
nmf_cluster.mex: $(NMF_SOURCES_MEX) $(NMF_HEADERS)
	$(MEX) $(CXXFLAGS) -o $@ $(NMF_SOURCES_OCT)

###########################################################
# Scripts to build from octave or matlab

comma := ,

scripts: make_lso_cluster_octave.m make_lso_cluster_matlab.m
scripts: make_nmf_cluster_octave.m make_nmf_cluster_matlab.m
.SILENT: make_lso_cluster_octave.m make_lso_cluster_matlab.m
.SILENT: make_nmf_cluster_octave.m make_nmf_cluster_matlab.m

make_lso_cluster_octave.m: Makefile
	echo "function make_lso_cluster_octave(varargin)" > $@
	echo "% Compiles the lso_cluster function for octave." >> $@
	echo "mkoctfile($(foreach x,$(LSO_SOURCES_OCT) $(CXXFLAGS_OCT),'$(x)'$(comma)) '-o', 'lso_cluster', varargin{:});" >> $@
make_nmf_cluster_octave.m: Makefile
	echo "function make_nmf_cluster_octave(varargin)" > $@
	echo "% Compiles the nmf_cluster function for octave." >> $@
	echo "mkoctfile($(foreach x,$(NMF_SOURCES_OCT) $(CXXFLAGS_OCT),'$(x)'$(comma)) '-o', 'nmf_cluster', varargin{:});" >> $@

make_lso_cluster_matlab.m: Makefile
	echo "function make_lso_cluster_matlab(varargin)" > $@
	echo "% Compiles the lso_cluster function for matlab (also octave compatible)." >> $@
	echo "if exist('OCTAVE_VERSION')" >> $@
	echo "	more_args = {'-DCATCH_EXCEPTIONS=1'};" >> $@
	echo "else" >> $@
	echo "	more_args = {'-largeArrayDims'};" >> $@
	echo "end" >> $@
	echo "mex($(foreach x,$(LSO_SOURCES_MEX) $(CXXFLAGS_MEX),'$(x)'$(comma)) '-o', 'lso_cluster', more_args{:}, varargin{:});" >> $@
make_nmf_cluster_matlab.m: Makefile
	echo "function make_nmf_cluster_matlab(varargin)" > $@
	echo "% Compiles the nmf_cluster function for matlab (also octave compatible)." >> $@
	echo "if exist('OCTAVE_VERSION')" >> $@
	echo "	more_args = {'-DCATCH_EXCEPTIONS=1'};" >> $@
	echo "else" >> $@
	echo "	more_args = {'-largeArrayDims'};" >> $@
	echo "end" >> $@
	echo "mex($(foreach x,$(NMF_SOURCES_MEX) $(CXXFLAGS_MEX),'$(x)'$(comma)) '-o', 'nmf_cluster', more_args{:}, varargin{:});" >> $@
