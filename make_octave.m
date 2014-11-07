function make_octave(varargin)
mkoctfile('src/lso/lso_cluster_impl.cpp', 'src/lso/loss_functions.cpp', 'src/lso/trace_file_io.cpp', 'src/lso/lso_cluster_octave.cpp', '-Wall', '-Isrc/common', '-o', 'lso_cluster', varargin{:});
