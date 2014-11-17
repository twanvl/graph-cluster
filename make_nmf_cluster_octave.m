function make_nmf_cluster_octave(varargin)
% Compiles the nmf_cluster function for octave.
mkoctfile('src/nmf/nmf_cluster_impl.cpp', 'src/common/util.cpp', 'src/nmf/nmf_cluster_octave.cpp', '-Wall', '-Isrc/common', '-DOCTAVE', '-o', 'nmf_cluster', varargin{:});
