function make_nmf_cluster_matlab(varargin)
% Compiles the nmf_cluster function for matlab (also octave compatible).
if exist('OCTAVE_VERSION')
	more_args = {'-DCATCH_EXCEPTIONS=1'};
else
	more_args = {'-largeArrayDims'};
end
mex('src/nmf/nmf_cluster_impl.cpp', 'src/common/util.cpp', 'src/nmf/nmf_cluster_mex.cpp', '-Wall', '-Isrc/common', '-o', 'nmf_cluster', more_args{:}, varargin{:});
