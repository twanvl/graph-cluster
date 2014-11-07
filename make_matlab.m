function make_matlab(varargin)
if exist('OCTAVE_VERSION')
	more_args = {'-DCATCH_EXCEPTIONS=1'};
else
	more_args = {'-largeArrayDims'};
end
mex('src/lso/lso_cluster_impl.cpp', 'src/lso/loss_functions.cpp', 'src/lso/trace_file_io.cpp', 'src/lso/lso_cluster_mex.cpp', '-Wall', '-Isrc/common', '-o', 'lso_cluster', more_args{:}, varargin{:});
