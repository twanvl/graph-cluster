function make_matlab(varargin)
	if exist('OCTAVE_VERSION')
		more_args = {'-DCATCH_EXCEPTIONS=1'};
	else
		more_args = {'-largeArrayDims'};
	end
	%mex('lso_cluster_mex.cpp','-o','lso_cluster',more_args{:},varargin{:});
	mex('nmf_cluster_mex.cpp','-o','nmf_cluster',more_args{:},varargin{:});

