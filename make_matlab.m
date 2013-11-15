function make_matlab(varargin)
mex('lso_cluster_mex.cpp','-o','lso_cluster','-largeArrayDims',varargin{:});
