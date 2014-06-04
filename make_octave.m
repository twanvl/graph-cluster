function make_octave(varargin)
mkoctfile('lso_cluster_octave.cpp','-o','lso_cluster','-Wall',varargin{:});

