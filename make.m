function make(varargin)
% Compiles the lso_cluster and nmf_cluster functions for matlab or octave
if exist('OCTAVE_VERSION')
	make_lso_cluster_octave(varargin{:});
	make_nmf_cluster_octave(varargin{:});
else
	make_lso_cluster_matlab(varargin{:});
	make_nmf_cluster_matlab(varargin{:});
end
