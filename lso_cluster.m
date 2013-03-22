function [c,l,p] = lso_cluster(varargin)
	% lso_cluster is not compiled, run lso_cluster once to compile it.
	
	% matlab/octave prefer the compiled versions to .m files, so this is our opportunity to compile
	fprintf('Notice: lso_cluster has not been compiled, compiling it now.\n');
	
	if exist('OCTAVE_VERSION') == 5
		make_octave
	else
		make_matlab
	end
	
	fprintf('Please re-run lso_cluster with the same arguments.\n');
	%[c,l,p] = lso_cluster(varargin{:});
