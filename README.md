Local Search Optimization for graph clustering
=============

Introduction
------------

This program uses a local search optimization to find an approximately optimal clustering of a undirected graph according to some objective function.
Many different objectives are supported.
There are bindings for octave and MATLAB.

Formally, the problem that is solved is to find a clustering C of the nodes in a graph that minimizes the loss

    loss(C) = f(sum_{c ∈ C} g(c))

The optimization method is the one introduced by Blondel et.al. \[2], and also used by \[1] and \[3].

Installation
------------

To build the stand alone version:

    make stand-alone

To build the octave interface:

    octave
    make_octave

To build the matlab interface:

    matlab
    make_matlab

Usage
------------

for the Octave and MATLAB bindings, the usage is:

    [c,l,k] = lso_cluster(A, <optional parameters>);

Where:
 * `A` is a symmetric weighted adjacency matrix. It can be sparse.
 * The optional parameters are a list of `'name',value` pairs, see below.
 * `c` is the labeling that represents the clustering. The first cluster will have label `0`, the next label `1`, etc.
 * `l` is the objective (loss) value of the clustering.
 * `k` is the number of clusters, `k = numel(unique(c))`.

Optional parameters regarding the clustering:

 * `'eval', clustering`
   Evaluate the objective function on `clustering`, do not optimize.
   
 * `'init', clustering`
   Use the given `clustering` as initial value for the optimization.
   Default: each node is initially in a separate cluster, i.e. `clustering=1:length(A)`.
   
Optional parameters regarding the objective:

 * `'loss', loss`
   Use the given loss/objective function. The loss is given a string name. See below for a list of supported loss functions.
   Default: `loss = 'modularity'`
   
 * `'total_volume', m`
   Replace the total volume (sum of edges) by `m`.
   Many objectives use the total volume for normalization, and changing it will change the scale at which clusters are found.
   Usually increasing the total volume will result in larger clusters.
   Default: `m = sum(sum(A))`
 
 * `'extra_loss', alpha`
   Add a term to the loss function that penalizes the volume of clusters, with weight `alpha`.
 
 * `'num_clusters', n`
   Force the solution to have the given number of clusters.
   The algorithm uses a binary search to alter the objective until it finds a solution with the given number of clusters.
   The alteration is the same as the one used by `extra_loss`.
   
 * `'min_num_cluster', n`
   Force the solution to have at least the given number of clusters.
   
 * `'max_num_cluster', n`
   Force the solution to have at most the given number of clusters.
   
Optional parameters about internal algorithm details, you only need these if you know what you are doing:

 * `'seed', random_seed`
   Use a given random seed.
   By default a fixed seed is used, so repeated runs with the same input give the same output.
   
 * `'num_repeats', n`
   Repeat the search n times from scratch with different random seeds and return the best result.
   Default: 1
 
 * `'num_partitions', n`
   Number of times to try and break apart the clusters and re-cluster them
   Default: 0
 
 * `'optimize_higher_level', bool`
   Use a hierarchical optimizer, where small clusters are considered as nodes of a higher level graph.
   Default: true.
    
 * `'always_consider_empty', bool`
   Always consider the move of a node into a new singleton cluster.
   Default: true.
    
 * `'num_loss_tweaks', n`
   Maximum number of iterations in the binary search to force the specified number of clusters.
   Default: 32
    
 * `'check_invariants', bool`
   Check invariants of the algorithm (for debugging). Default: false.
   
 * `'trace_file', filename`
   Write a trace of the steps performed by the optimization algorithm to a file in JSON format.

 * `'verbose', level`
   Level of debug output. Levels go up to 7 and are increasingly chatty.
   Default: `level = 0`, i.e. no output.


Loss functions
------------

The optional parameter `'loss'` specifies the loss function to use. The default is modularity.
In this section the notation is:

    v_c: The volume of cluster c, i.e. the sum of degrees/strengths of nodes in c.
         In MATLAB notation:
           v_c = sum(sum(A(C==c,:)))
    w_c: The weight of edges within cluster c
         In MATLAB notation:
           w_c = sum(sum(A(C==c,C==c))
    n_c: The number of nodes in cluster c
    m:   The total volume of the graph,
         m = v_V, where V is the set of all nodes.
         This is altered by the 'total_volume' option.

Since the loss is minimized, some objectives are negated compared to their usual definition.

Some of the supported loss functions are:

* `'modularity'`,
  `loss = -sum_c (w_c/m - v_c^2/m^2)`
  This is the negation of the usual definition

* `'infomap'`: The infomap objective by [3].

* `'ncut'`: Normalized cut,
  `loss = sum_c (v_c - w_c) / n_c`

* `'rcut'`: Ratio cut,
  `loss = sum_c (v_c - w_c) / v_c`

* `{'pmod',p}`: Modularity with a different power,
  `loss = -sum_c (w_c/m - (v_c/m)^2)`

* `{'mom',m}`: Monotonic variant of modularity,
  `loss = -sum_c (w_c/(m + 2v_c) - (v_c/(m + 2v_c))^2)`

* `'w-log-v'`,
  `loss = sum_c (w_c/m * log(v_c) )`

See `loss_functions.hpp` for the full list.
Some loss functions have parameters, these are passed as a cell array where the first element is the name.


Examples
------------

Find the clustering of a graph by optimizing modularity:

    % Build a graph that has a cluster structure
    A = (blkdiag(rand(4), rand(5), rand(6)) > 1-0.5) | (rand(15) > 1-0.05);
    A = A | A';
    % Find clusterings
    c = greedy_cluster(A);
    % c = [0 0 0 0 1 1 1 1 1 2 2 2 2 2 2]'

Evaluate the modularity of a given clustering:

    c = [0 0 0 0 5 5 5 1 1 2 2 2 2 2 2]
    [ignored, l] = greedy_cluster(A, 'eval',c);
    l
    % l = -0.40633

Optimize infomap:

    c = greedy_cluster(A, 'loss','infomap');

Find a solution with exactly 3 clusters:

    c = greedy_cluster(A, 'num_clusters',3);


Here is a larger example:

    % An LFR graph with mixing 0.6 and 1000 nodes (See [4])
    % The gaph is in A, the ground truth is in c
    > load example-LFR.mat
    
    > [d,l,k] = lso_cluster(A,'loss','modularity');
    > normalized_mutual_information(c,d)
    ans = 0.93011
    
    % This solution has too few clusters
    > k
    k = 25
    > numel(unique(c))
    ans = 41
    
    % So, force the number of clusters to be 41
    > [d,l,k] = lso_cluster(A,'loss','modularity','num_clusters',41);
    > normalized_mutual_information(c,d)
    ans = 1
    
    % Or use a different loss function
    > [d,l,k] = lso_cluster(A,'loss','w-log-v');
    > normalized_mutual_information(c,d)
    ans = 1

Usage, stand alone version
------------

Graphs should be given on stdin or as a text file in the first argument. Each line should look like:

     <from> <to> <weight>

Where `<from>` and `<to>` are non-negative integer node identifiers. The first node is node 0, the number of nodes will be set to 1 plus the largest node identifier.
If needed, you can add edges `n n 0` to force the graph to have `n+1` nodes.

The output is written as text to stdout, where each line gives a cluster label. With `-o <filename>` the output is directed to a file instead.

Other parameters are specified as `--parameter value`, with the same name as the matlab version parameters. For example `--loss infomap` uses the infomap loss function.

    $ cat example.in
    $ lso_cluster < example.in


References
----------

\[1] Graph clustering: does the optimization procedure matter more than the objective function?;
     Twan van Laarhoven and Elena Marchiori;
     Physical Review E 87, 012812 (2013)
     [\[pdf\]](http://cs.ru.nl/~T.vanLaarhoven/clustering2012/PhysRevE.87.012812.pdf)
     [\[publisher\]](http://link.aps.org/doi/10.1103/PhysRevE.87.012812)
     [\[website\]](http://cs.ru.nl/~T.vanLaarhoven/clustering2012/)

\[2] Fast unfolding of communities in large networks;
     Vincent D Blondel, Jean-Loup Guillaume, Renaud Lambiotte and Etienne Lefebvre;
     J. Stat. Mech. Theory Exp. 2008, P10008 (2008),
     [\[publisher\]](http://iopscience.iop.org/1742-5468/2008/10/P10008/)
     [\[arxiv\]](http://arxiv.org/abs/0803.0476)

\[3] Maps of random walks on complex networks reveal community structure;
     M. Rosvall and C. T. Bergstrom;
     Proc. Natl. Acad. Sci. USA 105, 1118 (2008).
     [\[publisher\]](http://dx.doi.org/10.1073/pnas.0706851105)
     [\[arxiv\]](http://arxiv.org/abs/0707.0609)
     [\[code\]](http://www.tp.umu.se/~rosvall/code.html)

\[4] Benchmark graphs for testing community detection algorithms;
     A. Lancichinetti, S. Fortunato and F. Radicchi;
     Physical Review E 78, 046110 (2008)
     [\[pdf\]](https://sites.google.com/site/santofortunato/benchmark.pdf?attredirects=0)
     [\[website\]](https://sites.google.com/site/santofortunato/inthepress2)
