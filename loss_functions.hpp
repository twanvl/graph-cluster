// -----------------------------------------------------------------------------
// Local search optimization clustering
// By Twan van Laarhoven
// License: BSD style, see file LICENSE
// -----------------------------------------------------------------------------

#ifndef HEADER_LSO_LOSS_FUNCTIONS
#define HEADER_LSO_LOSS_FUNCTIONS

#include "lso_cluster.hpp"
#include <memory>
#include <stdexcept>
#include <boost/math/special_functions/gamma.hpp>

namespace lso_cluster {

// -----------------------------------------------------------------------------
// Utility
// -----------------------------------------------------------------------------

inline double plogp(double p) {
	if (p < 1e-15) {
		return 0.;
	} else {
		return p * log(p);
	}
}
inline double sqr(double x) {
	return x*x;
}

// -----------------------------------------------------------------------------
// Loss functions
// -----------------------------------------------------------------------------

struct Modularity : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		// modularity = 1/2m * ∑{ij in same clus} A{ij} - d_i*d_j/2m
		//            = self/tot - deg/tot*deg/tot
		// this is a loss, so -modularity
		return -clus.self/total.degree + (clus.degree/total.degree)*(clus.degree/total.degree);
	}
};

struct SizeModularity : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		return -clus.self/total.degree + sqr(clus.size/total.size);
	}
};

struct Infomap : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		return plogp((clus.degree + clus.exit()) / total.degree) - 2.0 * plogp(clus.exit() / total.degree);
	}
	double global(Doubles const& sum_local, Stats const& total, int num_clusters) const {
		// plogp_node_degree = sum over nodes i, plogp(degree(i) / total.degree)
		// can't calculate this, but since it is a global constant, it doesn't matter
		double plogp_node_degree = 0;
		return sum_local[0] + plogp(total.exit() / total.degree) - plogp_node_degree;
	}
};

struct InfomapPaper : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-10) return 0.;
		double d = clus.degree / total.degree;
		double s = clus.self   / clus.degree;
		double e = clus.exit() / clus.degree;
		return Doubles(plogp(d), -d * (plogp(s) + plogp(e)));
	}
	double global(Doubles const& sum_local, Stats const& total, int num_clusters) const {
		// plogp_node_degree = sum over nodes i, plogp(degree(i) / total.degree)
		double plogp_node_degree = 0.;
		double plogp_degree  = sum_local[0];
		double entropy_inout = sum_local[1];
		double s = total.self / total.degree;
		return s * plogp_degree + entropy_inout - plogp_node_degree;
	}
};

struct InfomapTweak : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		return plogp((clus.degree + clus.exit()) / total.degree) - 2.0 * plogp(clus.exit() / total.degree);
	}
};

struct Conductance : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		// conductance(S) = e_S / min(v_S, v_!S)
		double denom = min(clus.degree, total.degree - clus.degree);
		if (clus.degree < 1e-6) return 0.;
		if (denom < 1e-6) return INFINITY;
		return clus.exit() / denom;
	}
};

struct SelfLogDegree : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6) return 0;
		//return clus.self / total.degree * log(clus.degree / total.degree);
		double l = clus.self / total.degree * log(clus.degree / total.degree);
//		if (l != l) {
//			octave_stdout << "Found a NaN in lossfun for " << clus.self << " / " << total.degree << " * log(" << clus.degree << " / " << total.degree << ")" << endl;
//		}
		return l;
	}
};

struct SelfLogSize : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.size < 1e-6) return 0;
		return clus.self / total.degree * log(clus.size / total.size);
	}
};

struct SelfLogDegreeSym : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6 || clus.degree == total.degree) return 0;
		return clus.self / total.degree * log(clus.degree / total.degree)
		     + clus.exit() / total.degree * log(1 - clus.degree / total.degree);
	}
};

struct SelfLogDegreeMoved : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6) return 0;
		return clus.self / total.degree * (log(clus.degree / total.degree) - log(total.degree));
	}
};

struct SelfLogSelf : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.self < 1e-6) return 0;
		return clus.self / total.degree * log(clus.self / total.degree);
	}
};

struct DegreeLogDegree : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6) return 0;
		return clus.degree / total.degree * log(clus.degree / total.degree);
	}
};

struct SelfOverDegree : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6) return 0;
		//return -clus.self / (clus.degree + 0.01*total.degree);
		return clus.self * (-total.degree / clus.degree + 1);
	}
};

struct SelfSqrtDegree : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6) return 0;
		return clus.self / total.degree * (sqrt(clus.degree / total.degree) - 1);
	}
};

struct SelfSqrtSelf : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.self < 1e-6) return 0;
		return clus.self / total.degree * (sqrt(clus.self / total.degree) - 1);
	}
};
struct SelfSqrtSelf2 : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		double s = clus.self / total.degree;
		return sqrt(s) * (s - 1);
	}
};
struct SelfSqrtSelf3 : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		double s = clus.self / total.degree;
		return sqrt(s) * (sqrt(s) - 1);
	}
};

struct DDS : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		double d  = clus.degree / total.degree;
		double ds = clus.self / total.degree;
		return d * ds - ds;
	}
};

struct SDDS : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		double d  = clus.degree / total.degree;
		double ds = clus.self / total.degree;
		return sqrt(d) * (ds - ds / d);
	}
};
/*
struct LDDS : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		double d = clus.degree / total.degree;
		double s = clus.self / clus.degree;
		return s * ();
	}
};*/

struct SelfLogDegreeGlobal : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		return plogp(clus.degree / total.degree);
	}
	double global(Doubles const& sum_deg_log_deg, Stats const& total, int num_clusters) const {
		return total.self / total.degree * sum_deg_log_deg[0];
	}
};

struct SelfLogDegreeGlobalSqr : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		return sqr(clus.degree / total.degree);
	}
	double global(Doubles const& sum_deg_deg, Stats const& total, int num_clusters) const {
		return total.self / total.degree * log(sum_deg_deg[0]);
	}
};

struct SelfLogDegreeAdjust : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6) return 0;
		return clus.self / total.degree + clus.degree / total.degree * log(clus.degree / total.degree);
	}
};
struct SelfLogDegreeAdjust2 : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6) return 0;
		return clus.degree / total.degree + clus.self / total.degree * log(clus.degree / total.degree);
	}
};
struct SelfLogDegreeAdjust3 : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree == 0) return 0;
		return -clus.self / total.degree + clus.degree / total.degree * log(clus.degree / total.degree);
	}
};
struct SelfLogDegreeAdjust4 : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree == 0) return 0;
		return -clus.self / total.degree + clus.degree / total.degree * (log(clus.degree / total.degree) + 1);
	}
};

struct ADS : public LossFunction {
	double pivot;
	ADS(double pivot = 0.5) : pivot(pivot) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6) return 0.;
		double d = clus.degree / total.degree;
		double w = clus.self / clus.degree;
		if (d < pivot) {
			return w * -d / pivot;
		} else {
			return w * ((d-pivot)/(1-pivot)-1);
		}
	}
};

struct PPD : public LossFunction {
	double q,k;
	PPD(double q = 2, double k = 0.3) : q(q), k(k) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6) return 0.;
		double d = clus.degree / total.degree;
		double w = clus.self / clus.degree;
		return -w * pow(d,q*k) * pow(1-d, q*(1-k));
	}
};

struct RatioDensityDifference : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6) return 0;
		double p_in  = clus.self / clus.degree;
		double p_out = clus.exit() / (total.degree - clus.degree);
		return p_out - p_in;
	}
};

struct NormalizedDensityDifference : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.size < 1e-6) return 0;
		double p_in  = clus.self / clus.size;
		double p_out = clus.exit() / (total.size - clus.size);
		return p_out - p_in;
	}
};

struct WeightedDensityDifference : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6) return 0;
		if (clus.degree > total.degree - 1e-6) return 0;
		double p_in  = clus.degree * clus.self / clus.degree;
		double p_out = clus.degree * clus.exit() / (total.degree - clus.degree);
		return p_out - p_in;
	}
};

struct NormalizedCut : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.size < 1e-6) return 0;
		//if (clus.size == total.size) return 1e5; // Hack: don't want just a single cluster
		return clus.exit() / clus.size;
	}
};

struct RatioCut : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6) return 0;
		//if (clus.degree == total.degree) return 1e5; // Hack: don't want just a single cluster
		return clus.exit() / clus.degree;
	}
};

struct MinusRatioCut : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree > total.degree + 1e-6) return 1e10;
		return clus.exit() / (total.degree - clus.degree);
	}
};

struct MinusNormalizedCut : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.size > total.size + 1e-6) return 1e10;
		return clus.exit() / (total.size - clus.size);
	}
};

struct PowerRatioCut : public LossFunction {
	double p;
	PowerRatioCut(double p = 0.5) : p(p) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree == 0) return 0;
		return pow(clus.exit() / clus.degree,p);
	}
};

struct ProbBlockModel : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		return clus.degree * clus.degree;
	}
	double global(Doubles const& self_deg_deg, Stats const& total, int num_clusters) const {
		double total_deg_deg = total.degree * total.degree;
		double self_deg_deg2 = self_deg_deg[0] / total_deg_deg;
		double exit_deg_deg2 = (total_deg_deg - self_deg_deg[0]) / total_deg_deg;
		return entropy2(total.self  /total.degree, self_deg_deg2)
		     + entropy2(total.exit()/total.degree, exit_deg_deg2);
	}
	inline static double entropy2(double a, double b) {
		return plogp(a / (a+b)) + plogp(b / (a+b));
	}
};

struct PBM2 : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		return clus.degree * clus.degree;
	}
	double global(Doubles const& self_deg_deg, Stats const& total, int num_clusters) const {
		double a = 2.*sqrt(self_deg_deg[0]/total.degree + 3./8);
		double b = 2.*sqrt(total.self + 3./8);
		// based on Anscombe transform
		return a - b;
	}
};

struct PBM3 : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		return clus.size * clus.size;
	}
	double global(Doubles const& self_deg_deg, Stats const& total, int num_clusters) const {
		double max_self = self_deg_deg[0];
		double max_exit = total.size*total.size - max_self;
		double a = total.self   / max_self;
		double b = total.exit() / max_exit;
		return b - a;
	}
};

struct PBM4 : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		return clus.degree * clus.degree;
	}
	double global(Doubles const& self_deg_deg, Stats const& total, int num_clusters) const {
		double max_self = self_deg_deg[0] / total.degree;
		double max_exit = total.degree - max_self;
		double a = total.self   / max_self;
		double b = total.exit() / max_exit;
		return b - a;
	}
};

struct NormalGivenDegree_wrong : public LossFunction {
	double Pb; // total normalized weight of between cluster edges
	Doubles local(Stats const& clus, Stats const& total) const {
		return sqr(clus.degree / total.degree);
	}
	double global(Doubles const& self_deg_deg, Stats const& total, int num_clusters) const {
		double Eb = Pb * (1 - self_deg_deg[0]);
		double Ew = 1 - Eb;
		return Ew*Eb; // sqrt of that, plus constant junk
	}
};

struct PoissonGivenDegree : public LossFunction {
	double self_loops;
	double prior_a, prior_b;
	PoissonGivenDegree() : self_loops(1.0), prior_a(1.0), prior_b(1.0) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		return sqr(clus.degree + self_loops*clus.size);
	}
	double global(Doubles const& within_deg_deg, Stats const& total, int num_clusters) const {
		// Assume a flat prior over clusterings
		// Assume that edge count has poisson distribution with rate proportional to di*dj
		// q(G,C) = ∑{i,j} ci==cj ? Log[(α*di*dj)^Eij/Eij! * Exp[-α*di*dj]]
		//                        : Log[(β*di*dj)^Eij/Eij! * Exp[-β*di*dj]]
		//        = ∑{i,j} ci==cj ? Eij*Log[α] - α*di*dj : ...
		double between_deg_deg = sqr(total.degree + self_loops*total.size) - within_deg_deg[0]; // ∑ di*dj for between
		// Note that the first merging of two singletons will not happen if we don't have self loops
		// therefore, add them here
		double within  = total.self   + self_loops*total.size;
		double between = total.degree - total.self; // + size - size
		// calculate a and b to maximize likelihood
		// do some smoothing to prevent div-by-0 and log(0),
		// this is probably some kind of prior (gamma?).
		double a = (within  + prior_a) / (within_deg_deg[0] + prior_b);
		double b = (between + prior_a) / (between_deg_deg   + prior_b);
		// loss
		double logp = within  * log(a) - a * within_deg_deg[0]
		            + between * log(b) - b * between_deg_deg;
		return -logp;
	}
};

// put a Chinese Restaurant Process prior instead of a flat prior
template <typename Base>
struct CRPPrior : public Base {
	double strength;
	CRPPrior() : strength(1) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.size <= 1e-6) return 0;
		Doubles out = Base::local(clus,total);
		out[1] = lgamma(clus.size);
		return out;
	}
	double global(Doubles const& local, Stats const& total, int num_clusters) const {
		double loss = Base::global(local, total, num_clusters);
		return loss - strength * (local[1] - lgamma(total.size+1));
	}
};

struct NumClusPlus : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		return clus.size > 0 ? 1 : 0;
	}
};
struct NumClusMinus : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		return clus.size > 0 ? -1 : 0;
	}
};
struct NumSelf : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		return -clus.self / total.degree;
	}
};

struct PowerModularity : public LossFunction {
	double p;
	PowerModularity(double p = 1.5) : p(p) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6) return 0;
		return -clus.self / total.degree + pow(clus.degree / total.degree,p) / (p-1);
	}
};
struct ShiftedParabola : public LossFunction {
	double z;
	ShiftedParabola(double z = 0.5) : z(z) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6) return 0;
		double s = clus.self/clus.degree;
		double d = clus.degree/total.degree;
		return s * d * (d - z);
	}
};
struct Cubic : public LossFunction {
	double a;
	Cubic(double a = -0.44) : a(a) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6) return 0;
		double x = clus.degree / total.degree;
		double b = 1.;
		double c = -1-a;
		double w = x * (c + x * (b + x * a));
		return clus.self / clus.degree * w;
	}
};
struct BetaLoss : public LossFunction {
	double a,b;
	BetaLoss(double a = 2, double b = 10) : a(a),b(b) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6) return 0;
		double d = clus.degree / total.degree;
		return -clus.self / clus.degree * pow(d,a) * pow(1-d,b);
	}
};


struct KL1 : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.self < 1e-6) return 0;
		return - clus.degree / total.degree * log(clus.degree / (clus.self / clus.degree));
	}
};
struct CrossEntropy : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6) return 0;
		return clus.self / total.degree * log(clus.degree / total.degree);
	}
	double global(Doubles const& sum_local, Stats const& total, int num_clusters) const {
		return sum_local[0] / max(1., total.self);
	}
};
struct CrossEntropy2 : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.degree < 1e-6) return 0;
		return clus.self / total.degree * log(clus.degree / total.degree);
	}
	double global(Doubles const& sum_local, Stats const& total, int num_clusters) const {
		double ts = total.self / total.degree;
		return sum_local[0] + plogp(ts) + plogp(1-ts);
	}
};

struct Circle : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.self < 1e-6) return 0;
		return clus.self / total.degree * (sqrt(1 - sqr(1 - clus.degree / clus.self)) - 1);
	}
};
struct Parabola2 : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.self < 1e-6) return 0;
		return -clus.self / total.degree * sqr(1 - clus.degree / clus.self);
	}
};
struct SSqrt : public LossFunction {
	Doubles local(Stats const& clus, Stats const& total) const {
		if (clus.self < 1e-6) return 0;
		double d = clus.degree / clus.self;
		return -clus.self / total.degree * sqrt(d * (1 - d));
	}
};

struct LogWithin : public LossFunction {
	double scale;
	LogWithin(double scale = 1) : scale(scale) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		double w = clus.self / total.degree;
		return -log(scale * w + 1);
	}
};
struct SqrtWithin : public LossFunction {
	double scale;
	SqrtWithin(double scale = 1) : scale(scale) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		double w = clus.self / total.degree;
		return -sqrt(scale * w + 1);
	}
};
struct PowerWithin : public LossFunction {
	double power;
	PowerWithin(double power = 1) : power(power) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		double w = clus.self / total.degree;
		return -pow(w, power);
	}
};

struct WLogLoss : public LossFunction {
	double scale;
	WLogLoss(double scale = 1) : scale(scale) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		double w = clus.self / total.degree;
		double v = clus.degree / total.degree;
		return -w * log(1 + exp(-scale * v));
	}
};
struct WHingeLoss : public LossFunction {
	double scale;
	WHingeLoss(double scale = 1) : scale(scale) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		double w = clus.self / total.degree;
		double v = clus.degree / total.degree;
		return -w * max(0.0, 1 - scale * v);
	}
};
struct LocalModularity : public LossFunction {
	double k;
	LocalModularity(double k = 1e-10) : k(k) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		double w = clus.self / total.degree;
		double v = clus.degree / total.degree;
		return -w + v * v / (w + k);
	}
};
struct ConstantPenalty : public LossFunction {
	double k;
	ConstantPenalty(double k = 1e-1) : k(k) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		double w = clus.self / total.degree;
		if (clus.degree < 1e-8) return 0;
		return -w + k;
	}
};
struct MonotonicModularity : public LossFunction {
	double m;
	double g;
	MonotonicModularity(double m = 1, double g = 2) : m(m), g(g) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		double w = clus.self / total.degree;
		double v = clus.degree / total.degree;
		double n = m + g*v;
		return -w/n + sqr(v/n);
	}
};



struct ExtraSelf : public LossFunction {
	shared_ptr<LossFunction> lossfun;
	double extra;
	ExtraSelf(shared_ptr<LossFunction>& lossfun, double extra) : lossfun(lossfun), extra(extra) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		return lossfun->local(clus, total);
	}
	double global(Doubles const& sum_local, Stats const& total, int num_clusters) const {
		return lossfun->global(sum_local, total, num_clusters) + extra * total.self / total.degree;
	}
};
struct ExtraNum : public ExtraSelf {
	ExtraNum(shared_ptr<LossFunction>& lossfun, double extra) : ExtraSelf(lossfun,extra) {}
	double global(Doubles const& sum_local, Stats const& total, int num_clusters) const {
		return lossfun->global(sum_local, total, num_clusters) + extra * num_clusters;
	}
};
// note: only works for additive loss functions
struct ExtraDegreeSqr : public LossFunction {
	shared_ptr<LossFunction> lossfun;
	double target,amount;
	ExtraDegreeSqr(shared_ptr<LossFunction>& lossfun, double target, double amount) : lossfun(lossfun), target(target), amount(amount) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		Doubles l = lossfun->local(clus, total);
		l[0] += amount * sqr(clus.degree / total.degree - target);
		return l;
	}
};
struct ExtraNumTarget : public LossFunction {
	shared_ptr<LossFunction> lossfun;
	double target,amount;
	ExtraNumTarget(shared_ptr<LossFunction>& lossfun, double target, double amount) : lossfun(lossfun), target(target), amount(amount) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		return lossfun->local(clus, total);
	}
	double global(Doubles const& sum_local, Stats const& total, int num_clusters) const {
		double lg = lossfun->global(sum_local, total, num_clusters);
		return lg + amount * sqr(num_clusters - target);
	}
};
struct ExtraNoSingleton : public LossFunction {
	shared_ptr<LossFunction> lossfun;
	double amount;
	ExtraNoSingleton(shared_ptr<LossFunction>& lossfun, double amount = 1.) : lossfun(lossfun), amount(amount) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		Doubles l = lossfun->local(clus, total);
		if (clus.size > 0 && clus.size < 1.5) l[0] += amount;
		return l;
	}
	double global(Doubles const& sum_local, Stats const& total, int num_clusters) const {
		return lossfun->global(sum_local, total, num_clusters);
	}
};
// note: only works for additive loss functions
struct ExtraMaxSize : public LossFunction {
	shared_ptr<LossFunction> lossfun;
	double max_size;
	double amount;
	ExtraMaxSize(shared_ptr<LossFunction>& lossfun, double max_size = 0, double amount = 1e10) : lossfun(lossfun), max_size(max_size), amount(amount) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		Doubles l = lossfun->local(clus, total);
		l[0] += max(clus.size - max_size, 0.) * amount;
		return l;
	}
};

// explicity set the volume of the graph
struct WithTotalVolume : public LossFunction {
	shared_ptr<LossFunction> lossfun;
	double vol;
	WithTotalVolume(shared_ptr<LossFunction>& lossfun, double vol = 1.) : lossfun(lossfun), vol(vol) {}
	Doubles local(Stats const& clus, Stats const& total) const {
		Stats total2 = total;
		total2.degree = total2.self = vol;
		return lossfun->local(clus, total2);
	}
	double global(Doubles const& sum_local, Stats const& total, int num_clusters) const {
		Stats total2 = total;
		total2.degree = total2.self = vol;
		return lossfun->global(sum_local, total2, num_clusters);
	}
};

// -----------------------------------------------------------------------------
// Factory for loss functions based on name
// -----------------------------------------------------------------------------

// Construct a loss function with the given name
// Throws an error if the loss is not supported
shared_ptr<LossFunction> loss_function_by_name(std::string const& name, size_t argc, double const* argv) {
	if (name == "mod" || name == "modularity") {
		return shared_ptr<LossFunction>(new Modularity);
	} else if (name == "smod" || name == "size modularity") {
		return shared_ptr<LossFunction>(new SizeModularity);
	} else if (name == "infomap") {
		return shared_ptr<LossFunction>(new Infomap);
	} else if (name == "infomap_paper" || name == "imp") {
		return shared_ptr<LossFunction>(new InfomapPaper);
	} else if (name == "cond" || name == "conductance") {
		return shared_ptr<LossFunction>(new Conductance);
	} else if (name == "self log degree" || name == "sld" || name == "w-log-v") {
		return shared_ptr<LossFunction>(new SelfLogDegree);
	} else if (name == "self log self" || name == "sls") {
		return shared_ptr<LossFunction>(new SelfLogSelf);
	} else if (name == "degree log degree" || name == "dld") {
		return shared_ptr<LossFunction>(new DegreeLogDegree);
	} else if (name == "self log size" || name == "sln" || name == "w-log-n") {
		return shared_ptr<LossFunction>(new SelfLogSize);
	} else if (name == "self over degree" || name == "sod") {
		return shared_ptr<LossFunction>(new SelfOverDegree);
	} else if (name == "self sqrt degree" || name == "ssd") {
		return shared_ptr<LossFunction>(new SelfSqrtDegree);
	} else if (name == "sss") {
		return shared_ptr<LossFunction>(new SelfSqrtSelf);
	} else if (name == "sss2") {
		return shared_ptr<LossFunction>(new SelfSqrtSelf2);
	} else if (name == "sss3") {
		return shared_ptr<LossFunction>(new SelfSqrtSelf3);
	} else if (name == "self log degree global" || name == "sldg") {
		return shared_ptr<LossFunction>(new SelfLogDegreeGlobal);
	} else if (name == "sldgs") {
		return shared_ptr<LossFunction>(new SelfLogDegreeGlobalSqr);
	} else if (name == "dds" || name == "parabola") {
		return shared_ptr<LossFunction>(new DDS);
	} else if (name == "sdds") {
		return shared_ptr<LossFunction>(new SDDS);
	} else if (name == "slda") {
		return shared_ptr<LossFunction>(new SelfLogDegreeAdjust);
	} else if (name == "slda2") {
		return shared_ptr<LossFunction>(new SelfLogDegreeAdjust2);
	} else if (name == "slda3") {
		return shared_ptr<LossFunction>(new SelfLogDegreeAdjust3);
	} else if (name == "slda4") {
		return shared_ptr<LossFunction>(new SelfLogDegreeAdjust4);
	} else if (name == "slds") {
		return shared_ptr<LossFunction>(new SelfLogDegreeSym);
	} else if (name == "sldm") {
		return shared_ptr<LossFunction>(new SelfLogDegreeMoved);
	} else if (name == "ads") {
		ADS* lossfun = new ADS;
		if (argc > 0) lossfun->pivot = argv[0];
		return shared_ptr<LossFunction>(lossfun);
	} else if (name == "ppd") {
		PPD* lossfun = new PPD;
		if (argc > 0) lossfun->q = argv[0];
		if (argc > 1) lossfun->k = argv[1];
		return shared_ptr<LossFunction>(lossfun);
	} else if (name == "density difference" || name == "ndd") {
		return shared_ptr<LossFunction>(new NormalizedDensityDifference);
	} else if (name == "ratio density difference" || name == "rdd") {
		return shared_ptr<LossFunction>(new RatioDensityDifference);
	} else if (name == "weighted density difference" || name == "wdd") {
		return shared_ptr<LossFunction>(new WeightedDensityDifference);
	} else if (name == "normalized cut" || name == "ncut") {
		return shared_ptr<LossFunction>(new NormalizedCut);
	} else if (name == "ratio cut" || name == "rcut") {
		return shared_ptr<LossFunction>(new RatioCut);
	} else if (name == "minus normalized cut" || name == "mncut") {
		return shared_ptr<LossFunction>(new MinusNormalizedCut);
	} else if (name == "minus ratio cut" || name == "mrcut") {
		return shared_ptr<LossFunction>(new MinusRatioCut);
	} else if (name == "pbm") {
		return shared_ptr<LossFunction>(new ProbBlockModel);
	} else if (name == "pbm2") {
		return shared_ptr<LossFunction>(new PBM2);
	} else if (name == "pbm3") {
		return shared_ptr<LossFunction>(new PBM3);
	} else if (name == "pbm4") {
		return shared_ptr<LossFunction>(new PBM4);
	} else if (name == "poisson" || name == "pgd") {
		PoissonGivenDegree* lossfun = new PoissonGivenDegree;
		if (argc > 0) lossfun->self_loops = argv[0];
		if (argc > 1) lossfun->prior_a = argv[1];
		if (argc > 2) lossfun->prior_b = argv[2];
		return shared_ptr<LossFunction>(lossfun);
	} else if (name == "crp-poisson" || name == "crp") {
		CRPPrior<PoissonGivenDegree>* lossfun = new CRPPrior<PoissonGivenDegree>;
		if (argc > 0) lossfun->strength = argv[0];
		if (argc > 1) lossfun->self_loops = argv[1];
		if (argc > 2) lossfun->prior_a = argv[2];
		if (argc > 3) lossfun->prior_b = argv[3];
		return shared_ptr<LossFunction>(lossfun);
	} else if (name == "num" || name == "num+") {
		return shared_ptr<LossFunction>(new NumClusPlus);
	} else if (name == "num-") {
		return shared_ptr<LossFunction>(new NumClusMinus);
	} else if (name == "self" || name == "num_self") {
		return shared_ptr<LossFunction>(new NumSelf);
	} else if (name == "prcut" || name == "powrcut" || name == "powercut") {
		PowerRatioCut* lossfun = new PowerRatioCut;
		if (argc > 0) lossfun->p = argv[0];
		return shared_ptr<LossFunction>(lossfun);
	} else if (name == "pmod") {
		PowerModularity* lossfun = new PowerModularity;
		if (argc > 0) lossfun->p = argv[0];
		return shared_ptr<LossFunction>(lossfun);
	} else if (name == "par" || name == "sp") {
		ShiftedParabola* lossfun = new ShiftedParabola;
		if (argc > 0) lossfun->z = argv[0];
		return shared_ptr<LossFunction>(lossfun);
	} else if (name == "cubic") {
		Cubic* lossfun = new Cubic;
		if (argc > 0) lossfun->a = argv[0];
		return shared_ptr<LossFunction>(lossfun);
	} else if (name == "kl1") {
		return shared_ptr<LossFunction>(new KL1);
	} else if (name == "ce1" || name == "crossent") {
		return shared_ptr<LossFunction>(new CrossEntropy);
	} else if (name == "ce2" || name == "crossent2") {
		return shared_ptr<LossFunction>(new CrossEntropy2);
	} else if (name == "circ" || name == "circle") {
		return shared_ptr<LossFunction>(new Circle);
	} else if (name == "par2" || name == "parabola2") {
		return shared_ptr<LossFunction>(new Parabola2);
	} else if (name == "ssqrt") {
		return shared_ptr<LossFunction>(new SSqrt);
	} else if (name == "beta") {
		BetaLoss* lossfun = new BetaLoss;
		if (argc > 0) lossfun->a = argv[0];
		if (argc > 1) lossfun->b = argv[1];
		return shared_ptr<LossFunction>(lossfun);
	} else if (name == "lw") {
		LogWithin* lossfun = new LogWithin;
		if (argc > 0) lossfun->scale = argv[0];
		return shared_ptr<LossFunction>(lossfun);
	} else if (name == "sw") {
		SqrtWithin* lossfun = new SqrtWithin;
		if (argc > 0) lossfun->scale = argv[0];
		return shared_ptr<LossFunction>(lossfun);
	} else if (name == "pw") {
		PowerWithin* lossfun = new PowerWithin;
		if (argc > 0) lossfun->power = argv[0];
		return shared_ptr<LossFunction>(lossfun);
	} else if (name == "wll") {
		WLogLoss* lossfun = new WLogLoss;
		if (argc > 0) lossfun->scale = argv[0];
		return shared_ptr<LossFunction>(lossfun);
	} else if (name == "whl") {
		WHingeLoss* lossfun = new WHingeLoss;
		if (argc > 0) lossfun->scale = argv[0];
		return shared_ptr<LossFunction>(lossfun);
	} else if (name == "lmod") {
		LocalModularity* lossfun = new LocalModularity;
		if (argc > 0) lossfun->k = argv[0];
		return shared_ptr<LossFunction>(lossfun);
	} else if (name == "cp") {
		ConstantPenalty* lossfun = new ConstantPenalty;
		if (argc > 0) lossfun->k = argv[0];
		return shared_ptr<LossFunction>(lossfun);
	} else if (name == "mom") {
		MonotonicModularity* lossfun = new MonotonicModularity;
		if (argc > 0) lossfun->m = argv[0];
		if (argc > 1) lossfun->g = argv[1];
		return shared_ptr<LossFunction>(lossfun);
	} else {
		throw std::invalid_argument("Unrecognized loss function: '" + name + "'");
	}
}

// -----------------------------------------------------------------------------
}
#endif
