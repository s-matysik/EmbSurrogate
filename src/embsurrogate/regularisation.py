"""Selection of lambda and k for the surrogate — measured values, not assumed a priori.

MECHANISM
---------
Throughout the package, the ridge system has the form

    (K + shift * I) alpha = Y,   where   shift = lambda * N

(see `dual._solve_ridge` and `run_primal.dual_cg`). This parameterization is
standard: `lambda` is a penalty PER TRAINING PAIR, so
at a fixed lambda, the diagonal shift grows proportionally to N.
The question that had to be settled by measurement was: which of these two
parameters is the invariant of the task — lambda (penalty per pair) or shift
(absolute shift)? The answer has direct consequences, because if
shift were the invariant, then when going from 9600 to 20000 pairs one would need
to decrease lambda by more than a factor of two.

Scan at two dataset sizes (validation 1200 pairs, sign features 3-4,
global Spearman rho on the similarity matrix):

    lambda     shift@9600   rho@9600      shift@20000   rho@20000
    1e-5         0.096        0.7550         0.20          0.7781
    3e-5         0.288      * 0.7608 *       0.60        * 0.7822 *
    6.25e-5      0.600        0.7582          —             —
    1e-4         0.960        0.7516         2.0           0.7657
    3e-4           —            —            6.0           0.7217
    1e-3           —            —           20.0          0.6251

The optimum occurs at THE SAME lambda (3e-5) for both sizes, and not at the
same shift (at N = 9600, the shift-equivalent of the optimum from 20000, i.e.
lambda = 6.25e-5, is worse by 0.0026 than lambda = 3e-5). The invariant is
thus the penalty per pair: lambda does NOT scale with N.

Practical conclusion: the previously used lambda = 1e-4 is simply too large, and its
cost grows with N, because at a fixed lambda the shift grows:

    N = 9600:   0.7608 -> 0.7516   (loss 0.0092)
    N = 20000:  0.7822 -> 0.7657   (loss 0.0165)

RANGE OF VALIDITY
---------------
The gain applies to pure ridge regression. After applying residual correction
(neighbourhood k, OOF coordinates), the difference almost disappears — the local
correction fixes most of what excessive regularization breaks:

    N = 20000, lambda 1e-4, k = 50:  rho 0.7905
    N = 20000, lambda 3e-5, k = 50:  rho 0.7912
    N = 20000, lambda 3e-5, k = 100: rho 0.7914   (best measured)

That is: if you compute the ridge alone, lambda = 3e-5 is worth 0.0165 rho; if
you compute the full variant with residual correction, it is worth about 0.001, i.e. it falls
within the noise (CI95 wide at about +-0.013). There is one cost: a smaller lambda gives a
worse-conditioned system, so the iterative solver needs more iterations —
at N = 20000 it grows from 58 (lambda 1e-4) to 102 (lambda 3e-5), and at
lambda 1e-5 to 173. For lambda 3e-5 this is still a reasonable price.

The neighbourhood k has a flat optimum between 50 and 100 (0.7912 / 0.7914); by default
50, because it is cheaper at inference time, and the difference lies within the noise."""

from __future__ import annotations

# Penalty per single training pair — invariant, DO NOT scale with N.
LAMBDA_DOMYSLNA = 3e-5

# Neighbourhood size in residuals correction; flat optimum 50-100.
K_DOMYSLNE = 50

# Measured global rho (validation 1200, seeds 3-4) — for regression comparisons.
ZMIERZONE_RIDGE = {
    (9600, 1e-5): 0.7550,
    (9600, 3e-5): 0.7608,
    (9600, 6.25e-5): 0.7582,
    (9600, 1e-4): 0.7516,
    (20000, 1e-5): 0.7781,
    (20000, 3e-5): 0.7822,
    (20000, 1e-4): 0.7657,
    (20000, 3e-4): 0.7217,
    (20000, 1e-3): 0.6251,
}


def parametry(n_par: int) -> dict:
    """Recommended (lambda, k) for a training set with n_par pairs.

    Also returns `shift`, since it is this number that actually goes onto the diagonal and,
    when diagnosing CG convergence, this is what is examined, not lambda.
    """
    return {
        "lam": LAMBDA_DOMYSLNA,
        "k": K_DOMYSLNE,
        "shift": LAMBDA_DOMYSLNA * n_par,
        "degree": 0,
    }
