"""Portability test of the method across other teachers (items 1-4 of round 3).

MECHANISM
---------
The question is: does the n-gram surrogate work because it hit one particular
teacher model, or because kernel regression on character features is a
good approximation of EVERY sentence encoder on this corpus. To
answer this, we hold EVERYTHING except the teacher constant: the same corpus
(data/korpus45k.jsonl), the same split (klon.run_big.indices), the same features
(characters 3-4), the same lambda, the same N. Only the target matrix Y changes.

Key computational observation: the kernel K = Phi Phi^T depends ONLY on the features,
so we compute the Cholesky decomposition ONCE and use it for all targets
(alpha_t = G^-1 Y_t is three back-substitutions, not three factorizations).
Three teachers therefore cost as much time as one plus 5%, and the comparison
is exact: different f_nl and different rho result solely from Y, because the
p^2 matrix is literally the same vector of numbers for every teacher.

We compute the error decomposition on the leave-one-out residuals from the closed-form
formula e_i = alpha_i / (G^-1)_ii and the regression ||e_i||^2 = a + b p_i^2, where p^2 is
the squared distance of the feature vector from the span of the rest. NOTE the definition of
p^2: it is pure feature geometry, so we compute it on the kernel WITHOUT ridge
penalty (jitter 1e-6):
    training:    p^2_i = 1 / (K^-1)_ii
    validation:  p^2_q = 1 - k_q^T K^-1 k_q      (||phi_q|| = 1)
The variant with G = K + lambda N I ("_lam" columns in the CSV) is WRONG as a measure
of coverage and is recorded only as a negative control: at lambda N ~ 1
it inflates the training leverage above unity (median p^2 = 1.58 at N = 9600),
which gives a negative intercept and f_nl outside [0, 1]. The residuals e_i always come from
the actual model, i.e. from lambda = 1e-4.

MEASURED EFFECT (validation 1200, N = 9600, features characters 3-4, lambda = 1e-4)
-------------------------------------------------------------------------
                                 floor  ridge  +k50 correction  f_nl
paraphrase-multi-L12-v2 (384D)     0.349  0.7598     0.7625    0.339
all-MiniLM-L6-v2        (384D)     0.673  0.8325     0.8311    0.276
distiluse-multi-v1      (512D)     0.351  0.7597     0.7588    0.345
(column is median cos; global rho ridge/k50: 0.7516/0.7714,
 0.6921/0.7215, 0.6521/0.6810)

Conclusions that follow from this and that this method is meant to establish:
1. Residual correction improves global rho for all three teachers
   (k = 50: +0.020 / +0.029 / +0.029) and for all three it is practically
   neutral on cosine alone (from -0.001 to +0.003). The direction and order of
   magnitude of the effect are the same, so the method is not a property of one
   (model, corpus) pair.
2. f_nl < 0.4 for every teacher and at every N: lack of coverage dominates.
   The spread between teachers (0.276-0.345) is smaller than the drift of
   f_nl itself with N (0.053 -> 0.339 for N = 2400 -> 9600), so "the target is nearly
   linear in n-grams" is a property of the feature space, not of the model.
3. The floor is MANDATORY in reporting: all-MiniLM-L6 has a shared
   component so strong that the mean training vector alone gives cos = 0.673,
   with the model's advantage being only +0.158. Without the floor its cos of 0.83
   would look better than the 0.76 of the multilingual teacher, when it is the opposite:
   on rho per query it comes out at 0.700 versus 0.761.

    python3 -m klon.teachers                      # N = 9600, full metrics
    KLON_N=4800 KLON_ONLY_DIAG=1 python3 -m klon.teachers   # error decomposition only

Environment variables: KLON_N, KLON_NBOOT, KLON_SUF, KLON_ONLY_DIAG.
Run as a separate process with OMP_NUM_THREADS=2, see run_big docstring."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
import gc, json, pathlib, sys, time
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cho_factor, cho_solve, lapack

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from klon import dual as D, evaluate as E, features as F
from klon.run_big import indices
from klon.run_experiment import load_dataset

N = int(os.environ.get("KLON_N", 9600))
LAM = 1e-4
EPS = 1e-6
KS = (20, 50)
NBOOT = int(os.environ.get("KLON_NBOOT", 200))
SUF = os.environ.get("KLON_SUF", "")
OUT = ROOT / "wyniki"
OUT.mkdir(exist_ok=True)

TEACHERS = [
    ("paraphrase-multilingual-MiniLM-L12-v2", ROOT / "data" / "ds_big3", 384),
    ("all-MiniLM-L6-v2", ROOT / "data" / "ds_t_minilm6", 384),
    ("distiluse-base-multilingual-cased-v1", ROOT / "data" / "ds_t_distiluse", 512),
]
TEACHERS = [t for t in TEACHERS if (t[1] / "vectors.npy").exists()]
print("teachers:", [t[0] for t in TEACHERS], flush=True)
assert len(TEACHERS) >= 2


def zeruj_gorny(L, n, B=2048):
    """cho_factor/trtri leave garbage, not zeros, in the unused triangle."""
    for c0 in range(0, n, B):
        c1 = min(c0 + B, n)
        if c0:
            L[:c0, c0:c1] = 0.0
        blk = L[c0:c1, c0:c1]
        blk[np.triu_indices(c1 - c0, 1)] = 0.0
    return L


def gram(Xp, n, dtype, ridge):
    Gm = np.empty((n, n), dtype=dtype, order="F")
    for i0 in range(0, n, 1000):
        i1 = min(i0 + 1000, n)
        Gm[i0:i1, :] = (Xp[i0:i1] @ Xp.T).toarray().astype(dtype)
    Gm.flat[:: n + 1] = 1.0 + ridge
    return Gm


# ---- text and split (from the largest set; the pool prefix is the same) ----
texts_ref, _, _ = load_dataset(TEACHERS[0][1])
val, t0, pool = indices(len(texts_ref))
tr = pool[:N]
assert len(np.intersect1d(tr, val)) == 0 and len(np.intersect1d(tr, t0)) == 0
assert len(np.intersect1d(val, t0)) == 0
print(f"N={N} validation={len(val)} (T0 unchanged)", flush=True)

Ys = {}
for nm, d, dim in TEACHERS:
    txt, Y, _ = load_dataset(d)
    assert Y.shape[1] == dim, (nm, Y.shape)
    assert Y.shape[0] >= max(tr.max(), val.max()) + 1, f"{nm}: za maly zbior"
    for i in list(tr[:50]) + list(val[:50]):
        assert txt[i] == texts_ref[i], f"{nm}: text divergence at {i}"
    assert np.allclose(np.linalg.norm(Y[val], axis=1), 1.0, atol=1e-3)
    Ys[nm] = Y
    print(f"  {nm}: {Y.shape}", flush=True)
del txt

# ---- features (common to all teachers) ----
t = time.time()
cfg = dict(F.DEFAULT_CONFIG)
Xp, idf = F.fit_transform([texts_ref[i] for i in tr], cfg)
Xv = F.transform([texts_ref[i] for i in val], idf, cfg)
nnz = round(Xp.nnz / N)
print(f"features {Xp.shape} nnz/text {nnz} ({time.time()-t:.0f}s)", flush=True)
Ks = np.asarray((Xv @ Xp.T).todense(), dtype=np.float64)          # 1200 x N

# ---- stage A: model (lambda = 1e-4), float32 ----
t = time.time()
G = gram(Xp, N, np.float32, LAM * N)
c, low = cho_factor(G, lower=True, overwrite_a=True, check_finite=False)
alphas = {nm: cho_solve((c, low), Ys[nm][tr].astype(np.float32),
                        check_finite=False).astype(np.float64) for nm, _, _ in TEACHERS}
Wl = cho_solve((c, low), Ks.T.astype(np.float32), check_finite=False)
p2_val_lam = 1.0 - np.einsum("ij,ij->j", Ks.T, Wl.astype(np.float64))
del Wl
Linv, info = lapack.strtri(c, lower=1, overwrite_c=1)
assert info == 0, info
diag_inv = np.einsum("ki,ki->i", zeruj_gorny(Linv, N), Linv, optimize=True).astype(np.float64)
p2_tr_lam = 1.0 / diag_inv
del Linv, G, c
gc.collect()
print(f"etap A: alpha + LOO ({time.time()-t:.0f}s)", flush=True)

# ---- stage B: pure feature geometry (jitter 1e-6), float64 ----
t = time.time()
KJ = gram(Xp, N, np.float64, EPS)
c2, low2 = cho_factor(KJ, lower=True, overwrite_a=True, check_finite=False)
W = cho_solve((c2, low2), Ks.T, check_finite=False)
p2_val = 1.0 - np.einsum("ij,ij->j", Ks.T, W)
del W
LinvK, info = lapack.dtrtri(c2, lower=1, overwrite_c=1)
assert info == 0, info
diag_invK = np.einsum("ki,ki->i", zeruj_gorny(LinvK, N), LinvK, optimize=True)
p2_tr = 1.0 / diag_invK
del LinvK, KJ, c2
gc.collect()
print(f"etap B: p^2 ({time.time()-t:.0f}s)\n  p^2 (jitter {EPS:g}):  trening med {np.median(p2_tr):.3f} "
      f"[{np.percentile(p2_tr,5):.3f}-{np.percentile(p2_tr,95):.3f}]  median validation {np.median(p2_val):.3f}\n"
      f"  p^2 (lambda={LAM:g}): trening med {np.median(p2_tr_lam):.3f}  "
      f"validation med {np.median(p2_val_lam):.3f}", flush=True)


def dopasuj(e2, p2t, p2v):
    A = np.column_stack([np.ones(len(e2)), p2t])
    coef, *_ = np.linalg.lstsq(A, e2, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    pred = A @ coef
    r2 = 1.0 - ((e2 - pred) ** 2).sum() / ((e2 - e2.mean()) ** 2).sum()
    ep2 = float(np.mean(p2v))
    return a, b, r2, ep2, a / (a + b * ep2)


rows, diag_rows = [], []
for nm, _, dim in TEACHERS:
    Ytr, Yval = Ys[nm][tr], Ys[nm][val]
    mu = Ytr.mean(0)
    alpha = alphas[nm]
    R = Ytr - (Xp @ (Xp.T @ alpha))
    z = D.l2norm(Ytr - alpha / diag_inv[:, None])
    b = Ks @ alpha

    e2 = (alpha ** 2).sum(1) / diag_inv ** 2            # ||e_i||^2, LOO formula
    a_nl, b_cov, r2f, ep2, f_nl = dopasuj(e2, p2_tr, p2_val)
    a_l, b_l, r2l, ep2l, f_l = dopasuj(e2, p2_tr_lam, p2_val_lam)
    diag_rows.append({
        "nauczyciel": nm, "wymiar": dim, "N": N, "nnz_na_tekst": nnz,
        "a_podloga_nieliniowosci": a_nl, "b_brak_pokrycia": b_cov,
        "R2_dopasowania": r2f, "srednia_p2_walidacja": ep2,
        "skladnik_pokrycia": b_cov * ep2, "f_nl": f_nl,
        "mediana_p2_trening": float(np.median(p2_tr)),
        "mediana_p2_walidacja": float(np.median(p2_val)),
        "mediana_e2_LOO": float(np.median(e2)), "srednia_e2_LOO": float(np.mean(e2)),
        "a_lam": a_l, "b_lam": b_l, "R2_dopasowania_lam": r2l,
        "srednia_p2_walidacja_lam": ep2l, "f_nl_lam": f_l})
    print(f"  {nm:40s} a={a_nl:.4f} b={b_cov:.4f} R2fit={r2f:.3f} E[p2]={ep2:.3f} "
          f"-> f_nl={f_nl:.4f}  (lam check: f_nl={f_l:.4f})", flush=True)

    if os.environ.get("KLON_ONLY_DIAG"):
        del Ytr, Yval, R, z, b, alpha
        gc.collect()
        continue
    cand = [("podloga_sredni_wektor", E.baseline_mean(Ytr, len(val))),
            ("ridge", D.l2norm(b))]
    for k in KS:
        cand.append((f"ridge+korekta_LOO_k{k}",
                     D.l2norm(b + D._local(D.l2norm(b) @ z.T, R, k, degree=0))))

    cos_floor = None
    for wariant, Yh in cand:
        tt = time.time()
        m = E.report_full(Yval, Yh, mu, n_boot=NBOOT)
        if cos_floor is None:
            cos_floor = m["cos_mediana"]
        rec = {"nauczyciel": nm, "wymiar": dim, "wariant": wariant, "N": N,
               "lambda": LAM, "cechy": "znaki 3-4"}
        rec.update({kk: vv for kk, vv in m.items() if kk != "rho_globalne_CI95"})
        rec["rho_globalne_CI_dol"], rec["rho_globalne_CI_gora"] = m["rho_globalne_CI95"]
        rec["cos_podloga"] = cos_floor
        rec["przewaga_nad_podloga"] = m["cos_mediana"] - cos_floor
        rec["f_nl"] = f_nl
        rec["sekundy"] = round(time.time() - tt, 1)
        rows.append(rec)
        print(f"  {nm:40s} {wariant:26s} cos {m['cos_mediana']:.4f} "
              f"(+{rec['przewaga_nad_podloga']:.4f}) cos_c {m['cos_centrowany']:.4f} "
              f"rho {m['rho_globalne']:.4f} rho_q {m['rho_per_zapytanie']:.4f} "
              f"R2 {m['R2']:.4f} rec@10 {m['recall@10']:.4f} ndcg {m['ndcg@10']:.4f}",
              flush=True)
    del Ytr, Yval, R, z, b, alpha
    gc.collect()

if rows:
    pd.DataFrame(rows).to_csv(OUT / f"nauczyciele{SUF}.csv", index=False)
pd.DataFrame(diag_rows).to_csv(OUT / f"nauczyciele_diagnostyka{SUF}.csv", index=False)

# ---- item 4: agreement of teachers on the same 1200 texts, without regression ----
iu = np.triu_indices(len(val), 1)
uppers = {nm: (Ys[nm][val] @ Ys[nm][val].T)[iu] for nm, _, _ in TEACHERS}
names = [nm for nm, _, _ in TEACHERS]
Z = np.ones((len(names), len(names)))
zrows = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        r = float(stats.spearmanr(uppers[names[i]], uppers[names[j]]).statistic)
        Z[i, j] = Z[j, i] = r
        zrows.append({"nauczyciel_A": names[i], "nauczyciel_B": names[j],
                      "rho_spearman_macierzy_podobienstw": r,
                      "pearson": float(np.corrcoef(uppers[names[i]], uppers[names[j]])[0, 1])})
pd.DataFrame(Z, index=names, columns=names).to_csv(OUT / f"nauczyciele_zgodnosc{SUF}.csv")
pd.DataFrame(zrows).to_csv(OUT / f"nauczyciele_zgodnosc_pary{SUF}.csv", index=False)
print("\n--- agreement (Spearman's rho of similarity matrices, 1200 texts) ---", flush=True)
for r in zrows:
    print(f"  {r['nauczyciel_A']} vs {r['nauczyciel_B']}: "
          f"{r['rho_spearman_macierzy_podobienstw']:.4f}", flush=True)
roz = []
for nm, _, _ in TEACHERS:
    u = uppers[nm]
    roz.append({"nauczyciel": nm, "cos_mediana_par": float(np.median(u)),
                "cos_p95_par": float(np.percentile(u, 95)),
                "cos_p5_par": float(np.percentile(u, 5))})
    print(f"  teacher cosine {nm}: mediana par {np.median(u):.3f} "
          f"p95 {np.percentile(u,95):.3f}", flush=True)
pd.DataFrame(roz).to_csv(OUT / f"nauczyciele_rozklad_cos{SUF}.csv", index=False)
json.dump({"N": N, "lam": LAM, "eps": EPS, "k": list(KS), "n_walidacja": len(val),
           "n_boot": NBOOT, "cechy": "znaki 3-4", "nnz_na_tekst": nnz,
           "nauczyciele": names}, open(OUT / f"nauczyciele_konfig{SUF}.json", "w"), indent=2)
print("KONIEC", flush=True)
