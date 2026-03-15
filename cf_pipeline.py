"""
=======================================================================
Airbnb Collaborative Filtering Pipeline
=======================================================================

Implements ALS and BPR collaborative filtering on top of the existing
content-based pipeline. Also includes a hybrid (CF + content) fusion.

Requirements:
    pip install implicit

Run AFTER recommendation_pipeline.py (uses its saved outputs).
=======================================================================
"""

import json
import os
import pickle
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Check implicit is available ──────────────────────────────────────
try:
    import implicit
    from implicit.als import AlternatingLeastSquares
    from implicit.bpr import BayesianPersonalizedRanking
    print("implicit library found:", implicit.__version__)
except ImportError:
    raise ImportError(
        "Please install the 'implicit' library first:\n"
        "    pip install implicit"
    )

# =====================================================================
# PATHS — mirrors recommendation_pipeline.py
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EDA_DIR  = os.path.join(BASE_DIR, "eda_outputs_new1")
CB_DIR   = os.path.join(BASE_DIR, "recommendation_outputs_new1")  # content-based outputs
OUT_DIR  = os.path.join(BASE_DIR, "recommendation_outputs_cf")
os.makedirs(OUT_DIR, exist_ok=True)

K_VALUES = [5, 10, 15, 20]


# =====================================================================
# STEP 1: LOAD SAVED ARTIFACTS FROM CONTENT-BASED PIPELINE
# =====================================================================
print("=" * 70)
print("STEP 1: LOAD SAVED ARTIFACTS")
print("=" * 70)

# Listing IDs and fused embeddings saved by the original pipeline
listing_ids  = np.load(os.path.join(CB_DIR, "listing_ids.npy"))
fused_emb    = np.load(os.path.join(CB_DIR, "fused_embeddings_256d.npy"))
lid_to_idx   = {lid: i for i, lid in enumerate(listing_ids)}
n_listings   = len(listing_ids)
print(f"  Listings: {n_listings}  |  Fused embedding: {fused_emb.shape}")

# Train / test splits
train_df = pd.read_csv(os.path.join(CB_DIR, "train_chrono.csv"))
test_df  = pd.read_csv(os.path.join(CB_DIR, "test_chrono.csv"))
print(f"  Train interactions: {len(train_df)}")
print(f"  Test  interactions: {len(test_df)}")

# Filter to valid listings only
valid_lids  = set(listing_ids)
train_df    = train_df[train_df["listing_id"].isin(valid_lids)].copy()
test_df     = test_df[test_df["listing_id"].isin(valid_lids)].copy()

# Warm / cold test users
train_users   = set(train_df["reviewer_id"].unique())
warm_test_df  = test_df[test_df["reviewer_id"].isin(train_users)]
print(f"  Warm test users: {warm_test_df['reviewer_id'].nunique()}")
print(f"  Cold test users: {test_df[~test_df['reviewer_id'].isin(train_users)]['reviewer_id'].nunique()}")


# =====================================================================
# STEP 2: BUILD USER / ITEM INDEX MAPS & SPARSE MATRIX
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2: BUILD INTERACTION MATRIX")
print("=" * 70)

# Map reviewer_id -> integer user index (training users only)
all_train_users = train_df["reviewer_id"].unique()
user_to_idx     = {uid: i for i, uid in enumerate(all_train_users)}
n_users         = len(all_train_users)
print(f"  Training users: {n_users}  |  Items: {n_listings}")

# Build COO then convert to CSR (users x items)
rows = train_df["reviewer_id"].map(user_to_idx).values
cols = train_df["listing_id"].map(lid_to_idx).values
data = np.ones(len(rows), dtype=np.float32)

user_item_csr = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_listings))
item_user_csr = user_item_csr.T.tocsr()   # implicit library expects items x users
print(f"  User-item matrix: {user_item_csr.shape}  "
      f"nnz={user_item_csr.nnz}  "
      f"density={user_item_csr.nnz / (n_users * n_listings):.6f}")


# =====================================================================
# STEP 3: TRAIN ALS
# =====================================================================
print("\n" + "=" * 70)
print("STEP 3: TRAIN ALS (Alternating Least Squares)")
print("=" * 70)

als_model = AlternatingLeastSquares(
    factors=128,          # latent dimension
    regularization=0.05,
    alpha=40,             # confidence scaling for implicit feedback
    iterations=30,
    use_gpu=False,
    random_state=42,
)

t0 = time.time()
als_model.fit(item_user_csr)
print(f"  ALS training time: {time.time()-t0:.1f}s")
print(f"  User factors: {als_model.user_factors.shape}")
print(f"  Item factors: {als_model.item_factors.shape}")

# Save model
with open(os.path.join(OUT_DIR, "als_model.pkl"), "wb") as f:
    pickle.dump(als_model, f)
print("  Saved als_model.pkl")


# =====================================================================
# STEP 4: TRAIN BPR
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4: TRAIN BPR (Bayesian Personalized Ranking)")
print("=" * 70)

bpr_model = BayesianPersonalizedRanking(
    factors=128,
    learning_rate=0.01,
    regularization=0.01,
    iterations=100,
    use_gpu=False,
    random_state=42,
)

t0 = time.time()
bpr_model.fit(item_user_csr)
print(f"  BPR training time: {time.time()-t0:.1f}s")
print(f"  User factors: {bpr_model.user_factors.shape}")
print(f"  Item factors: {bpr_model.item_factors.shape}")

with open(os.path.join(OUT_DIR, "bpr_model.pkl"), "wb") as f:
    pickle.dump(bpr_model, f)
print("  Saved bpr_model.pkl")


# =====================================================================
# STEP 5: EVALUATION HELPERS
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5: EVALUATION")
print("=" * 70)

def compute_ranks(sim_scores, true_indices):
    """Rank of true item in each row of sim_scores (lower = better)."""
    true_scores = sim_scores[np.arange(len(true_indices)), true_indices][:, None]
    BATCH = 2000
    ranks = []
    for b in range(0, len(true_indices), BATCH):
        bs = sim_scores[b:b+BATCH]
        bt = true_scores[b:b+BATCH]
        ranks.extend((bs > bt).sum(axis=1) + 1)
    return np.array(ranks, dtype=float)


def metrics_from_ranks(all_ranks, is_warm):
    res = {}
    for k in K_VALUES:
        hit  = (all_ranks <= k).astype(float)
        ndcg = np.where(all_ranks <= k, 1.0 / np.log2(all_ranks + 1), 0.0)
        mapk = np.where(all_ranks <= k, 1.0 / all_ranks, 0.0)
        prec = hit / k

        res[f"Hit@{k}"]    = hit.mean()
        res[f"Prec@{k}"]   = prec.mean()
        res[f"Recall@{k}"] = hit.mean()
        res[f"NDCG@{k}"]   = ndcg.mean()
        res[f"MAP@{k}"]    = mapk.mean()

        if is_warm.sum() > 0:
            res[f"Hit@{k}_W"]    = hit[is_warm].mean()
            res[f"Prec@{k}_W"]   = prec[is_warm].mean()
            res[f"Recall@{k}_W"] = hit[is_warm].mean()
            res[f"NDCG@{k}_W"]   = ndcg[is_warm].mean()
            res[f"MAP@{k}_W"]    = mapk[is_warm].mean()
        else:
            for m in ["Hit", "Prec", "Recall", "NDCG", "MAP"]:
                res[f"{m}@{k}_W"] = 0.0

    rr = 1.0 / all_ranks
    res["MRR"]   = rr.mean()
    res["MRR_W"] = rr[is_warm].mean() if is_warm.sum() > 0 else 0.0
    return res



def get_factors(model):
    """
    Return (user_factors, item_factors) correctly oriented.
    implicit is fit on item_user_csr, so internally the naming can be swapped
    depending on the library version. We detect orientation by matching shapes.
    """
    a, b = model.user_factors, model.item_factors
    if a.shape[0] == n_users and b.shape[0] == n_listings:
        return a, b   # correct: user_factors=(n_users,F), item_factors=(n_items,F)
    elif a.shape[0] == n_listings and b.shape[0] == n_users:
        return b, a   # swapped (common in implicit >= 0.6)
    else:
        raise ValueError(
            f"Cannot determine factor orientation: "
            f"model.user_factors={a.shape}, model.item_factors={b.shape}, "
            f"n_users={n_users}, n_listings={n_listings}"
        )


def evaluate_cf(model, name, fallback_emb=None):
    """
    Evaluate a CF model (ALS or BPR) on the chronological test set.

    For warm users  -> use model's learned user factors directly.
    For cold users  -> fall back to content-based mean-pooled embedding
                       (projected to same space via dot with item factors).
                       If fallback_emb is None, cold users are skipped.
    """
    t0 = time.time()

    user_factors, item_factors = get_factors(model)
    print(f"    user_factors: {user_factors.shape}  item_factors: {item_factors.shape}")

    # Normalise for cosine-style scoring
    item_norm = item_factors / (np.linalg.norm(item_factors, axis=1, keepdims=True) + 1e-9)

    train_hist = train_df.groupby("reviewer_id")["listing_id"].apply(list).to_dict()

    test_uids = test_df["reviewer_id"].values
    test_lids = test_df["listing_id"].values

    user_vecs, is_warm_list = [], []
    for uid in test_uids:
        if uid in user_to_idx:
            # Warm: pull the trained user vector
            uidx = user_to_idx[uid]
            uv = user_factors[uidx]
            is_warm_list.append(True)
        else:
            # Cold: mean of item factors of history (if any), else global mean
            if uid in train_hist:
                idxs = [lid_to_idx[l] for l in train_hist[uid] if l in lid_to_idx]
            else:
                idxs = []
            if idxs:
                uv = item_factors[idxs].mean(axis=0)
            else:
                uv = item_factors.mean(axis=0)
            is_warm_list.append(False)
        user_vecs.append(uv)

    user_mat  = np.array(user_vecs)
    is_warm   = np.array(is_warm_list)
    user_norm = user_mat / (np.linalg.norm(user_mat, axis=1, keepdims=True) + 1e-9)

    # Score matrix: (n_test, n_items)
    scores = user_norm @ item_norm.T

    # Filter to test rows where true listing is in our catalog
    valid_rows = [i for i, l in enumerate(test_lids) if l in lid_to_idx]
    true_idx   = np.array([lid_to_idx[test_lids[i]] for i in valid_rows])
    valid_rows = np.array(valid_rows)

    if len(valid_rows) == 0:
        print(f"  {name}: no valid test rows!")
        return {}

    scores_v = scores[valid_rows]
    warm_v   = is_warm[valid_rows]
    ranks    = compute_ranks(scores_v, true_idx)

    res = {"Modality": name}
    res.update(metrics_from_ranks(ranks, warm_v))
    res["time_sec"] = time.time() - t0

    print(f"  {name:<35} ({res['time_sec']:.1f}s) | "
          f"H@5={res['Hit@5']:.4f}  H@10={res['Hit@10']:.4f}  "
          f"H@15={res['Hit@15']:.4f}  H@20={res['Hit@20']:.4f} | "
          f"H@10_W={res['Hit@10_W']:.4f}  MRR_W={res['MRR_W']:.4f}")
    return res


# ── ALS standalone ──────────────────────────────────────────────────
print("\nRunning ALS evaluation...")
als_res = evaluate_cf(als_model, "ALS (factors=128)")

# ── BPR standalone ──────────────────────────────────────────────────
print("\nRunning BPR evaluation...")
bpr_res = evaluate_cf(bpr_model, "BPR (factors=128)")


# =====================================================================
# STEP 6: HYBRID FUSION (CF score + Content-Based score)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 6: HYBRID FUSION (CF + Content-Based)")
print("=" * 70)

# ── Content-based similarity scores ──────────────────────────────────
def _build_cb_scores():
    """Build content-based score matrix for test users (reused from pipeline)."""
    cb_norm     = fused_emb / (np.linalg.norm(fused_emb, axis=1, keepdims=True) + 1e-9)
    train_hist  = train_df.groupby("reviewer_id")["listing_id"].apply(list).to_dict()
    global_mean = fused_emb.mean(axis=0)

    test_uids = test_df["reviewer_id"].values
    test_lids = test_df["listing_id"].values

    user_vecs, is_warm_list = [], []
    for uid in test_uids:
        if uid in train_hist:
            idxs = [lid_to_idx[l] for l in train_hist[uid] if l in lid_to_idx]
            uv   = fused_emb[idxs].mean(axis=0) if idxs else global_mean
            is_warm_list.append(bool(idxs))
        else:
            uv = global_mean
            is_warm_list.append(False)
        user_vecs.append(uv)

    user_mat  = np.array(user_vecs)
    is_warm   = np.array(is_warm_list)
    user_norm = user_mat / (np.linalg.norm(user_mat, axis=1, keepdims=True) + 1e-9)
    scores    = user_norm @ cb_norm.T          # (n_test, n_items)

    valid_rows = np.array([i for i, l in enumerate(test_lids) if l in lid_to_idx])
    true_idx   = np.array([lid_to_idx[test_lids[i]] for i in valid_rows])
    return scores, valid_rows, true_idx, is_warm


def _cf_scores(model):
    """Build CF score matrix for all test users."""
    user_f, item_f = get_factors(model)
    item_norm = item_f / (np.linalg.norm(item_f, axis=1, keepdims=True) + 1e-9)

    train_hist = train_df.groupby("reviewer_id")["listing_id"].apply(list).to_dict()
    test_uids  = test_df["reviewer_id"].values

    user_vecs = []
    for uid in test_uids:
        if uid in user_to_idx:
            uv = user_f[user_to_idx[uid]]
        else:
            idxs = [lid_to_idx[l] for l in train_hist.get(uid, []) if l in lid_to_idx]
            uv   = item_f[idxs].mean(axis=0) if idxs else item_f.mean(axis=0)
        user_vecs.append(uv)

    user_mat  = np.array(user_vecs)
    user_norm = user_mat / (np.linalg.norm(user_mat, axis=1, keepdims=True) + 1e-9)
    return user_norm @ item_norm.T


print("  Building content-based scores...")
cb_scores, valid_rows, true_idx, is_warm = _build_cb_scores()
warm_v = is_warm[valid_rows]

print("  Building ALS scores...")
als_scores = _cf_scores(als_model)

print("  Building BPR scores...")
bpr_scores = _cf_scores(bpr_model)

# ── Grid search over lambda (hybrid weight) ──────────────────────────
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

hybrid_results = []

def eval_hybrid(cf_sc, cb_sc, lam, name):
    combined = lam * cf_sc[valid_rows] + (1 - lam) * cb_sc[valid_rows]
    ranks    = compute_ranks(combined, true_idx)
    m = {"Modality": name, "lambda": lam}
    m.update(metrics_from_ranks(ranks, warm_v))
    m["time_sec"] = 0.0
    return m


print("\n  ALS Hybrid grid search (lambda = CF weight):")
best_als_h10_w, best_als_lam = -1, 0
for lam in LAMBDAS:
    m = eval_hybrid(als_scores, cb_scores, lam, f"ALS+CB (lam={lam:.1f})")
    hybrid_results.append(m)
    flag = ""
    if m["Hit@10_W"] > best_als_h10_w:
        best_als_h10_w = m["Hit@10_W"]
        best_als_lam   = lam
        best_als_m     = m
        flag = "  <-- best"
    print(f"    lam={lam:.1f}  H@10_W={m['Hit@10_W']:.4f}  MRR_W={m['MRR_W']:.4f}{flag}")

print(f"\n  >> Best ALS+CB: lambda={best_als_lam}  Hit@10_W={best_als_h10_w:.4f}")

print("\n  BPR Hybrid grid search (lambda = CF weight):")
best_bpr_h10_w, best_bpr_lam = -1, 0
for lam in LAMBDAS:
    m = eval_hybrid(bpr_scores, cb_scores, lam, f"BPR+CB (lam={lam:.1f})")
    hybrid_results.append(m)
    flag = ""
    if m["Hit@10_W"] > best_bpr_h10_w:
        best_bpr_h10_w = m["Hit@10_W"]
        best_bpr_lam   = lam
        best_bpr_m     = m
        flag = "  <-- best"
    print(f"    lam={lam:.1f}  H@10_W={m['Hit@10_W']:.4f}  MRR_W={m['MRR_W']:.4f}{flag}")

print(f"\n  >> Best BPR+CB: lambda={best_bpr_lam}  Hit@10_W={best_bpr_h10_w:.4f}")


# =====================================================================
# STEP 7: RESULTS SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("STEP 7: RESULTS SUMMARY")
print("=" * 70)

# Combine all results
all_results = [als_res, bpr_res, best_als_m, best_bpr_m] + hybrid_results
results_df  = pd.DataFrame(all_results).round(4)
results_df.to_csv(os.path.join(OUT_DIR, "evaluation_results_cf.csv"), index=False)
print(f"  Saved evaluation_results_cf.csv  ({len(results_df)} rows)")

# Console table
print(f"\n{'Model':40s} | {'H@5':>6s} {'H@10':>6s} {'H@15':>6s} {'H@20':>6s} | "
      f"{'NDCG@10':>7s} {'MRR':>6s} | {'H@10_W':>7s} {'MRR_W':>6s}")
print("-" * 110)

summary_rows = [
    als_res,
    bpr_res,
    best_als_m,
    best_bpr_m,
]
for row in summary_rows:
    print(f"{row['Modality']:40s} | "
          f"{row.get('Hit@5',0):>6.4f} {row.get('Hit@10',0):>6.4f} "
          f"{row.get('Hit@15',0):>6.4f} {row.get('Hit@20',0):>6.4f} | "
          f"{row.get('NDCG@10',0):>7.4f} {row.get('MRR',0):>6.4f} | "
          f"{row.get('Hit@10_W',0):>7.4f} {row.get('MRR_W',0):>6.4f}")

print("-" * 110)
best_overall = max(summary_rows, key=lambda r: r.get("Hit@10_W", 0))
print(f"\n  Best Hit@10 (Warm): {best_overall['Modality']}  "
      f"= {best_overall['Hit@10_W']:.4f}")

# Baseline reference from content-based pipeline
CB_BASELINE_H10W = 0.1504
print(f"  Content-Based Baseline (Weighted): {CB_BASELINE_H10W:.4f}")
delta = best_overall["Hit@10_W"] - CB_BASELINE_H10W
sign  = "+" if delta >= 0 else ""
print(f"  Delta vs Baseline: {sign}{delta:.4f}  ({sign}{delta/CB_BASELINE_H10W*100:.1f}%)")


# =====================================================================
# STEP 8: GENERATE RECOMMENDATIONS FOR ALL USERS (ALS)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 8: GENERATE TOP-5 RECOMMENDATIONS (ALS, all training users)")
print("=" * 70)

interactions_all = pd.read_json(os.path.join(EDA_DIR, "user_listing_interactions.json"))
interactions_all = interactions_all[interactions_all["listing_id"].isin(valid_lids)]
all_users        = interactions_all["reviewer_id"].unique()
user_hist_all    = interactions_all.groupby("reviewer_id")["listing_id"].apply(list).to_dict()

TOP_K   = 5
user_f, item_f = get_factors(als_model)
i_norm  = item_f / (np.linalg.norm(item_f, axis=1, keepdims=True) + 1e-9)

all_recs = []
BATCH    = 1000
t_start  = time.time()

user_batches = [all_users[i:i+BATCH] for i in range(0, len(all_users), BATCH)]
for b_idx, batch in enumerate(user_batches):
    if b_idx % 20 == 0:
        print(f"    Batch {b_idx+1}/{len(user_batches)}...")

    batch_vecs, batch_uids, batch_hist = [], [], []
    for uid in batch:
        if uid in user_to_idx:
            uv = user_f[user_to_idx[uid]]
        else:
            idxs = [lid_to_idx[l] for l in user_hist_all.get(uid, []) if l in lid_to_idx]
            uv   = item_f[idxs].mean(axis=0) if idxs else item_f.mean(axis=0)
        batch_vecs.append(uv)
        batch_uids.append(uid)
        batch_hist.append(set(user_hist_all.get(uid, [])))

    u_mat  = np.array(batch_vecs)
    u_norm = u_mat / (np.linalg.norm(u_mat, axis=1, keepdims=True) + 1e-9)
    scores = u_norm @ i_norm.T  # (B, N_items)

    for i, uid in enumerate(batch_uids):
        row = scores[i].copy()
        # Mask already-seen items
        for seen_lid in batch_hist[i]:
            if seen_lid in lid_to_idx:
                row[lid_to_idx[seen_lid]] = -np.inf
        top_idxs = np.argpartition(-row, TOP_K)[:TOP_K]
        top_idxs = top_idxs[np.argsort(-row[top_idxs])]
        recs     = [int(listing_ids[idx]) for idx in top_idxs]
        all_recs.append({
            "user_id": uid,
            **{f"rec_{j+1}": recs[j] for j in range(TOP_K)}
        })

print(f"  Inference done in {time.time()-t_start:.1f}s")
recs_df = pd.DataFrame(all_recs)
recs_df.to_csv(os.path.join(OUT_DIR, "recommendations_all_users_als.csv"), index=False)
print(f"  Saved recommendations for {len(recs_df)} users -> recommendations_all_users_als.csv")

print("\n" + "=" * 70)
print("CF PIPELINE COMPLETE!")
print(f"Outputs saved to: {OUT_DIR}")
print("=" * 70)
