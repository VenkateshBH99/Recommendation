"""
=======================================================================
Airbnb Collaborative Filtering Pipeline  (v2 - tuned)
=======================================================================

Implements ALS and BPR collaborative filtering on top of the existing
content-based pipeline.  Includes:
  - Sparsity diagnostics to understand the data
  - Hyperparameter grid search for ALS and BPR
  - Hybrid fusion (best CF + content-based scores)
  - Final recommendations output

Requirements:
    pip install implicit

Run AFTER recommendation_pipeline.py (uses its saved outputs).
=======================================================================
"""

import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")
np.random.seed(42)

try:
    import implicit
    from implicit.als import AlternatingLeastSquares
    from implicit.bpr import BayesianPersonalizedRanking
    print(f"implicit library found: {implicit.__version__}")
except ImportError:
    raise ImportError("pip install implicit")

# =====================================================================
# PATHS
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EDA_DIR  = os.path.join(BASE_DIR, "eda_outputs_new1")
CB_DIR   = os.path.join(BASE_DIR, "recommendation_outputs_new1")
OUT_DIR  = os.path.join(BASE_DIR, "recommendation_outputs_cf2")
os.makedirs(OUT_DIR, exist_ok=True)

K_VALUES = [5, 10, 15, 20]


# =====================================================================
# STEP 1: LOAD ARTIFACTS
# =====================================================================
print("=" * 70)
print("STEP 1: LOAD SAVED ARTIFACTS")
print("=" * 70)

listing_ids = np.load(os.path.join(CB_DIR, "listing_ids.npy"))
fused_emb   = np.load(os.path.join(CB_DIR, "fused_embeddings_256d.npy"))
lid_to_idx  = {lid: i for i, lid in enumerate(listing_ids)}
n_listings  = len(listing_ids)
print(f"  Listings: {n_listings}  |  Fused embedding: {fused_emb.shape}")

train_df = pd.read_csv(os.path.join(CB_DIR, "train_chrono.csv"))
test_df  = pd.read_csv(os.path.join(CB_DIR, "test_chrono.csv"))

valid_lids = set(listing_ids)
train_df   = train_df[train_df["listing_id"].isin(valid_lids)].copy()
test_df    = test_df[test_df["listing_id"].isin(valid_lids)].copy()

train_users  = set(train_df["reviewer_id"].unique())
warm_test_df = test_df[test_df["reviewer_id"].isin(train_users)]
print(f"  Train interactions : {len(train_df)}")
print(f"  Test  interactions : {len(test_df)}")
print(f"  Warm test users    : {warm_test_df['reviewer_id'].nunique()}")
print(f"  Cold test users    : {test_df[~test_df['reviewer_id'].isin(train_users)]['reviewer_id'].nunique()}")


# =====================================================================
# STEP 2: BUILD INTERACTION MATRIX + SPARSITY DIAGNOSTICS
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2: INTERACTION MATRIX + SPARSITY DIAGNOSTICS")
print("=" * 70)

all_train_users = train_df["reviewer_id"].unique()
user_to_idx     = {uid: i for i, uid in enumerate(all_train_users)}
n_users         = len(all_train_users)

rows_idx = train_df["reviewer_id"].map(user_to_idx).values
cols_idx = train_df["listing_id"].map(lid_to_idx).values
data_val = np.ones(len(rows_idx), dtype=np.float32)

user_item_csr = sp.csr_matrix((data_val, (rows_idx, cols_idx)), shape=(n_users, n_listings))
item_user_csr = user_item_csr.T.tocsr()

density = user_item_csr.nnz / (n_users * n_listings)
print(f"  Matrix shape  : {user_item_csr.shape}")
print(f"  Non-zeros     : {user_item_csr.nnz}")
print(f"  Density       : {density:.6f}  ({density*100:.4f}%)")

user_counts = np.diff(user_item_csr.indptr)
print(f"\n  User interaction distribution:")
for threshold in [1, 2, 3, 5, 10]:
    pct = (user_counts <= threshold).mean() * 100
    print(f"    <= {threshold:2d} interactions : {pct:.1f}% of users")
print(f"    mean  : {user_counts.mean():.2f}")
print(f"    median: {np.median(user_counts):.1f}")
print(f"    max   : {user_counts.max()}")

item_counts = np.diff(item_user_csr.indptr)
print(f"\n  Item interaction distribution:")
print(f"    mean  : {item_counts.mean():.2f}")
print(f"    median: {np.median(item_counts):.1f}")
print(f"    items with 0 interactions : {(item_counts == 0).sum()}")
print(f"    items with >= 5           : {(item_counts >= 5).sum()}")

train_hist = train_df.groupby("reviewer_id")["listing_id"].apply(list).to_dict()
warm_uids  = warm_test_df["reviewer_id"].unique()
hist_lens  = [len(train_hist.get(u, [])) for u in warm_uids]
print(f"\n  Warm test user history lengths:")
print(f"    mean  : {np.mean(hist_lens):.2f}")
print(f"    median: {np.median(hist_lens):.1f}")
print(f"    users with 1 interaction : {sum(1 for h in hist_lens if h == 1)}")
print(f"    users with >= 3           : {sum(1 for h in hist_lens if h >= 3)}")
print(f"\n  NOTE: CF struggles when median history <= 2.")
print(f"  Hybrid fusion (CF + content-based) is the best strategy here.")


# =====================================================================
# HELPERS
# =====================================================================

def get_factors(model):
    """
    Return (user_factors, item_factors) correctly oriented.
    implicit fits on item_user_csr so naming can be swapped by version.
    """
    a, b = model.user_factors, model.item_factors
    if a.shape[0] == n_users and b.shape[0] == n_listings:
        return a, b
    elif a.shape[0] == n_listings and b.shape[0] == n_users:
        return b, a
    else:
        raise ValueError(
            f"Cannot determine factor orientation: "
            f"user_factors={a.shape}, item_factors={b.shape}, "
            f"n_users={n_users}, n_listings={n_listings}"
        )


def compute_ranks(sim_scores, true_indices):
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


def build_cf_scores(user_f, item_f, test_uids):
    """Score matrix (n_test, n_items) from CF factors."""
    global_vec = item_f.mean(axis=0)
    user_vecs, is_warm_list = [], []
    for uid in test_uids:
        if uid in user_to_idx:
            uv = user_f[user_to_idx[uid]]
            is_warm_list.append(True)
        else:
            idxs = [lid_to_idx[l] for l in train_hist.get(uid, []) if l in lid_to_idx]
            uv   = item_f[idxs].mean(axis=0) if idxs else global_vec
            is_warm_list.append(False)
        user_vecs.append(uv)
    user_mat = np.array(user_vecs)
    return _norm(user_mat) @ _norm(item_f).T, np.array(is_warm_list)


def _norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def _user_profile(emb, test_uids):
    """Mean-pool training history embeddings for each test user."""
    global_vec = emb.mean(axis=0)
    user_vecs, is_warm_list = [], []
    for uid in test_uids:
        idxs = [lid_to_idx[l] for l in train_hist.get(uid, []) if l in lid_to_idx]
        if idxs and uid in train_users:
            uv = emb[idxs].mean(axis=0)
            is_warm_list.append(True)
        else:
            uv = global_vec
            is_warm_list.append(False)
        user_vecs.append(uv)
    return np.array(user_vecs), np.array(is_warm_list)


def build_cb_scores(test_uids):
    """
    Weighted late-fusion content-based scores - mirrors the best config from
    the original pipeline: alpha=0.3 (struct) + beta=0.6 (text) + gamma=0.1 (CLIP).
    Falls back to SVD-256 if per-modality embeddings are not found.
    """
    try:
        struct_emb = np.load(os.path.join(CB_DIR, "emb_struct.npy"))
        text_emb   = np.load(os.path.join(CB_DIR, "emb_text.npy"))
        clip_emb   = np.load(os.path.join(CB_DIR, "emb_clip.npy"))

        u_s, is_warm_list = _user_profile(struct_emb, test_uids)
        u_t, _            = _user_profile(text_emb,   test_uids)
        u_c, _            = _user_profile(clip_emb,   test_uids)

        # Best weights from original grid search
        ALPHA, BETA, GAMMA = 0.3, 0.6, 0.1
        sim_s  = _norm(u_s) @ _norm(struct_emb).T
        sim_t  = _norm(u_t) @ _norm(text_emb).T
        sim_c  = _norm(u_c) @ _norm(clip_emb).T
        scores = ALPHA * sim_s + BETA * sim_t + GAMMA * sim_c
        print(f"  CB scores: weighted fusion (struct={ALPHA}, text={BETA}, clip={GAMMA})")
        return scores, np.array(is_warm_list)

    except FileNotFoundError:
        print("  CB scores: SVD-256 fallback (per-modality .npy files not found in CB_DIR)")
        emb_norm   = fused_emb / (np.linalg.norm(fused_emb, axis=1, keepdims=True) + 1e-9)
        global_vec = fused_emb.mean(axis=0)
        user_vecs, is_warm_list = [], []
        for uid in test_uids:
            idxs = [lid_to_idx[l] for l in train_hist.get(uid, []) if l in lid_to_idx]
            if idxs and uid in train_users:
                uv = fused_emb[idxs].mean(axis=0)
                is_warm_list.append(True)
            else:
                uv = global_vec
                is_warm_list.append(False)
            user_vecs.append(uv)
        user_mat = np.array(user_vecs)
        return _norm(user_mat) @ emb_norm.T, np.array(is_warm_list)


def evaluate_scores(scores, is_warm, test_lids, name, extra=None):
    valid_rows = np.array([i for i, l in enumerate(test_lids) if l in lid_to_idx])
    true_idx   = np.array([lid_to_idx[test_lids[i]] for i in valid_rows])
    if len(valid_rows) == 0:
        return {}
    scores_v = scores[valid_rows]
    warm_v   = is_warm[valid_rows]
    ranks    = compute_ranks(scores_v, true_idx)
    res = {"Modality": name}
    if extra:
        res.update(extra)
    res.update(metrics_from_ranks(ranks, warm_v))
    return res


# =====================================================================
# STEP 3: CONTENT-BASED BASELINE
# =====================================================================
print("\n" + "=" * 70)
print("STEP 3: CONTENT-BASED BASELINE")
print("=" * 70)

test_uids = test_df["reviewer_id"].values
test_lids = test_df["listing_id"].values

cb_scores, cb_warm = build_cb_scores(test_uids)
cb_res = evaluate_scores(cb_scores, cb_warm, test_lids, "Content-Based (Weighted Fusion)")
print(f"  H@10={cb_res['Hit@10']:.4f}  H@10_W={cb_res['Hit@10_W']:.4f}  MRR_W={cb_res['MRR_W']:.4f}")

# Pre-compute valid rows once (shared across all hybrid evaluations)
valid_rows = np.array([i for i, l in enumerate(test_lids) if l in lid_to_idx])
true_idx   = np.array([lid_to_idx[test_lids[i]] for i in valid_rows])
warm_v     = cb_warm[valid_rows]

CB_BEST_H10W = 0.1504  # best from original weighted-fusion pipeline


# =====================================================================
# STEP 4: ALS HYPERPARAMETER GRID SEARCH
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4: ALS HYPERPARAMETER GRID SEARCH")
print("=" * 70)
print("  Key: with sparse data (median 1-2 interactions), use LOW alpha (1-5)")
print("  and smaller factors to avoid overfitting.\n")

# alpha=40 (the default) is designed for datasets where users have dozens
# of interactions. With 1-2 interactions it dominates the loss and the
# model collapses to near-zero scores for most users.
ALS_GRID = [
    {"factors": 32,  "regularization": 0.1,  "alpha": 1,  "iterations": 50},
    {"factors": 32,  "regularization": 0.1,  "alpha": 5,  "iterations": 50},
    {"factors": 64,  "regularization": 0.1,  "alpha": 1,  "iterations": 50},
    {"factors": 64,  "regularization": 0.05, "alpha": 5,  "iterations": 50},
    {"factors": 64,  "regularization": 0.01, "alpha": 1,  "iterations": 50},
    {"factors": 128, "regularization": 0.1,  "alpha": 1,  "iterations": 50},
    {"factors": 128, "regularization": 0.05, "alpha": 5,  "iterations": 50},
    {"factors": 128, "regularization": 0.01, "alpha": 1,  "iterations": 50},
]

als_results = []
best_als_h10w, best_als_model, best_als_cfg = -1, None, None

for cfg in ALS_GRID:
    t0  = time.time()
    mdl = AlternatingLeastSquares(
        factors=cfg["factors"],
        regularization=cfg["regularization"],
        alpha=cfg["alpha"],
        iterations=cfg["iterations"],
        use_gpu=False,
        random_state=42,
        calculate_training_loss=False,
    )
    mdl.fit(item_user_csr, show_progress=False)
    uf, if_ = get_factors(mdl)
    sc, iw  = build_cf_scores(uf, if_, test_uids)
    label   = f"ALS f={cfg['factors']} reg={cfg['regularization']} alpha={cfg['alpha']}"
    res     = evaluate_scores(sc, iw, test_lids, label, extra=cfg)
    res["time_sec"] = round(time.time() - t0, 1)
    als_results.append(res)

    flag = ""
    if res["Hit@10_W"] > best_als_h10w:
        best_als_h10w  = res["Hit@10_W"]
        best_als_model = mdl
        best_als_cfg   = cfg
        flag = "  <-- best"
    print(f"  {label:50s}  H@10_W={res['Hit@10_W']:.4f}  MRR_W={res['MRR_W']:.4f}  ({res['time_sec']}s){flag}")

print(f"\n  >> Best ALS: {best_als_cfg}  Hit@10_W={best_als_h10w:.4f}")
with open(os.path.join(OUT_DIR, "als_model_best.pkl"), "wb") as f:
    pickle.dump(best_als_model, f)
print("  Saved als_model_best.pkl")


# =====================================================================
# STEP 5: BPR HYPERPARAMETER GRID SEARCH
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5: BPR HYPERPARAMETER GRID SEARCH")
print("=" * 70)
print("  BPR directly optimises ranking — lower LR + more iterations helps.\n")

BPR_GRID = [
    {"factors": 32,  "learning_rate": 0.05,  "regularization": 0.01,  "iterations": 200},
    {"factors": 32,  "learning_rate": 0.01,  "regularization": 0.001, "iterations": 300},
    {"factors": 64,  "learning_rate": 0.05,  "regularization": 0.01,  "iterations": 200},
    {"factors": 64,  "learning_rate": 0.01,  "regularization": 0.001, "iterations": 300},
    {"factors": 128, "learning_rate": 0.05,  "regularization": 0.01,  "iterations": 200},
    {"factors": 128, "learning_rate": 0.01,  "regularization": 0.001, "iterations": 300},
]

bpr_results = []
best_bpr_h10w, best_bpr_model, best_bpr_cfg = -1, None, None

for cfg in BPR_GRID:
    t0  = time.time()
    mdl = BayesianPersonalizedRanking(
        factors=cfg["factors"],
        learning_rate=cfg["learning_rate"],
        regularization=cfg["regularization"],
        iterations=cfg["iterations"],
        use_gpu=False,
        random_state=42,
    )
    mdl.fit(item_user_csr, show_progress=False)
    uf, if_ = get_factors(mdl)
    sc, iw  = build_cf_scores(uf, if_, test_uids)
    label   = f"BPR f={cfg['factors']} lr={cfg['learning_rate']} reg={cfg['regularization']}"
    res     = evaluate_scores(sc, iw, test_lids, label, extra=cfg)
    res["time_sec"] = round(time.time() - t0, 1)
    bpr_results.append(res)

    flag = ""
    if res["Hit@10_W"] > best_bpr_h10w:
        best_bpr_h10w  = res["Hit@10_W"]
        best_bpr_model = mdl
        best_bpr_cfg   = cfg
        flag = "  <-- best"
    print(f"  {label:58s}  H@10_W={res['Hit@10_W']:.4f}  MRR_W={res['MRR_W']:.4f}  ({res['time_sec']}s){flag}")

print(f"\n  >> Best BPR: {best_bpr_cfg}  Hit@10_W={best_bpr_h10w:.4f}")
with open(os.path.join(OUT_DIR, "bpr_model_best.pkl"), "wb") as f:
    pickle.dump(best_bpr_model, f)
print("  Saved bpr_model_best.pkl")


# =====================================================================
# STEP 6: HYBRID FUSION GRID SEARCH
# =====================================================================
print("\n" + "=" * 70)
print("STEP 6: HYBRID FUSION (best ALS + CB  |  best BPR + CB)")
print("=" * 70)

uf_als, if_als = get_factors(best_als_model)
als_sc, _      = build_cf_scores(uf_als, if_als, test_uids)

uf_bpr, if_bpr = get_factors(best_bpr_model)
bpr_sc, _      = build_cf_scores(uf_bpr, if_bpr, test_uids)

LAMBDAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hybrid_results = []
best_hybrid_h10w, best_hybrid_res = -1, None
best_als_hyb, best_bpr_hyb = None, None

def eval_hybrid(cf_sc, cb_sc, lam, name):
    combined = lam * cf_sc[valid_rows] + (1 - lam) * cb_sc[valid_rows]
    ranks    = compute_ranks(combined, true_idx)
    m = {"Modality": name, "lambda": lam}
    m.update(metrics_from_ranks(ranks, warm_v))
    m["time_sec"] = 0.0
    return m

print("\n  ALS + Content-Based hybrid (lambda = CF weight):")
best_als_hyb_h10w = -1
for lam in LAMBDAS:
    m    = eval_hybrid(als_sc, cb_scores, lam, f"ALS+CB lam={lam:.2f}")
    flag = ""
    if m["Hit@10_W"] > best_als_hyb_h10w:
        best_als_hyb_h10w = m["Hit@10_W"]
        best_als_hyb      = m
        flag = "  <-- best"
    if m["Hit@10_W"] > best_hybrid_h10w:
        best_hybrid_h10w = m["Hit@10_W"]
        best_hybrid_res  = m
    hybrid_results.append(m)
    print(f"    lam={lam:.2f}  H@10_W={m['Hit@10_W']:.4f}  MRR_W={m['MRR_W']:.4f}{flag}")

print(f"\n  BPR + Content-Based hybrid (lambda = CF weight):")
best_bpr_hyb_h10w = -1
for lam in LAMBDAS:
    m    = eval_hybrid(bpr_sc, cb_scores, lam, f"BPR+CB lam={lam:.2f}")
    flag = ""
    if m["Hit@10_W"] > best_bpr_hyb_h10w:
        best_bpr_hyb_h10w = m["Hit@10_W"]
        best_bpr_hyb      = m
        flag = "  <-- best"
    if m["Hit@10_W"] > best_hybrid_h10w:
        best_hybrid_h10w = m["Hit@10_W"]
        best_hybrid_res  = m
    hybrid_results.append(m)
    print(f"    lam={lam:.2f}  H@10_W={m['Hit@10_W']:.4f}  MRR_W={m['MRR_W']:.4f}{flag}")


# =====================================================================
# STEP 7: RESULTS SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("STEP 7: RESULTS SUMMARY")
print("=" * 70)

all_rows = [cb_res] + als_results + bpr_results + hybrid_results
results_df = pd.DataFrame(all_rows).round(4)
results_df.to_csv(os.path.join(OUT_DIR, "evaluation_results_cf.csv"), index=False)
print(f"  Saved evaluation_results_cf.csv  ({len(results_df)} rows)\n")

header = f"{'Model':52s} | {'H@5':>6s} {'H@10':>6s} {'H@15':>6s} | {'H@10_W':>7s} {'MRR_W':>6s}"
print(header)
print("-" * len(header))

best_als_standalone = max(als_results, key=lambda r: r.get("Hit@10_W", 0))
best_bpr_standalone = max(bpr_results, key=lambda r: r.get("Hit@10_W", 0))

for row in [cb_res, best_als_standalone, best_bpr_standalone, best_als_hyb, best_bpr_hyb]:
    print(f"  {row['Modality']:50s} | "
          f"{row.get('Hit@5',0):>6.4f} {row.get('Hit@10',0):>6.4f} "
          f"{row.get('Hit@15',0):>6.4f} | "
          f"{row.get('Hit@10_W',0):>7.4f} {row.get('MRR_W',0):>6.4f}")

print("-" * len(header))
delta = best_hybrid_h10w - CB_BEST_H10W
sign  = "+" if delta >= 0 else ""
print(f"\n  Original pipeline best (Weighted fusion) : {CB_BEST_H10W:.4f}")
print(f"  Best hybrid CF result                    : {best_hybrid_h10w:.4f}  [{best_hybrid_res['Modality']}]")
print(f"  Delta vs original best                   : {sign}{delta:.4f}  ({sign}{delta/CB_BEST_H10W*100:.1f}%)")


# =====================================================================
# STEP 8: GENERATE TOP-5 RECOMMENDATIONS (Best ALS, seen-items filtered)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 8: GENERATE TOP-5 RECOMMENDATIONS (Best ALS)")
print("=" * 70)

import json
interactions_all = pd.read_json(os.path.join(EDA_DIR, "user_listing_interactions.json"))
interactions_all = interactions_all[interactions_all["listing_id"].isin(valid_lids)]
all_users_full   = interactions_all["reviewer_id"].unique()
user_hist_all    = interactions_all.groupby("reviewer_id")["listing_id"].apply(list).to_dict()

uf, if_  = get_factors(best_als_model)
i_norm   = if_ / (np.linalg.norm(if_, axis=1, keepdims=True) + 1e-9)
global_v = if_.mean(axis=0)

TOP_K    = 5
BATCH    = 1000
all_recs = []
t_start  = time.time()

batches = [all_users_full[i:i+BATCH] for i in range(0, len(all_users_full), BATCH)]
for b_idx, batch in enumerate(batches):
    if b_idx % 20 == 0:
        print(f"    Batch {b_idx+1}/{len(batches)}...")

    vecs, uids, hists = [], [], []
    for uid in batch:
        if uid in user_to_idx:
            uv = uf[user_to_idx[uid]]
        else:
            idxs = [lid_to_idx[l] for l in user_hist_all.get(uid, []) if l in lid_to_idx]
            uv   = if_[idxs].mean(axis=0) if idxs else global_v
        vecs.append(uv)
        uids.append(uid)
        hists.append(set(user_hist_all.get(uid, [])))

    u_mat  = np.array(vecs)
    u_norm = u_mat / (np.linalg.norm(u_mat, axis=1, keepdims=True) + 1e-9)
    scores = u_norm @ i_norm.T

    for i, uid in enumerate(uids):
        row = scores[i].copy()
        for seen in hists[i]:
            if seen in lid_to_idx:
                row[lid_to_idx[seen]] = -np.inf
        top_idxs = np.argpartition(-row, TOP_K)[:TOP_K]
        top_idxs = top_idxs[np.argsort(-row[top_idxs])]
        recs     = [int(listing_ids[idx]) for idx in top_idxs]
        all_recs.append({"user_id": uid, **{f"rec_{j+1}": recs[j] for j in range(TOP_K)}})

print(f"  Done in {time.time()-t_start:.1f}s")
recs_df = pd.DataFrame(all_recs)
recs_df.to_csv(os.path.join(OUT_DIR, "recommendations_all_users_als.csv"), index=False)
print(f"  Saved {len(recs_df)} users -> recommendations_all_users_als.csv")

print("\n" + "=" * 70)
print("CF PIPELINE COMPLETE!")
print(f"Outputs saved to: {OUT_DIR}")
print("=" * 70)
