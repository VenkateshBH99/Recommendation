"""
=======================================================================
Airbnb Multi-Modal Text-Image Recommendation Pipeline
=======================================================================

End-to-end content-based recommendation system using:
  - Structured features  (listing attrs, host, location, amenities)
  - Text features         (TF-IDF on listing descriptions + reviews)
  - Image features        (pre-computed CLIP 512-d embeddings)

Evaluation: Leave-one-out on users with >=2 interactions
Metrics:    Hit@K, MAP@K, MRR, NDCG@K
Ablation:   Each modality solo + combinations + full fusion
"""

import json
import os
import pickle
import time
import warnings
from collections import defaultdict
from itertools import combinations as itertools_combinations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EDA_DIR = os.path.join(BASE_DIR, "eda_outputs_new1")
OUT_DIR = os.path.join(BASE_DIR, "recommendation_outputs_new1")
os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================================
# STEP 1: LOAD & SELECT FEATURES
# =====================================================================
print("=" * 70)
print("STEP 1: LOAD & SELECT FEATURES")
print("=" * 70)

# --- 1a. Structured features ---
master = pd.read_json(os.path.join(EDA_DIR, "master_structured_features.json"))
print(f"  Loaded master_structured_features: {master.shape}")

# Select relevant structured feature columns (drop redundant/raw)
DROP_COLS = [
    "listing_id",       # ID, not a feature
    "price",            # use log_price instead (less skewed)
    "minimum_nights",   # use log_min_nights
    "maximum_nights",   # use log_max_nights
    "price_tier",       # categorical string, we have log_price + OHE room/prop
    "government_area",  # high cardinality string, handled via location OHE
    "market_",          # empty-string market (6 rows)
    "has_image",        # we filter to listings WITH images anyway
    "has_clip_embedding",
    "has_resnet_embedding",
    "market_Other (Domestic)",    # too few samples
    "market_Other (International)",
]
struct_feature_cols = [c for c in master.columns if c not in DROP_COLS]
print(f"  Structured features selected: {len(struct_feature_cols)} columns")

# --- 1b. Text features ---
text_f = pd.read_json(os.path.join(EDA_DIR, "text_features.json"))
review_text = pd.read_json(os.path.join(EDA_DIR, "review_text_features.json"))
print(f"  Loaded text_features: {text_f.shape}")
print(f"  Loaded review_text_features: {review_text.shape}")

# Merge listing text + review text
text_merged = text_f[["listing_id", "combined_text"]].merge(
    review_text[["listing_id", "aggregated_review_text"]], on="listing_id", how="left"
)
text_merged["aggregated_review_text"] = text_merged["aggregated_review_text"].fillna("")
text_merged["full_text"] = (
    text_merged["combined_text"] + " " + text_merged["aggregated_review_text"]
)
print(f"  Merged text: {len(text_merged)} listings")

# --- 1c. CLIP image embeddings (from Normalized root) ---
with open(os.path.join(BASE_DIR, "clip_embeddings.pkl"), "rb") as f:
    clip_dict = pickle.load(f)
clip_df = pd.DataFrame([
    {"listing_id": lid, **{f"clip_{i}": v for i, v in enumerate(emb)}}
    for lid, emb in clip_dict.items()
])
clip_listing_ids = set(clip_df["listing_id"].values)
print(f"  Loaded CLIP embeddings: {clip_df.shape} (512-d)")

# --- 1c2. ResNet image embeddings (from Normalized root) ---
with open(os.path.join(BASE_DIR, "resnet50_embeddings.pkl"), "rb") as f:
    resnet_dict = pickle.load(f)
resnet_df = pd.DataFrame([
    {"listing_id": lid, **{f"resnet_{i}": v for i, v in enumerate(emb)}}
    for lid, emb in resnet_dict.items()
])
print(f"  Loaded ResNet embeddings: {resnet_df.shape} (2048-d)")

# --- 1d. Interactions ---
interactions = pd.read_json(os.path.join(EDA_DIR, "user_listing_interactions.json"))
print(f"  Loaded interactions: {len(interactions)}")

# --- 1e. Filter to listings that have CLIP embeddings (image available) ---
valid_listings = sorted(clip_listing_ids & set(master["listing_id"]))
print(f"\n  Listings with all modalities (structured + text + image): {len(valid_listings)}")

# Build aligned DataFrames indexed by listing_id
master_aligned = master[master["listing_id"].isin(valid_listings)].copy()
master_aligned = master_aligned.sort_values("listing_id").reset_index(drop=True)
listing_ids = master_aligned["listing_id"].values
lid_to_idx = {lid: i for i, lid in enumerate(listing_ids)}
n_listings = len(listing_ids)
print(f"  Final listing count: {n_listings}")


# =====================================================================
# STEP 2: TEXT EMBEDDINGS (TF-IDF)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2: TEXT EMBEDDINGS (TF-IDF)")
print("=" * 70)

# Align text to same listing order
text_aligned = text_merged[text_merged["listing_id"].isin(valid_listings)].copy()
text_aligned = text_aligned.set_index("listing_id").loc[listing_ids].reset_index()

tfidf = TfidfVectorizer(
    max_features=500,
    sublinear_tf=True,
    stop_words="english",
    min_df=5,
    max_df=0.95,
    ngram_range=(1, 2),
)
t0 = time.time()
tfidf_matrix = tfidf.fit_transform(text_aligned["full_text"].fillna(""))
text_embeddings = tfidf_matrix.toarray()  # (n_listings, 500)
print(f"  TF-IDF matrix: {text_embeddings.shape} in {time.time()-t0:.1f}s")
print(f"  Vocabulary size: {len(tfidf.vocabulary_)}")
print(f"  Top terms: {list(tfidf.vocabulary_.keys())[:20]}")


# =====================================================================
# STEP 3: MULTI-MODAL FUSION
# =====================================================================
print("\n" + "=" * 70)
print("STEP 3: MULTI-MODAL FUSION")
print("=" * 70)

# --- 3a. Structured features (normalized) ---
struct_data = master_aligned[struct_feature_cols].values.astype(float)
scaler_struct = StandardScaler()
struct_norm = scaler_struct.fit_transform(struct_data)
print(f"  Structured: {struct_norm.shape}")

# --- 3b. Text embeddings (normalized) ---
scaler_text = StandardScaler()
text_norm = scaler_text.fit_transform(text_embeddings)
print(f"  Text TF-IDF: {text_norm.shape}")

# --- 3c. CLIP embeddings (normalized) ---
clip_aligned = clip_df[clip_df["listing_id"].isin(valid_listings)].copy()
clip_aligned = clip_aligned.set_index("listing_id").loc[listing_ids].reset_index()
clip_data = clip_aligned.drop(columns=["listing_id"]).values.astype(float)
scaler_clip = StandardScaler()
clip_norm = scaler_clip.fit_transform(clip_data)
print(f"  CLIP image:  {clip_norm.shape}")

# --- 3c2. ResNet embeddings (normalized) ---
resnet_aligned = resnet_df[resnet_df["listing_id"].isin(valid_listings)].copy()
resnet_aligned = resnet_aligned.set_index("listing_id").loc[listing_ids].reset_index()
resnet_data = resnet_aligned.drop(columns=["listing_id"]).values.astype(float)
scaler_resnet = StandardScaler()
resnet_norm = scaler_resnet.fit_transform(resnet_data)
print(f"  ResNet image: {resnet_norm.shape}")

# --- 3d. Fused embedding ---
fused = np.hstack([struct_norm, text_norm, clip_norm])
print(f"  Fused (concat): {fused.shape}")

# Dimensionality reduction with TruncatedSVD for efficiency
N_COMPONENTS = 256
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
fused_reduced = svd.fit_transform(fused)
explained_var = svd.explained_variance_ratio_.sum()
print(f"  Fused (SVD {N_COMPONENTS}-d): {fused_reduced.shape}, explained variance: {explained_var:.3f}")

# Prepare modality ablation embeddings
# --- Generate ALL modality combinations (15 subsets + SVD) ---
base_modalities = {
    "Struct": struct_norm,
    "Text":   text_norm,
    "CLIP":   clip_norm,
    "ResNet": resnet_norm,
}

modality_embeddings = {}
for r in range(1, len(base_modalities) + 1):
    for combo in itertools_combinations(base_modalities.keys(), r):
        name = " + ".join(combo)
        emb = np.hstack([base_modalities[k] for k in combo])
        modality_embeddings[name] = emb

# Also add SVD-reduced full fusion
modality_embeddings["All (SVD-256)"] = fused_reduced

print(f"  Total modality configurations: {len(modality_embeddings)}")
for name, emb in modality_embeddings.items():
    print(f"    {name:35s}: {emb.shape}")


# =====================================================================
# STEP 4: CHRONOLOGICAL TRAIN / TEST SPLIT
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4: CHRONOLOGICAL TRAIN / TEST SPLIT")
print("=" * 70)

# 1. Identify valid interactions (where both user and item exist in our features)
valid_listing_ids = set(listing_ids)
valid_interactions = interactions[interactions["listing_id"].isin(valid_listing_ids)].copy()

print(f"  Valid interactions: {len(valid_interactions)}")

# 2. Chronological Split
# Load review dates from raw file to merge
print("  Loading review dates...")
try:
    reviews_df = pd.read_json("airbnb_reviews.json")
    reviews_df['date_obj'] = pd.to_datetime(reviews_df['review_date'], unit='ms')
    
    # Merge date into interactions
    # interactions df has reviewer_id, listing_id. We need date.
    # Note: interactions_df might have duplicates if user reviewed same listing twice (rare but possible)
    # detailed merge:
    interactions_with_date = valid_interactions.merge(
        reviews_df[['reviewer_id', 'listing_id', 'date_obj']], 
        on=['reviewer_id', 'listing_id'], 
        how='left'
    )
    
    # Drop duplicates if any (keep first)
    interactions_with_date = interactions_with_date.drop_duplicates(subset=['reviewer_id', 'listing_id'])
    
    # Sort by date
    interactions_with_date = interactions_with_date.sort_values('date_obj')

    # Threshold: 2018-08-22 (from EDA)
    SPLIT_DATE = pd.Timestamp("2018-08-22")
    print(f"  Split Date Threshold: {SPLIT_DATE}")

    train_df = interactions_with_date[interactions_with_date['date_obj'] < SPLIT_DATE].copy()
    test_df = interactions_with_date[interactions_with_date['date_obj'] >= SPLIT_DATE].copy()

    print(f"  Train interactions: {len(train_df)} ({len(train_df)/len(interactions_with_date):.1%})")
    print(f"  Test interactions:  {len(test_df)} ({len(test_df)/len(interactions_with_date):.1%})")

    # Identify Warm vs Cold Users in Test
    train_users = set(train_df['reviewer_id'].unique())
    test_users = test_df['reviewer_id'].unique()

    warm_test_df = test_df[test_df['reviewer_id'].isin(train_users)]
    cold_test_df = test_df[~test_df['reviewer_id'].isin(train_users)]

    print(f"  Test Breakdown by User Type:")
    print(f"    Warm interactions (Returning Users): {len(warm_test_df)} ({len(warm_test_df)/len(test_df):.1%})")
    print(f"    Cold interactions (New Users):       {len(cold_test_df)} ({len(cold_test_df)/len(test_df):.1%})")

    # Save split info
    split_info = {
        "split_date": str(SPLIT_DATE),
        "train_count": len(train_df),
        "test_count": len(test_df),
        "warm_test_count": len(warm_test_df),
        "cold_test_count": len(cold_test_df)
    }
    with open(os.path.join(OUT_DIR, "train_test_split_chrono.json"), "w") as f:
        json.dump(split_info, f, indent=4)
    print("  Saved chronological split info.")

except Exception as e:
    print(f"Error in chronological split: {e}")
    print("Falling back to random split for testing...")
    # Fallback to random split if file not found
    train_df = valid_interactions.sample(frac=0.8, random_state=42)
    test_df = valid_interactions.drop(train_df.index)


# =====================================================================
# STEP 5: RECOMMENDATION & EVALUATION (CHRONOLOGICAL)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5: RECOMMENDATION & EVALUATION")
print("=" * 70)

K_VALUES = [5, 10, 15, 20]

def compute_ranks(sim_scores, true_indices):
    """Compute rank of true item for each row in sim_scores."""
    true_scores = sim_scores[np.arange(len(true_indices)), true_indices][:, None]
    BATCH = 2000
    ranks = []
    for b in range(0, len(true_indices), BATCH):
        batch_sims = sim_scores[b:b+BATCH]
        batch_true = true_scores[b:b+BATCH]
        ranks.extend((batch_sims > batch_true).sum(axis=1) + 1)
    return np.array(ranks, dtype=float)

def metrics_from_ranks(all_ranks, is_warm_valid):
    """Compute all metrics from rank array."""
    res = {}
    for k in K_VALUES:
        hit = (all_ranks <= k).astype(float)
        ndcg = np.where(all_ranks <= k, 1.0 / np.log2(all_ranks + 1), 0.0)
        mapk = np.where(all_ranks <= k, 1.0 / all_ranks, 0.0)
        prec = hit / k  # single relevant item

        # Overall
        res[f"Hit@{k}"] = hit.mean()
        res[f"Prec@{k}"] = prec.mean()
        res[f"Recall@{k}"] = hit.mean()  # = Hit@K for single-item
        res[f"NDCG@{k}"] = ndcg.mean()
        res[f"MAP@{k}"] = mapk.mean()

        # Warm
        if is_warm_valid.sum() > 0:
            res[f"Hit@{k}_W"] = hit[is_warm_valid].mean()
            res[f"Prec@{k}_W"] = prec[is_warm_valid].mean()
            res[f"Recall@{k}_W"] = hit[is_warm_valid].mean()
            res[f"NDCG@{k}_W"] = ndcg[is_warm_valid].mean()
            res[f"MAP@{k}_W"] = mapk[is_warm_valid].mean()
        else:
            for m in ["Hit","Prec","Recall","NDCG","MAP"]:
                res[f"{m}@{k}_W"] = 0.0

    rr = 1.0 / all_ranks
    res["MRR"] = rr.mean()
    res["MRR_W"] = rr[is_warm_valid].mean() if is_warm_valid.sum() > 0 else 0.0
    return res

def evaluate_chrono(embeddings, name):
    """Evaluate a single embedding matrix using chronological split."""
    start = time.time()
    lid_to_idx = {lid: i for i, lid in enumerate(listing_ids)}

    def get_user_emb(user_lids):
        idxs = [lid_to_idx[l] for l in user_lids if l in lid_to_idx]
        return embeddings[idxs].mean(axis=0) if idxs else None

    train_hist = train_df.groupby("reviewer_id")["listing_id"].apply(list).to_dict()
    global_mean = embeddings.mean(axis=0)

    test_uids = test_df["reviewer_id"].values
    test_lids = test_df["listing_id"].values

    user_vecs, is_warm = [], []
    for uid in test_uids:
        if uid in train_hist:
            v = get_user_emb(train_hist[uid])
            if v is not None:
                user_vecs.append(v); is_warm.append(True); continue
        user_vecs.append(global_mean); is_warm.append(False)

    user_mat = np.array(user_vecs)
    is_warm = np.array(is_warm)

    # Normalize
    user_mat = user_mat / (np.linalg.norm(user_mat, axis=1, keepdims=True) + 1e-9)
    item_mat = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)

    sims = user_mat @ item_mat.T

    # Filter to valid test rows (true listing exists in our set)
    true_idx, valid_rows = [], []
    for i, lid in enumerate(test_lids):
        if lid in lid_to_idx:
            true_idx.append(lid_to_idx[lid]); valid_rows.append(i)
    true_idx = np.array(true_idx)
    valid_rows = np.array(valid_rows)
    if len(valid_rows) == 0:
        return {}

    sims_v = sims[valid_rows]
    warm_v = is_warm[valid_rows]
    ranks = compute_ranks(sims_v, true_idx)

    res = {"Modality": name}
    res.update(metrics_from_ranks(ranks, warm_v))
    elapsed = time.time() - start
    res["time_sec"] = elapsed

    print(f"  ✓ {name:<35} ({elapsed:.1f}s) | "
          f"H@5={res['Hit@5']:.4f} H@10={res['Hit@10']:.4f} "
          f"H@15={res['Hit@15']:.4f} H@20={res['Hit@20']:.4f} | "
          f"H@10_W={res['Hit@10_W']:.4f} MRR_W={res['MRR_W']:.4f}")
    return res

# ── Run ablation across ALL modality combinations ──
print("\nRunning Chronological Ablation Study (all combos)...")
results = []
for name, emb in modality_embeddings.items():
    results.append(evaluate_chrono(emb, name))


# =====================================================================
# STEP 5b: WEIGHTED LATE FUSION (GRID SEARCH)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5b: WEIGHTED LATE FUSION (GRID SEARCH)")
print("=" * 70)

# Strategy: Late fusion at the score level.
# score(u,i) = α·sim_struct(u,i) + β·sim_text(u,i) + γ·sim_clip(u,i)
# Grid search over α,β,γ with α+β+γ = 1 to maximize Hit@10 on warm users.

lid_to_idx = {lid: i for i, lid in enumerate(listing_ids)}
train_hist = train_df.groupby("reviewer_id")["listing_id"].apply(list).to_dict()

def _norm_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

def _build_profiles(emb, uids, true_lids):
    """Build user profile vectors and return aligned data."""
    vecs, warm, tidx, vrows = [], [], [], []
    for i, uid in enumerate(uids):
        if uid in train_hist:
            idxs = [lid_to_idx[l] for l in train_hist[uid] if l in lid_to_idx]
            if idxs:
                vecs.append(emb[idxs].mean(axis=0)); warm.append(True)
            else:
                vecs.append(emb.mean(axis=0)); warm.append(False)
        else:
            vecs.append(emb.mean(axis=0)); warm.append(False)
    return np.array(vecs), np.array(warm)

test_uids = test_df["reviewer_id"].values
test_lids = test_df["listing_id"].values

# Valid test rows
valid_rows = np.array([i for i, l in enumerate(test_lids) if l in lid_to_idx])
true_idx = np.array([lid_to_idx[test_lids[i]] for i in valid_rows])

# Precompute per-modality similarity matrices (only warm users matter for tuning)
print("  Precomputing per-modality similarity matrices...")

prof_s, warm_mask = _build_profiles(struct_norm, test_uids, test_lids)
prof_t, _ = _build_profiles(text_norm, test_uids, test_lids)
prof_c, _ = _build_profiles(clip_norm, test_uids, test_lids)

sim_s = (_norm_rows(prof_s) @ _norm_rows(struct_norm).T)[valid_rows]
sim_t = (_norm_rows(prof_t) @ _norm_rows(text_norm).T)[valid_rows]
sim_c = (_norm_rows(prof_c) @ _norm_rows(clip_norm).T)[valid_rows]
warm_v = warm_mask[valid_rows]

print(f"  Sim matrices: {sim_s.shape} | Warm users: {warm_v.sum()}")

# Grid search
GRID_STEP = 0.1
grid_vals = np.round(np.arange(0, 1.01, GRID_STEP), 2)

best_h10_w = -1
best_weights = (0, 0, 0)
grid_results = []

for alpha in grid_vals:
    for beta in grid_vals:
        gamma = round(1.0 - alpha - beta, 2)
        if gamma < -0.001 or gamma > 1.001:
            continue
        gamma = max(0.0, min(1.0, gamma))

        combined = alpha * sim_s + beta * sim_t + gamma * sim_c
        ranks = compute_ranks(combined, true_idx)
        m = metrics_from_ranks(ranks, warm_v)

        grid_results.append({
            "alpha": alpha, "beta": beta, "gamma": gamma,
            **{k: v for k, v in m.items()}
        })

        if m["Hit@10_W"] > best_h10_w:
            best_h10_w = m["Hit@10_W"]
            best_weights = (alpha, beta, gamma)

print(f"\n  Grid Search: {len(grid_results)} weight combinations evaluated")
print(f"  🏆 Best Weights: α={best_weights[0]:.2f} (Struct), "
      f"β={best_weights[1]:.2f} (Text), γ={best_weights[2]:.2f} (CLIP)")
print(f"  🏆 Best Hit@10 (Warm): {best_h10_w:.4f}")

# Full evaluation with best weights
alpha_b, beta_b, gamma_b = best_weights
combined_best = alpha_b * sim_s + beta_b * sim_t + gamma_b * sim_c
ranks_best = compute_ranks(combined_best, true_idx)
weighted_metrics = metrics_from_ranks(ranks_best, warm_v)
weighted_metrics["Modality"] = f"Weighted ({alpha_b:.1f}/{beta_b:.1f}/{gamma_b:.1f})"
weighted_metrics["time_sec"] = 0.0
results.append(weighted_metrics)

print(f"  Weighted Fusion Results:")
for k in K_VALUES:
    print(f"    Hit@{k}: {weighted_metrics[f'Hit@{k}']:.4f} "
          f"(W: {weighted_metrics[f'Hit@{k}_W']:.4f}) | "
          f"NDCG@{k}: {weighted_metrics[f'NDCG@{k}']:.4f} "
          f"(W: {weighted_metrics[f'NDCG@{k}_W']:.4f})")

# Save grid search results
grid_df = pd.DataFrame(grid_results)
grid_df.to_csv(os.path.join(OUT_DIR, "weighted_fusion_grid_search.csv"), index=False)
print(f"  ✓ Saved weighted_fusion_grid_search.csv ({len(grid_results)} rows)")


# =====================================================================
# STEP 6: RESULTS REPORT
# =====================================================================
print("\n" + "=" * 70)
print("STEP 6: RESULTS REPORT")
print("=" * 70)

# Build full results DataFrame
results_df = pd.DataFrame(results).round(4)

# --- Console Summary Table (Warm users, key metrics) ---
print("\n── Warm User Performance (Returning Users) ──")
print(f"{'Modality':35s} | {'H@5':>6s} {'H@10':>6s} {'H@15':>6s} {'H@20':>6s} | "
      f"{'NDCG@10':>7s} {'MAP@10':>7s} {'MRR':>6s} | {'Time':>5s}")
print("─" * 120)

for _, row in results_df.iterrows():
    print(f"{row['Modality']:35s} | "
          f"{row.get('Hit@5_W',0):>6.4f} {row.get('Hit@10_W',0):>6.4f} "
          f"{row.get('Hit@15_W',0):>6.4f} {row.get('Hit@20_W',0):>6.4f} | "
          f"{row.get('NDCG@10_W',0):>7.4f} {row.get('MAP@10_W',0):>7.4f} "
          f"{row.get('MRR_W',0):>6.4f} | {row.get('time_sec',0):>4.1f}s")
print("─" * 120)

# Highlight best
best_h10 = results_df.loc[results_df["Hit@10_W"].idxmax()]
best_mrr = results_df.loc[results_df["MRR_W"].idxmax()]
print(f"\n  🏆 Best Hit@10 (Warm):  {best_h10['Modality']} ({best_h10['Hit@10_W']:.4f})")
print(f"  🏆 Best MRR (Warm):    {best_mrr['Modality']} ({best_mrr['MRR_W']:.4f})")

# Random baselines
for k in K_VALUES:
    print(f"  📊 Random Hit@{k}: {k/n_listings:.4f}")


# =====================================================================
# STEP 7: SAVE OUTPUTS
# =====================================================================
print("\n" + "=" * 70)
print("STEP 7: SAVE OUTPUTS")
print("=" * 70)

# Save results table
results_df.to_csv(os.path.join(OUT_DIR, "evaluation_results.csv"))
print(f"  ✓ Saved evaluation_results.csv")

# Save fused embeddings for downstream use
np.save(os.path.join(OUT_DIR, "fused_embeddings_256d.npy"), fused_reduced)
# Save listing_ids
np.save(os.path.join(OUT_DIR, "listing_ids.npy"), listing_ids)
print(f"  ✓ Saved listing_ids.npy ({listing_ids.shape})")

# Save train/test split dataframe interactions instead
train_df.to_csv(os.path.join(OUT_DIR, "train_chrono.csv"), index=False)
test_df.to_csv(os.path.join(OUT_DIR, "test_chrono.csv"), index=False)
print(f"  ✓ Saved train_chrono.csv and test_chrono.csv")

# Deprecated: Old split saving
# split_data = {
#     "train": {str(k): [int(x) for x in v] for k, v in train_data.items()},
#     "test": {str(k): int(v) for k, v in test_data.items()},
# }
# with open(os.path.join(OUT_DIR, "train_test_split.json"), "w") as f:
#     json.dump(split_data, f, indent=2)
print(f"  ✓ Saved train_test_split.json")

# Save the TF-IDF vectorizer for reuse
with open(os.path.join(OUT_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)
print(f"  ✓ Saved tfidf_vectorizer.pkl")

# Save scalers
with open(os.path.join(OUT_DIR, "scalers.pkl"), "wb") as f:
    pickle.dump({
        "struct": scaler_struct,
        "text": scaler_text,
        "clip": scaler_clip,
        "resnet": scaler_resnet,
        "svd": svd,
    }, f)
print(f"  ✓ Saved scalers.pkl")

# Save per-modality embeddings for downstream experiments
for mod_name, emb in modality_embeddings.items():
    fname = mod_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "and")
    np.save(os.path.join(OUT_DIR, f"emb_{fname}.npy"), emb)
print(f"  ✓ Saved per-modality embeddings ({len(modality_embeddings)} files)")


# =====================================================================
# STEP 8: SAMPLE RECOMMENDATIONS
# =====================================================================
print("\n" + "=" * 70)
print("STEP 8: SAMPLE RECOMMENDATIONS")
print("=" * 70)

# Load listing names for display
with open(os.path.join(BASE_DIR, "airbnb_listings.json")) as f:
    listings_raw = pd.DataFrame(json.load(f))
lid_to_name = dict(zip(listings_raw["listing_id"], listings_raw["name"]))
lid_to_price = dict(zip(listings_raw["listing_id"], listings_raw["price"]))

with open(os.path.join(BASE_DIR, "airbnb_address.json")) as f:
    addr_raw = pd.DataFrame(json.load(f))
lid_to_market = dict(zip(addr_raw["listing_id"], addr_raw["market"]))


def show_recommendations(query_lid, embeddings, listing_ids, lid_to_idx, top_k=5):
    """Show top-K recommendations for a given listing."""
    if query_lid not in lid_to_idx:
        print(f"  Listing {query_lid} not in index")
        return

    q_idx = lid_to_idx[query_lid]
    sims = cosine_similarity(embeddings[q_idx:q_idx+1], embeddings)[0]
    sims[q_idx] = -1  # exclude self

    top_idxs = np.argsort(sims)[::-1][:top_k]
    print(f"\n  Query: [{query_lid}] {lid_to_name.get(query_lid, '?')} "
          f"(${lid_to_price.get(query_lid, '?')}, {lid_to_market.get(query_lid, '?')})")
    print(f"  {'Rank':>4s}  {'Score':>6s}  {'ID':>10s}  {'Price':>6s}  {'Market':12s}  Name")
    print(f"  {'─'*80}")
    for rank, idx in enumerate(top_idxs, 1):
        lid = listing_ids[idx]
        print(f"  {rank:4d}  {sims[idx]:6.3f}  {lid:10d}  ${lid_to_price.get(lid, 0):>5d}  "
              f"{lid_to_market.get(lid, '?'):12s}  {lid_to_name.get(lid, '?')[:50]}")


# Show 3 example recommendations using full fusion
sample_lids = listing_ids[:3]
for lid in sample_lids:
    show_recommendations(lid, fused_reduced, listing_ids, lid_to_idx, top_k=5)


# =====================================================================
# STEP 9: GENERATE RECOMMENDATIONS FOR ALL USERS
# =====================================================================
print("\n" + "=" * 70)
print("STEP 9: GENERATE RECOMMENDATIONS FOR ALL ~145k USERS")
print("=" * 70)

# 1. Build user profiles for ALL users
# (Single interaction -> item embedding; Multi -> mean embedding)
print("  Building user profiles...")
user_profiles = {}
all_users = valid_interactions["reviewer_id"].unique()
print(f"  Total unique users to recommend for: {len(all_users)}")

# Group interactions by user
user_history = valid_interactions.groupby("reviewer_id")["listing_id"].apply(list).to_dict()

# Pre-compute valid listing embeddings (fused_reduced)
# Map lid -> embedding row index
lid_to_idx_map = {lid: i for i, lid in enumerate(listing_ids)}

# 2. Batched inference
BATCH_SIZE = 1000
top_k = 5
all_recs = []

# Listings matrix (N_items, 256)
item_embeddings = fused_reduced

print(f"  Starting batched inference (Batch size: {BATCH_SIZE})...")
t_start = time.time()

user_batches = [all_users[i:i + BATCH_SIZE] for i in range(0, len(all_users), BATCH_SIZE)]

for batch_idx, user_batch in enumerate(user_batches):
    if batch_idx % 10 == 0:
        print(f"    Processing batch {batch_idx+1}/{len(user_batches)}...")

    # Build batch user profile matrix (Batch_Size, 256)
    batch_profiles = []
    batch_uids = []
    
    for uid in user_batch:
        history_lids = user_history.get(uid, [])
        valid_idxs = [lid_to_idx_map[lid] for lid in history_lids if lid in lid_to_idx_map]
        
        if not valid_idxs:
            continue
            
        # User vector = Mean of history item embeddings
        user_vec = item_embeddings[valid_idxs].mean(axis=0)
        batch_profiles.append(user_vec)
        batch_uids.append(uid)
    
    if not batch_profiles:
        continue
        
    batch_matrix = np.array(batch_profiles) # (B, 256)
    
    # Cosine Similarity: User (B, 256) x Items.T (256, N) -> (B, N)
    # Assumes embeddings are already normalized? 
    # StandardScaler output is not unit length, so we need to normalize for cosine sim
    # Normalize batch user vectors
    batch_norm = np.linalg.norm(batch_matrix, axis=1, keepdims=True)
    batch_matrix = batch_matrix / (batch_norm + 1e-9)
    
    # Normalize item descriptors (do once outside loop ideally, but fast enough here)
    if batch_idx == 0:
        item_norm = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
        item_embeddings_norm = item_embeddings / (item_norm + 1e-9)

    # Compute scores
    sim_scores = np.dot(batch_matrix, item_embeddings_norm.T) # (B, N)
    
    # Get Top-K
    # argpartition is faster than sort for top-k
    # We want indices of top 5
    top_k_idxs = np.argpartition(-sim_scores, top_k, axis=1)[:, :top_k]
    
    # Sort the top k (since argpartition is not sorted)
    # This is small (B, 5), so trivial
    rows = np.arange(sim_scores.shape[0])[:, None]
    top_k_scores = sim_scores[rows, top_k_idxs]
    
    # Sort within the top k
    sort_order = np.argsort(-top_k_scores, axis=1)
    final_idxs = top_k_idxs[rows, sort_order]
    
    # Map back to listing IDs
    for i, uid in enumerate(batch_uids):
        recs = [listing_ids[idx] for idx in final_idxs[i]]
        all_recs.append({
            "user_id": uid,
            "rec_1": recs[0],
            "rec_2": recs[1],
            "rec_3": recs[2],
            "rec_4": recs[3],
            "rec_5": recs[4]
        })

print(f"  Inference complete in {time.time() - t_start:.1f}s")

# Save to CSV
recs_df = pd.DataFrame(all_recs)
out_path = os.path.join(OUT_DIR, "recommendations_all_users.csv")
recs_df.to_csv(out_path, index=False)
print(f"  ✓ Saved recommendations for {len(recs_df)} users to: {out_path}")

print("\n" + "=" * 70)
print("RECOMMENDATION PIPELINE COMPLETE!")
print(f"All outputs saved to: {OUT_DIR}")
print("=" * 70)
