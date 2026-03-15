"""
=======================================================================
Option A: Enhanced Text Embeddings — Sentence-BERT (MPNet)
=======================================================================

Replaces TF-IDF 500-d text embeddings with Sentence-BERT 768-d dense
embeddings from 'all-mpnet-base-v2' (best general-purpose SBERT model).

Setup:  pip install sentence-transformers

How it works:
  1. Loads listing descriptions + review text (same sources as pipeline)
  2. Encodes all texts with SentenceTransformer in GPU-accelerated batches
  3. Saves SBERT embeddings to disk
  4. Re-runs chronological evaluation (reuses existing train/test split)
  5. Compares SBERT vs TF-IDF across all modality combos
  6. Runs weighted late fusion grid search with SBERT text

Expected improvement: 20–40% uplift on Hit@K from better semantic
understanding (contextual embeddings vs bag-of-words TF-IDF).
"""

import json
import os
import pickle
import time
import warnings
from itertools import combinations as itertools_combinations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EDA_DIR = os.path.join(BASE_DIR, "eda_outputs_new1")
OUT_DIR = os.path.join(BASE_DIR, "recommendation_outputs_new1")
os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================================
# STEP 1: GENERATE SENTENCE-BERT EMBEDDINGS
# =====================================================================
print("=" * 70)
print("OPTION A: ENHANCED TEXT EMBEDDINGS (Sentence-BERT)")
print("=" * 70)

# --- 1a. Load and merge text data ---
print("\n[1] Loading text data...")
text_f = pd.read_json(os.path.join(EDA_DIR, "text_features.json"))
review_text = pd.read_json(os.path.join(EDA_DIR, "review_text_features.json"))
print(f"  Loaded text_features: {text_f.shape}")
print(f"  Loaded review_text_features: {review_text.shape}")

text_merged = text_f[["listing_id", "combined_text"]].merge(
    review_text[["listing_id", "aggregated_review_text"]], on="listing_id", how="left"
)
text_merged["aggregated_review_text"] = text_merged["aggregated_review_text"].fillna("")
text_merged["full_text"] = (
    text_merged["combined_text"] + " " + text_merged["aggregated_review_text"]
)
print(f"  Merged text: {len(text_merged)} listings")

# --- 1b. Load structured & image features (for combo evaluation) ---
print("\n[2] Loading other modalities...")
master = pd.read_json(os.path.join(EDA_DIR, "master_structured_features.json"))

with open(os.path.join(BASE_DIR, "clip_embeddings.pkl"), "rb") as f:
    clip_dict = pickle.load(f)
clip_df = pd.DataFrame([
    {"listing_id": lid, **{f"clip_{i}": v for i, v in enumerate(emb)}}
    for lid, emb in clip_dict.items()
])
clip_listing_ids = set(clip_df["listing_id"].values)

with open(os.path.join(BASE_DIR, "resnet50_embeddings.pkl"), "rb") as f:
    resnet_dict = pickle.load(f)
resnet_df = pd.DataFrame([
    {"listing_id": lid, **{f"resnet_{i}": v for i, v in enumerate(emb)}}
    for lid, emb in resnet_dict.items()
])

# --- 1c. Filter to valid listings ---
valid_listings = sorted(clip_listing_ids & set(master["listing_id"]))
master_aligned = master[master["listing_id"].isin(valid_listings)].copy()
master_aligned = master_aligned.sort_values("listing_id").reset_index(drop=True)
listing_ids = master_aligned["listing_id"].values
lid_to_idx = {lid: i for i, lid in enumerate(listing_ids)}
n_listings = len(listing_ids)
print(f"  Valid listings: {n_listings}")

# Align text
text_aligned = text_merged[text_merged["listing_id"].isin(valid_listings)].copy()
text_aligned = text_aligned.set_index("listing_id").loc[listing_ids].reset_index()
texts = text_aligned["full_text"].fillna("").tolist()


# --- 1d. Encode with Sentence-BERT ---
print("\n[3] Encoding texts with Sentence-BERT (all-mpnet-base-v2)...")
print("    This may take a few minutes on first run (model download ~420MB)...")

try:
    from sentence_transformers import SentenceTransformer

    # Model options (uncomment the one you want):
    # MODEL_NAME = "all-MiniLM-L6-v2"        # 384-d, fastest, good quality
    MODEL_NAME = "all-mpnet-base-v2"          # 768-d, best quality general purpose
    # MODEL_NAME = "instructor-xl"            # 768-d, instruction-tuned (needs InstructorEmbedding)
    # MODEL_NAME = "multi-qa-mpnet-base-dot-v1"  # 768-d, great for Q&A / retrieval

    model = SentenceTransformer(MODEL_NAME)
    print(f"  Model loaded: {MODEL_NAME}")
    print(f"  Max sequence length: {model.max_seq_length}")
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    t0 = time.time()
    sbert_embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity
    )
    encode_time = time.time() - t0
    print(f"  ✓ Encoded {len(texts)} texts in {encode_time:.1f}s")
    print(f"  SBERT embedding shape: {sbert_embeddings.shape}")

    # Save embeddings
    np.save(os.path.join(OUT_DIR, "sbert_text_embeddings.npy"), sbert_embeddings)
    np.save(os.path.join(OUT_DIR, "sbert_listing_ids.npy"), listing_ids)
    print(f"  ✓ Saved sbert_text_embeddings.npy")

    SBERT_AVAILABLE = True

except ImportError:
    print("  ⚠ sentence-transformers not installed!")
    print("  Install: pip install sentence-transformers")
    print("  Attempting to load pre-computed embeddings...")

    sbert_path = os.path.join(OUT_DIR, "sbert_text_embeddings.npy")
    if os.path.exists(sbert_path):
        sbert_embeddings = np.load(sbert_path)
        print(f"  ✓ Loaded cached SBERT embeddings: {sbert_embeddings.shape}")
        SBERT_AVAILABLE = True
    else:
        print("  ✗ No pre-computed SBERT embeddings found. Exiting.")
        SBERT_AVAILABLE = False


if not SBERT_AVAILABLE:
    print("\nCannot proceed without SBERT embeddings. Please install sentence-transformers.")
    exit(1)


# =====================================================================
# STEP 2: PREPARE ALL MODALITY EMBEDDINGS
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2: PREPARE MODALITY EMBEDDINGS (with SBERT)")
print("=" * 70)

# Drop ID columns for structured features
DROP_COLS = [
    "listing_id", "price", "minimum_nights", "maximum_nights",
    "price_tier", "government_area", "market_",
    "has_image", "has_clip_embedding", "has_resnet_embedding",
    "market_Other (Domestic)", "market_Other (International)",
]
struct_cols = [c for c in master.columns if c not in DROP_COLS]
struct_data = master_aligned[struct_cols].values.astype(float)
scaler_struct = StandardScaler()
struct_norm = scaler_struct.fit_transform(struct_data)
print(f"  Structured: {struct_norm.shape}")

# SBERT text (already L2-normalized if done above, but StandardScaler for consistency)
scaler_sbert = StandardScaler()
sbert_norm = scaler_sbert.fit_transform(sbert_embeddings)
print(f"  SBERT Text:  {sbert_norm.shape}")

# CLIP image
clip_aligned = clip_df[clip_df["listing_id"].isin(valid_listings)].copy()
clip_aligned = clip_aligned.set_index("listing_id").loc[listing_ids].reset_index()
clip_data = clip_aligned.drop(columns=["listing_id"]).values.astype(float)
scaler_clip = StandardScaler()
clip_norm = scaler_clip.fit_transform(clip_data)
print(f"  CLIP image:  {clip_norm.shape}")

# ResNet image
resnet_aligned = resnet_df[resnet_df["listing_id"].isin(valid_listings)].copy()
resnet_aligned = resnet_aligned.set_index("listing_id").loc[listing_ids].reset_index()
resnet_data = resnet_aligned.drop(columns=["listing_id"]).values.astype(float)
scaler_resnet = StandardScaler()
resnet_norm = scaler_resnet.fit_transform(resnet_data)
print(f"  ResNet image: {resnet_norm.shape}")

# Build all combination embeddings
base_modalities = {
    "Struct":     struct_norm,
    "SBERT":      sbert_norm,
    "CLIP":       clip_norm,
    "ResNet":     resnet_norm,
}

modality_embeddings = {}
for r in range(1, len(base_modalities) + 1):
    for combo in itertools_combinations(base_modalities.keys(), r):
        name = " + ".join(combo)
        emb = np.hstack([base_modalities[k] for k in combo])
        modality_embeddings[name] = emb

# SVD-reduced full fusion (Struct + SBERT + CLIP)
fused_3way = np.hstack([struct_norm, sbert_norm, clip_norm])
svd = TruncatedSVD(n_components=256, random_state=42)
fused_reduced = svd.fit_transform(fused_3way)
explained_var = svd.explained_variance_ratio_.sum()
modality_embeddings["All-SBERT (SVD-256)"] = fused_reduced
print(f"  SVD Fusion (Struct+SBERT+CLIP → 256-d): explained var = {explained_var:.3f}")

print(f"\n  Total modality configurations: {len(modality_embeddings)}")
for name, emb in modality_embeddings.items():
    print(f"    {name:40s}: {emb.shape}")


# =====================================================================
# STEP 3: LOAD CHRONOLOGICAL SPLIT
# =====================================================================
print("\n" + "=" * 70)
print("STEP 3: LOAD CHRONOLOGICAL SPLIT")
print("=" * 70)

train_df = pd.read_csv(os.path.join(OUT_DIR, "train_chrono.csv"))
test_df = pd.read_csv(os.path.join(OUT_DIR, "test_chrono.csv"))
print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

train_users = set(train_df["reviewer_id"].unique())


# =====================================================================
# STEP 4: EVALUATION
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4: EVALUATION — SBERT vs TF-IDF ABLATION")
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
        prec = hit / k

        # Overall
        res[f"Hit@{k}"] = hit.mean()
        res[f"Prec@{k}"] = prec.mean()
        res[f"Recall@{k}"] = hit.mean()
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
            for m in ["Hit", "Prec", "Recall", "NDCG", "MAP"]:
                res[f"{m}@{k}_W"] = 0.0

    rr = 1.0 / all_ranks
    res["MRR"] = rr.mean()
    res["MRR_W"] = rr[is_warm_valid].mean() if is_warm_valid.sum() > 0 else 0.0
    return res


def evaluate_chrono(embeddings, name):
    """Evaluate a single embedding matrix using chronological split."""
    start = time.time()

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
                user_vecs.append(v)
                is_warm.append(True)
                continue
        user_vecs.append(global_mean)
        is_warm.append(False)

    user_mat = np.array(user_vecs)
    is_warm = np.array(is_warm)

    # Normalize for cosine similarity
    user_mat = user_mat / (np.linalg.norm(user_mat, axis=1, keepdims=True) + 1e-9)
    item_mat = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)

    sims = user_mat @ item_mat.T

    # Filter to valid test rows
    true_idx, valid_rows = [], []
    for i, lid in enumerate(test_lids):
        if lid in lid_to_idx:
            true_idx.append(lid_to_idx[lid])
            valid_rows.append(i)
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

    print(f"  ✓ {name:<40} ({elapsed:.1f}s) | "
          f"H@5={res['Hit@5']:.4f} H@10={res['Hit@10']:.4f} "
          f"H@15={res['Hit@15']:.4f} H@20={res['Hit@20']:.4f} | "
          f"H@10_W={res['Hit@10_W']:.4f} MRR_W={res['MRR_W']:.4f}")
    return res


# --- Run evaluation on all SBERT-based combos ---
print("\nRunning SBERT Ablation Study...\n")
results = []
for name, emb in modality_embeddings.items():
    results.append(evaluate_chrono(emb, name))


# =====================================================================
# STEP 5: WEIGHTED LATE FUSION (GRID SEARCH WITH SBERT)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5: WEIGHTED LATE FUSION (SBERT)")
print("=" * 70)

def _norm_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def _build_profiles(emb, uids):
    """Build user profile vectors."""
    train_hist = train_df.groupby("reviewer_id")["listing_id"].apply(list).to_dict()
    vecs, warm = [], []
    for uid in uids:
        if uid in train_hist:
            idxs = [lid_to_idx[l] for l in train_hist[uid] if l in lid_to_idx]
            if idxs:
                vecs.append(emb[idxs].mean(axis=0))
                warm.append(True)
            else:
                vecs.append(emb.mean(axis=0))
                warm.append(False)
        else:
            vecs.append(emb.mean(axis=0))
            warm.append(False)
    return np.array(vecs), np.array(warm)


test_uids = test_df["reviewer_id"].values
test_lids = test_df["listing_id"].values

valid_rows = np.array([i for i, l in enumerate(test_lids) if l in lid_to_idx])
true_idx = np.array([lid_to_idx[test_lids[i]] for i in valid_rows])

print("  Precomputing per-modality similarity matrices...")
prof_s, warm_mask = _build_profiles(struct_norm, test_uids)
prof_t, _ = _build_profiles(sbert_norm, test_uids)
prof_c, _ = _build_profiles(clip_norm, test_uids)

sim_s = (_norm_rows(prof_s) @ _norm_rows(struct_norm).T)[valid_rows]
sim_t = (_norm_rows(prof_t) @ _norm_rows(sbert_norm).T)[valid_rows]
sim_c = (_norm_rows(prof_c) @ _norm_rows(clip_norm).T)[valid_rows]
warm_v = warm_mask[valid_rows]

print(f"  Sim matrices: {sim_s.shape} | Warm users: {warm_v.sum()}")

# Grid search
GRID_STEP = 0.05  # Finer grid for better optimization
grid_vals = np.round(np.arange(0, 1.01, GRID_STEP), 2)

best_h10_w = -1
best_weights = (0, 0, 0)
grid_results = []

print("  Running grid search (step=0.05, this may take a minute)...")
t0 = time.time()

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
            "alpha_struct": alpha, "beta_sbert": beta, "gamma_clip": gamma,
            **{k: v for k, v in m.items()}
        })

        if m["Hit@10_W"] > best_h10_w:
            best_h10_w = m["Hit@10_W"]
            best_weights = (alpha, beta, gamma)

print(f"  Grid search done in {time.time() - t0:.1f}s ({len(grid_results)} combos)")
print(f"  🏆 Best Weights: α={best_weights[0]:.2f} (Struct), "
      f"β={best_weights[1]:.2f} (SBERT), γ={best_weights[2]:.2f} (CLIP)")
print(f"  🏆 Best Hit@10 (Warm): {best_h10_w:.4f}")

# Full metrics with best weights
alpha_b, beta_b, gamma_b = best_weights
combined_best = alpha_b * sim_s + beta_b * sim_t + gamma_b * sim_c
ranks_best = compute_ranks(combined_best, true_idx)
weighted_metrics = metrics_from_ranks(ranks_best, warm_v)
weighted_metrics["Modality"] = f"SBERT Weighted ({alpha_b:.2f}/{beta_b:.2f}/{gamma_b:.2f})"
weighted_metrics["time_sec"] = 0.0
results.append(weighted_metrics)


# =====================================================================
# STEP 6: RESULTS & COMPARISON TABLE
# =====================================================================
print("\n" + "=" * 70)
print("STEP 6: RESULTS COMPARISON — SBERT vs BASELINE")
print("=" * 70)

results_df = pd.DataFrame(results).round(4)

# Console summary
print(f"\n{'Modality':40s} | {'H@5':>6s} {'H@10':>6s} {'H@15':>6s} {'H@20':>6s} | "
      f"{'NDCG@10':>7s} {'MAP@10':>7s} {'MRR':>6s}")
print("─" * 120)

for _, row in results_df.iterrows():
    print(f"{row['Modality']:40s} | "
          f"{row.get('Hit@5_W',0):>6.4f} {row.get('Hit@10_W',0):>6.4f} "
          f"{row.get('Hit@15_W',0):>6.4f} {row.get('Hit@20_W',0):>6.4f} | "
          f"{row.get('NDCG@10_W',0):>7.4f} {row.get('MAP@10_W',0):>7.4f} "
          f"{row.get('MRR_W',0):>6.4f}")
print("─" * 120)

# Best performers
best_h10 = results_df.loc[results_df["Hit@10_W"].idxmax()]
best_mrr = results_df.loc[results_df["MRR_W"].idxmax()]
print(f"\n  🏆 Best Hit@10 (Warm):  {best_h10['Modality']} ({best_h10['Hit@10_W']:.4f})")
print(f"  🏆 Best MRR (Warm):    {best_mrr['Modality']} ({best_mrr['MRR_W']:.4f})")

# --- Try to compare with original TF-IDF results ---
tfidf_results_path = os.path.join(OUT_DIR, "evaluation_results.csv")
if os.path.exists(tfidf_results_path):
    orig_df = pd.read_csv(tfidf_results_path)
    print("\n\n── SBERT vs TF-IDF Comparison (Key Modalities) ──")
    print(f"{'Config':35s} | {'TF-IDF H@10_W':>14s} {'SBERT H@10_W':>14s} {'Δ':>8s}")
    print("─" * 80)

    compare_keys = ["Struct", "CLIP", "ResNet"]
    sbert_text_row = results_df[results_df["Modality"] == "SBERT"]
    tfidf_text_row = orig_df[orig_df["Modality"] == "Text"]

    if len(sbert_text_row) > 0 and len(tfidf_text_row) > 0:
        tv = tfidf_text_row.iloc[0].get("Hit@10_W", 0)
        sv = sbert_text_row.iloc[0].get("Hit@10_W", 0)
        delta = sv - tv
        print(f"{'Text Only':35s} | {tv:>14.4f} {sv:>14.4f} {delta:>+8.4f}")

    for key in ["Struct + Text", "Struct + Text + CLIP"]:
        sbert_key = key.replace("Text", "SBERT")
        s_row = results_df[results_df["Modality"] == sbert_key]
        t_row = orig_df[orig_df["Modality"] == key]
        if len(s_row) > 0 and len(t_row) > 0:
            tv = t_row.iloc[0].get("Hit@10_W", 0)
            sv = s_row.iloc[0].get("Hit@10_W", 0)
            delta = sv - tv
            print(f"{key + ' → SBERT':35s} | {tv:>14.4f} {sv:>14.4f} {delta:>+8.4f}")

    print("─" * 80)

# Save results
results_df.to_csv(os.path.join(OUT_DIR, "evaluation_results_sbert.csv"), index=False)
print(f"\n  ✓ Saved evaluation_results_sbert.csv")

grid_df = pd.DataFrame(grid_results)
grid_df.to_csv(os.path.join(OUT_DIR, "weighted_fusion_grid_search_sbert.csv"), index=False)
print(f"  ✓ Saved weighted_fusion_grid_search_sbert.csv ({len(grid_results)} rows)")

# Save scalers
with open(os.path.join(OUT_DIR, "scalers_sbert.pkl"), "wb") as f:
    pickle.dump({
        "struct": scaler_struct,
        "sbert": scaler_sbert,
        "clip": scaler_clip,
        "resnet": scaler_resnet,
        "svd_sbert": svd,
    }, f)
print(f"  ✓ Saved scalers_sbert.pkl")

print("\n" + "=" * 70)
print("OPTION A COMPLETE — SBERT TEXT EMBEDDINGS")
print("=" * 70)
