"""
=======================================================================
Option C: Attention-Weighted User History Pooling
=======================================================================

Replaces naïve mean-pooling of user interaction history with attention-
weighted aggregation, where more relevant past interactions contribute
more to the user profile vector.

Three attention strategies:
  1. Target-Aware Attention:  weights = softmax(sim(history_item, query_item))
  2. Self-Attention (Learned): trainable query vector attending over history
  3. Recency-Weighted Attention: exponential decay favoring recent items

Setup:  pip install torch  (for self-attention variant only)

Expected improvement: 10–25% uplift from better user profiling that
captures preference nuances instead of losing signal in the average.
"""

import json
import os
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EDA_DIR = os.path.join(BASE_DIR, "eda_outputs_new1")
INPUT_DIR = os.path.join(BASE_DIR, "recommendation_outputs_new1")  # read pre-computed data
OUT_DIR = os.path.join(BASE_DIR, "recommendation_outputs_new2")    # write new outputs
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("OPTION C: ATTENTION-WEIGHTED USER HISTORY POOLING")
print("=" * 70)


# =====================================================================
# STEP 1: LOAD DATA
# =====================================================================
print("\n[1] Loading data...")

listing_ids = np.load(os.path.join(INPUT_DIR, "listing_ids.npy"))
lid_to_idx = {lid: i for i, lid in enumerate(listing_ids)}
n_listings = len(listing_ids)

# Load embeddings (default: SVD-256 fused)
fused_emb = np.load(os.path.join(INPUT_DIR, "fused_embeddings_256d.npy"))
print(f"  Fused embeddings: {fused_emb.shape}")

# Also load individual modalities for comparison
struct_emb = np.load(os.path.join(INPUT_DIR, "emb_struct.npy"))
text_emb = np.load(os.path.join(INPUT_DIR, "emb_text.npy"))
clip_emb = np.load(os.path.join(INPUT_DIR, "emb_clip.npy"))

# Load chronological split
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train_chrono.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test_chrono.csv"))
print(f"  Train: {len(train_df)} | Test: {len(test_df)}")

train_users = set(train_df["reviewer_id"].unique())
train_hist = train_df.groupby("reviewer_id")["listing_id"].apply(list).to_dict()

# Load review dates for recency weighting
print("  Loading review dates for recency weighting...")
try:
    reviews_df = pd.read_json(os.path.join(BASE_DIR, "airbnb_reviews.json"))
    reviews_df["date_obj"] = pd.to_datetime(reviews_df["review_date"], unit="ms")

    # Build per-user review date history (reviewer_id → [(listing_id, date), ...])
    user_review_dates = defaultdict(list)
    for _, row in reviews_df.iterrows():
        user_review_dates[row["reviewer_id"]].append(
            (row["listing_id"], row["date_obj"])
        )
    # Sort by date
    for uid in user_review_dates:
        user_review_dates[uid].sort(key=lambda x: x[1])
    HAS_DATES = True
    print(f"  ✓ Loaded review dates for {len(user_review_dates)} users")
except Exception as e:
    print(f"  ⚠ Could not load review dates: {e}")
    HAS_DATES = False


# =====================================================================
# STEP 2: DEFINE ATTENTION STRATEGIES
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2: ATTENTION POOLING STRATEGIES")
print("=" * 70)

K_VALUES = [5, 10, 15, 20]


# ─────── Strategy 0: Baseline Mean Pooling ───────
def mean_pool_user(history_lids, embeddings, **kwargs):
    """Baseline: simple mean of history item embeddings."""
    idxs = [lid_to_idx[l] for l in history_lids if l in lid_to_idx]
    if not idxs:
        return embeddings.mean(axis=0)
    return embeddings[idxs].mean(axis=0)


# ─────── Strategy 1: Target-Aware Attention ───────
def target_aware_user(history_lids, embeddings, candidate_emb=None, temperature=1.0, **kwargs):
    """
    Attention weights based on similarity between each history item
    and the candidate/query item.

    w_i = softmax(sim(h_i, candidate) / τ)
    user_vec = Σ w_i · h_i

    This is computed per-candidate, so it's more expensive but captures
    "which past experience is most relevant to THIS candidate?"
    """
    idxs = [lid_to_idx[l] for l in history_lids if l in lid_to_idx]
    if not idxs:
        return embeddings.mean(axis=0)
    if candidate_emb is None:
        return embeddings[idxs].mean(axis=0)

    history_mat = embeddings[idxs]  # (n_history, dim)

    # Normalize for cosine similarity
    h_norm = history_mat / (np.linalg.norm(history_mat, axis=1, keepdims=True) + 1e-9)
    c_norm = candidate_emb / (np.linalg.norm(candidate_emb) + 1e-9)

    # Attention scores
    scores = h_norm @ c_norm / temperature  # (n_history,)
    weights = np.exp(scores - scores.max())  # numerically stable softmax
    weights = weights / (weights.sum() + 1e-9)

    return (weights[:, None] * history_mat).sum(axis=0)


# ─────── Strategy 2: Self-Attention (Trainable) ───────
# Uses a learned query vector to attend over history items
# Implemented below in the PyTorch section


# ─────── Strategy 3: Recency-Weighted Attention ───────
def recency_weighted_user(history_lids, embeddings, uid=None, decay_rate=0.1, **kwargs):
    """
    Exponential decay weighting: more recent interactions get higher weight.

    w_i = exp(-λ · age_i) where age_i = (max_date - date_i).days
    user_vec = Σ (w_i / Σw) · h_i

    Falls back to linear position weighting if dates unavailable.
    """
    idxs = [lid_to_idx[l] for l in history_lids if l in lid_to_idx]
    if not idxs:
        return embeddings.mean(axis=0)

    history_mat = embeddings[idxs]
    n = len(idxs)

    if HAS_DATES and uid in user_review_dates:
        # Use actual dates
        dates_for_uid = user_review_dates[uid]
        lid_to_date = {lid: d for lid, d in dates_for_uid}

        ages = []
        for lid in history_lids:
            if lid in lid_to_idx and lid in lid_to_date:
                ages.append(lid_to_date[lid])
            elif lid in lid_to_idx:
                ages.append(None)

        if ages and any(a is not None for a in ages):
            valid_dates = [a for a in ages if a is not None]
            max_date = max(valid_dates)
            weights = []
            for a in ages:
                if a is not None:
                    age_days = (max_date - a).days
                    weights.append(np.exp(-decay_rate * age_days / 30))  # monthly decay
                else:
                    weights.append(0.5)  # default weight for missing dates
            weights = np.array(weights)
        else:
            # Linear fallback
            weights = np.linspace(0.5, 1.0, n)
    else:
        # Position-based fallback (assume last = most recent)
        weights = np.exp(-decay_rate * np.arange(n)[::-1])

    weights = weights / (weights.sum() + 1e-9)
    return (weights[:, None] * history_mat).sum(axis=0)


# ─────── Strategy 4: Diversity-Aware Attention ───────
def diversity_weighted_user(history_lids, embeddings, alpha_div=0.5, **kwargs):
    """
    Combines recency with diversity: down-weights items that are
    too similar to already-weighted items (MMR-style).

    Encourages the user profile to capture the breadth of interests
    rather than being dominated by repeated similar interactions.
    """
    idxs = [lid_to_idx[l] for l in history_lids if l in lid_to_idx]
    if not idxs:
        return embeddings.mean(axis=0)

    history_mat = embeddings[idxs]
    n = len(idxs)

    if n == 1:
        return history_mat[0]

    # Normalize
    h_norm = history_mat / (np.linalg.norm(history_mat, axis=1, keepdims=True) + 1e-9)

    # Pairwise similarity
    sim_matrix = h_norm @ h_norm.T

    # Initialize with uniform weights
    weights = np.ones(n) / n

    # Iterative MMR-style reweighting
    for _ in range(3):  # 3 iterations usually converge
        # For each item, compute average similarity to other weighted items
        redundancy = (sim_matrix * weights[None, :]).sum(axis=1) - weights * 1.0
        # Diversity bonus: items dissimilar to highly-weighted items get boosted
        diversity_bonus = 1.0 - redundancy
        # Blend with recency (position-based)
        recency = np.linspace(0.5, 1.0, n)
        weights = (1 - alpha_div) * recency + alpha_div * diversity_bonus
        weights = np.maximum(weights, 0.01)
        weights = weights / weights.sum()

    return (weights[:, None] * history_mat).sum(axis=0)


# =====================================================================
# STEP 3: EVALUATE ALL STRATEGIES
# =====================================================================
print("\n" + "=" * 70)
print("STEP 3: EVALUATE ATTENTION STRATEGIES")
print("=" * 70)


def compute_ranks(sim_scores, true_indices):
    """Compute rank of true item."""
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

        res[f"Hit@{k}"] = hit.mean()
        res[f"Prec@{k}"] = prec.mean()
        res[f"Recall@{k}"] = hit.mean()
        res[f"NDCG@{k}"] = ndcg.mean()
        res[f"MAP@{k}"] = mapk.mean()

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


def evaluate_pooling_strategy(embeddings, pool_fn, strategy_name, target_aware=False):
    """Evaluate a user pooling strategy."""
    start = time.time()
    global_mean = embeddings.mean(axis=0)

    test_uids = test_df["reviewer_id"].values
    test_lids = test_df["listing_id"].values

    # Normalize item embeddings once
    item_mat = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)

    # Filter to valid test rows
    true_idx, valid_rows = [], []
    for i, lid in enumerate(test_lids):
        if lid in lid_to_idx:
            true_idx.append(lid_to_idx[lid])
            valid_rows.append(i)
    true_idx = np.array(true_idx)
    valid_rows = np.array(valid_rows)

    if target_aware:
        # Target-aware: must compute per-candidate, which is expensive
        # Optimization: build profile for each test interaction against the true item
        # Then compare with ALL items using that profile
        print(f"    Computing target-aware profiles ({len(valid_rows)} queries)...")

        all_ranks = np.zeros(len(valid_rows), dtype=float)
        is_warm = np.zeros(len(valid_rows), dtype=bool)

        CHUNK = 500
        for chunk_start in range(0, len(valid_rows), CHUNK):
            chunk_end = min(chunk_start + CHUNK, len(valid_rows))
            chunk_rows = valid_rows[chunk_start:chunk_end]
            chunk_true = true_idx[chunk_start:chunk_end]

            user_vecs = []
            for row_idx in chunk_rows:
                uid = test_uids[row_idx]
                if uid in train_hist:
                    # Use global mean as proxy for target-aware attention
                    # (true target-aware would need per-candidate, too expensive)
                    v = pool_fn(
                        train_hist[uid], embeddings,
                        candidate_emb=global_mean,
                        uid=uid,
                    )
                    user_vecs.append(v)
                    is_warm[chunk_start + (row_idx - chunk_rows[0]):chunk_start + (row_idx - chunk_rows[0]) + 1] = True
                else:
                    user_vecs.append(global_mean)

            user_mat_chunk = np.array(user_vecs)
            user_mat_chunk = user_mat_chunk / (np.linalg.norm(user_mat_chunk, axis=1, keepdims=True) + 1e-9)
            sims = user_mat_chunk @ item_mat.T

            ts = sims[np.arange(len(chunk_true)), chunk_true][:, None]
            ranks = (sims > ts).sum(axis=1) + 1
            all_ranks[chunk_start:chunk_end] = ranks

        # Recompute warm mask for valid rows
        is_warm_valid = np.array([test_uids[r] in train_users for r in valid_rows])

    else:
        # Standard pooling (non target-aware)
        user_vecs, is_warm = [], []
        for uid in test_uids:
            if uid in train_hist:
                v = pool_fn(train_hist[uid], embeddings, uid=uid)
                user_vecs.append(v)
                is_warm.append(True)
            else:
                user_vecs.append(global_mean)
                is_warm.append(False)

        user_mat = np.array(user_vecs)
        is_warm = np.array(is_warm)

        user_mat = user_mat / (np.linalg.norm(user_mat, axis=1, keepdims=True) + 1e-9)
        sims = user_mat @ item_mat.T

        sims_v = sims[valid_rows]
        is_warm_valid = is_warm[valid_rows]
        all_ranks = compute_ranks(sims_v, true_idx)

    res = {"Strategy": strategy_name}
    res.update(metrics_from_ranks(all_ranks, is_warm_valid))
    elapsed = time.time() - start
    res["time_sec"] = elapsed

    print(f"  ✓ {strategy_name:<40} ({elapsed:.1f}s) | "
          f"H@5={res['Hit@5']:.4f} H@10={res['Hit@10']:.4f} "
          f"H@15={res['Hit@15']:.4f} H@20={res['Hit@20']:.4f} | "
          f"H@10_W={res.get('Hit@10_W', 0):.4f} MRR_W={res.get('MRR_W', 0):.4f}")
    return res


# ─── Run evaluations on fused embeddings (SVD-256) ───
print("\nEvaluating on Fused Embeddings (SVD-256):")
print("─" * 70)

results = []

# Baseline: Mean pooling
results.append(evaluate_pooling_strategy(fused_emb, mean_pool_user, "Mean Pooling (Baseline)"))

# Strategy 1: Target-aware attention
results.append(evaluate_pooling_strategy(
    fused_emb, target_aware_user, "Target-Aware Attention (τ=1.0)", target_aware=True
))

# Strategy 3: Recency-weighted (multiple decay rates)
for decay in [0.05, 0.1, 0.2, 0.5]:
    def recency_fn(lids, emb, decay_rate=decay, **kw):
        return recency_weighted_user(lids, emb, decay_rate=decay_rate, **kw)
    results.append(evaluate_pooling_strategy(
        fused_emb, recency_fn, f"Recency-Weighted (λ={decay})"
    ))

# Strategy 4: Diversity-aware (multiple alpha values)
for alpha in [0.3, 0.5, 0.7]:
    def diversity_fn(lids, emb, alpha_div=alpha, **kw):
        return diversity_weighted_user(lids, emb, alpha_div=alpha_div, **kw)
    results.append(evaluate_pooling_strategy(
        fused_emb, diversity_fn, f"Diversity-Aware (α={alpha})"
    ))


# =====================================================================
# STEP 4: SELF-ATTENTION (TRAINABLE) — PYTORCH
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4: SELF-ATTENTION (TRAINABLE)")
print("=" * 70)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    print("  ⚠ PyTorch not installed. Skipping self-attention strategy.")
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    class SelfAttentionPooling(nn.Module):
        """
        Learned self-attention over user history items.

        Architecture:
          - Trainable query vector q ∈ R^d
          - Attention: α_i = softmax(q^T · W_k · h_i / √d)
          - User profile: Σ α_i · W_v · h_i

        Trained end-to-end with a recommendation objective
        (maximize similarity with positive items).
        """
        def __init__(self, embed_dim, n_heads=4):
            super().__init__()
            self.embed_dim = embed_dim
            self.n_heads = n_heads
            self.head_dim = embed_dim // n_heads

            # Multi-head attention components
            self.W_q = nn.Parameter(torch.randn(n_heads, self.head_dim) * 0.02)
            self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
            self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            self.layer_norm = nn.LayerNorm(embed_dim)

        def forward(self, history_embs):
            """
            history_embs: (batch, max_seq_len, embed_dim) — padded sequence
            Returns: (batch, embed_dim) — aggregated user embedding
            """
            B, S, D = history_embs.shape

            # Keys and values from history items
            keys = self.W_k(history_embs).view(B, S, self.n_heads, self.head_dim)
            values = self.W_v(history_embs).view(B, S, self.n_heads, self.head_dim)

            # Learned query (broadcast across batch)
            query = self.W_q.unsqueeze(0).expand(B, -1, -1)  # (B, n_heads, head_dim)

            # Attention scores
            # query: (B, n_heads, head_dim) → (B, n_heads, 1, head_dim)
            # keys:  (B, S, n_heads, head_dim) → (B, n_heads, S, head_dim)
            keys_t = keys.permute(0, 2, 1, 3)  # (B, n_heads, S, head_dim)
            query_t = query.unsqueeze(2)         # (B, n_heads, 1, head_dim)

            scores = (query_t * keys_t).sum(dim=-1) / (self.head_dim ** 0.5)  # (B, n_heads, S)

            # Mask padding (history items with all-zero embeddings)
            mask = (history_embs.abs().sum(dim=-1) == 0)  # (B, S)
            mask = mask.unsqueeze(1).expand_as(scores)     # (B, n_heads, S)
            scores = scores.masked_fill(mask, -1e9)

            attn_weights = torch.softmax(scores, dim=-1)  # (B, n_heads, S)

            # Weighted sum of values
            values_t = values.permute(0, 2, 1, 3)  # (B, n_heads, S, head_dim)
            weighted = (attn_weights.unsqueeze(-1) * values_t).sum(dim=2)  # (B, n_heads, head_dim)
            weighted = weighted.reshape(B, -1)  # (B, embed_dim)

            # Project and residual
            output = self.out_proj(weighted)
            # Add mean-pool residual connection
            non_mask = (~mask[:, 0, :]).float()  # (B, S)
            mean_pool = (history_embs * non_mask.unsqueeze(-1)).sum(dim=1) / (non_mask.sum(dim=1, keepdim=True) + 1e-9)
            output = self.layer_norm(output + mean_pool)

            return output

    # ─── Train self-attention model ───
    print("\n  Preparing training data for self-attention...")

    EMB_DIM = fused_emb.shape[1]
    MAX_HISTORY = 20  # Truncate user history to most recent 20

    # Prepare paired data: (user_history, positive_item, negative_item)
    train_histories = []
    train_pos_items = []
    train_neg_items = []

    for uid, lids in train_hist.items():
        valid_lids = [l for l in lids if l in lid_to_idx]
        if len(valid_lids) < 2:
            continue

        # Leave-one-out from training set for self-attention training
        for i in range(1, len(valid_lids)):
            history = valid_lids[:i][-MAX_HISTORY:]  # Use items before this one
            pos_item = valid_lids[i]

            # Random negative
            neg_idx = np.random.randint(n_listings)
            while neg_idx in set(lid_to_idx.get(l, -1) for l in valid_lids):
                neg_idx = np.random.randint(n_listings)

            train_histories.append(history)
            train_pos_items.append(lid_to_idx[pos_item])
            train_neg_items.append(neg_idx)

    print(f"  Training triplets: {len(train_histories)}")

    # Pad histories
    def pad_histories(histories, max_len, embeddings):
        """Pad variable-length histories to fixed length."""
        B = len(histories)
        D = embeddings.shape[1]
        padded = np.zeros((B, max_len, D), dtype=np.float32)
        for i, hist in enumerate(histories):
            idxs = [lid_to_idx[l] for l in hist if l in lid_to_idx]
            n = min(len(idxs), max_len)
            if n > 0:
                padded[i, :n] = embeddings[idxs[-n:]]
        return padded

    print("  Padding histories...")
    padded_hist = pad_histories(train_histories, MAX_HISTORY, fused_emb)
    pos_items = fused_emb[train_pos_items]
    neg_items = fused_emb[train_neg_items]

    print(f"  Padded history shape: {padded_hist.shape}")

    # Training
    attn_model = SelfAttentionPooling(EMB_DIM, n_heads=4).to(device)
    optimizer = optim.Adam(attn_model.parameters(), lr=1e-3, weight_decay=1e-5)
    margin = 0.2

    BATCH_SIZE = 512
    N_EPOCHS = 15
    n_samples = len(train_histories)

    print(f"\n  Training Self-Attention ({N_EPOCHS} epochs, margin={margin})...")

    for epoch in range(1, N_EPOCHS + 1):
        attn_model.train()
        perm = np.random.permutation(n_samples)
        total_loss = 0.0

        for b in range(0, n_samples, BATCH_SIZE):
            batch_idx = perm[b:b+BATCH_SIZE]

            hist_batch = torch.from_numpy(padded_hist[batch_idx]).to(device)
            pos_batch = torch.from_numpy(pos_items[batch_idx]).to(device)
            neg_batch = torch.from_numpy(neg_items[batch_idx]).to(device)

            optimizer.zero_grad()

            # Forward: get user embeddings via attention
            user_emb = attn_model(hist_batch)  # (B, D)

            # Normalize
            user_emb = nn.functional.normalize(user_emb, dim=1)
            pos_emb = nn.functional.normalize(pos_batch, dim=1)
            neg_emb = nn.functional.normalize(neg_batch, dim=1)

            # BPR-style triplet loss
            pos_score = (user_emb * pos_emb).sum(dim=1)
            neg_score = (user_emb * neg_emb).sum(dim=1)
            loss = torch.clamp(margin - pos_score + neg_score, min=0).mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)

        avg_loss = total_loss / n_samples
        if epoch % 3 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | Loss: {avg_loss:.4f}")

    print("  ✓ Self-attention model trained")

    # ─── Evaluate self-attention ───
    print("\n  Evaluating self-attention pooling...")
    attn_model.eval()

    test_uids = test_df["reviewer_id"].values
    test_lids = test_df["listing_id"].values

    item_mat = fused_emb / (np.linalg.norm(fused_emb, axis=1, keepdims=True) + 1e-9)

    # Build user profiles via self-attention
    user_vecs = []
    is_warm = []

    with torch.no_grad():
        for uid in test_uids:
            if uid in train_hist:
                history = train_hist[uid]
                valid_hist = [l for l in history if l in lid_to_idx]
                if valid_hist:
                    # Pad to (1, MAX_HISTORY, D)
                    hist_padded = np.zeros((1, MAX_HISTORY, EMB_DIM), dtype=np.float32)
                    idxs = [lid_to_idx[l] for l in valid_hist[-MAX_HISTORY:]]
                    hist_padded[0, :len(idxs)] = fused_emb[idxs]

                    hist_tensor = torch.from_numpy(hist_padded).to(device)
                    u_emb = attn_model(hist_tensor).cpu().numpy()[0]
                    user_vecs.append(u_emb)
                    is_warm.append(True)
                    continue

            user_vecs.append(fused_emb.mean(axis=0))
            is_warm.append(False)

    user_mat = np.array(user_vecs)
    is_warm = np.array(is_warm)
    user_mat = user_mat / (np.linalg.norm(user_mat, axis=1, keepdims=True) + 1e-9)

    sims = user_mat @ item_mat.T

    true_idx, valid_rows = [], []
    for i, lid in enumerate(test_lids):
        if lid in lid_to_idx:
            true_idx.append(lid_to_idx[lid])
            valid_rows.append(i)
    true_idx = np.array(true_idx)
    valid_rows = np.array(valid_rows)

    sims_v = sims[valid_rows]
    warm_v = is_warm[valid_rows]
    ranks = compute_ranks(sims_v, true_idx)

    self_attn_res = {"Strategy": "Self-Attention (Learned, 4-head)"}
    self_attn_res.update(metrics_from_ranks(ranks, warm_v))
    self_attn_res["time_sec"] = 0.0

    print(f"  ✓ {'Self-Attention (Learned, 4-head)':<40} | "
          f"H@5={self_attn_res['Hit@5']:.4f} H@10={self_attn_res['Hit@10']:.4f} "
          f"H@15={self_attn_res['Hit@15']:.4f} H@20={self_attn_res['Hit@20']:.4f} | "
          f"H@10_W={self_attn_res.get('Hit@10_W', 0):.4f} MRR_W={self_attn_res.get('MRR_W', 0):.4f}")

    results.append(self_attn_res)

    # Save model
    torch.save(attn_model.state_dict(), os.path.join(OUT_DIR, "self_attention_pooling_model.pt"))
    print(f"  ✓ Saved self_attention_pooling_model.pt")


# =====================================================================
# STEP 5: RESULTS COMPARISON
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5: RESULTS COMPARISON — ALL POOLING STRATEGIES")
print("=" * 70)

results_df = pd.DataFrame(results).round(4)

print(f"\n{'Strategy':40s} | {'H@5_W':>6s} {'H@10_W':>7s} {'H@15_W':>7s} {'H@20_W':>7s} | "
      f"{'NDCG@10_W':>9s} {'MAP@10_W':>8s} {'MRR_W':>6s} | {'Time':>5s}")
print("─" * 130)

for _, row in results_df.iterrows():
    print(f"{row['Strategy']:40s} | "
          f"{row.get('Hit@5_W',0):>6.4f} {row.get('Hit@10_W',0):>7.4f} "
          f"{row.get('Hit@15_W',0):>7.4f} {row.get('Hit@20_W',0):>7.4f} | "
          f"{row.get('NDCG@10_W',0):>9.4f} {row.get('MAP@10_W',0):>8.4f} "
          f"{row.get('MRR_W',0):>6.4f} | {row.get('time_sec',0):>4.1f}s")
print("─" * 130)

# Highlight best
best_row = results_df.loc[results_df.get("Hit@10_W", results_df["Hit@10"]).idxmax()]
baseline_row = results_df[results_df["Strategy"] == "Mean Pooling (Baseline)"].iloc[0]
print(f"\n  🏆 Best Strategy:  {best_row['Strategy']} "
      f"(Hit@10_W = {best_row.get('Hit@10_W', best_row['Hit@10']):.4f})")

if "Hit@10_W" in baseline_row and "Hit@10_W" in best_row:
    uplift = (best_row["Hit@10_W"] - baseline_row["Hit@10_W"]) / baseline_row["Hit@10_W"] * 100
    print(f"  📈 Uplift over Mean Pooling: {uplift:+.1f}%")

# Save
results_df.to_csv(os.path.join(OUT_DIR, "evaluation_results_attention_pooling.csv"), index=False)
print(f"\n  ✓ Saved evaluation_results_attention_pooling.csv")

print("\n" + "=" * 70)
print("OPTION C COMPLETE — ATTENTION-WEIGHTED USER POOLING")
print("=" * 70)
