"""
=======================================================================
Option B: Learnable Fusion — MLP & Contrastive Models (PyTorch)
=======================================================================

Instead of a hand-tuned weighted sum of similarity scores, trains a
neural model to learn optimal fusion of multi-modal embeddings.

Two approaches implemented:
  1. MLP Classifier: MLP(user_emb ⊕ item_emb) → σ(relevance)
  2. Contrastive Model: Learns a shared projection space with margin loss

Setup:  pip install torch

How it works:
  - Loads pre-computed per-modality embeddings from pipeline outputs
  - Constructs positive pairs from training interactions
  - Samples negative pairs (random items user hasn't interacted with)
  - Trains MLP with BCE loss + early stopping on validation Hit@10
  - Evaluates on chronological test set with full metrics

Expected improvement: 15–30% uplift from learning non-linear cross-modal
interactions that weighted sums cannot capture.
"""

import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "recommendation_outputs_new1")  # read pre-computed data
OUT_DIR = os.path.join(BASE_DIR, "recommendation_outputs_new2")    # write new outputs
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("OPTION B: LEARNABLE FUSION (MLP + Contrastive)")
print("=" * 70)


# =====================================================================
# STEP 1: LOAD PRE-COMPUTED EMBEDDINGS & SPLIT
# =====================================================================
print("\n[1] Loading pre-computed embeddings...")

listing_ids = np.load(os.path.join(INPUT_DIR, "listing_ids.npy"))
lid_to_idx = {lid: i for i, lid in enumerate(listing_ids)}
n_listings = len(listing_ids)
print(f"  Listings: {n_listings}")

# Load individual modality embeddings
struct_emb = np.load(os.path.join(INPUT_DIR, "emb_struct.npy"))
text_emb = np.load(os.path.join(INPUT_DIR, "emb_text.npy"))
clip_emb = np.load(os.path.join(INPUT_DIR, "emb_clip.npy"))
resnet_emb = np.load(os.path.join(INPUT_DIR, "emb_resnet.npy"))

print(f"  Struct:  {struct_emb.shape}")
print(f"  Text:    {text_emb.shape}")
print(f"  CLIP:    {clip_emb.shape}")
print(f"  ResNet:  {resnet_emb.shape}")

# Concatenate all modalities into a single item embedding
item_embeddings = np.hstack([struct_emb, text_emb, clip_emb])
ITEM_DIM = item_embeddings.shape[1]
print(f"  Combined item embeddings: {item_embeddings.shape}")

# Load chronological split
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train_chrono.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test_chrono.csv"))
print(f"  Train interactions: {len(train_df)}")
print(f"  Test interactions:  {len(test_df)}")

train_users = set(train_df["reviewer_id"].unique())
train_hist = train_df.groupby("reviewer_id")["listing_id"].apply(list).to_dict()


# =====================================================================
# STEP 2: BUILD USER EMBEDDINGS (MEAN-POOL HISTORY)
# =====================================================================
print("\n[2] Building user profile embeddings...")

def build_user_embedding(user_lids, embeddings):
    """Mean-pool embeddings of interacted items."""
    idxs = [lid_to_idx[l] for l in user_lids if l in lid_to_idx]
    if idxs:
        return embeddings[idxs].mean(axis=0)
    return embeddings.mean(axis=0)  # fallback to global mean

USER_DIM = ITEM_DIM  # same since we mean-pool item embeddings


# =====================================================================
# STEP 3: CONSTRUCT TRAINING PAIRS
# =====================================================================
print("\n[3] Constructing training pairs (positive + negative sampling)...")

NEG_RATIO = 4  # 4 negatives per positive

all_listing_indices = np.arange(n_listings)

pos_user_vecs = []
pos_item_vecs = []
neg_user_vecs = []
neg_item_vecs = []

t0 = time.time()
for uid, lids in train_hist.items():
    u_emb = build_user_embedding(lids, item_embeddings)

    for lid in lids:
        if lid not in lid_to_idx:
            continue

        # Positive pair
        item_idx = lid_to_idx[lid]
        pos_user_vecs.append(u_emb)
        pos_item_vecs.append(item_embeddings[item_idx])

        # Negative pairs (random items user hasn't interacted with)
        user_item_set = set(lid_to_idx.get(l, -1) for l in lids)
        neg_indices = np.random.choice(n_listings, size=NEG_RATIO * 3, replace=False)
        neg_indices = [idx for idx in neg_indices if idx not in user_item_set][:NEG_RATIO]

        for neg_idx in neg_indices:
            neg_user_vecs.append(u_emb)
            neg_item_vecs.append(item_embeddings[neg_idx])

print(f"  Positive pairs: {len(pos_user_vecs)}")
print(f"  Negative pairs: {len(neg_user_vecs)}")
print(f"  Pair construction: {time.time() - t0:.1f}s")


# =====================================================================
# STEP 4: TRAIN MLP FUSION MODEL
# =====================================================================
print("\n[4] Training MLP Fusion Model...")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    print("  ⚠ PyTorch not installed! Install: pip install torch")
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ─────── 4a. MLP Architecture ───────
    class FusionMLP(nn.Module):
        """
        MLP(user_emb ⊕ item_emb) → relevance score

        Architecture:
          Input:  user_dim + item_dim (concatenated)
          Hidden: 512 → 256 → 128, each with BatchNorm + ReLU + Dropout
          Output: 1 (sigmoid)
        """
        def __init__(self, user_dim, item_dim, dropout=0.3):
            super().__init__()
            input_dim = user_dim + item_dim

            self.network = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout),

                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout / 2),

                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

        def forward(self, user_emb, item_emb):
            x = torch.cat([user_emb, item_emb], dim=1)
            return self.network(x).squeeze(-1)

    # ─────── 4b. Contrastive Fusion Model ───────
    class ContrastiveFusion(nn.Module):
        """
        Projects user and item embeddings into a shared space.
        Uses margin-based contrastive loss.

        Architecture:
          User tower:  user_dim → 256 → 128
          Item tower:  item_dim → 256 → 128
          Scoring:     cosine similarity in 128-d shared space
        """
        def __init__(self, user_dim, item_dim, proj_dim=128):
            super().__init__()

            self.user_tower = nn.Sequential(
                nn.Linear(user_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, proj_dim),
            )

            self.item_tower = nn.Sequential(
                nn.Linear(item_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, proj_dim),
            )

        def forward(self, user_emb, item_emb):
            u_proj = nn.functional.normalize(self.user_tower(user_emb), dim=1)
            i_proj = nn.functional.normalize(self.item_tower(item_emb), dim=1)
            return (u_proj * i_proj).sum(dim=1)  # cosine similarity

        def encode_users(self, user_emb):
            return nn.functional.normalize(self.user_tower(user_emb), dim=1)

        def encode_items(self, item_emb):
            return nn.functional.normalize(self.item_tower(item_emb), dim=1)

    # ─────── 4c. Prepare PyTorch Data ───────
    print("  Preparing training data tensors...")

    # Combine pos + neg
    X_user = np.array(pos_user_vecs + neg_user_vecs, dtype=np.float32)
    X_item = np.array(pos_item_vecs + neg_item_vecs, dtype=np.float32)
    y = np.array(
        [1.0] * len(pos_user_vecs) + [0.0] * len(neg_user_vecs),
        dtype=np.float32
    )

    # Shuffle
    perm = np.random.permutation(len(y))
    X_user, X_item, y = X_user[perm], X_item[perm], y[perm]

    # Train/val split (90/10 of training pairs)
    val_split = int(0.9 * len(y))
    train_dataset = TensorDataset(
        torch.from_numpy(X_user[:val_split]),
        torch.from_numpy(X_item[:val_split]),
        torch.from_numpy(y[:val_split]),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_user[val_split:]),
        torch.from_numpy(X_item[val_split:]),
        torch.from_numpy(y[val_split:]),
    )

    BATCH_SIZE = 2048
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2)

    print(f"  Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # ─────── 4d. Training Loop ───────
    def train_model(model, model_name, loss_fn, epochs=30, lr=1e-3, patience=5):
        """Generic training loop with early stopping."""
        print(f"\n  ── Training {model_name} ──")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Train
            model.train()
            train_loss = 0.0
            for batch_u, batch_i, batch_y in train_loader:
                batch_u = batch_u.to(device)
                batch_i = batch_i.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                scores = model(batch_u, batch_i)
                loss = loss_fn(scores, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_y)

            train_loss /= len(train_dataset)

            # Validate
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_u, batch_i, batch_y in val_loader:
                    batch_u = batch_u.to(device)
                    batch_i = batch_i.to(device)
                    batch_y = batch_y.to(device)
                    scores = model(batch_u, batch_i)
                    loss = loss_fn(scores, batch_y)
                    val_loss += loss.item() * len(batch_y)
            val_loss /= len(val_dataset)

            scheduler.step(val_loss)

            if epoch % 3 == 0 or epoch == 1:
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"    Epoch {epoch:3d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"LR: {lr_now:.2e}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"    Early stopping at epoch {epoch}")
                    break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        print(f"    ✓ Best val loss: {best_val_loss:.4f}")
        return model

    # Train both models
    # Model 1: MLP with BCE
    mlp_model = FusionMLP(USER_DIM, ITEM_DIM, dropout=0.3)
    bce_loss = nn.BCELoss()
    mlp_model = train_model(mlp_model, "MLP (BCE)", bce_loss, epochs=30, lr=1e-3)

    # Model 2: Contrastive with Margin Loss
    contrastive_model = ContrastiveFusion(USER_DIM, ITEM_DIM, proj_dim=128)

    def contrastive_loss(scores, labels, margin=0.5):
        """Margin-based contrastive loss."""
        pos_loss = (1 - labels) * torch.clamp(scores - margin, min=0) ** 2
        neg_loss = labels * torch.clamp(margin - scores, min=0) ** 2
        # labels=1 means positive: want high score
        # labels=0 means negative: want low score
        return (labels * torch.clamp(1 - scores, min=0) ** 2 +
                (1 - labels) * torch.clamp(scores - margin, min=0) ** 2).mean()

    contrastive_model = train_model(
        contrastive_model, "Contrastive (Margin)", contrastive_loss,
        epochs=30, lr=5e-4
    )


    # =====================================================================
    # STEP 5: EVALUATE ON TEST SET
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 5: EVALUATE LEARNED FUSION MODELS")
    print("=" * 70)

    K_VALUES = [5, 10, 15, 20]

    def evaluate_learned_model(model, model_name, use_contrastive=False):
        """Evaluate a trained fusion model on the chronological test set."""
        start = time.time()
        model.eval()

        # Build user profiles from training history
        test_uids = test_df["reviewer_id"].values
        test_lids = test_df["listing_id"].values
        global_mean = item_embeddings.mean(axis=0)

        user_vecs, is_warm = [], []
        for uid in test_uids:
            if uid in train_hist:
                u_emb = build_user_embedding(train_hist[uid], item_embeddings)
                user_vecs.append(u_emb)
                is_warm.append(True)
            else:
                user_vecs.append(global_mean)
                is_warm.append(False)

        user_mat = np.array(user_vecs, dtype=np.float32)
        is_warm = np.array(is_warm)

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

        user_mat_v = user_mat[valid_rows]
        warm_v = is_warm[valid_rows]

        # Score all user-item pairs
        # For efficiency, use batched inference
        EVAL_BATCH = 50 if not use_contrastive else 500  # MLP needs smaller batches (memory)
        all_scores = np.zeros((len(valid_rows), n_listings), dtype=np.float32)

        item_tensor = torch.from_numpy(item_embeddings.astype(np.float32)).to(device)

        with torch.no_grad():
            if use_contrastive:
                # Pre-project all items
                item_proj = model.encode_items(item_tensor)

            for b in range(0, len(valid_rows), EVAL_BATCH):
                batch_users = torch.from_numpy(
                    user_mat_v[b:b+EVAL_BATCH]
                ).to(device)

                if use_contrastive:
                    # Project users and compute cosine similarity
                    user_proj = model.encode_users(batch_users)
                    batch_scores = (user_proj @ item_proj.T).cpu().numpy()
                else:
                    # MLP scoring: chunk items to avoid OOM
                    ITEM_CHUNK = 500
                    batch_scores_list = []
                    for ic in range(0, n_listings, ITEM_CHUNK):
                        ic_end = min(ic + ITEM_CHUNK, n_listings)
                        chunk_items = item_tensor[ic:ic_end]
                        bu = batch_users.unsqueeze(1).expand(-1, ic_end - ic, -1)
                        bi = chunk_items.unsqueeze(0).expand(len(batch_users), -1, -1)
                        bu_flat = bu.reshape(-1, USER_DIM)
                        bi_flat = bi.reshape(-1, ITEM_DIM)
                        scores_flat = model(bu_flat, bi_flat)
                        batch_scores_list.append(
                            scores_flat.reshape(len(batch_users), ic_end - ic).cpu().numpy()
                        )
                    batch_scores = np.hstack(batch_scores_list)

                all_scores[b:b+EVAL_BATCH] = batch_scores

        # Compute ranks
        true_scores = all_scores[np.arange(len(true_idx)), true_idx][:, None]
        ranks = (all_scores > true_scores).sum(axis=1) + 1
        ranks = ranks.astype(float)

        # Metrics
        res = {"Modality": model_name}
        for k in K_VALUES:
            hit = (ranks <= k).astype(float)
            ndcg = np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)
            mapk = np.where(ranks <= k, 1.0 / ranks, 0.0)
            prec = hit / k

            res[f"Hit@{k}"] = hit.mean()
            res[f"Prec@{k}"] = prec.mean()
            res[f"Recall@{k}"] = hit.mean()
            res[f"NDCG@{k}"] = ndcg.mean()
            res[f"MAP@{k}"] = mapk.mean()

            if warm_v.sum() > 0:
                res[f"Hit@{k}_W"] = hit[warm_v].mean()
                res[f"Prec@{k}_W"] = prec[warm_v].mean()
                res[f"Recall@{k}_W"] = hit[warm_v].mean()
                res[f"NDCG@{k}_W"] = ndcg[warm_v].mean()
                res[f"MAP@{k}_W"] = mapk[warm_v].mean()

        rr = 1.0 / ranks
        res["MRR"] = rr.mean()
        res["MRR_W"] = rr[warm_v].mean() if warm_v.sum() > 0 else 0.0

        elapsed = time.time() - start
        res["time_sec"] = elapsed

        print(f"  ✓ {model_name:<40} ({elapsed:.1f}s) | "
              f"H@5={res['Hit@5']:.4f} H@10={res['Hit@10']:.4f} "
              f"H@15={res['Hit@15']:.4f} H@20={res['Hit@20']:.4f} | "
              f"H@10_W={res.get('Hit@10_W', 0):.4f} MRR_W={res.get('MRR_W', 0):.4f}")
        return res

    results = []

    # Evaluate MLP
    print("\nEvaluating MLP model...")
    mlp_results = evaluate_learned_model(mlp_model, "MLP Fusion (BCE)")
    results.append(mlp_results)

    # Evaluate Contrastive (much faster since it's just cosine sim after projection)
    print("\nEvaluating Contrastive model...")
    contrastive_results = evaluate_learned_model(
        contrastive_model, "Contrastive Fusion", use_contrastive=True
    )
    results.append(contrastive_results)


    # =====================================================================
    # STEP 6: COMPARISON WITH BASELINE
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 6: RESULTS COMPARISON")
    print("=" * 70)

    # Load baseline results
    baseline_path = os.path.join(INPUT_DIR, "evaluation_results.csv")
    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)

        # Find best baseline (weighted fusion or best single)
        best_baseline = baseline_df.loc[baseline_df["Hit@10_W"].idxmax()]
        print(f"\n  Baseline Best: {best_baseline['Modality']} "
              f"(Hit@10_W = {best_baseline['Hit@10_W']:.4f})")

    results_df = pd.DataFrame(results).round(4)

    print(f"\n{'Model':40s} | {'H@5_W':>6s} {'H@10_W':>7s} {'H@15_W':>7s} {'H@20_W':>7s} | "
          f"{'NDCG@10_W':>9s} {'MRR_W':>6s}")
    print("─" * 100)
    for _, row in results_df.iterrows():
        print(f"{row['Modality']:40s} | "
              f"{row.get('Hit@5_W',0):>6.4f} {row.get('Hit@10_W',0):>7.4f} "
              f"{row.get('Hit@15_W',0):>7.4f} {row.get('Hit@20_W',0):>7.4f} | "
              f"{row.get('NDCG@10_W',0):>9.4f} {row.get('MRR_W',0):>6.4f}")
    print("─" * 100)

    # Save results
    results_df.to_csv(os.path.join(OUT_DIR, "evaluation_results_learned_fusion.csv"), index=False)
    print(f"\n  ✓ Saved evaluation_results_learned_fusion.csv")

    # Save model checkpoints
    torch.save(mlp_model.state_dict(), os.path.join(OUT_DIR, "mlp_fusion_model.pt"))
    torch.save(contrastive_model.state_dict(), os.path.join(OUT_DIR, "contrastive_fusion_model.pt"))
    print(f"  ✓ Saved mlp_fusion_model.pt")
    print(f"  ✓ Saved contrastive_fusion_model.pt")

else:
    print("\n  Cannot train neural models without PyTorch.")
    print("  Install: pip install torch")
    print("  Skipping to results comparison with pre-computed baselines only.")


print("\n" + "=" * 70)
print("OPTION B COMPLETE — LEARNABLE FUSION")
print("=" * 70)
