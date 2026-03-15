"""
PCA Analysis for Airbnb Listings JSON Data
==========================================
Usage:
    python pca_analysis.py --input your_data.json
    python pca_analysis.py --input your_data.json --n_components 10 --top_n 20 --output results/
    python pca_analysis.py --input your_data.json --download_images --image_url_col picture_url

Image storage:
    - All images found in URL columns are downloaded to --output/images/
    - A manifest CSV (images/image_manifest.csv) logs every image: listing_id, url, local path, status
    - Embedded image arrays (CLIP / ResNet) are saved as .npy files to --output/embeddings/
    - Failed downloads are logged to --output/images/failed_downloads.txt
"""

import argparse
import json
import os
import time
import hashlib
import warnings
warnings.filterwarnings("ignore")

import csv
import mimetypes
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ── Config ────────────────────────────────────────────────────────────────────
EXCLUDE_COLS = {"listing_id", "price_tier", "government_area"}

STYLE = {
    "bg":       "#0d0f14",
    "panel":    "#13161e",
    "border":   "#252830",
    "text":     "#e2e8f0",
    "muted":    "#64748b",
    "accent1":  "#f0c040",   # gold
    "accent2":  "#38bdf8",   # sky
    "accent3":  "#f87171",   # red
    "accent4":  "#4ade80",   # green
    "accent5":  "#c084fc",   # purple
}

PALETTE = [STYLE["accent1"], STYLE["accent2"], STYLE["accent3"],
           STYLE["accent4"], STYLE["accent5"], "#fb923c", "#34d399", "#e879f9"]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load JSON file (array of objects or single object)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = [raw]
    df = pd.DataFrame(raw)
    print(f"✔ Loaded {len(df)} records, {df.shape[1]} columns")
    return df


def prepare_features(df: pd.DataFrame):
    """Select numeric columns, drop excluded ones, fill NaN with column mean."""
    drop = [c for c in df.columns if c in EXCLUDE_COLS]
    df = df.drop(columns=drop, errors="ignore")

    # Keep only numeric columns
    numeric = df.select_dtypes(include=[np.number])
    # Drop columns that are all-NaN or zero-variance
    numeric = numeric.dropna(axis=1, how="all")
    numeric = numeric.loc[:, numeric.std() > 0]
    numeric = numeric.fillna(numeric.mean())

    print(f"✔ Numeric features used: {numeric.shape[1]}  |  Samples: {numeric.shape[0]}")
    return numeric


# ── PCA ───────────────────────────────────────────────────────────────────────
def run_pca(X: pd.DataFrame, n_components: int):
    """Standardise and run PCA; returns fitted objects + transformed data."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    scores = pca.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=[f"PC{i+1}" for i in range(n_comp)],
    )
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print(f"\n{'─'*50}")
    print(f"  Components computed : {n_comp}")
    print(f"  Variance explained  : {cumulative[-1]*100:.1f}% (all {n_comp} PCs)")
    for i, (ev, cv) in enumerate(zip(explained, cumulative)):
        print(f"    PC{i+1}: {ev*100:6.2f}%   cumulative: {cv*100:6.2f}%")
    print(f"{'─'*50}\n")

    return pca, scaler, scores, loadings, explained, cumulative


# ── Plotting ──────────────────────────────────────────────────────────────────
def apply_dark_style(ax):
    ax.set_facecolor(STYLE["panel"])
    for spine in ax.spines.values():
        spine.set_edgecolor(STYLE["border"])
    ax.tick_params(colors=STYLE["muted"], labelsize=8)
    ax.xaxis.label.set_color(STYLE["muted"])
    ax.yaxis.label.set_color(STYLE["muted"])
    ax.title.set_color(STYLE["text"])
    ax.grid(color=STYLE["border"], linewidth=0.5, alpha=0.6)


def plot_scree(ax, explained, cumulative):
    x = np.arange(1, len(explained) + 1)
    bars = ax.bar(x, explained * 100, color=STYLE["accent1"],
                  alpha=0.85, edgecolor="none", zorder=3)
    ax.plot(x, cumulative * 100, "o-", color=STYLE["accent2"],
            linewidth=2, markersize=5, zorder=4, label="Cumulative %")
    ax.axhline(80, color=STYLE["accent3"], linewidth=0.8,
               linestyle="--", alpha=0.6, label="80% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("Scree Plot", fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i}" for i in x], fontsize=7)
    ax.legend(fontsize=7, facecolor=STYLE["panel"],
              labelcolor=STYLE["text"], framealpha=0.8)
    for bar, val in zip(bars, explained):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val*100:.1f}%", ha="center", va="bottom",
                fontsize=6.5, color=STYLE["text"])
    apply_dark_style(ax)


def plot_biplot(ax, scores, pc_x=0, pc_y=1, labels=None):
    x, y = scores[:, pc_x], scores[:, pc_y]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(x))]
    ax.scatter(x, y, c=colors, s=60, zorder=5, edgecolors="none", alpha=0.9)
    if labels is not None:
        for i, lbl in enumerate(labels):
            ax.annotate(str(lbl)[:10], (x[i], y[i]),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=6, color=PALETTE[i % len(PALETTE)], alpha=0.9)
    ax.axhline(0, color=STYLE["border"], linewidth=0.7)
    ax.axvline(0, color=STYLE["border"], linewidth=0.7)
    ax.set_xlabel(f"PC{pc_x+1}")
    ax.set_ylabel(f"PC{pc_y+1}")
    ax.set_title(f"Score Plot  PC{pc_x+1} vs PC{pc_y+1}", fontweight="bold", fontsize=10)
    apply_dark_style(ax)


def plot_loadings(ax, loadings, pc=0, top_n=20):
    col = f"PC{pc+1}"
    series = loadings[col].abs().sort_values(ascending=False).head(top_n)
    features = series.index
    vals = loadings.loc[features, col].values
    colors = [STYLE["accent1"] if v >= 0 else STYLE["accent3"] for v in vals]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, alpha=0.85, edgecolor="none")
    ax.axvline(0, color=STYLE["border"], linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f.replace("amenity_", "").replace("_", " ")[:30]
                        for f in features], fontsize=7)
    ax.set_xlabel("Loading")
    ax.set_title(f"PC{pc+1} Feature Loadings (Top {top_n})", fontweight="bold", fontsize=10)
    ax.invert_yaxis()
    apply_dark_style(ax)


def plot_variance_heatmap(ax, loadings, top_n=15):
    # Show loadings heatmap for top features across all PCs
    importance = loadings.abs().max(axis=1).sort_values(ascending=False).head(top_n)
    subset = loadings.loc[importance.index]
    feat_labels = [f.replace("amenity_", "").replace("_", " ")[:28] for f in subset.index]
    im = ax.imshow(subset.values, cmap="RdYlBu_r", aspect="auto",
                   vmin=-1, vmax=1, interpolation="nearest")
    ax.set_yticks(range(len(feat_labels)))
    ax.set_yticklabels(feat_labels, fontsize=7)
    ax.set_xticks(range(subset.shape[1]))
    ax.set_xticklabels(subset.columns, fontsize=7, rotation=45, ha="right")
    ax.set_title("Loadings Heatmap (Top Features)", fontweight="bold", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(labelsize=7)
    apply_dark_style(ax)


def plot_cumvar(ax, explained):
    x = np.arange(1, len(explained) + 1)
    cum = np.cumsum(explained) * 100
    ax.plot(x, cum, "o-", color=STYLE["accent4"], linewidth=2, markersize=5)
    ax.fill_between(x, cum, alpha=0.15, color=STYLE["accent4"])
    for thresh, color in [(80, STYLE["accent3"]), (90, STYLE["accent2"]), (95, STYLE["accent1"])]:
        ax.axhline(thresh, linestyle="--", linewidth=0.8, color=color, alpha=0.7, label=f"{thresh}%")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Explained (%)")
    ax.set_title("Cumulative Explained Variance", fontweight="bold", fontsize=10)
    ax.set_xticks(x)
    ax.legend(fontsize=7, facecolor=STYLE["panel"],
              labelcolor=STYLE["text"], framealpha=0.8)
    apply_dark_style(ax)


def build_figure(pca, scores, loadings, explained, cumulative,
                 labels=None, top_n=15, output_dir="."):
    fig = plt.figure(figsize=(18, 14), facecolor=STYLE["bg"])
    fig.suptitle("PCA Analysis — Airbnb Listings",
                 fontsize=16, color=STYLE["text"], fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38,
                           left=0.06, right=0.97, top=0.93, bottom=0.06)

    plot_scree(fig.add_subplot(gs[0, 0]), explained, cumulative)
    plot_cumvar(fig.add_subplot(gs[0, 1]), explained)
    plot_biplot(fig.add_subplot(gs[0, 2]), scores, pc_x=0, pc_y=1, labels=labels)
    plot_loadings(fig.add_subplot(gs[1, 0]), loadings, pc=0, top_n=top_n)
    plot_loadings(fig.add_subplot(gs[1, 1]), loadings, pc=1, top_n=top_n)
    plot_variance_heatmap(fig.add_subplot(gs[1, 2]), loadings, top_n=top_n)

    out_path = os.path.join(output_dir, "pca_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=STYLE["bg"])
    print(f"✔ Figure saved → {out_path}")
    plt.close()


# ── CSV export ────────────────────────────────────────────────────────────────
def export_results(scores, loadings, explained, feature_cols, output_dir):
    # Scores
    scores_df = pd.DataFrame(
        scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])]
    )
    scores_path = os.path.join(output_dir, "pca_scores.csv")
    scores_df.to_csv(scores_path, index=False)
    print(f"✔ Scores saved  → {scores_path}")

    # Loadings
    load_path = os.path.join(output_dir, "pca_loadings.csv")
    loadings.to_csv(load_path)
    print(f"✔ Loadings saved → {load_path}")

    # Variance summary
    var_df = pd.DataFrame({
        "component": [f"PC{i+1}" for i in range(len(explained))],
        "explained_variance_ratio": explained,
        "cumulative_variance_ratio": np.cumsum(explained),
    })
    var_path = os.path.join(output_dir, "pca_variance.csv")
    var_df.to_csv(var_path, index=False)
    print(f"✔ Variance saved → {var_path}")


# ── Image storage ─────────────────────────────────────────────────────────────

# Columns that commonly contain image URLs in Airbnb data
IMAGE_URL_COLUMNS = [
    "picture_url", "xl_picture_url", "thumbnail_url",
    "medium_url", "host_picture_url", "host_thumbnail_url",
]

# Columns that may contain raw image array data (CLIP / ResNet embeddings stored as lists)
EMBEDDING_COLUMNS = [
    "clip_embedding", "resnet_embedding",
    "has_clip_embedding", "has_resnet_embedding",  # flag columns — skip if just bool
]


def _url_to_filename(listing_id, url: str, col: str) -> str:
    """Derive a safe local filename from listing id + url."""
    ext = os.path.splitext(url.split("?")[0])[-1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
        ext = ".jpg"
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{listing_id}__{col}__{url_hash}{ext}"


def _download_one(task: dict) -> dict:
    """Download a single image. Returns a result dict for the manifest."""
    url, dest, listing_id, col = task["url"], task["dest"], task["listing_id"], task["col"]
    if os.path.exists(dest):
        return {**task, "status": "skipped (exists)", "size_bytes": os.path.getsize(dest)}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp, open(dest, "wb") as f:
            data = resp.read()
            f.write(data)
        return {**task, "status": "ok", "size_bytes": len(data)}
    except Exception as e:
        return {**task, "status": f"error: {e}", "size_bytes": 0}


def download_images(df: pd.DataFrame, output_dir: str,
                    url_cols: list = None, workers: int = 8) -> pd.DataFrame:
    """
    Scan df for URL columns, download all images, save manifest CSV.
    Returns the manifest as a DataFrame.
    """
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Auto-detect URL columns if not specified
    if url_cols is None:
        url_cols = [c for c in df.columns if c in IMAGE_URL_COLUMNS]
        # Also detect any column whose name contains 'url' or 'image'
        for c in df.columns:
            if c not in url_cols and ("url" in c.lower() or "image" in c.lower()):
                url_cols.append(c)
    url_cols = [c for c in url_cols if c in df.columns]

    if not url_cols:
        print("⚠  No image URL columns found. Skipping image download.")
        return pd.DataFrame()

    print(f"\n📷 Image URL columns detected: {url_cols}")

    # Build download task list
    tasks = []
    for _, row in df.iterrows():
        lid = row.get("listing_id", row.name)
        for col in url_cols:
            url = row.get(col)
            if not isinstance(url, str) or not url.startswith("http"):
                continue
            fname = _url_to_filename(lid, url, col)
            dest = os.path.join(img_dir, fname)
            tasks.append({"listing_id": lid, "col": col, "url": url,
                           "dest": dest, "local_path": os.path.join("images", fname)})

    if not tasks:
        print("⚠  No valid image URLs found in data.")
        return pd.DataFrame()

    print(f"   Downloading {len(tasks)} images with {workers} workers …")
    results = []
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download_one, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            status = "✔" if res["status"] == "ok" else ("↷" if "skipped" in res["status"] else "✘")
            print(f"   [{i:>4}/{len(tasks)}] {status}  {res['url'][:70]}", end="\r")
            if res["status"].startswith("error"):
                failed.append(res["url"])

    print()  # newline after \r progress

    manifest = pd.DataFrame(results)[["listing_id", "col", "url", "local_path", "status", "size_bytes"]]
    manifest_path = os.path.join(img_dir, "image_manifest.csv")
    manifest.to_csv(manifest_path, index=False)
    print(f"✔ Manifest saved → {manifest_path}")

    ok = manifest[manifest["status"] == "ok"]
    sk = manifest[manifest["status"].str.startswith("skipped")]
    er = manifest[manifest["status"].str.startswith("error")]
    print(f"   Downloaded: {len(ok)}  |  Skipped (exist): {len(sk)}  |  Failed: {len(er)}")

    if failed:
        fail_path = os.path.join(img_dir, "failed_downloads.txt")
        Path(fail_path).write_text("\n".join(failed))
        print(f"   Failed URLs → {fail_path}")

    return manifest


def save_embeddings(df: pd.DataFrame, output_dir: str):
    """
    Save any list/array columns that look like embeddings (CLIP, ResNet, etc.) as .npy files.
    Ignores boolean flag columns like has_clip_embedding.
    """
    emb_dir = os.path.join(output_dir, "embeddings")
    saved_any = False

    for col in df.columns:
        # Must look like an embedding column
        if not any(kw in col.lower() for kw in ("embedding", "vector", "clip", "resnet")):
            continue
        # Skip if it's just a boolean flag
        if df[col].dtype in (bool, np.bool_) or df[col].isin([0, 1, True, False]).all():
            continue
        # Try to parse the first non-null value as an array
        sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
        if sample is None:
            continue
        try:
            arr = np.array([
                np.array(v, dtype=np.float32) if not isinstance(v, (int, float)) else [float(v)]
                for v in df[col].fillna(0)
            ])
            os.makedirs(emb_dir, exist_ok=True)
            out = os.path.join(emb_dir, f"{col}.npy")
            np.save(out, arr)
            print(f"✔ Embedding saved → {out}  shape: {arr.shape}")
            saved_any = True
        except Exception as e:
            print(f"⚠  Could not save embedding '{col}': {e}")

    if not saved_any:
        print("   No array-type embedding columns found (has_clip_embedding / has_resnet_embedding are boolean flags).")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="PCA Analysis for Airbnb Listings JSON")
    p.add_argument("--input",  "-i", required=True, help="Path to input JSON file")
    p.add_argument("--output", "-o", default=".",   help="Output directory (default: current dir)")
    p.add_argument("--n_components", "-n", type=int, default=10,
                   help="Max number of PCA components (default: 10)")
    p.add_argument("--top_n", "-t", type=int, default=15,
                   help="Top N features to show in loading plots (default: 15)")
    p.add_argument("--label_col", "-l", default="listing_id",
                   help="Column to use as point labels in biplot (default: listing_id)")
    p.add_argument("--no_plot", action="store_true", help="Skip figure generation")
    p.add_argument("--no_export", action="store_true", help="Skip CSV exports")
    p.add_argument("--download_images", action="store_true",
                   help="Download & store all listing images found in URL columns")
    p.add_argument("--image_url_col", nargs="+", default=None,
                   help="Specific URL column(s) to use (default: auto-detect)")
    p.add_argument("--image_workers", type=int, default=8,
                   help="Parallel download workers (default: 8)")
    p.add_argument("--save_embeddings", action="store_true",
                   help="Save CLIP/ResNet embedding columns as .npy files")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load
    df = load_data(args.input)

    # Labels for biplot
    labels = df[args.label_col].astype(str).tolist() if args.label_col in df.columns else None

    # Prepare features
    X = prepare_features(df)

    # PCA
    pca, scaler, scores, loadings, explained, cumulative = run_pca(X, args.n_components)

    # Plot
    if not args.no_plot:
        build_figure(pca, scores, loadings, explained, cumulative,
                     labels=labels, top_n=args.top_n, output_dir=args.output)

    # Export
    if not args.no_export:
        export_results(scores, loadings, explained, X.columns.tolist(), args.output)

    # Images
    if args.download_images:
        download_images(df, args.output,
                        url_cols=args.image_url_col,
                        workers=args.image_workers)

    if args.save_embeddings:
        print("\n💾 Saving embeddings …")
        save_embeddings(df, args.output)

    print("\n✅ PCA analysis complete!")


if __name__ == "__main__":
    main()
