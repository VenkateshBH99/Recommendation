## Airbnb Multi-Modal Recommendation System — End-to-End Walkthrough

---

### 1. Problem Statement & High-Level Approach

- **Goal**: Build a **multi-modal, content-based recommendation system** for Airbnb listings that can leverage **structured attributes, listing text, translated reviews, and images**.
- **Key challenge**: The interaction matrix is **extremely sparse** — 98.2% of reviewers have only 1 review — making pure collaborative filtering ineffective.
- **Solution**: Learn **per-listing embeddings** from multiple modalities (structured, text, image), then:
  - Construct **user profiles** as the mean of embeddings of items they reviewed.
  - Rank candidate listings via **cosine similarity** between user profiles and listing embeddings.
  - Explore different **fusion strategies** (early concatenation vs. weighted late fusion) for modalities.

---

### 2. Data & EDA Summary

- **Source**: 8 normalized JSON tables (listings, hosts, addresses, images, reviews, translated reviews, amenities, listing-host map) plus pre-computed image embeddings.
- **Scale**:
  - 5,555 listings across **9 countries** and **12 markets**.
  - 149,792 reviews (20+ languages) translated to English via **M2M-100**.
  - 4,545 listings with **CLIP (512-d)** and **ResNet (2048-d)** embeddings.
  - 186 unique amenities.

**Key EDA findings (from `eda_feature_engineering.py` and `eda_outputs_new/`):**

- **Price**:
  - Median ≈ **$129/night**, strongly right-skewed (max ≈ $48,842).
  - **Log-transform** (`log_price`) used for modelling; prices capped at the 99th percentile (~$1,500) for visualization.
- **Property & room types**:
  - ~**65% Apartments**, ~11% Houses, ~7% Condominiums.
  - ~63% Entire home/apt, ~36% Private room, ~1–2% Shared room.
- **Geography**:
  - Listings spread across 12 markets; largest markets include **Istanbul, Montreal, Barcelona** and the Hawaiian islands.
  - Geo scatter plots show clear price gradients by neighborhood and city.
- **Hosts**:
  - 5,104 unique hosts; around **20% superhosts**.
  - ~64% of hosts respond within 1 hour; **multi-listing hosts** captured as a special feature.
- **Reviews & cold-start**:
  - Listings: median ~15 reviews, with a long tail to 500+ reviews.
  - Reviewers: **98.2% have only 1 review**, confirming a severe **user cold-start** issue.
  - Reviews span 20+ languages, all **translated to English** using LLM-based translation.
- **Amenities**:
  - Mean ≈ 22 amenities per listing.
  - Core amenities (WiFi, Essentials, Kitchen) appear in >90% of listings.
  - Differentiating amenities (Pool, Gym, etc.) are much rarer and thus useful for recommendation.
- **Correlations**:
  - Price correlates with `accommodates`, `bedrooms`, `bathrooms`, and total amenities.
  - Amenity count shows a moderate positive correlation with price.
- **Modality coverage**:
  - 100% listings have structured data.
  - ~81–82% have images + CLIP/ResNet embeddings.
  - ~71% have reviews.
  - ~57% have **all modalities** available.

---

### 3. Feature Engineering Overview

All feature engineering is implemented in `eda_feature_engineering.py`, with outputs saved into `eda_outputs_new/`.

#### 3.1 Structured Features (`master_structured_features.json`)

Per-listing structured feature vector (~101 dimensions), including:

- **Listing numerics**:
  - `accommodates`, `bedrooms`, `beds`, `bathrooms`
  - `price`, `log_price`, `minimum_nights`, `maximum_nights`, `log_min_nights`, `log_max_nights`
  - Categorical **price tier** (`budget`, `economy`, `mid-range`, `premium`, `luxury`).
- **Room & property types**:
  - One-hot room type (Entire, Private, Shared).
  - Property type grouped to top-8 categories + `Other`, then one-hot encoded.
- **Cancellation & bed type**:
  - Ordinal `cancel_strictness` (flexible → super strict).
  - `is_real_bed` flag for higher perceived comfort.
- **Host features** (merged via `listing_host_map.json` and `airbnb_hosts.json`):
  - `is_superhost`, `response_rate`, `response_speed` (ordinal).
  - `is_multi_listing_host`, `log_listing_count`.
  - `has_host_about` (profile completeness).
- **Location features** (from `airbnb_address.json`):
  - Continuous `longitude`, `latitude`.
  - One-hot **market** and **country**.
  - Raw `government_area` retained for potential neighborhood-level analysis.
- **Amenity features**:
  - Top-40 amenities as binary indicators.
  - Total amenity count.
  - Category scores (0–1) for **safety**, **comfort**, **kitchen**, **entertainment**, **family** amenities.
- **Review summary stats**:
  - `review_count`, `unique_reviewers`, `review_span_days`, `reviews_per_month`, `log_review_count`.
- **Image availability flags**:
  - `has_image`, `has_clip_embedding`, `has_resnet_embedding`.

All missing numeric values are imputed (e.g., medians); amenity and review fields are filled with 0 where absent.

#### 3.2 Text Features (`text_features.json`, `review_text_features.json`)

From `eda_feature_engineering.py`:

- **Listing text**:
  - `combined_text` field concatenates:
    - `name`, `summary`, `description`, `space`,
    - `neighborhood_overview`, `transit`.
  - Additional metadata: `text_length`, `word_count`.
- **Aggregated review text**:
  - For each listing, up to **50 translated reviews** concatenated into `aggregated_review_text`.
  - Plus `review_text_length`.

These are later merged into a single `full_text` per listing in `recommendation_pipeline.py` and encoded using **TF-IDF**:

- Max 500 features, log-scaled term frequency, English stop-word removal.
- N-grams: unigrams + bigrams.
- Min document frequency 5, max document frequency 95%.

#### 3.3 Image Features (`clip_embeddings.pkl`, `image_embeddings.pkl`)

- **CLIP embeddings**:
  - 512-dimensional embeddings from a **CLIP** model, providing a **shared text–image space**.
- **ResNet embeddings**:
  - 2048-dimensional embeddings from a **ResNet-50** backbone, capturing deep visual patterns but without direct text alignment.
- These are converted to aligned DataFrames (`clip_features.pkl`, `resnet_features.pkl`) and merged with listings.

#### 3.4 User–Item Interactions (`user_listing_interactions.json`)

- Built from unique `(reviewer_id, listing_id)` pairs from `airbnb_reviews.json`.
- Binary implicit feedback: **user reviewed listing = positive interaction**.
- Sparsity is ~99.9997%, motivating a **content-based** approach with user profiles rather than matrix factorization.

---

### 4. Modelling & Recommendation Pipeline

The full recommendation workflow is implemented in `recommendation_pipeline.py`.

#### 4.1 Per-Listing Embeddings (Multi-Modal)

1. **Align listings**:
   - Restrict to listings that have at least CLIP embeddings (plus structured + text) to ensure a consistent multi-modal space.
2. **Structured embeddings**:
   - Extract the 101 structured features.
   - Standardize with `StandardScaler` → `struct_norm`.
3. **Text embeddings (TF-IDF)**:
   - Merge `text_features.json` and `review_text_features.json` into `full_text`.
   - Fit `TfidfVectorizer` (500-dim) and obtain dense matrix → `text_embeddings`.
   - Standardize → `text_norm`.
4. **Image embeddings**:
   - CLIP: 512-dim, standardized → `clip_norm`.
   - ResNet: 2048-dim, standardized → `resnet_norm`.
5. **Early fusion (concatenation)**:
   - Concatenate selected modalities to create high-dimensional embeddings.
   - Build **15 configurations** corresponding to all non-empty subsets of {Struct, Text, CLIP, ResNet}, plus an SVD-reduced “All” configuration.
6. **Dimensionality reduction (SVD)**:
   - Apply `TruncatedSVD` to the full concatenated vector (Struct + Text + CLIP) down to **256 dimensions**.
   - ~71% explained variance.
   - This forms `fused_reduced`, a compact multi-modal embedding for each listing.

All per-modality and fused embeddings are saved to `recommendation_outputs_new/emb_*.npy` for reuse.

#### 4.2 User Profiles & Scoring

- For each user:
  - Aggregate their **training interactions** to a listing set \(S_u\).
  - Compute user vector \(u\) as the **mean** of the embeddings \(v_i\) of items \(i ∈ S_u\).
- For each user–item pair:
  - Compute **cosine similarity** between \(u\) and each listing embedding \(v_j\).
  - Exclude already-interacted listings when generating recommendations.
  - Rank all candidates by similarity to form the recommendation list.
- **Warm vs cold users**:
  - **Warm**: user appears in training interactions → a personalized profile is built.
  - **Cold**: user has no history in training → fallback to **global mean item embedding**.

---

### 5. Training, Testing & Evaluation Protocol

#### 5.1 Chronological Split

- To simulate real deployment, the pipeline enforces a **time-based split**:
  - Load review timestamps from `airbnb_reviews.json`.
  - Convert to `date_obj` and sort interactions chronologically.
  - **Split date**: **2018-08-22** (≈80th percentile of review dates).
  - **Training set**: interactions strictly **before** the split date.
  - **Test set**: interactions **on or after** the split date.
- Statistics (approximate, based on current run):
  - Training: ~80% of interactions.
  - Test: ~20% of interactions.
  - In the test set:
    - **Warm interactions** (returning users): ~0.5%.
    - **Cold interactions** (new users): ~99.5%.
- All split metadata is saved as `recommendation_outputs_new/train_test_split_chrono.json`.

#### 5.2 Evaluation Setup

- **Evaluation scope**:
  - Only consider test interactions whose items exist in the final features index.
  - Evaluate for all **17 modality configurations** (15 subsets + SVD + weighted fusion).
- **Metrics** (computed at K = 5, 10, 15, 20):
  - **Hit@K**: whether the true item appears in the top-K list.
  - **Precision@K**, **Recall@K** (equivalent to Hit@K for a single relevant item).
  - **NDCG@K**: discounted gain based on the position of the true item.
  - **MAP@K**: average precision for the single relevant item.
  - **MRR**: mean reciprocal rank over all test interactions.
- Metrics are computed **overall** and specifically for **warm users** (suffix `_W`).
- Full results table is saved as `recommendation_outputs_new/evaluation_results.csv`.

---

### 6. Results & Insights

#### 6.1 Single-Modality Performance (Warm Users)

From the ablation results:

- **Text (TF-IDF)**:
  - Best single modality.
  - Hit@10 (warm) ≈ **9.7%**, significantly above random.
- **Structured features**:
  - Reasonable performance: Hit@10 (warm) ≈ **6.2%**.
  - Capture price, size, host quality, location, and amenities.
- **CLIP image embeddings**:
  - Modest performance (Hit@10 warm ≈ 2.7%).
  - Helpful for style/appearance but weaker for functional similarity alone.
- **ResNet embeddings**:
  - Near-random behavior (Hit@10 warm ≈ 0.9%).
  - Very high dimensionality and lack of text alignment make them noisy for this task.

**Conclusion**: **Text** is the dominant signal; structured features add complementary value; CLIP is modestly helpful; ResNet alone is not effective.

#### 6.2 Multi-Modal Concatenation

- **Struct + Text**:
  - Best performing concatenation baseline.
  - Hit@10 (warm) ≈ **12.4%**, clearly outperforming either modality alone.
- Adding **ResNet** to any combination generally **hurts performance**, likely due to:
  - Curse of dimensionality.
  - Noisy, high-dimensional signal dominating the embedding space.
- Including **CLIP** in naive concatenation is neutral or slightly mixed.

#### 6.3 Weighted Late Fusion

- The pipeline performs a **grid search** over weights \((α, β, γ)\) for **structured, text, CLIP** similarities:
  - Step 0.1 for each weight, with constraint \(α + β + γ = 1\).
  - 66 valid combinations evaluated.
- **Best weights found**:
  - \(α = 0.30\) (**Structured**),
  - \(β = 0.60\) (**Text**),
  - \(γ = 0.10\) (**CLIP**).
- **Performance**:
  - Hit@10 (warm) ≈ **15.0%**, compared to:
    - ~12.4% for Struct+Text concatenation.
    - Much lower for single modalities.
  - This represents a **substantial lift** and roughly **68× improvement over random** (Hit@10 random ≈ 0.22%).

**Interpretation**:

- Text drives ~60% of predictive power.
- Structured features (price, capacity, location, amenities) contribute ~30%.
- Visual CLIP signal is weak alone but adds a useful **10% marginal boost** in a weighted ensemble.
- **Late fusion** is more robust than concatenation because each modality keeps its own similarity geometry, avoiding issues with scale and dimensionality.

#### 6.4 Warm vs Cold Users

- **Warm users** (very small fraction of all users) benefit strongly:
  - Personalized, history-based embeddings produce meaningful ranking gains.
- **Cold users**:
  - With no training history, the fallback is a **global mean embedding**, which yields **near-random performance**.
  - This emphasizes the need for:
    - Alternative strategies (e.g., popularity, contextual bandits, geo-aware heuristics) for the majority of first-time users.

---

### 7. Full-System Outputs & Artifacts

Everything is organized under the project root:

- **EDA & features** (`eda_outputs_new/`):
  - `master_structured_features.json`: main tabular feature matrix.
  - `text_features.json`, `review_text_features.json`: listing and review text.
  - `user_listing_interactions.json`: binary interaction dataset.
  - `clip_features.pkl`, `resnet_features.pkl`: aligned image embeddings.
  - Multiple `.png` plots: price, room types, geography, amenities, correlations, text coverage, etc.
- **Recommendation pipeline outputs** (`recommendation_outputs_new/`):
  - `evaluation_results.csv`: metrics for all 17 embedding configurations.
  - `weighted_fusion_grid_search.csv`: full grid of weights and corresponding metrics.
  - `train_chrono.csv`, `test_chrono.csv`: chronological interaction splits.
  - `train_test_split_chrono.json`: summary stats of the split and warm/cold breakdown.
  - `fused_embeddings_256d.npy`: 256-d fused embeddings per listing.
  - `listing_ids.npy`: listing id index aligned with embeddings.
  - `emb_*.npy`: per-modality and combination embeddings.
  - `tfidf_vectorizer.pkl`, `scalers.pkl`: saved pre-processing models.
  - `recommendations_all_users.csv`: top-5 recommendations for ~145k users using fused embeddings.

---

### 8. How to Reproduce & Extend

#### 8.1 Reproducing the Pipeline

1. **Run EDA & feature engineering**:
   - Execute `eda_feature_engineering.py` to:
     - Load normalized JSON tables.
     - Run visual EDA, generate plots into `eda_outputs_new/`.
     - Build and save all structured, text, image, and interaction features.
2. **Run recommendation pipeline**:
   - Execute `recommendation_pipeline.py` to:
     - Load engineered features from `eda_outputs_new/`.
     - Build TF-IDF and multi-modal embeddings.
     - Perform chronological train/test split.
     - Run ablation over all modality configurations.
     - Run weighted late fusion grid search.
     - Save evaluation tables, embeddings, splits, and recommendations into `recommendation_outputs_new/`.

#### 8.2 Possible Extensions

- **Better cold-start handling**:
  - Incorporate popularity-based priors or context-aware rankings (e.g., by market, price range, travel season).
- **Richer text models**:
  - Replace TF-IDF with transformer-based embeddings (e.g., Sentence-BERT) for deeper semantics.
- **Learned fusion**:
  - Train a small neural net to learn fusion weights per user or per segment, instead of fixed global weights.
- **Online serving**:
  - Export `fused_embeddings_256d.npy` and indexes to a vector database (e.g., FAISS) to support real-time recommendations.

---

### 9. Exporting This Walkthrough to PDF

- This file (`walkthrough_pdf_report.md`) is designed to be **directly exported to PDF**.
- You can:
  - Open it in your editor and use the built-in **“Export as PDF”** / **“Print to PDF”** functionality, or
  - Use a command-line tool like `pandoc`, for example:

```bash
pandoc walkthrough_pdf_report.md -o walkthrough_pdf_report.pdf
```

This will produce a single PDF that documents the **entire pipeline**: EDA, feature engineering, text analytics, multi-modal modelling, training/testing setup, evaluation results, and how to reproduce/extend the system.

