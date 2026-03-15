import json
import os
import pickle
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="husl", font_scale=1.1)
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["figure.dpi"] = 120

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "eda_outputs_new1")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# SECTION 1: DATA LOADING
# =====================================================================
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

with open(os.path.join(BASE_DIR, "airbnb_listings.json")) as f:
    listings = pd.DataFrame(json.load(f))
with open(os.path.join(BASE_DIR, "airbnb_hosts.json")) as f:
    hosts = pd.DataFrame(json.load(f))
with open(os.path.join(BASE_DIR, "airbnb_address.json")) as f:
    address = pd.DataFrame(json.load(f))
with open(os.path.join(BASE_DIR, "airbnb_images.json")) as f:
    images = pd.DataFrame(json.load(f))
with open(os.path.join(BASE_DIR, "airbnb_reviews.json")) as f:
    reviews = pd.DataFrame(json.load(f))
with open(os.path.join(BASE_DIR, "airbnb_amenities.json")) as f:
    amenities = pd.DataFrame(json.load(f))
with open(os.path.join(BASE_DIR, "listing_host_map.json")) as f:
    lh_map = pd.DataFrame(json.load(f))

# Load translated reviews (JSONL)
translated_reviews = []
with open(os.path.join(BASE_DIR, "airbnb_reviews_combined_llm_lang_ldjson.json")) as f:
    for line in f:
        translated_reviews.append(json.loads(line))
translated_reviews = pd.DataFrame(translated_reviews)

# Load pre-computed embeddings
with open(os.path.join(BASE_DIR, "clip_embeddings.pkl"), "rb") as f:
    clip_embeddings = pickle.load(f)
with open(os.path.join(BASE_DIR, "resnet50_embeddings.pkl"), "rb") as f:
    image_embeddings = pickle.load(f)

# Image filenames
image_files = set(
    int(f.replace(".jpg", ""))
    for f in os.listdir(os.path.join(BASE_DIR, "images"))
    if f.endswith(".jpg")
)

print(f"  Listings:            {len(listings):>7,} records × {len(listings.columns)} cols")
print(f"  Hosts:               {len(hosts):>7,} records × {len(hosts.columns)} cols")
print(f"  Addresses:           {len(address):>7,} records × {len(address.columns)} cols")
print(f"  Images (metadata):   {len(images):>7,} records × {len(images.columns)} cols")
print(f"  Reviews (raw):       {len(reviews):>7,} records × {len(reviews.columns)} cols")
print(f"  Reviews (translated):{len(translated_reviews):>7,} records × {len(translated_reviews.columns)} cols")
print(f"  Amenities:           {len(amenities):>7,} records × {len(amenities.columns)} cols")
print(f"  Listing-Host Map:    {len(lh_map):>7,} records × {len(lh_map.columns)} cols")
print(f"  Downloaded images:   {len(image_files):>7,}")
print(f"  CLIP embeddings:     {len(clip_embeddings):>7,} (dim=512)")
print(f"  ResNet embeddings:   {len(image_embeddings):>7,} (dim=2048)")


# =====================================================================
# SECTION 2: EXPLORATORY DATA ANALYSIS
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 2: EXPLORATORY DATA ANALYSIS")
print("=" * 70)


# --- 2.1 Listings Overview ---
print("\n--- 2.1 Listings Overview ---")
print(f"\nMissing values:\n{listings.isnull().sum()[listings.isnull().sum() > 0]}")
print(f"\nNumeric columns summary:")
numeric_cols = ["accommodates", "bedrooms", "beds", "bathrooms", "price", "minimum_nights", "maximum_nights"]
print(listings[numeric_cols].describe().round(2).to_string())


# --- 2.2 Price Distribution ---
print("\n--- 2.2 Price Distribution ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Raw price
axes[0].hist(listings["price"], bins=100, color="#4C72B0", edgecolor="black", alpha=0.7)
axes[0].set_title("Price Distribution (Raw)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Price ($)")
axes[0].set_ylabel("Count")

# Log price
listings["log_price"] = np.log1p(listings["price"])
axes[1].hist(listings["log_price"], bins=50, color="#55A868", edgecolor="black", alpha=0.7)
axes[1].set_title("Log(Price+1) Distribution", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Log(Price)")
axes[1].set_ylabel("Count")

# Price capped at 99th percentile
cap = listings["price"].quantile(0.99)
axes[2].hist(listings[listings["price"] <= cap]["price"], bins=50, color="#C44E52", edgecolor="black", alpha=0.7)
axes[2].set_title(f"Price ≤ ${cap:.0f} (99th pctl)", fontsize=13, fontweight="bold")
axes[2].set_xlabel("Price ($)")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "price_distribution.png"), bbox_inches="tight")
plt.close()
print("  → Saved price_distribution.png")


# --- 2.3 Property Type & Room Type ---
print("\n--- 2.3 Property & Room Type ---")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

prop_counts = listings["property_type"].value_counts().head(10)
prop_counts.plot.barh(ax=axes[0], color=sns.color_palette("viridis", len(prop_counts)))
axes[0].set_title("Top 10 Property Types", fontsize=13, fontweight="bold")
axes[0].invert_yaxis()

room_counts = listings["room_type"].value_counts()
colors = ["#4C72B0", "#55A868", "#C44E52"]
room_counts.plot.pie(ax=axes[1], autopct="%1.1f%%", colors=colors, startangle=90)
axes[1].set_title("Room Type Distribution", fontsize=13, fontweight="bold")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "property_room_types.png"), bbox_inches="tight")
plt.close()
print("  → Saved property_room_types.png")


# --- 2.4 Geographic Distribution ---
print("\n--- 2.4 Geographic Distribution ---")
merged_geo = listings.merge(address, on="listing_id")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

market_counts = merged_geo["market"].value_counts().head(12)
market_counts.plot.barh(ax=axes[0], color=sns.color_palette("coolwarm", len(market_counts)))
axes[0].set_title("Listings by Market", fontsize=13, fontweight="bold")
axes[0].invert_yaxis()

country_counts = merged_geo["country"].value_counts()
country_counts.plot.barh(ax=axes[1], color=sns.color_palette("Set2", len(country_counts)))
axes[1].set_title("Listings by Country", fontsize=13, fontweight="bold")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "geographic_distribution.png"), bbox_inches="tight")
plt.close()
print("  → Saved geographic_distribution.png")

# Geo scatter
fig, ax = plt.subplots(figsize=(14, 8))
scatter = ax.scatter(
    merged_geo["longitude"], merged_geo["latitude"],
    c=merged_geo["price"].clip(upper=cap), cmap="YlOrRd", s=8, alpha=0.6
)
plt.colorbar(scatter, label="Price ($)")
ax.set_title("Listings Geospatial Distribution (colored by price)", fontsize=14, fontweight="bold")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "geo_scatter.png"), bbox_inches="tight")
plt.close()
print("  → Saved geo_scatter.png")


# --- 2.5 Accommodates / Bedrooms / Beds / Bathrooms ---
print("\n--- 2.5 Room Capacity Features ---")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, col in zip(axes.flat, ["accommodates", "bedrooms", "beds", "bathrooms"]):
    data = listings[col].dropna()
    ax.hist(data, bins=range(int(data.min()), int(data.max()) + 2), color="#4C72B0", edgecolor="black", alpha=0.7)
    ax.set_title(f"{col.title()} Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel(col.title())
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "room_capacity.png"), bbox_inches="tight")
plt.close()
print("  → Saved room_capacity.png")


# --- 2.6 Host Analysis ---
print("\n--- 2.6 Host Analysis ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Superhost
sh = hosts["host_is_superhost"].value_counts()
sh.index = ["No", "Yes"]
sh.plot.bar(ax=axes[0], color=["#C44E52", "#55A868"])
axes[0].set_title("Superhost Distribution", fontsize=12, fontweight="bold")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# Response rate
valid_rr = hosts["host_response_rate"].dropna()
axes[1].hist(valid_rr, bins=20, color="#4C72B0", edgecolor="black", alpha=0.7)
axes[1].set_title("Host Response Rate (%)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Response Rate")

# Response time
rt = hosts["host_response_time"].dropna().value_counts()
rt.plot.barh(ax=axes[2], color=sns.color_palette("Set2", len(rt)))
axes[2].set_title("Response Time", fontsize=12, fontweight="bold")
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "host_analysis.png"), bbox_inches="tight")
plt.close()
print("  → Saved host_analysis.png")


# --- 2.7 Review Analysis ---
print("\n--- 2.7 Review Analysis ---")
review_counts = translated_reviews.groupby("listing_id").size().reset_index(name="review_count")
reviewer_counts = translated_reviews.groupby("reviewer_id").size().reset_index(name="reviews_given")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Reviews per listing
axes[0].hist(review_counts["review_count"].clip(upper=200), bins=50, color="#4C72B0", edgecolor="black", alpha=0.7)
axes[0].set_title("Reviews per Listing", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Number of Reviews")

# Reviews per reviewer (cold start visualization)
rc_dist = reviewer_counts["reviews_given"].value_counts().sort_index().head(10)
rc_dist.plot.bar(ax=axes[1], color="#C44E52")
axes[1].set_title("Reviews per Reviewer\n(98.2% have only 1 review)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Number of Reviews Given")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

# Review language distribution
lang_counts = translated_reviews["detected_language"].value_counts().head(10)
lang_counts.plot.barh(ax=axes[2], color=sns.color_palette("tab10", len(lang_counts)))
axes[2].set_title("Top 10 Review Languages", fontsize=12, fontweight="bold")
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "review_analysis.png"), bbox_inches="tight")
plt.close()
print("  → Saved review_analysis.png")


# --- 2.8 Amenities Analysis ---
print("\n--- 2.8 Amenities Analysis ---")
amenity_counts = amenities["amenity"].value_counts()
amenities_per_listing = amenities.groupby("listing_id").size().reset_index(name="num_amenities")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

amenity_counts.head(25).plot.barh(ax=axes[0], color=sns.color_palette("viridis", 25))
axes[0].set_title("Top 25 Amenities", fontsize=12, fontweight="bold")
axes[0].invert_yaxis()
axes[0].set_xlabel("Number of Listings")

axes[1].hist(amenities_per_listing["num_amenities"], bins=40, color="#55A868", edgecolor="black", alpha=0.7)
axes[1].set_title("Amenities per Listing", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Number of Amenities")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "amenities_analysis.png"), bbox_inches="tight")
plt.close()
print("  → Saved amenities_analysis.png")


# --- 2.9 Text Field Coverage ---
print("\n--- 2.9 Text Field Coverage ---")
text_cols = ["name", "summary", "description", "space", "neighborhood_overview",
             "notes", "transit", "access", "interaction", "house_rules"]

text_stats = []
for col in text_cols:
    non_empty = listings[col].str.strip() != ""
    lengths = listings.loc[non_empty, col].str.len()
    text_stats.append({
        "field": col,
        "fill_rate_%": non_empty.mean() * 100,
        "avg_length": lengths.mean(),
        "median_length": lengths.median(),
    })
text_stats_df = pd.DataFrame(text_stats)
print(text_stats_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].barh(text_stats_df["field"], text_stats_df["fill_rate_%"], color=sns.color_palette("RdYlGn", len(text_stats_df)))
axes[0].set_title("Text Field Fill Rate (%)", fontsize=12, fontweight="bold")
axes[0].set_xlim(0, 105)
axes[0].invert_yaxis()

axes[1].barh(text_stats_df["field"], text_stats_df["avg_length"], color=sns.color_palette("coolwarm", len(text_stats_df)))
axes[1].set_title("Average Text Length (chars)", fontsize=12, fontweight="bold")
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "text_coverage.png"), bbox_inches="tight")
plt.close()
print("  → Saved text_coverage.png")


# --- 2.10 Image & Embedding Coverage ---
print("\n--- 2.10 Image & Embedding Coverage ---")
all_listing_ids = set(listings["listing_id"])
ids_with_images = all_listing_ids & image_files
ids_with_clip = all_listing_ids & set(clip_embeddings.keys())
ids_with_resnet = all_listing_ids & set(image_embeddings.keys())
ids_with_reviews_set = set(translated_reviews["listing_id"].unique())

coverage = {
    "Total Listings": len(all_listing_ids),
    "With Downloaded Image": len(ids_with_images),
    "With CLIP (512-d)": len(ids_with_clip),
    "With ResNet (2048-d)": len(ids_with_resnet),
    "With Reviews": len(ids_with_reviews_set & all_listing_ids),
    "With Image + Reviews": len(ids_with_images & ids_with_reviews_set),
    "With ALL modalities": len(ids_with_images & ids_with_reviews_set & ids_with_clip),
}
for k, v in coverage.items():
    print(f"  {k:30s}: {v:5d} ({v/len(all_listing_ids)*100:.1f}%)")


# --- 2.11 Correlation Analysis ---
print("\n--- 2.11 Feature Correlations ---")
merged_full = listings.merge(address, on="listing_id").merge(lh_map, on="listing_id").merge(
    hosts.astype({"host_id": str}), on="host_id", how="left"
)
merged_full = merged_full.merge(review_counts, on="listing_id", how="left")
merged_full = merged_full.merge(amenities_per_listing, on="listing_id", how="left")
merged_full["review_count"] = merged_full["review_count"].fillna(0)
merged_full["has_image"] = merged_full["listing_id"].isin(image_files).astype(int)

corr_cols = ["price", "accommodates", "bedrooms", "beds", "bathrooms",
             "minimum_nights", "review_count", "num_amenities",
             "host_is_superhost", "host_response_rate", "host_total_listings_count"]
merged_full["host_is_superhost"] = merged_full["host_is_superhost"].astype(int)

fig, ax = plt.subplots(figsize=(12, 10))
corr_matrix = merged_full[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, square=True, linewidths=0.5)
ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"), bbox_inches="tight")
plt.close()
print("  → Saved correlation_matrix.png")


# --- 2.12 Price vs Features ---
print("\n--- 2.12 Price vs Key Features ---")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for ax, col in zip(axes.flat, ["accommodates", "bedrooms", "room_type", "property_type", "cancellation_policy", "num_amenities"]):
    if col in ["room_type", "property_type", "cancellation_policy"]:
        if col == "property_type":
            top_vals = merged_full[col].value_counts().head(6).index
            data = merged_full[merged_full[col].isin(top_vals)]
        else:
            data = merged_full
        data.boxplot(column="price", by=col, ax=ax, showfliers=False)
        ax.set_title(f"Price by {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Price ($)")
        plt.sca(ax)
        plt.xticks(rotation=35, ha="right")
    else:
        ax.scatter(merged_full[col], merged_full["price"].clip(upper=cap), alpha=0.2, s=5)
        ax.set_title(f"Price vs {col}", fontsize=11, fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Price ($)")

plt.suptitle("")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "price_vs_features.png"), bbox_inches="tight")
plt.close()
print("  → Saved price_vs_features.png")


# =====================================================================
# SECTION 3: FEATURE ENGINEERING
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 3: FEATURE ENGINEERING FOR TEXT-IMAGE RECOMMENDATION")
print("=" * 70)
print("""
RECOMMENDATION APPROACH: Multi-Modal Content-Based Filtering
(with text + image + structured features)

Rationale:
  - 98.2% of reviewers have only 1 review → collaborative filtering alone fails
  - Rich text (descriptions, translated reviews) → NLP embeddings
  - 4,545 listing images → visual embeddings (CLIP 512-d, ResNet 2048-d)
  - 186 amenities + structured attributes → tabular features
  - Combined multi-modal embeddings enable similarity-based recommendation
""")


# === 3.1 Structured Listing Features ===
print("--- 3.1 Structured Listing Features ---")

listing_features = listings[["listing_id"]].copy()

# Numeric features (fill NaN with median)
for col in ["accommodates", "bedrooms", "beds", "bathrooms", "price", "minimum_nights", "maximum_nights"]:
    listing_features[col] = listings[col].fillna(listings[col].median())

# Log-transform skewed features
listing_features["log_price"] = np.log1p(listing_features["price"])
listing_features["log_min_nights"] = np.log1p(listing_features["minimum_nights"])
listing_features["log_max_nights"] = np.log1p(listing_features["maximum_nights"])

# Price bins (for categorical price-tier matching)
listing_features["price_tier"] = pd.qcut(
    listing_features["price"], q=5,
    labels=["budget", "economy", "mid-range", "premium", "luxury"]
)

# Room type one-hot encoding
room_dummies = pd.get_dummies(listings["room_type"], prefix="room")
listing_features = pd.concat([listing_features, room_dummies], axis=1)

# Property type (top categories + "Other")
top_props = listings["property_type"].value_counts().head(8).index
listings["property_type_grouped"] = listings["property_type"].where(
    listings["property_type"].isin(top_props), "Other"
)
prop_dummies = pd.get_dummies(listings["property_type_grouped"], prefix="prop")
listing_features = pd.concat([listing_features, prop_dummies], axis=1)

# Cancellation policy encoding (ordinal)
cancel_map = {
    "flexible": 1, "moderate": 2,
    "strict_14_with_grace_period": 3,
    "super_strict_30": 4, "super_strict_60": 5
}
listing_features["cancel_strictness"] = listings["cancellation_policy"].map(cancel_map).fillna(3)

# Bed type encoding
bed_map = {"Real Bed": 1, "Pull-out Sofa": 0, "Futon": 0, "Couch": 0, "Airbed": 0}
listing_features["is_real_bed"] = listings["bed_type"].map(bed_map).fillna(1)

print(f"  Structured features: {listing_features.shape[1] - 1} columns")
print(f"  Columns: {list(listing_features.columns[1:])}")


# === 3.2 Host Features ===
print("\n--- 3.2 Host Features ---")

host_features = lh_map.copy()
hosts_typed = hosts.copy()
hosts_typed["host_id"] = hosts_typed["host_id"].astype(str)
host_features["host_id"] = host_features["host_id"].astype(str)
host_features = host_features.merge(hosts_typed, on="host_id", how="left")

host_features["is_superhost"] = host_features["host_is_superhost"].astype(int)
host_features["response_rate"] = host_features["host_response_rate"].fillna(
    host_features["host_response_rate"].median()
)

# Response time ordinal
rt_map = {"within an hour": 4, "within a few hours": 3, "within a day": 2, "a few days or more": 1}
host_features["response_speed"] = host_features["host_response_time"].map(rt_map).fillna(2)

# Multi-listing host flag (professional host indicator)
host_features["is_multi_listing_host"] = (host_features["host_total_listings_count"] > 1).astype(int)
host_features["log_listing_count"] = np.log1p(host_features["host_total_listings_count"])

# Host has "about" text
host_features["has_host_about"] = (host_features["host_about"].str.strip() != "").astype(int)

host_feat_cols = ["listing_id", "is_superhost", "response_rate", "response_speed",
                  "is_multi_listing_host", "log_listing_count", "has_host_about"]
host_features = host_features[host_feat_cols]
print(f"  Host features: {len(host_feat_cols) - 1} columns")


# === 3.3 Location Features ===
print("\n--- 3.3 Location Features ---")

location_features = address[["listing_id", "longitude", "latitude"]].copy()

# Market one-hot (important for geographical matching)
market_dummies = pd.get_dummies(address["market"], prefix="market")
location_features = pd.concat([location_features, market_dummies], axis=1)

# Country one-hot
country_dummies = pd.get_dummies(address["country_code"], prefix="country")
location_features = pd.concat([location_features, country_dummies], axis=1)

# Government area (high cardinality → use for neighborhood embedding later)
location_features["government_area"] = address["government_area"]

print(f"  Location features: {location_features.shape[1] - 1} columns")


# === 3.4 Amenity Features ===
print("\n--- 3.4 Amenity Features ---")

# Binary amenity matrix for top-N amenities
TOP_N_AMENITIES = 40
top_amenities = amenities["amenity"].value_counts().head(TOP_N_AMENITIES).index.tolist()

amenity_pivot = amenities[amenities["amenity"].isin(top_amenities)].copy()
amenity_pivot["present"] = 1
amenity_matrix = amenity_pivot.pivot_table(
    index="listing_id", columns="amenity", values="present", fill_value=0
).reset_index()
amenity_matrix.columns = ["listing_id"] + [f"amenity_{c.lower().replace(' ', '_').replace('/', '_')}" for c in amenity_matrix.columns[1:]]

# Total amenity count
amenity_total = amenities.groupby("listing_id").size().reset_index(name="total_amenities")
amenity_matrix = amenity_matrix.merge(amenity_total, on="listing_id", how="left")

# Amenity category features (grouped)
safety_amenities = {"Smoke detector", "Carbon monoxide detector", "Fire extinguisher", "First aid kit", "Lock on bedroom door"}
comfort_amenities = {"Air conditioning", "Heating", "Hair dryer", "Iron", "Hot water", "Washer", "Dryer"}
kitchen_amenities = {"Kitchen", "Refrigerator", "Microwave", "Cooking basics", "Dishes and silverware", "Stove", "Oven", "Coffee maker"}
entertainment_amenities = {"TV", "Cable TV", "Internet", "Wifi", "Laptop friendly workspace"}
family_amenities = {"Family/kid friendly", "Crib", "High chair", "Children's books and toys"}

listing_amenity_sets = amenities.groupby("listing_id")["amenity"].apply(set)
category_features = pd.DataFrame({"listing_id": listing_amenity_sets.index})
category_features["safety_score"] = listing_amenity_sets.apply(lambda x: len(x & safety_amenities) / len(safety_amenities)).values
category_features["comfort_score"] = listing_amenity_sets.apply(lambda x: len(x & comfort_amenities) / len(comfort_amenities)).values
category_features["kitchen_score"] = listing_amenity_sets.apply(lambda x: len(x & kitchen_amenities) / len(kitchen_amenities)).values
category_features["entertainment_score"] = listing_amenity_sets.apply(lambda x: len(x & entertainment_amenities) / len(entertainment_amenities)).values
category_features["family_score"] = listing_amenity_sets.apply(lambda x: len(x & family_amenities) / len(family_amenities)).values

amenity_matrix = amenity_matrix.merge(category_features, on="listing_id", how="left")
print(f"  Amenity features: {amenity_matrix.shape[1] - 1} columns (top {TOP_N_AMENITIES} binary + 5 category scores + total count)")


# === 3.5 Review / Text Features ===
print("\n--- 3.5 Review / Text Features ---")

# Review statistics per listing
review_stats = translated_reviews.groupby("listing_id").agg(
    review_count=("review_id", "count"),
    unique_reviewers=("reviewer_id", "nunique"),
    first_review_ts=("review_date", "min"),
    last_review_ts=("review_date", "max"),
).reset_index()

review_stats["review_span_days"] = (review_stats["last_review_ts"] - review_stats["first_review_ts"]) / (1000 * 60 * 60 * 24)
review_stats["reviews_per_month"] = np.where(
    review_stats["review_span_days"] > 0,
    review_stats["review_count"] / (review_stats["review_span_days"] / 30),
    0
)
review_stats["log_review_count"] = np.log1p(review_stats["review_count"])
review_stats = review_stats.drop(columns=["first_review_ts", "last_review_ts"])

# Combined listing text (for text embedding)
text_features = listings[["listing_id"]].copy()
text_features["combined_text"] = (
    listings["name"].fillna("") + " | " +
    listings["summary"].fillna("") + " | " +
    listings["description"].fillna("") + " | " +
    listings["space"].fillna("") + " | " +
    listings["neighborhood_overview"].fillna("") + " | " +
    listings["transit"].fillna("")
)
text_features["text_length"] = text_features["combined_text"].str.len()
text_features["word_count"] = text_features["combined_text"].str.split().str.len()

# Aggregated translated review text per listing (for review-based text embedding)
translated_review_text = translated_reviews.groupby("listing_id")["translated_comments"].apply(
    lambda x: " ".join(x.dropna().astype(str).head(50))  # cap at 50 reviews to manage size
).reset_index()
translated_review_text.columns = ["listing_id", "aggregated_review_text"]
translated_review_text["review_text_length"] = translated_review_text["aggregated_review_text"].str.len()

print(f"  Review stats: {review_stats.shape[1] - 1} columns")
print(f"  Text features: combined listing text + aggregated review text")
print(f"  Listings with combined text: {(text_features['text_length'] > 10).sum()}")
print(f"  Listings with aggregated reviews: {len(translated_review_text)}")


# === 3.6 Image Features ===
print("\n--- 3.6 Image / Visual Features ---")

# CLIP embeddings DataFrame (512-d)
clip_ids = sorted(set(clip_embeddings.keys()) & all_listing_ids)
clip_matrix = np.stack([clip_embeddings[lid] for lid in clip_ids])
clip_df = pd.DataFrame(clip_matrix, columns=[f"clip_{i}" for i in range(512)])
clip_df.insert(0, "listing_id", clip_ids)

# ResNet embeddings DataFrame (2048-d)
resnet_ids = sorted(set(image_embeddings.keys()) & all_listing_ids)
resnet_matrix = np.stack([image_embeddings[lid] for lid in resnet_ids])
resnet_df = pd.DataFrame(resnet_matrix, columns=[f"resnet_{i}" for i in range(2048)])
resnet_df.insert(0, "listing_id", resnet_ids)

# Image availability flag
image_flag = pd.DataFrame({
    "listing_id": list(all_listing_ids),
    "has_image": [1 if lid in image_files else 0 for lid in all_listing_ids],
    "has_clip_embedding": [1 if lid in set(clip_embeddings.keys()) else 0 for lid in all_listing_ids],
    "has_resnet_embedding": [1 if lid in set(image_embeddings.keys()) else 0 for lid in all_listing_ids],
})

print(f"  CLIP embeddings: {len(clip_ids)} listings × 512 dims")
print(f"  ResNet embeddings: {len(resnet_ids)} listings × 2048 dims")
print(f"  Listings with image flag: {len(image_flag)}")


# === 3.7 User-Listing Interaction Matrix ===
print("\n--- 3.7 User-Listing Interaction Matrix ---")

# Binary interaction (implicit feedback: user reviewed a listing = positive interaction)
interactions = translated_reviews[["reviewer_id", "listing_id"]].drop_duplicates()
interactions["interaction"] = 1

print(f"  Unique interactions: {len(interactions):,}")
print(f"  Unique users: {interactions['reviewer_id'].nunique():,}")
print(f"  Unique listings: {interactions['listing_id'].nunique():,}")
print(f"  Sparsity: {1 - len(interactions) / (interactions['reviewer_id'].nunique() * interactions['listing_id'].nunique()):.6f}")
print(f"  → Matrix is 99.9997% sparse → Content-based approach is essential!")


# =====================================================================
# SECTION 4: MERGE & SAVE ENGINEERED FEATURES
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 4: MERGING & SAVING ENGINEERED FEATURES")
print("=" * 70)

# Master structured feature table (one row per listing)
master_features = listing_features.copy()
master_features = master_features.merge(host_features, on="listing_id", how="left")
master_features = master_features.merge(location_features, on="listing_id", how="left")
master_features = master_features.merge(amenity_matrix, on="listing_id", how="left")
master_features = master_features.merge(review_stats, on="listing_id", how="left")
master_features = master_features.merge(image_flag, on="listing_id", how="left")

# Fill NaN for listings without reviews
for col in ["review_count", "unique_reviewers", "review_span_days", "reviews_per_month", "log_review_count"]:
    if col in master_features.columns:
        master_features[col] = master_features[col].fillna(0)

# Fill NaN amenity columns
amenity_cols = [c for c in master_features.columns if c.startswith("amenity_")]
for col in amenity_cols + ["total_amenities", "safety_score", "comfort_score", "kitchen_score", "entertainment_score", "family_score"]:
    if col in master_features.columns:
        master_features[col] = master_features[col].fillna(0)

print(f"\nMaster Feature Table: {master_features.shape[0]} listings × {master_features.shape[1]} columns")
print(f"Columns: {list(master_features.columns)}")

# Save all outputs
master_features.to_json(os.path.join(OUTPUT_DIR, "master_structured_features.json"), orient="records", indent=2)
text_features.to_json(os.path.join(OUTPUT_DIR, "text_features.json"), orient="records")
translated_review_text.to_json(os.path.join(OUTPUT_DIR, "review_text_features.json"), orient="records")
interactions.to_json(os.path.join(OUTPUT_DIR, "user_listing_interactions.json"), orient="records")

# Save CLIP and ResNet embeddings as separate files
clip_df.to_pickle(os.path.join(OUTPUT_DIR, "clip_features.pkl"))
resnet_df.to_pickle(os.path.join(OUTPUT_DIR, "resnet_features.pkl"))

print(f"\n✓ Saved master_structured_features.json ({master_features.shape[1]} cols)")
print(f"✓ Saved text_features.json")
print(f"✓ Saved review_text_features.json")
print(f"✓ Saved user_listing_interactions.json")
print(f"✓ Saved clip_features.pkl (512-d)")
print(f"✓ Saved resnet_features.pkl (2048-d)")


# =====================================================================
# SECTION 5: FEATURE SUMMARY FOR RECOMMENDATION
# =====================================================================
print("\n" + "=" * 70)
print("SECTION 5: FEATURE SUMMARY FOR TEXT-IMAGE RECOMMENDATION")
print("=" * 70)

summary = """
╔══════════════════════════════════════════════════════════════════════╗
║           FEATURES FOR TEXT-IMAGE RECOMMENDATION SYSTEM             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  A. STRUCTURED FEATURES (per listing)                                ║
║  ───────────────────────────────────────────────────────────────────  ║
║  • Listing:    accommodates, bedrooms, beds, bathrooms, log_price,   ║
║               price_tier, room_type(OHE), property_type(OHE),        ║
║               cancel_strictness, is_real_bed, min/max_nights         ║
║  • Host:      is_superhost, response_rate, response_speed,           ║
║               is_multi_listing_host, has_host_about                  ║
║  • Location:  lat/lon, market(OHE), country(OHE)                     ║
║  • Amenities: top-40 binary + 5 category scores + total_count        ║
║  • Reviews:   review_count, reviews_per_month, review_span_days      ║
║                                                                      ║
║  B. TEXT FEATURES (for NLP embedding)                                ║
║  ───────────────────────────────────────────────────────────────────  ║
║  • Combined listing text: name + summary + description + space +     ║
║    neighborhood_overview + transit                                    ║
║  • Aggregated translated review text (top 50 per listing)            ║
║  → Generate embeddings using: Sentence-BERT / CLIP text encoder      ║
║                                                                      ║
║  C. IMAGE FEATURES (pre-computed)                                    ║
║  ───────────────────────────────────────────────────────────────────  ║
║  • CLIP embeddings (512-d): aligned text-image space                 ║
║  • ResNet embeddings (2048-d): deep visual features                  ║
║  → 4,545 listings have image embeddings                              ║
║                                                                      ║
║  D. USER INTERACTION DATA                                            ║
║  ───────────────────────────────────────────────────────────────────  ║
║  • Binary interaction matrix (reviewer → listing)                    ║
║  • 146,640 users × 3,923 listings                                    ║
║  • 99.9997% sparse → content-based is primary strategy               ║
║                                                                      ║
║  RECOMMENDED APPROACH:                                               ║
║  ─────────────────────                                               ║
║  1. Text embedding:   Encode listing text via SBERT/CLIP text enc.   ║
║  2. Image embedding:  Use CLIP 512-d (text-image aligned space)      ║
║  3. Structured:       Normalize & concat structured features         ║
║  4. Multi-modal fusion: Concatenate or weighted-sum all modalities   ║
║  5. Similarity search: Cosine similarity / FAISS for fast retrieval  ║
║  6. Optional hybrid:  Add collaborative signal for repeat users      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
print(summary)

print("=" * 70)
print("EDA & FEATURE ENGINEERING COMPLETE!")
print(f"All outputs saved to: {OUTPUT_DIR}")
print("=" * 70)
