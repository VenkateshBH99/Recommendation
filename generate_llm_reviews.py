"""
=======================================================================
LLM-Generated Reviews for Unreviewed Listings
=======================================================================

Generates 20,000 fully original reviews using Qwen2.5-1.5B (mlx-lm)
for the 1,632 listings that have zero reviews.

Each review is generated fresh by the LLM based on listing context
(market, room type, price, property type, amenities, name).
No templates, no sentence recombination — 100% LLM-authored text.

Output:
  airbnb_reviews_llm_unreviewed.json  — LLM-generated reviews for unreviewed listings
  
Usage:
  nohup /opt/anaconda3/bin/python generate_llm_reviews.py > llm_reviews.log 2>&1 &
"""

import json
import os
import random
import hashlib
import time
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("LLM REVIEW GENERATION FOR UNREVIEWED LISTINGS")
print("=" * 70)

# =====================================================================
# 1. LOAD DATA & IDENTIFY UNREVIEWED LISTINGS
# =====================================================================
print("\n[1/4] Loading data...")

original_reviews = []
with open(os.path.join(BASE_DIR, "m2m_translated_checkpoint.json")) as f:
    for line in f:
        original_reviews.append(json.loads(line))
reviews_df = pd.DataFrame(original_reviews)

with open(os.path.join(BASE_DIR, "airbnb_listings.json")) as f:
    listings = pd.DataFrame(json.load(f))
with open(os.path.join(BASE_DIR, "airbnb_address.json")) as f:
    address = pd.DataFrame(json.load(f))
with open(os.path.join(BASE_DIR, "amenities_df.json")) as f:
    amenities = pd.DataFrame(json.load(f))

listing_info = listings[["listing_id", "room_type", "property_type", "price",
                          "accommodates", "name", "bedrooms", "beds",
                          "bathrooms", "summary"]].merge(
    address[["listing_id", "market", "country", "suburb"]], on="listing_id"
)

# Top amenities per listing (for context)
top_amenities_per_listing = (
    amenities.groupby("listing_id")["amenity"]
    .apply(lambda x: ", ".join(x.head(8)))
    .to_dict()
)

all_listing_ids = set(listing_info["listing_id"])
reviewed_listing_ids = set(reviews_df["listing_id"].unique())
unreviewed_ids = sorted(all_listing_ids - reviewed_listing_ids)

lid_to_info = listing_info.set_index("listing_id").to_dict("index")

print(f"  Total listings: {len(all_listing_ids):,}")
print(f"  Reviewed: {len(reviewed_listing_ids):,}")
print(f"  Unreviewed: {len(unreviewed_ids):,}")

# =====================================================================
# 2. LOAD LLM
# =====================================================================
print("\n[2/4] Loading LLM...")

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
t0 = time.time()
model, tokenizer = load(MODEL_ID)
sampler = make_sampler(temp=0.8, top_p=0.9)
print(f"  Model loaded in {time.time() - t0:.1f}s")

# =====================================================================
# 3. GENERATE REVIEWS
# =====================================================================
print("\n[3/4] Generating LLM reviews...")

TARGET_REVIEWS = 2_000
n_unreviewed = len(unreviewed_ids)
base_per_listing = TARGET_REVIEWS // n_unreviewed
extra = TARGET_REVIEWS % n_unreviewed

print(f"  Target: {TARGET_REVIEWS:,} reviews across {n_unreviewed:,} listings "
      f"(~{base_per_listing} per listing)")

# Sentiment distribution: 60% positive, 25% neutral, 15% negative
SENTIMENTS = ["positive"] * 60 + ["neutral"] * 25 + ["negative"] * 15

SENTIMENT_INSTRUCTIONS = {
    "positive": "Write a positive, enthusiastic review. Mention things you loved.",
    "neutral": "Write a balanced, matter-of-fact review. Mention both pros and minor cons.",
    "negative": "Write a critical review. Mention specific issues and disappointments.",
}

# Max review ID from existing data
max_review_id = int(reviews_df["review_id"].astype(int).max()) + 200_000
review_id_counter = max_review_id + 1

generated_hashes = set()
llm_reviews = []
failed = 0
t_start = time.time()

for i, lid in enumerate(unreviewed_ids):
    n_reviews = base_per_listing + (1 if i < extra else 0)
    info = lid_to_info.get(lid, {})

    # Build listing context
    listing_name = info.get("name", "Airbnb listing")
    room_type = info.get("room_type", "Entire home/apt")
    prop_type = info.get("property_type", "Apartment")
    price = info.get("price", 100)
    market = info.get("market", "")
    city = info.get("suburb", market)
    country = info.get("country", "")
    accommodates = info.get("accommodates", 2)
    bedrooms = info.get("bedrooms", 1)
    beds = info.get("beds", 1)
    bathrooms = info.get("bathrooms", 1)
    summary = str(info.get("summary", ""))[:150]
    top_amen = top_amenities_per_listing.get(lid, "WiFi, Kitchen")

    context_str = (
        f"Listing: \"{listing_name}\"\n"
        f"Type: {room_type} ({prop_type})\n"
        f"Location: {city}, {country}\n"
        f"Price: ${price}/night\n"
        f"Accommodates: {accommodates} guests, {bedrooms} bedroom(s), "
        f"{beds} bed(s), {bathrooms} bathroom(s)\n"
        f"Amenities: {top_amen}"
    )
    if summary and summary != "nan" and len(summary) > 10:
        context_str += f"\nDescription: {summary}"

    for j in range(n_reviews):
        sentiment = random.choice(SENTIMENTS)
        sent_instruction = SENTIMENT_INSTRUCTIONS[sentiment]

        messages = [
            {"role": "system",
             "content": (
                 "You are an Airbnb guest who just stayed at a property. "
                 "Write a short, authentic review (3-5 sentences). "
                 "Be specific and natural — like a real traveler. "
                 "Output ONLY the review text, nothing else. "
                 "Do NOT include any labels, headers, or formatting."
             )},
            {"role": "user",
             "content": (
                 f"Here is the listing you stayed at:\n\n{context_str}\n\n"
                 f"{sent_instruction}\n\n"
                 f"Write your review:"
             )},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        try:
            response = generate(
                model, tokenizer, prompt=prompt,
                max_tokens=180, sampler=sampler, verbose=False,
            )
            text = response.strip()

            # Quality gate
            if (len(text) < 20 or len(text) > 1500
                    or "review" in text[:15].lower()
                    or "listing" in text[:15].lower()
                    or "here" in text[:10].lower()
                    or text.startswith('"') and text.endswith('"')):
                # Strip wrapping quotes if present
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1].strip()
                elif len(text) < 20:
                    failed += 1
                    continue

            # Uniqueness check
            h = hashlib.md5(text.encode()).hexdigest()
            if h in generated_hashes:
                failed += 1
                continue
            generated_hashes.add(h)

            synth_uid = f"llm_user_{review_id_counter}"
            synth_name = f"Traveler_{review_id_counter}"
            synth_date = random.randint(1256616000000, 1552276800000)

            llm_reviews.append({
                "listing_id": lid,
                "review_id": str(review_id_counter),
                "review_date": synth_date,
                "reviewer_id": synth_uid,
                "reviewer_name": synth_name,
                "comments": text,
                "source": "llm_generated",
                "sentiment": sentiment,
            })
            review_id_counter += 1

        except Exception as e:
            failed += 1

    # Progress reporting
    total_done = len(llm_reviews) + failed
    if (i + 1) % 100 == 0 or i == n_unreviewed - 1:
        elapsed = time.time() - t_start
        rate = len(llm_reviews) / elapsed if elapsed > 0 else 0
        eta = (TARGET_REVIEWS - len(llm_reviews)) / rate if rate > 0 else 0
        print(f"  [{i+1}/{n_unreviewed} listings] "
              f"{len(llm_reviews):,} reviews generated, "
              f"{failed} failed, "
              f"{rate:.1f} rev/s, "
              f"ETA {eta/3600:.1f}h")

elapsed_total = time.time() - t_start
print(f"\n  Generation complete:")
print(f"    Reviews generated: {len(llm_reviews):,}")
print(f"    Failed/rejected:   {failed}")
print(f"    Unique texts:      {len(generated_hashes):,}")
print(f"    Time:              {elapsed_total:.0f}s ({elapsed_total/3600:.1f}h)")

# Free memory
import gc
del model, tokenizer, sampler
gc.collect()
print("  Model unloaded")

# =====================================================================
# 4. VALIDATE & SAVE
# =====================================================================
print("\n[4/4] Validating and saving...")

llm_df = pd.DataFrame(llm_reviews)

# Stats
print(f"\n  --- LLM Review Stats ---")
print(f"  Total reviews:    {len(llm_df):,}")
print(f"  Unique listings:  {llm_df['listing_id'].nunique():,}")
print(f"  Unique texts:     {len(generated_hashes):,}")

word_counts = [len(r["comments"].split()) for r in llm_reviews]
print(f"  Word count: mean={np.mean(word_counts):.1f}, "
      f"median={np.median(word_counts):.0f}, "
      f"std={np.std(word_counts):.1f}")

# Sentiment distribution
sent_dist = llm_df["sentiment"].value_counts()
print(f"\n  Sentiment distribution:")
for s, c in sent_dist.items():
    print(f"    {s}: {c:,} ({c/len(llm_df)*100:.1f}%)")

# Sample reviews
print(f"\n  --- Sample LLM-Generated Reviews ---")
samples = llm_df.sample(min(5, len(llm_df)), random_state=42)
for _, row in samples.iterrows():
    lid = row["listing_id"]
    info = lid_to_info.get(lid, {})
    market = info.get("market", "?")
    print(f"\n  [{lid} / {market} / {row['sentiment']}]")
    print(f"  {row['comments'][:300]}")

# Save
output_path = os.path.join(BASE_DIR, "airbnb_reviews_llm_unreviewed.json")
# Drop the helper columns before saving
save_records = [{k: v for k, v in r.items() if k not in ("source", "sentiment")}
                for r in llm_reviews]
with open(output_path, "w") as f:
    json.dump(save_records, f, indent=2, ensure_ascii=False)

print(f"\n  ✓ Saved {len(save_records):,} LLM reviews to airbnb_reviews_llm_unreviewed.json")
print(f"  ✓ File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")

# Also save with metadata for analysis
meta_path = os.path.join(BASE_DIR, "airbnb_reviews_llm_unreviewed_meta.json")
with open(meta_path, "w") as f:
    json.dump(llm_reviews, f, indent=2, ensure_ascii=False)
print(f"  ✓ Saved with metadata to airbnb_reviews_llm_unreviewed_meta.json")

print("\n" + "=" * 70)
print("LLM REVIEW GENERATION COMPLETE")
print("=" * 70)
print(f"""
Method: 100% LLM-generated (Qwen2.5-1.5B-Instruct-4bit)
  → Each review generated from scratch using listing context
  → Context includes: name, type, location, price, amenities, description
  → Sentiment balanced: 60% positive, 25% neutral, 15% negative
  → Every review is unique (MD5 verified)

Output: airbnb_reviews_llm_unreviewed.json ({len(save_records):,} reviews)
Time:   {elapsed_total/3600:.1f} hours at {len(llm_reviews)/elapsed_total:.1f} rev/s
""")
