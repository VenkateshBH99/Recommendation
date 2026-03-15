"""
Generate balanced synthetic reviews for the 1,632 listings that have zero reviews.
Creates new synthetic reviewer IDs and assigns 3-8 reviews per listing.
Output: appended to airbnb_reviews_balanced.json (overwritten with combined).
"""

import json
import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict

random.seed(123)
np.random.seed(123)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("GENERATING REVIEWS FOR UNREVIEWED LISTINGS")
print("=" * 70)

# Load data
with open(os.path.join(BASE_DIR, "airbnb_reviews.json")) as f:
    original_reviews = json.load(f)
with open(os.path.join(BASE_DIR, "airbnb_reviews_balanced.json")) as f:
    existing_synthetic = json.load(f)
with open(os.path.join(BASE_DIR, "airbnb_listings.json")) as f:
    listings = json.load(f)

original_df = pd.DataFrame(original_reviews)
all_listing_ids = set(r["listing_id"] for r in listings)
reviewed_ids = set(original_df["listing_id"].unique())
unreviewed_ids = sorted(all_listing_ids - reviewed_ids)

print(f"  Total listings: {len(all_listing_ids)}")
print(f"  Already reviewed: {len(reviewed_ids)}")
print(f"  Unreviewed: {len(unreviewed_ids)}")

# Get max IDs from all existing data to avoid collisions
all_existing = original_reviews + existing_synthetic
max_review_id = max(int(r["review_id"]) for r in all_existing)
max_reviewer_id = max(int(r["reviewer_id"]) for r in all_existing)

print(f"  Max existing review_id: {max_review_id}")
print(f"  Max existing reviewer_id: {max_reviewer_id}")

# Balanced review templates (60% positive, 25% neutral, 15% negative)
TEMPLATES = {
    "positive": [
        "Great place to stay! Exactly as described and very clean.",
        "Wonderful experience. The host was very responsive and helpful.",
        "Loved it! Would definitely recommend to anyone visiting the area.",
        "Perfect location, clean and comfortable. Everything we needed.",
        "Amazing stay! The apartment was beautiful and well-equipped.",
        "Very nice place, great value for money. Would come back again.",
        "Fantastic! The listing was accurate and the neighborhood was lovely.",
        "Really enjoyed our stay. Comfortable, clean, and well-located.",
        "Excellent accommodation! Host was very welcoming and communicative.",
        "Had a wonderful time. The space was exactly what we were looking for.",
        "Perfect location, walking distance to everything we wanted to see.",
        "Incredible location! Right in the heart of the city.",
        "Great value for the price. Very clean and well-maintained.",
        "Spacious apartment with all the amenities we needed.",
        "Cozy and charming place. Felt like a home away from home.",
    ],
    "neutral": [
        "Nice place, good for the price. A few minor things but overall fine.",
        "Decent stay. The location was good but the space was a bit small.",
        "It was okay. Clean and functional but nothing special.",
        "Good enough for a short stay. Met our basic needs.",
        "Average experience. The listing was mostly accurate.",
        "The place was fine. Nothing stood out but nothing was wrong either.",
        "Mixed feelings. Some things were great, others could be improved.",
        "Adequate for the price. Would consider alternatives next time.",
        "The stay was alright. Check-in was smooth but the space felt dated.",
        "Reasonable accommodation. Not luxury but served its purpose.",
    ],
    "negative": [
        "Disappointing stay. The photos were misleading and the place was smaller than expected.",
        "Not great. The apartment was noisy and the walls were thin.",
        "Would not recommend. Cleanliness was below expectations.",
        "The host was unresponsive and check-in was very difficult.",
        "Overpriced for what you get. Many better options in the area.",
        "Had issues with the heating/AC and the host took too long to respond.",
        "The neighborhood felt unsafe at night. Would not stay here again.",
        "Listing description was inaccurate. Missing several advertised amenities.",
        "Too noisy to sleep well. Street noise was constant throughout the night.",
        "Uncomfortable beds and outdated furniture. Not worth the price.",
        "Plumbing issues during our stay. Hot water was unreliable.",
        "The place smelled musty and wasn't properly cleaned before arrival.",
        "WiFi didn't work and the host didn't seem to care.",
        "Bugs in the apartment. Very unpleasant experience overall.",
        "Way too far from public transport despite the listing saying otherwise.",
    ],
}

# Weighted pool: 60% positive, 25% neutral, 15% negative
WEIGHTED_POOL = []
for _ in range(12):
    WEIGHTED_POOL.extend(TEMPLATES["positive"])
for _ in range(8):
    WEIGHTED_POOL.extend(TEMPLATES["neutral"])
for _ in range(3):
    WEIGHTED_POOL.extend(TEMPLATES["negative"])

# Reviewer name pool (diverse international names)
FIRST_NAMES = [
    "James", "Maria", "Chen", "Fatima", "Yuki", "Oliver", "Sofia", "Mohammed",
    "Priya", "Lucas", "Emma", "Wei", "Ana", "Kenji", "Isabella", "Ahmed",
    "Lena", "Carlos", "Mei", "David", "Sarah", "Hiroshi", "Olga", "Pierre",
    "Nina", "Thomas", "Aisha", "Diego", "Sakura", "Michael", "Julia", "Raj",
    "Elena", "Hans", "Chloe", "Ali", "Ingrid", "Marco", "Yuna", "Alex",
    "Katarina", "Ravi", "Camille", "Sven", "Lucia", "Boris", "Hana", "Felix",
    "Nadia", "Jorge", "Mika", "Teresa", "Viktor", "Lina", "Bruno", "Amara",
    "Leo", "Rosa", "Sam", "Daria", "Paolo", "Min", "Zara", "Ivan",
]

# Date range from dataset (2009-10-27 to 2019-03-11 in ms)
DATE_MIN = 1256616000000
DATE_MAX = 1552276800000

# Generate reviews for unreviewed listings
print(f"\nGenerating reviews for {len(unreviewed_ids)} unreviewed listings...")

new_reviews = []
review_id_counter = max_review_id + 1
reviewer_id_counter = max_reviewer_id + 1

REVIEWS_PER_LISTING_MIN = 3
REVIEWS_PER_LISTING_MAX = 8

for i, listing_id in enumerate(unreviewed_ids):
    n_reviews = random.randint(REVIEWS_PER_LISTING_MIN, REVIEWS_PER_LISTING_MAX)

    for _ in range(n_reviews):
        # Create a new unique reviewer
        reviewer_name = random.choice(FIRST_NAMES)
        review_date = random.randint(DATE_MIN, DATE_MAX)
        comment = random.choice(WEIGHTED_POOL)

        new_reviews.append({
            "listing_id": listing_id,
            "review_id": str(review_id_counter),
            "review_date": review_date,
            "reviewer_id": str(reviewer_id_counter),
            "reviewer_name": reviewer_name,
            "comments": comment,
        })
        review_id_counter += 1
        reviewer_id_counter += 1

    if (i + 1) % 500 == 0:
        print(f"  Processed {i+1}/{len(unreviewed_ids)} listings, {len(new_reviews)} reviews so far...")

print(f"  Generated {len(new_reviews):,} reviews for {len(unreviewed_ids)} listings")

# Sentiment counts
sentiment = {"positive": 0, "neutral": 0, "negative": 0}
for r in new_reviews:
    c = r["comments"]
    if c in TEMPLATES["negative"]:
        sentiment["negative"] += 1
    elif c in TEMPLATES["neutral"]:
        sentiment["neutral"] += 1
    else:
        sentiment["positive"] += 1

total_s = sum(sentiment.values())
print(f"\n  --- Sentiment Distribution (New Unreviewed-Listing Reviews) ---")
for s, cnt in sentiment.items():
    print(f"  {s:>10s}: {cnt:>6,} ({cnt/total_s:.1%})")

# Merge with existing synthetic reviews
combined_synthetic = existing_synthetic + new_reviews
print(f"\n  --- Combined Synthetic Stats ---")
print(f"  Previous synthetic (user augmentation): {len(existing_synthetic):,}")
print(f"  New (unreviewed listings):              {len(new_reviews):,}")
print(f"  Total synthetic:                        {len(combined_synthetic):,}")

# Save updated balanced file
output_path = os.path.join(BASE_DIR, "airbnb_reviews_balanced.json")
with open(output_path, "w") as f:
    json.dump(combined_synthetic, f, indent=2, ensure_ascii=False)
print(f"\n  ✓ Saved {len(combined_synthetic):,} to airbnb_reviews_balanced.json ({os.path.getsize(output_path)/(1024*1024):.1f} MB)")

# Save full combined (original + all synthetic)
full_combined = original_reviews + combined_synthetic
combined_path = os.path.join(BASE_DIR, "airbnb_reviews_combined.json")
with open(combined_path, "w") as f:
    json.dump(full_combined, f, indent=2, ensure_ascii=False)
print(f"  ✓ Saved {len(full_combined):,} to airbnb_reviews_combined.json ({os.path.getsize(combined_path)/(1024*1024):.1f} MB)")

# Final coverage check
full_df = pd.DataFrame(full_combined)
print(f"\n  --- Final Coverage ---")
print(f"  Total reviews: {len(full_df):,}")
print(f"  Unique listings with reviews: {full_df.listing_id.nunique()}")
print(f"  Unique reviewers: {full_df.reviewer_id.nunique():,}")
print(f"  All {len(all_listing_ids)} listings now have reviews: {full_df.listing_id.nunique() >= len(all_listing_ids)}")

print("\n" + "=" * 70)
print("DONE — All 5,555 listings now have reviews")
print("=" * 70)
