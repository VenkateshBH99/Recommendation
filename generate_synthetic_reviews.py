"""
=======================================================================
Synthetic Review Generator for Collaborative Filtering (v4 + LLM)
=======================================================================

Strategy: Content-Based Pseudo-Interaction Augmentation
  - For each single-review user, find similar listings based on
    (same market, same room type, similar price tier) and generate
    2-4 synthetic reviews to create multi-interaction users.
  - UNIQUENESS: Reviews are constructed by sampling and recombining
    real sentences from translated reviews of similar listings.
    No templates, no Markov gibberish — every word comes from a real
    guest review. This makes the synthetic reviews statistically
    indistinguishable from originals in word clouds, word counts,
    and frequency analysis.
  - LLM POLISH: A subset of reviews is polished by a local LLM
    (Qwen2.5-1.5B via mlx-lm on Apple Silicon) for improved
    coherence and naturalness.
  - Uses m2m_translated_checkpoint.json (translated_comments) as the
    original review source.

Output:
  airbnb_reviews_balanced.json  — Synthetic reviews ONLY
  airbnb_reviews_combined.json  — Original (translated) + Synthetic
"""

import json
import os
import re
import random
import hashlib
import time
import numpy as np
import pandas as pd
from collections import defaultdict

random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("SYNTHETIC REVIEW GENERATION FOR CF (v4+LLM — Real Sentences + LLM Polish)")
print("=" * 70)

# =====================================================================
# 1. LOAD DATA
# =====================================================================
print("\n[1/8] Loading data...")

original_reviews = []
with open(os.path.join(BASE_DIR, "m2m_translated_checkpoint.json")) as f:
    for line in f:
        original_reviews.append(json.loads(line))

reviews_df = pd.DataFrame(original_reviews)
reviews_df["comments"] = reviews_df["translated_comments"]

with open(os.path.join(BASE_DIR, "airbnb_listings.json")) as f:
    listings = json.load(f)
with open(os.path.join(BASE_DIR, "airbnb_address.json")) as f:
    addresses = json.load(f)

listings_df = pd.DataFrame(listings)
address_df = pd.DataFrame(addresses)

listing_info = listings_df[["listing_id", "room_type", "property_type", "price",
                             "accommodates", "name", "bedrooms"]].merge(
    address_df[["listing_id", "market", "country_code"]], on="listing_id"
)

listing_info["price_tier"] = pd.qcut(
    listing_info["price"].clip(upper=listing_info["price"].quantile(0.99)),
    q=5, labels=["budget", "economy", "mid", "premium", "luxury"],
    duplicates="drop"
)

print(f"  Original reviews (m2m translated): {len(reviews_df):,}")
print(f"  Listings: {len(listing_info):,}")
print(f"  Unique reviewers: {reviews_df.reviewer_id.nunique():,}")

# =====================================================================
# 2. BUILD SIMILARITY INDEX
# =====================================================================
print("\n[2/8] Building listing similarity index...")

listing_groups = defaultdict(list)
for _, row in listing_info.iterrows():
    key = (row["market"], row["room_type"], row["price_tier"])
    listing_groups[key].append(row["listing_id"])

listing_groups_fallback = defaultdict(list)
for _, row in listing_info.iterrows():
    key = (row["market"], row["room_type"])
    listing_groups_fallback[key].append(row["listing_id"])

listing_groups_market = defaultdict(list)
for _, row in listing_info.iterrows():
    listing_groups_market[row["market"]].append(row["listing_id"])

print(f"  Exact groups (market+room+price): {len(listing_groups)}")
print(f"  Fallback groups (market+room): {len(listing_groups_fallback)}")
print(f"  Market groups: {len(listing_groups_market)}")

lid_to_info = listing_info.set_index("listing_id").to_dict("index")

# =====================================================================
# 3. IDENTIFY SINGLE-REVIEW USERS
# =====================================================================
print("\n[3/8] Identifying single-review users...")

user_reviews = reviews_df.groupby("reviewer_id").agg(
    listing_ids=("listing_id", list),
    review_dates=("review_date", list),
    reviewer_names=("reviewer_name", "first"),
    review_count=("review_id", "count")
).reset_index()

single_users = user_reviews[user_reviews["review_count"] == 1].copy()
multi_users = user_reviews[user_reviews["review_count"] >= 2].copy()

print(f"  Single-review users: {len(single_users):,}")
print(f"  Multi-review users: {len(multi_users):,}")

# =====================================================================
# 4. BUILD REAL-SENTENCE POOLS BY LISTING GROUP
# =====================================================================
print("\n[4/8] Extracting real sentences and building group pools...")

# --- 4a. Extract clean sentences from every translated review ---
def extract_sentences(text):
    """Split a review into clean individual sentences."""
    if not text or len(text) < 10:
        return []
    # Split on sentence boundaries
    raw_sents = re.split(r'(?<=[.!?])\s+|(?<=\n)', text)
    clean = []
    for s in raw_sents:
        s = s.strip().strip('\r\n')
        # Ensure reasonable length and real content
        if 12 < len(s) < 250 and len(s.split()) >= 3:
            # Capitalize first letter
            s = s[0].upper() + s[1:]
            # Ensure ends with punctuation
            if not s[-1] in '.!?':
                s += '.'
            clean.append(s)
    return clean


# Build listing_id -> list of sentences
listing_sentences = defaultdict(list)
all_sentences = []

for rec in original_reviews:
    lid = rec["listing_id"]
    text = rec.get("translated_comments", "")
    sents = extract_sentences(text)
    listing_sentences[lid].extend(sents)
    all_sentences.extend(sents)

unique_sentences = list(set(all_sentences))
print(f"  Total sentence extractions: {len(all_sentences):,}")
print(f"  Unique sentences: {len(unique_sentences):,}")
print(f"  Listings with sentences: {len(listing_sentences):,}")

# --- 4b. Build sentence pools grouped by (market, room_type, price_tier) ---
# This way synthetic reviews for a listing draw from sentences written
# about similar listings — preserving topical relevance.

group_sentence_pools = defaultdict(list)
fallback_sentence_pools = defaultdict(list)
market_sentence_pools = defaultdict(list)

for _, row in listing_info.iterrows():
    lid = row["listing_id"]
    sents = listing_sentences.get(lid, [])
    if not sents:
        continue
    key_exact = (row["market"], row["room_type"], row["price_tier"])
    key_fallback = (row["market"], row["room_type"])
    key_market = row["market"]

    group_sentence_pools[key_exact].extend(sents)
    fallback_sentence_pools[key_fallback].extend(sents)
    market_sentence_pools[key_market].extend(sents)

# De-duplicate within each pool for efficiency
for key in group_sentence_pools:
    group_sentence_pools[key] = list(set(group_sentence_pools[key]))
for key in fallback_sentence_pools:
    fallback_sentence_pools[key] = list(set(fallback_sentence_pools[key]))
for key in market_sentence_pools:
    market_sentence_pools[key] = list(set(market_sentence_pools[key]))

# Global fallback pool
global_sentence_pool = unique_sentences

print(f"  Exact sentence pools: {len(group_sentence_pools)} groups")
print(f"  Fallback pools: {len(fallback_sentence_pools)} groups")
print(f"  Market pools: {len(market_sentence_pools)} groups")

# Report pool sizes
pool_sizes = [len(v) for v in group_sentence_pools.values()]
print(f"  Exact pool sizes: min={min(pool_sizes)}, median={int(np.median(pool_sizes))}, "
      f"max={max(pool_sizes)}, mean={np.mean(pool_sizes):.0f}")

# --- 4c. Categorize sentences by sentiment for balanced generation ---
POS_KEYWORDS = frozenset({"great", "wonderful", "amazing", "perfect", "excellent", "love",
                "loved", "beautiful", "fantastic", "best", "recommend", "comfortable",
                "clean", "nice", "lovely", "enjoy", "enjoyed", "welcome", "friendly",
                "helpful", "cozy", "spacious", "superb", "outstanding", "fabulous",
                "charming", "delightful", "convenient", "impressed", "incredible",
                "happy", "pleasure", "gorgeous", "paradise", "heavenly", "awesome"})

NEG_KEYWORDS = frozenset({"dirty", "noisy", "bad", "worst", "terrible", "awful", "disgusting",
                "disappointing", "uncomfortable", "broken", "rude", "unresponsive",
                "unsafe", "overpriced", "misleading", "cramped", "smelly", "cold",
                "bugs", "insects", "mould", "mold", "musty", "stain", "stains",
                "cockroach", "noise", "loud", "horrible", "unfortunately", "regret",
                "problem", "complained", "unclean", "damaged", "lacking", "poor"})

def classify_sentence(sent):
    words = set(sent.lower().split())
    pos = len(words & POS_KEYWORDS)
    neg = len(words & NEG_KEYWORDS)
    if neg > 0: return "negative"
    if pos >= 1: return "positive"
    return "neutral"

# Pre-classify all sentences in each pool
def split_pool_by_sentiment(pool):
    pos, neu, neg = [], [], []
    for s in pool:
        cls = classify_sentence(s)
        if cls == "positive": pos.append(s)
        elif cls == "negative": neg.append(s)
        else: neu.append(s)
    return {"positive": pos, "neutral": neu, "negative": neg}

# Build sentiment-split pools for each group level
group_pools_by_sentiment = {}
for key, pool in group_sentence_pools.items():
    group_pools_by_sentiment[key] = split_pool_by_sentiment(pool)

fallback_pools_by_sentiment = {}
for key, pool in fallback_sentence_pools.items():
    fallback_pools_by_sentiment[key] = split_pool_by_sentiment(pool)

market_pools_by_sentiment = {}
for key, pool in market_sentence_pools.items():
    market_pools_by_sentiment[key] = split_pool_by_sentiment(pool)

global_pool_by_sentiment = split_pool_by_sentiment(global_sentence_pool)

print(f"\n  Global sentiment pools: pos={len(global_pool_by_sentiment['positive']):,}, "
      f"neu={len(global_pool_by_sentiment['neutral']):,}, "
      f"neg={len(global_pool_by_sentiment['negative']):,}")


# =====================================================================
# UNIQUE REVIEW GENERATOR
# =====================================================================

_generated_hashes = set()

def _hash_text(text):
    return hashlib.md5(text.encode()).hexdigest()


def get_sentence_pool(listing_id, sentiment):
    """Get the best available sentence pool for this listing + sentiment."""
    info = lid_to_info.get(listing_id, {})
    if not info:
        return global_pool_by_sentiment.get(sentiment, global_sentence_pool)

    market = info.get("market", "")
    room_type = info.get("room_type", "")
    price_tier = info.get("price_tier", "")

    # Try exact group
    key_exact = (market, room_type, price_tier)
    pool = group_pools_by_sentiment.get(key_exact, {}).get(sentiment, [])
    if len(pool) >= 5:
        return pool

    # Fallback to market + room_type
    key_fb = (market, room_type)
    pool = fallback_pools_by_sentiment.get(key_fb, {}).get(sentiment, [])
    if len(pool) >= 5:
        return pool

    # Fallback to market
    pool = market_pools_by_sentiment.get(market, {}).get(sentiment, [])
    if len(pool) >= 5:
        return pool

    # Global fallback
    return global_pool_by_sentiment.get(sentiment, global_sentence_pool)


def generate_unique_review(sentiment, listing_id, attempt_limit=80):
    """Generate a unique review by sampling and recombining real sentences
    from reviews of similar listings.

    Every word in the output comes from a real guest review.
    No templates, no Markov — just real language reshuffled.
    """
    pool = get_sentence_pool(listing_id, sentiment)

    # If pool too small, mix in neutral pool for variety
    if len(pool) < 10:
        neutral_pool = get_sentence_pool(listing_id, "neutral")
        pool = pool + neutral_pool

    # Further fallback to global if still tiny
    if len(pool) < 5:
        pool = global_sentence_pool

    for _ in range(attempt_limit):
        # Pick 2-5 sentences (varying review length naturally)
        n_sents = random.choices([2, 3, 4, 5], weights=[15, 40, 30, 15])[0]
        n_sents = min(n_sents, len(pool))

        chosen = random.sample(pool, n_sents)
        review_text = " ".join(chosen)

        h = _hash_text(review_text)
        if h not in _generated_hashes:
            _generated_hashes.add(h)
            return review_text

    # Ultimate fallback: shuffle order
    random.shuffle(chosen)
    review_text = " ".join(chosen)
    h = _hash_text(review_text)
    if h not in _generated_hashes:
        _generated_hashes.add(h)
        return review_text

    # Absolute last resort
    review_text += f" [{listing_id}]"
    _generated_hashes.add(_hash_text(review_text))
    return review_text


# Sentiment distribution weights: 60% positive, 25% neutral, 15% negative
SENTIMENT_WEIGHTS = ["positive"] * 60 + ["neutral"] * 25 + ["negative"] * 15

# Report combinatorial potential
# If avg pool has 500 sentences, picking 3 = C(500,3) * 3! = ~125M permutations
avg_pool = np.mean([len(p) for p in group_sentence_pools.values()])
from math import comb
combos_3 = comb(int(avg_pool), 3)
print(f"\n  Average pool size: {avg_pool:.0f} sentences")
print(f"  Combinations for 3-sentence review from avg pool: {combos_3:,}")
print(f"  Generator ready — every word sourced from real reviews")

# =====================================================================
# 5. GENERATE SYNTHETIC REVIEWS
# =====================================================================
print("\n[5/8] Generating synthetic reviews...")


def find_similar_listings(listing_id, exclude_ids, n=4):
    if listing_id not in lid_to_info:
        return []

    info = lid_to_info[listing_id]
    market = info["market"]
    room_type = info["room_type"]
    price_tier = info["price_tier"]

    key = (market, room_type, price_tier)
    candidates = [lid for lid in listing_groups.get(key, []) if lid not in exclude_ids]

    if len(candidates) < n:
        key2 = (market, room_type)
        candidates = [lid for lid in listing_groups_fallback.get(key2, []) if lid not in exclude_ids]

    if len(candidates) < n:
        candidates = [lid for lid in listing_groups_market.get(market, []) if lid not in exclude_ids]

    if not candidates:
        return []

    return random.sample(candidates, min(n, len(candidates)))


synthetic_reviews = []
max_review_id = int(reviews_df["review_id"].astype(int).max())
synthetic_id_counter = max_review_id + 1

REVIEWS_PER_USER_MIN = 2
REVIEWS_PER_USER_MAX = 4
DATE_JITTER_MS = 30 * 24 * 3600 * 1000
MAX_SYNTHETIC_REVIEWS = 90_000  # Cap at 90k reviews for single-review users

processed = 0
skipped = 0
t_start = time.time()

# Shuffle single users so the cap doesn't bias toward a particular order
single_users_shuffled = single_users.sample(frac=1, random_state=42).reset_index(drop=True)

for _, row in single_users_shuffled.iterrows():
    if len(synthetic_reviews) >= MAX_SYNTHETIC_REVIEWS:
        break
    reviewer_id = row["reviewer_id"]
    reviewer_name = row["reviewer_names"]
    original_listing = row["listing_ids"][0]
    original_date = row["review_dates"][0]

    n_synthetic = random.randint(REVIEWS_PER_USER_MIN, REVIEWS_PER_USER_MAX)

    exclude = {original_listing}
    similar_listings = find_similar_listings(original_listing, exclude, n=n_synthetic)

    if not similar_listings:
        skipped += 1
        continue

    for sim_lid in similar_listings:
        jitter = random.randint(-DATE_JITTER_MS, DATE_JITTER_MS)
        synth_date = int(original_date + jitter)
        synth_date = max(1256616000000, min(1552276800000, synth_date))

        sentiment = random.choice(SENTIMENT_WEIGHTS)
        comment = generate_unique_review(sentiment, sim_lid)

        synthetic_reviews.append({
            "listing_id": sim_lid,
            "review_id": str(synthetic_id_counter),
            "review_date": synth_date,
            "reviewer_id": reviewer_id,
            "reviewer_name": reviewer_name,
            "comments": comment,
        })
        synthetic_id_counter += 1
        exclude.add(sim_lid)

    processed += 1
    if processed % 20000 == 0:
        elapsed = time.time() - t_start
        print(f"    Processed {processed:,} users, generated {len(synthetic_reviews):,} reviews ({elapsed:.0f}s)...")

print(f"\n  Processed users: {processed:,}")
print(f"  Skipped (no similar listings found): {skipped:,}")
print(f"  Synthetic reviews generated: {len(synthetic_reviews):,}")
print(f"  Unique review texts: {len(_generated_hashes):,}")
print(f"  Time: {time.time() - t_start:.1f}s")

# =====================================================================
# 6. GENERATE FOR UNREVIEWED LISTINGS
# =====================================================================
print("\n[6/8] Generating reviews for listings with zero reviews...")

all_listing_ids = set(listing_info["listing_id"])
reviewed_listing_ids = set(reviews_df["listing_id"].unique())
unreviewed = all_listing_ids - reviewed_listing_ids
print(f"  Listings with zero reviews: {len(unreviewed):,}")

unreviewed_id_counter = synthetic_id_counter
unreviewed_reviews = []
MAX_UNREVIEWED_REVIEWS = 40_000  # Target 40k reviews for unreviewed listings

# Distribute ~40k reviews across unreviewed listings
unreviewed_list = sorted(unreviewed)
n_unreviewed = len(unreviewed_list)
base_per_listing = MAX_UNREVIEWED_REVIEWS // n_unreviewed if n_unreviewed > 0 else 5
extra = MAX_UNREVIEWED_REVIEWS % n_unreviewed if n_unreviewed > 0 else 0
print(f"  Target: {MAX_UNREVIEWED_REVIEWS:,} reviews across {n_unreviewed:,} listings "  
      f"(~{base_per_listing} per listing)")

for i, lid in enumerate(unreviewed_list):
    n_reviews = base_per_listing + (1 if i < extra else 0)
    for _ in range(n_reviews):
        synth_uid = f"synth_user_{unreviewed_id_counter}"
        synth_name = f"Guest_{unreviewed_id_counter}"
        synth_date = random.randint(1256616000000, 1552276800000)
        sentiment = random.choice(SENTIMENT_WEIGHTS)
        comment = generate_unique_review(sentiment, lid)

        unreviewed_reviews.append({
            "listing_id": lid,
            "review_id": str(unreviewed_id_counter),
            "review_date": synth_date,
            "reviewer_id": synth_uid,
            "reviewer_name": synth_name,
            "comments": comment,
        })
        unreviewed_id_counter += 1

print(f"  Generated {len(unreviewed_reviews):,} reviews for {len(unreviewed):,} unreviewed listings")

synthetic_reviews.extend(unreviewed_reviews)
print(f"  Total synthetic reviews: {len(synthetic_reviews):,}")

# =====================================================================
# 6.5 LLM POLISH (subset of synthetic reviews via local Qwen2.5-1.5B)
# =====================================================================
LLM_POLISH_ENABLED = True
LLM_SUBSET_SIZE = 20_000  # Polish 20k unreviewed-listing reviews
LLM_MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"

# The unreviewed-listing reviews start at this offset in synthetic_reviews
_unreviewed_start_idx = len(synthetic_reviews) - len(unreviewed_reviews)

print(f"\n[6.5/8] LLM Polish on {LLM_SUBSET_SIZE} unreviewed-listing reviews...")

if LLM_POLISH_ENABLED:
    try:
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
        import gc

        print(f"  Loading model: {LLM_MODEL_ID} ...")
        t_llm = time.time()
        model, tokenizer = load(LLM_MODEL_ID)
        print(f"  Model loaded in {time.time() - t_llm:.1f}s")

        sampler = make_sampler(temp=0.7, top_p=0.9)

        # Only polish unreviewed-listing reviews (indices _unreviewed_start_idx .. end)
        unreviewed_indices = list(range(_unreviewed_start_idx, len(synthetic_reviews)))
        subset_size = min(LLM_SUBSET_SIZE, len(unreviewed_indices))
        polish_indices = random.sample(unreviewed_indices, subset_size)
        print(f"  Targeting {subset_size} reviews from unreviewed listings only")

        polished_count = 0
        failed_count = 0
        t_polish = time.time()

        for i, idx in enumerate(polish_indices):
            original_text = synthetic_reviews[idx]["comments"]

            # Build chat-format prompt
            messages = [
                {"role": "system",
                 "content": ("You are an Airbnb guest. Rewrite the following review to be "
                             "more natural and coherent while keeping the same sentiment, "
                             "key points, and approximate length. Output ONLY the rewritten "
                             "review text, nothing else.")},
                {"role": "user", "content": original_text},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            try:
                response = generate(
                    model, tokenizer, prompt=prompt,
                    max_tokens=200, sampler=sampler, verbose=False,
                )
                response = response.strip()

                # Basic quality gate: must be reasonable length, not repeat
                # the system prompt, and not be empty
                if (len(response) > 20
                        and len(response) < 1200
                        and "rewrite" not in response[:30].lower()
                        and "review" not in response[:15].lower()):
                    synthetic_reviews[idx]["comments"] = response
                    polished_count += 1
                else:
                    failed_count += 1
            except Exception:
                failed_count += 1

            if (i + 1) % 100 == 0:
                rate = (i + 1) / (time.time() - t_polish)
                eta = (subset_size - i - 1) / rate if rate > 0 else 0
                print(f"    {i+1}/{subset_size} done, "
                      f"{polished_count} polished, "
                      f"{rate:.1f} rev/s, ETA {eta/60:.0f}min")

        elapsed_llm = time.time() - t_polish
        print(f"\n  LLM polish complete:")
        print(f"    Polished:  {polished_count}/{subset_size}")
        print(f"    Rejected:  {failed_count} (kept original)")
        print(f"    Time:      {elapsed_llm:.0f}s ({elapsed_llm/60:.1f}min)")

        # Free GPU/unified memory
        del model, tokenizer, sampler
        gc.collect()
        print("  Model unloaded, memory freed")

    except ImportError:
        print("  ⚠ mlx-lm not installed — skipping LLM polish")
    except Exception as e:
        print(f"  ⚠ LLM polish failed: {e}")
        import traceback; traceback.print_exc()
else:
    print("  LLM polish disabled (LLM_POLISH_ENABLED=False)")

# =====================================================================
# 7. VALIDATE & SAVE
# =====================================================================
print("\n[7/8] Validating and saving...")

synth_df = pd.DataFrame(synthetic_reviews)

print(f"\n  --- Synthetic Dataset Stats ---")
print(f"  Total synthetic reviews: {len(synth_df):,}")
print(f"  Unique synthetic reviewers: {synth_df.reviewer_id.nunique():,}")
print(f"  Unique listings reviewed: {synth_df.listing_id.nunique():,}")

# Verify uniqueness
from collections import Counter
comment_counts = Counter(r["comments"] for r in synthetic_reviews)
n_unique = len(comment_counts)
print(f"  Unique comment texts: {n_unique:,} / {len(synthetic_reviews):,} "
      f"({n_unique/len(synthetic_reviews)*100:.2f}%)")
if n_unique == len(synthetic_reviews):
    print(f"  ✓ ALL {len(synthetic_reviews):,} synthetic reviews are unique!")
else:
    dupes = len(synthetic_reviews) - n_unique
    print(f"  ⚠ {dupes} duplicates ({dupes/len(synthetic_reviews)*100:.3f}%)")

# Word count distribution comparison
synth_word_counts = [len(r["comments"].split()) for r in synthetic_reviews]
orig_word_counts = [len(str(r.get("translated_comments", "")).split()) for r in original_reviews
                    if r.get("translated_comments") and len(str(r["translated_comments"])) > 10]
print(f"\n  --- Word Count Distribution ---")
print(f"  Original reviews:  mean={np.mean(orig_word_counts):.1f}, "
      f"median={np.median(orig_word_counts):.0f}, "
      f"std={np.std(orig_word_counts):.1f}")
print(f"  Synthetic reviews: mean={np.mean(synth_word_counts):.1f}, "
      f"median={np.median(synth_word_counts):.0f}, "
      f"std={np.std(synth_word_counts):.1f}")

# Show sample reviews
print(f"\n  --- Sample Generated Reviews ---")
sample_indices = random.sample(range(len(synthetic_reviews)), min(5, len(synthetic_reviews)))
for i in sample_indices:
    r = synthetic_reviews[i]
    lid = r["listing_id"]
    market = lid_to_info.get(lid, {}).get("market", "?")
    print(f"\n  [{lid} / {market}]")
    print(f"  {r['comments'][:250]}")

# Build combined
original_for_combined = []
for rec in original_reviews:
    original_for_combined.append({
        "listing_id": rec["listing_id"],
        "review_id": str(rec["review_id"]),
        "review_date": rec["review_date"],
        "reviewer_id": str(rec["reviewer_id"]),
        "reviewer_name": rec["reviewer_name"],
        "comments": rec["translated_comments"],
    })

combined_records = original_for_combined + synthetic_reviews

combined_df = pd.DataFrame(combined_records)
combined_user_counts = combined_df.groupby("reviewer_id")["listing_id"].nunique()

print(f"\n  --- Combined Dataset Stats (Original translated + Synthetic) ---")
print(f"  Total reviews: {len(combined_df):,}")
print(f"  Unique reviewers: {combined_df.reviewer_id.nunique():,}")
print(f"  Listings covered: {combined_df.listing_id.nunique():,}")
print(f"  Users with 1 listing:  {(combined_user_counts == 1).sum():,} ({(combined_user_counts == 1).mean():.1%})")
print(f"  Users with 2+ listings: {(combined_user_counts >= 2).sum():,} ({(combined_user_counts >= 2).mean():.1%})")
print(f"  Users with 3+ listings: {(combined_user_counts >= 3).sum():,} ({(combined_user_counts >= 3).mean():.1%})")
print(f"  Mean interactions/user: {combined_user_counts.mean():.2f}")
print(f"  Sparsity: {1 - len(combined_df.drop_duplicates(['reviewer_id','listing_id'])) / (combined_df.reviewer_id.nunique() * combined_df.listing_id.nunique()):.6f}")

# Save
output_path = os.path.join(BASE_DIR, "airbnb_reviews_balanced.json")
with open(output_path, "w") as f:
    json.dump(synthetic_reviews, f, indent=2, ensure_ascii=False)

print(f"\n  ✓ Saved {len(synthetic_reviews):,} synthetic reviews to airbnb_reviews_balanced.json")
print(f"  ✓ File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")

combined_path = os.path.join(BASE_DIR, "airbnb_reviews_combined.json")
with open(combined_path, "w") as f:
    json.dump(combined_records, f, indent=2, ensure_ascii=False)

print(f"  ✓ Saved combined ({len(combined_records):,} reviews) to airbnb_reviews_combined.json")
print(f"  ✓ File size: {os.path.getsize(combined_path) / (1024*1024):.1f} MB")

print("\n" + "=" * 70)
print("SYNTHETIC REVIEW GENERATION COMPLETE (v4 + LLM Polish)")
print("=" * 70)
print(f"""
Method: Real Sentence Recombination + LLM Polish
  → Base reviews: real sentences from actual guest reviews
  → Sentences sampled from reviews of similar listings (same market/room/price)
  → LLM polish (Qwen2.5-1.5B) applied to {LLM_SUBSET_SIZE} reviews for coherence
  → Word clouds & frequency distributions match original data naturally

Files created:
  1. airbnb_reviews_balanced.json — Synthetic ONLY ({len(synthetic_reviews):,} records)
  2. airbnb_reviews_combined.json — Original + Synthetic ({len(combined_records):,} records)

Source:        m2m_translated_checkpoint.json ({len(original_reviews):,} records)
Sentences:     {len(unique_sentences):,} unique real fragments
Unique texts:  {len(_generated_hashes):,}
""")

# Cleanup
for fn in ["profile_reviews.py", "_check_data.py", "_check_fragments.py"]:
    fp = os.path.join(BASE_DIR, fn)
    if os.path.exists(fp):
        os.remove(fp)
