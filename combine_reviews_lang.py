import json

# Load original reviews (line-delimited JSON)
with open('m2m_translated_checkpoint.json') as f:
    orig = [json.loads(line) for line in f]

# Load LLM reviews with language field
with open('airbnb_reviews_llm_unreviewed_overwrite_lang.json') as f:
    llm = json.load(f)

# Combine
combined = orig + llm

# Save as a single JSON array
with open('airbnb_reviews_combined_llm_lang.json', 'w') as f:
    json.dump(combined, f, indent=2, ensure_ascii=False)

print(f"Combined file written: {len(combined)} reviews")
