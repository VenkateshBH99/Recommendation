import json

with open('airbnb_reviews_combined_llm_lang.json') as f:
    reviews = json.load(f)

with open('airbnb_reviews_combined_llm_lang_ldjson.json', 'w') as f:
    for r in reviews:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"Wrote {len(reviews)} reviews as line-delimited JSON.")
