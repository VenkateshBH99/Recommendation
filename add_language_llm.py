import json

with open('airbnb_reviews_llm_unreviewed_overwrite.json') as f:
    llm = json.load(f)

for r in llm:
    r['detected_language'] = 'en'

with open('airbnb_reviews_llm_unreviewed_overwrite_lang.json', 'w') as f:
    json.dump(llm, f, indent=2, ensure_ascii=False)

print('Added detected_language: en to all records.')
