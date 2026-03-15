import json

with open('airbnb_reviews_llm_unreviewed.json') as f:
    llm = json.load(f)
with open('m2m_translated_checkpoint.json') as f2:
    orig = [json.loads(line) for line in f2]

uids = [(r['review_id'], r['reviewer_name']) for r in orig]

for i, r in enumerate(llm):
    r['review_id'], r['reviewer_name'] = uids[i % len(uids)]
    r['translated_comments'] = r['comments']
    r.pop('reviewer_id', None)

with open('airbnb_reviews_llm_unreviewed_overwrite.json', 'w') as f:
    json.dump(llm, f, indent=2, ensure_ascii=False)

print('Done. Example:', llm[0])
