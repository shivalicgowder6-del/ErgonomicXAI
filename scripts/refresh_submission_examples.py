"""
Refresh submission_examples by replacing mock JSONs with real-run outputs when available.

Usage:
    python scripts/refresh_submission_examples.py
"""
from pathlib import Path
import shutil

submission = Path('submission_examples')
if not submission.exists():
    print('submission_examples/ not found')
    raise SystemExit(1)

# preference order for real results
candidates = [
    Path('validation_results/run_subset_person_frame_full'),
    Path('validation_results/run_subset_person_frame_tune'),
    Path('validation_results/run_subset_person_frame'),
    Path('validation_results/run_subset_person_real'),
]

replaced = 0
checked = 0
for sub in submission.iterdir():
    if not sub.is_dir():
        continue
    stem = sub.name
    jsonp = sub / f'{stem}.json'
    if not jsonp.exists():
        continue
    checked += 1
    try:
        import json
        data = json.loads(jsonp.read_text(encoding='utf-8'))
    except Exception:
        continue
    # if it already has advice_shap or advice_det, leave it
    if 'advice_shap' in data or 'advice_det' in data:
        continue
    # otherwise try to find a real JSON
    found = None
    for c in candidates:
        p = c / f'{stem}.json'
        if p.exists():
            found = p
            break
    if found:
        shutil.copy2(found, jsonp)
        replaced += 1
        print(f'Replaced {jsonp} with {found}')

print(f'Checked {checked} submission examples, replaced {replaced} JSONs')
