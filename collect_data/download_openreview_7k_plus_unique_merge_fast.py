import json
import os

import openreview

OUT_DIR = 'data/openreview'
VENUE_ID = 'ICLR.cc/2024/Conference'
N_FIRST = 7000
N_SECOND_REQUESTED = 3000


def get_value(content, key):
    val = (content or {}).get(key)
    if isinstance(val, dict):
        return val.get('value')
    return val


def row_from_submission(paper):
    content = paper.content or {}
    return {
        'paper_id': paper.id,
        'number': paper.number,
        'title': get_value(content, 'title'),
        'abstract': get_value(content, 'abstract'),
        'keywords': get_value(content, 'keywords'),
        'pdf': get_value(content, 'pdf'),
        'n_reviews': None,
        'reviews': [],
        'decision': None,
        'decision_comment': None,
    }


def save_json(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
    submissions = client.get_all_notes(invitation=f'{VENUE_ID}/-/Submission')
    submissions = sorted(submissions, key=lambda p: ((p.number is None), p.number, p.id))

    if len(submissions) < N_FIRST:
        raise RuntimeError(f'Not enough submissions. Found {len(submissions)}, need at least {N_FIRST}.')

    first_7k = submissions[:N_FIRST]
    first_ids = {p.id for p in first_7k}

    remaining = [p for p in submissions if p.id not in first_ids]
    second_n = min(N_SECOND_REQUESTED, len(remaining))
    second_set = remaining[:second_n]

    second_ids = {p.id for p in second_set}
    overlap = first_ids.intersection(second_ids)
    if overlap:
        raise RuntimeError(f'Overlap detected: {len(overlap)}')

    rows_7k = [row_from_submission(p) for p in first_7k]
    rows_2nd = [row_from_submission(p) for p in second_set]
    merged = rows_7k + rows_2nd

    merged_ids = [r['paper_id'] for r in merged]
    if len(merged_ids) != len(set(merged_ids)):
        raise RuntimeError('Merged file contains duplicate paper_id values.')

    out_7k = os.path.join(OUT_DIR, 'iclr2024_openreview_7000_fresh.json')
    out_3k = os.path.join(OUT_DIR, 'iclr2024_openreview_3000_nonoverlap.json')
    out_merged = os.path.join(OUT_DIR, 'iclr2024_openreview_10000_merged.json')

    save_json(out_7k, rows_7k)
    save_json(out_3k, rows_2nd)
    save_json(out_merged, merged)

    print('Total submissions:', len(submissions))
    print('7k size:', len(rows_7k))
    print('Requested second size:', N_SECOND_REQUESTED)
    print('Actual second size:', len(rows_2nd))
    print('Overlap:', len(overlap))
    print('Merged size:', len(merged))
    print('Saved:', out_7k)
    print('Saved:', out_3k)
    print('Saved:', out_merged)


if __name__ == '__main__':
    main()
