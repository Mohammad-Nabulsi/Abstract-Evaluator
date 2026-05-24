import json
import os
import time
from typing import Any, Dict, List

import openreview

OUT_DIR = 'data/openreview'
VENUE_ID = 'ICLR.cc/2024/Conference'
N_FIRST = 7000
N_SECOND = 3000
SLEEP_PER_FORUM = 0.05


def get_value(content: Dict[str, Any], key: str):
    val = content.get(key)
    if isinstance(val, dict):
        return val.get('value')
    return val


def build_row(client: openreview.api.OpenReviewClient, paper, include_replies: bool = True) -> Dict[str, Any]:
    content = paper.content or {}

    title = get_value(content, 'title')
    abstract = get_value(content, 'abstract')
    keywords = get_value(content, 'keywords')
    pdf = get_value(content, 'pdf')

    reviews: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []

    if include_replies:
        replies = client.get_all_notes(forum=paper.id)
        time.sleep(SLEEP_PER_FORUM)

        for r in replies:
            invitation = r.invitations[0] if r.invitations else ''
            r_content = r.content or {}

            if 'Official_Review' in invitation:
                reviews.append(
                    {
                        'rating': get_value(r_content, 'rating'),
                        'confidence': get_value(r_content, 'confidence'),
                        'summary': get_value(r_content, 'summary'),
                        'strengths': get_value(r_content, 'strengths'),
                        'weaknesses': get_value(r_content, 'weaknesses'),
                    }
                )

            if 'Decision' in invitation:
                decisions.append(
                    {
                        'decision': get_value(r_content, 'decision'),
                        'comment': get_value(r_content, 'comment'),
                    }
                )

    return {
        'paper_id': paper.id,
        'number': paper.number,
        'title': title,
        'abstract': abstract,
        'keywords': keywords,
        'pdf': pdf,
        'n_reviews': len(reviews),
        'reviews': reviews,
        'decision': decisions[0]['decision'] if decisions else None,
        'decision_comment': decisions[0]['comment'] if decisions else None,
    }


def save_json(path: str, rows: List[Dict[str, Any]]):
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
    effective_second = min(N_SECOND, len(remaining))
    next_3k = remaining[:effective_second]

    overlap = first_ids.intersection({p.id for p in next_3k})
    if overlap:
        raise RuntimeError(f'Overlap detected between 7k and second set: {len(overlap)}')

    print(f'Total submissions: {len(submissions)}')
    print(f'7k set size: {len(first_7k)}')
    print(f'Requested second set size: {N_SECOND}')
    print(f'Actual second set size: {len(next_3k)}')
    print(f'Overlap size: {len(overlap)}')

    rows_7k = []
    for i, p in enumerate(first_7k, start=1):
        if i % 100 == 0 or i == 1:
            print(f'[7k] Processing {i}/{len(first_7k)}')
        rows_7k.append(build_row(client, p, include_replies=True))

    rows_3k = []
    for i, p in enumerate(next_3k, start=1):
        if i % 100 == 0 or i == 1:
            print(f'[2nd] Processing {i}/{len(next_3k)}')
        rows_3k.append(build_row(client, p, include_replies=True))

    merged = rows_7k + rows_3k

    ids_merged = [r['paper_id'] for r in merged if r.get('paper_id')]
    if len(ids_merged) != len(set(ids_merged)):
        raise RuntimeError('Merged output has duplicate paper_id values.')

    p7 = os.path.join(OUT_DIR, 'iclr2024_openreview_7000_fresh.json')
    p3 = os.path.join(OUT_DIR, 'iclr2024_openreview_3000_nonoverlap.json')
    p10 = os.path.join(OUT_DIR, 'iclr2024_openreview_10000_merged.json')

    save_json(p7, rows_7k)
    save_json(p3, rows_3k)
    save_json(p10, merged)

    print('Saved:')
    print(' -', p7, len(rows_7k))
    print(' -', p3, len(rows_3k))
    print(' -', p10, len(merged))


if __name__ == '__main__':
    main()
