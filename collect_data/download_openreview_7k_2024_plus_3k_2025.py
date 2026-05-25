import json
import os

import openreview

OUT_DIR = 'data/openreview'
VENUE_2024 = 'ICLR.cc/2024/Conference'
VENUE_2025 = 'ICLR.cc/2025/Conference'
N_2024 = 7000
N_SECOND_TOTAL = 3000


def get_value(content, key):
    val = (content or {}).get(key)
    if isinstance(val, dict):
        return val.get('value')
    return val


def row_from_submission(paper, source_venue):
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
        'source_venue': source_venue,
    }


def save_json(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')

    subs_2024 = client.get_all_notes(invitation=f'{VENUE_2024}/-/Submission')
    subs_2024 = sorted(subs_2024, key=lambda p: ((p.number is None), p.number, p.id))
    if len(subs_2024) < N_2024:
        raise RuntimeError(f'ICLR 2024 has {len(subs_2024)} submissions, need at least {N_2024}.')

    first_7k = subs_2024[:N_2024]
    ids_2024 = {p.id for p in first_7k}

    remaining_2024 = [p for p in subs_2024 if p.id not in ids_2024]
    take_2024 = min(len(remaining_2024), N_SECOND_TOTAL)
    second_rows = [row_from_submission(p, VENUE_2024) for p in remaining_2024[:take_2024]]

    need_more = N_SECOND_TOTAL - take_2024
    take_2025 = []
    if need_more > 0:
        subs_2025 = client.get_all_notes(invitation=f'{VENUE_2025}/-/Submission')
        subs_2025 = sorted(subs_2025, key=lambda p: ((p.number is None), p.number, p.id))
        candidates_2025 = [p for p in subs_2025 if p.id not in ids_2024]
        if len(candidates_2025) < need_more:
            raise RuntimeError(f'ICLR 2025 candidates={len(candidates_2025)} but need {need_more}.')
        take_2025 = candidates_2025[:need_more]
        second_rows.extend(row_from_submission(p, VENUE_2025) for p in take_2025)

    rows_7k = [row_from_submission(p, VENUE_2024) for p in first_7k]

    ids_second = {r['paper_id'] for r in second_rows}
    overlap = ids_2024.intersection(ids_second)
    if overlap:
        raise RuntimeError(f'Overlap detected between first and second set: {len(overlap)}')

    merged = rows_7k + second_rows
    merged_ids = [r['paper_id'] for r in merged]
    if len(merged_ids) != len(set(merged_ids)):
        raise RuntimeError('Merged output has duplicate paper_id values.')

    out_7k = os.path.join(OUT_DIR, 'iclr2024_openreview_7000_fresh.json')
    out_3k = os.path.join(OUT_DIR, 'openreview_second_3000_nonoverlap.json')
    out_10k = os.path.join(OUT_DIR, 'openreview_10000_merged_2024plus2025.json')

    save_json(out_7k, rows_7k)
    save_json(out_3k, second_rows)
    save_json(out_10k, merged)

    n_second_2024 = sum(1 for r in second_rows if r['source_venue'] == VENUE_2024)
    n_second_2025 = sum(1 for r in second_rows if r['source_venue'] == VENUE_2025)

    print('ICLR 2024 submissions:', len(subs_2024))
    print('First set (2024):', len(rows_7k))
    print('Second set total:', len(second_rows))
    print(' - from ICLR 2024:', n_second_2024)
    print(' - from ICLR 2025:', n_second_2025)
    print('Overlap first/second:', len(overlap))
    print('Merged total:', len(merged))
    print('Saved:', out_7k)
    print('Saved:', out_3k)
    print('Saved:', out_10k)


if __name__ == '__main__':
    main()
