# SPDX-License-Identifier: Apache-2.0
"""Score saved classifier rankings or taxonomy-based naming-pipeline outputs."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def base_form(name):
    if not name:
        return ''
    result = re.sub(r'[（(].*?[)）]', '', name).strip()
    return result or name


def lenient_equal(a, b, keywords):
    if not a or not b:
        return False
    if a == b:
        return True
    ba, bb = base_form(a), base_form(b)
    if ba and ba == bb:
        return True
    if bb and (bb in a or bb in keywords.get(a, '')):
        return True
    if ba and (ba in b or ba in keywords.get(b, '')):
        return True
    return False


def ranked_names(record):
    candidates = record['candidates']
    if not isinstance(candidates, list):
        raise ValueError('candidates must be a ranked list of name/score objects')
    names, previous = [], math.inf
    for candidate in candidates:
        name, score = candidate['name'], candidate['score']
        if not isinstance(name, str) or not name:
            raise ValueError('Candidate names must be nonempty strings')
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1 or score > previous):
            raise ValueError('Candidate scores must be finite probabilities in descending order')
        if name in names:
            raise ValueError('Duplicate candidate names')
        names.append(name)
        previous = score
    return names


def evaluate(ground_truth, predictions, mode='classifier', keywords=None):
    if not isinstance(ground_truth, dict) or not ground_truth:
        raise ValueError('Ground truth must be a nonempty object keyed by crop ID')
    if not isinstance(predictions, dict):
        raise ValueError('Predictions must be an object keyed by crop ID')
    if mode not in ('classifier', 'pipeline'):
        raise ValueError('Unsupported scoring mode')
    missing = ground_truth.keys() - predictions.keys()
    if missing:
        raise ValueError(f'{len(missing)} prediction records are missing; use explicit empty candidates for failures')
    if mode == 'pipeline' and (not isinstance(keywords, dict) or
                             any(not isinstance(k, str) or not isinstance(v, str) for k, v in keywords.items())):
        raise ValueError('Pipeline scoring requires an explicit name-to-keyword string mapping')
    top1 = top5 = lenient = no_match_n = no_match_correct = 0
    for crop_id, target in ground_truth.items():
        if target is not None and (not isinstance(target, str) or not target):
            raise ValueError('Targets must be nonempty strings, or null for pipeline no-match GT')
        names = ranked_names(predictions[crop_id])
        if mode == 'classifier':
            if target is None:
                raise ValueError('Classifier GT must contain only in-vocabulary string labels')
            top1 += bool(names) and names[0] == target
            top5 += target in names[:5]
        else:
            probability = predictions[crop_id]['none_probability']
            if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                    or not math.isfinite(probability) or not 0 <= probability <= 1):
                raise ValueError('none_probability must be a finite probability in [0, 1]')
            if target is None:
                no_match_n += 1
                accepted = probability >= 0.5 or not names
                no_match_correct += accepted
                top1 += accepted
                lenient += accepted
            else:
                top1 += bool(names) and names[0] == target
                lenient += bool(names) and lenient_equal(names[0], target, keywords)
    count = len(ground_truth)
    report = {'n': count, 'ignored_prediction_records': len(predictions.keys() - ground_truth.keys())}
    if mode == 'classifier':
        report.update(top1_correct=top1, top5_correct=top5, exact_top1=top1 / count, exact_top5=top5 / count)
    else:
        report.update(strict_correct=top1, lenient_correct=lenient, strict_top1=top1 / count,
                      lenient_top1=lenient / count, no_match_gt=no_match_n, no_match_correct=no_match_correct)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, nargs='+', required=True, help='One file per run/seed')
    parser.add_argument('--mode', choices=['classifier', 'pipeline'], default='classifier')
    parser.add_argument('--keywords', type=Path, help='Name-to-keyword JSON mapping; required for pipeline mode')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        if args.mode == 'pipeline' and args.keywords is None:
            raise ValueError('--keywords is required for pipeline mode (use {} for a keyword-free taxonomy)')
        gt = load_json(args.gt)
        keywords = load_json(args.keywords) if args.keywords else None
        runs = [evaluate(gt, load_json(path), args.mode, keywords) for path in args.predictions]
        metrics = ['exact_top1', 'exact_top5'] if args.mode == 'classifier' else ['strict_top1', 'lenient_top1']
        aggregate = {metric: {'mean': statistics.mean(r[metric] for r in runs),
                              'sample_std': statistics.stdev(r[metric] for r in runs) if len(runs) > 1 else None}
                     for metric in metrics}
        report = {'mode': args.mode, 'runs': runs, 'aggregate': aggregate}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
