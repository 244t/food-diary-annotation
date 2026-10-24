# SPDX-License-Identifier: Apache-2.0
"""Score saved dish boxes using the paper's output-order greedy IoU matching."""
import argparse
import json
import math
from pathlib import Path


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def iou(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - intersection)
    return intersection / union if union > 0 else 0.0


def greedy_matches(predictions, ground_truth, threshold=0.5):
    """Preserve prediction order; equal IoUs prefer the earlier GT box."""
    used = set()
    pairs = []
    for pred_index, prediction in enumerate(predictions):
        best_iou, best_index = -1, -1
        for index, target in enumerate(ground_truth):
            if index in used:
                continue
            overlap = iou(prediction, target)
            if overlap > best_iou:
                best_iou, best_index = overlap, index
        if best_index >= 0 and best_iou >= threshold:
            used.add(best_index)
            pairs.append((pred_index, best_index))
    return pairs


def boxes(record):
    value = record['boxes']
    if not isinstance(value, list):
        raise ValueError('boxes must be a list')
    for box in value:
        if (not isinstance(box, list) or len(box) != 4
                or any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) for x in box)):
            raise ValueError('Each box must have four finite numeric xyxy coordinates')
    return value


def prediction_pixels(record, target, coordinates, mode):
    result = boxes(record)
    if coordinates == 'normalized' and result:
        w, h = record.get('w', target.get('w')), record.get('h', target.get('h'))
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not math.isfinite(x) or x <= 0 for x in (w, h)):
            raise ValueError('Normalized predictions require positive finite w and h')
        if mode == 'crop':
            result = [[min(max(x, 0), 1) for x in b] for b in result]
        result = [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in result]
    # The original crop/photo scorers discard predictions at most one pixel wide/high.
    kept = [b for b in result if b[2] - b[0] > 1 and b[3] - b[1] > 1]
    return kept, len(result) - len(kept)


def summarize(rows):
    tp = sum(r['tp'] for r in rows)
    ngt = sum(r['gt_boxes'] for r in rows)
    npred = sum(r['predicted_boxes'] for r in rows)
    precision, recall = tp / max(1, npred), tp / max(1, ngt)
    return {'images': len(rows), 'gt_boxes': ngt, 'predicted_boxes': npred,
            'tp': tp, 'missed': ngt - tp, 'extra': npred - tp,
            'discarded_predictions': sum(r['discarded_predictions'] for r in rows),
            'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / max(1e-9, precision + recall)}


def evaluate(ground_truth, predictions, coordinates='normalized', mode='crop', threshold=0.5):
    if not isinstance(ground_truth, dict) or not ground_truth:
        raise ValueError('Ground truth must be a nonempty object keyed by image/crop ID')
    if not isinstance(predictions, dict):
        raise ValueError('Predictions must be an object keyed by image/crop ID')
    missing = ground_truth.keys() - predictions.keys()
    if missing:
        raise ValueError(f'{len(missing)} prediction records are missing; use explicit empty boxes for failures')
    if coordinates not in ('pixels', 'normalized') or mode not in ('crop', 'photo'):
        raise ValueError('Unsupported coordinate system or scoring mode')
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be in (0, 1]')
    rows, strata = [], {}
    for image_id, target in ground_truth.items():
        gt = boxes(target)
        if any(b[2] <= b[0] or b[3] <= b[1] for b in gt):
            raise ValueError(f'Invalid ground-truth box in {image_id}')
        predicted, discarded = prediction_pixels(predictions[image_id], target, coordinates, mode)
        row = {'gt_boxes': len(gt), 'predicted_boxes': len(predicted),
               'tp': len(greedy_matches(predicted, gt, threshold)), 'discarded_predictions': discarded}
        rows.append(row)
        if 'stratum' in target:
            if not isinstance(target['stratum'], str) or not target['stratum']:
                raise ValueError('stratum must be a nonempty string')
            strata.setdefault(target['stratum'], []).append(row)
    return {'iou_threshold': threshold, 'mode': mode, 'prediction_coordinates': coordinates,
            'ignored_prediction_records': len(predictions.keys() - ground_truth.keys()),
            'overall': summarize(rows),
            'by_gt_count': {key: summarize([r for r in rows if
                             (r['gt_boxes'] >= 3 if key == '3+' else r['gt_boxes'] == int(key))])
                            for key in ['0', '1', '2', '3+']},
            'by_stratum': {key: summarize(value) for key, value in sorted(strata.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--prediction-coordinates', choices=['pixels', 'normalized'], default='normalized')
    parser.add_argument('--mode', choices=['crop', 'photo'], default='crop')
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        report = evaluate(load_json(args.gt), load_json(args.predictions),
                          args.prediction_coordinates, args.mode, args.iou)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
