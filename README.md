# Automating Dish-Level Annotation of Real Food-Diary Photos

Project repository for our paper accepted at **MADiMa 2026 (ACCV Workshop)**.

**Authors:** Kyosuke Nishishita, Sosuke Amano, Kiyoharu Aizawa, and Yoko Yamakata.

## Overview

We study dish-level automatic annotation of real food-diary photos using a multi-stage pipeline for food/non-food classification, dish detection, single/multi-dish classification and splitting, and dish-name classification.

## Release status

- Dish splitting model: [Hugging Face](https://huggingface.co/nishishita/food-diary-dish-splitter)
- Single/multi-dish classifier: [Hugging Face](https://huggingface.co/nishishita/food-diary-single-multi-classifier)
- Dish-name classification demo: in preparation.

## Evaluation

Score saved predictions with Python 3.10+ and the standard library only. Input formats are shown in `examples/`; all examples are synthetic.

**Dish boxes: F1, missed and extra boxes**

```bash
python evaluation/evaluate_boxes.py --gt examples/boxes_gt.json --predictions examples/boxes_predictions.json
```

GT boxes use pixels. Predictions use normalized coordinates by default; specify `--prediction-coordinates pixels` for pixel coordinates.

**Dish-name classifier: exact top-1 and top-5 accuracy**

```bash
python evaluation/evaluate_names.py --gt examples/classifier_gt.json --predictions examples/classifier_predictions.json
```

**Naming pipeline: strict and lenient accuracy**

```bash
python evaluation/evaluate_names.py --mode pipeline --gt examples/names_gt.json --predictions examples/names_predictions.json --keywords examples/keywords.json
```

Use `--help` for additional options. Non-public photos, GT annotations, and foo.log Inc.'s dish-name master data are not included.

## Annotation guidelines

See [ANNOTATION.md](ANNOTATION.md) for dish-counting, bounding-box, and naming rules.

## License

The code is licensed under [Apache-2.0](LICENSE). See each model repository for its model license.
