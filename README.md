# Moondream_R4V

Local `moondream2` grounding pipeline for:

- object detection
- pointing
- structured JSON output
- PNG visualization

## Install

```bash
pip install "transformers>=4.51.1" "torch>=2.7.0" "accelerate>=1.10.0" "Pillow>=11.0.0"
```

## Run

Detection + pointing:

```bash
python inference_moondream_grounding.py \
  --model-path vikhyatk/moondream2 \
  --revision 2025-06-21 \
  --image path/to/image.jpg \
  --query "red box" \
  --mode both \
  --device cuda \
  --output-dir output/red_box
```

Repo demo image:

```bash
python inference_moondream_grounding.py \
  --model-path vikhyatk/moondream2 \
  --revision 2025-06-21 \
  --image examples/red_box_scene.png \
  --query "red box" \
  --mode both \
  --device cuda \
  --output-dir output/red_box_scene
```

Detection only:

```bash
python inference_moondream_grounding.py \
  --image path/to/image.jpg \
  --query "person" \
  --mode detect \
  --device cuda \
  --output-dir output/person_detect
```

Pointing only:

```bash
python inference_moondream_grounding.py \
  --image path/to/image.jpg \
  --query "person" \
  --mode point \
  --device cuda \
  --output-dir output/person_point
```

Repeat the same image 5 times to test whether outputs are identical:

```bash
python inference_moondream_grounding.py \
  --image examples/red_box_scene.png \
  --query "red box" \
  --mode both \
  --device cuda \
  --repeat-count 5 \
  --output-dir output/red_box_scene_repeat5
```

## Outputs

Each run writes:

- `result.json`
- `detect.png` when `mode` includes `detect`
- `point.png` when `mode` includes `point`

When `--repeat-count > 1`, it also writes:

- `run_000/result.json`, `run_001/result.json`, ...
- `batch_summary.json` with equality checks across runs

## REPP on Detection IoU

Use the clean detection box as the reference box, search for a bias-field perturbation, and mark success when perturbed IoU drops below `0.8`:

```bash
python inference_moondream_repp.py \
  --image demo/car.jpg \
  --query "car" \
  --device cpu \
  --epsilon 0.1 \
  --iou-threshold 0.8 \
  --count-particles 16 \
  --count-mh-steps 20 \
  --output-dir output/car_repp
```

Outputs:

- `result.json`
- `clean_box.png`
- `adv_box.png`

## Notes

- This uses the local open-source `vikhyatk/moondream2` model.
- It does not call the Moondream API.
- Box and point coordinates are converted to pixels for visualization.
- A minimal synthetic example image is included at `examples/red_box_scene.png`.
