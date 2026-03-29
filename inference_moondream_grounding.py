import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="vikhyatk/moondream2")
    parser.add_argument("--revision", type=str, default="2025-06-21")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--mode", choices=["detect", "point", "both"], default="both")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_model(model_path, revision, device, local_files_only):
    if device == "cuda":
        device_map = {"": "cuda"}
    else:
        device_map = {"": "cpu"}
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        trust_remote_code=True,
        device_map=device_map,
        local_files_only=local_files_only,
    )


def image_size(image):
    width, height = image.size
    return width, height


def normalize_point_to_pixels(point, width, height):
    if isinstance(point, dict):
        if "x" in point and "y" in point:
            x, y = float(point["x"]), float(point["y"])
        elif "point" in point and len(point["point"]) == 2:
            x, y = float(point["point"][0]), float(point["point"][1])
        elif "xy" in point and len(point["xy"]) == 2:
            x, y = float(point["xy"][0]), float(point["xy"][1])
        elif "coordinates" in point and len(point["coordinates"]) == 2:
            x, y = float(point["coordinates"][0]), float(point["coordinates"][1])
        else:
            raise ValueError(f"Expected point dict with x/y or point, got {point}")
    else:
        if len(point) != 2:
            raise ValueError(f"Expected 2D point, got {point}")
        x, y = float(point[0]), float(point[1])
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return x * width, y * height
    return x, y


def normalize_box_to_pixels(box, width, height):
    if len(box) != 4:
        raise ValueError(f"Expected 4D box, got {box}")
    x0, y0, x1, y1 = [float(v) for v in box]
    if all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
        return x0 * width, y0 * height, x1 * width, y1 * height
    return x0, y0, x1, y1


def draw_points(image, points, save_path):
    width, height = image.size
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    skipped_points = []
    for idx, point in enumerate(points):
        try:
            x, y = normalize_point_to_pixels(point, width, height)
        except (TypeError, ValueError, KeyError, IndexError):
            skipped_points.append({"index": idx, "raw": point})
            continue
        r = 7
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 80, 40), outline=(255, 255, 255), width=2)
        label = point.get("label") if isinstance(point, dict) else None
        draw.text((x + 8, y - 8), label or f"p{idx}", fill=(255, 80, 40))
    canvas.save(save_path)
    return skipped_points


def draw_boxes(image, objects, save_path):
    width, height = image.size
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    for idx, obj in enumerate(objects):
        box = obj.get("bbox") or obj.get("box") or obj.get("bounding_box")
        if box is None:
            continue
        x0, y0, x1, y1 = normalize_box_to_pixels(box, width, height)
        draw.rectangle((x0, y0, x1, y1), outline=(54, 104, 191), width=3)
        label = obj.get("label") or obj.get("name") or f"obj{idx}"
        draw.text((x0 + 4, y0 + 4), label, fill=(54, 104, 191))
    canvas.save(save_path)


def run_single_inference(model, image, image_path, query, mode, output_dir, run_index=None):
    result = {
        "image_path": image_path,
        "query": query,
        "mode": mode,
        "run_index": run_index,
        "detect": None,
        "point": None,
    }

    if mode in ("detect", "both"):
        detect_result = model.detect(image, query)
        result["detect"] = detect_result
        draw_boxes(image, detect_result.get("objects", []), output_dir / "detect.png")

    if mode in ("point", "both"):
        point_result = model.point(image, query)
        result["point"] = point_result
        skipped_points = draw_points(image, point_result.get("points", []), output_dir / "point.png")
        if skipped_points:
            result["point_visualization_skipped"] = skipped_points

    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def compare_runs(results):
    if not results:
        return {
            "num_runs": 0,
            "all_detect_equal": True,
            "all_point_equal": True,
            "all_equal": True,
            "pairwise": [],
        }

    baseline = results[0]
    pairwise = []
    all_detect_equal = True
    all_point_equal = True
    for result in results[1:]:
        detect_equal = result.get("detect") == baseline.get("detect")
        point_equal = result.get("point") == baseline.get("point")
        all_detect_equal = all_detect_equal and detect_equal
        all_point_equal = all_point_equal and point_equal
        pairwise.append(
            {
                "run": result["run_index"],
                "detect_equal_vs_run0": detect_equal,
                "point_equal_vs_run0": point_equal,
                "all_equal_vs_run0": detect_equal and point_equal,
            }
        )
    return {
        "num_runs": len(results),
        "all_detect_equal": all_detect_equal,
        "all_point_equal": all_point_equal,
        "all_equal": all_detect_equal and all_point_equal,
        "pairwise": pairwise,
    }


def main():
    args = parse_args()
    if args.repeat_count < 1:
        raise ValueError("--repeat-count must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_path, args.revision, args.device, args.local_files_only)
    image = Image.open(args.image).convert("RGB")

    results = []
    for run_index in range(args.repeat_count):
        run_output_dir = output_dir if args.repeat_count == 1 else output_dir / f"run_{run_index:03d}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        result = run_single_inference(
            model=model,
            image=image,
            image_path=args.image,
            query=args.query,
            mode=args.mode,
            output_dir=run_output_dir,
            run_index=run_index,
        )
        result["model_path"] = args.model_path
        result["revision"] = args.revision
        results.append(result)

    if args.repeat_count == 1:
        print(json.dumps(results[0], indent=2))
        return

    summary = {
        "model_path": args.model_path,
        "revision": args.revision,
        "image_path": args.image,
        "query": args.query,
        "mode": args.mode,
        "repeat_count": args.repeat_count,
        "determinism": compare_runs(results),
    }
    with open(output_dir / "batch_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
