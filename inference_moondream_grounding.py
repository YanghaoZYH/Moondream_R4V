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
    for idx, point in enumerate(points):
        x, y = normalize_point_to_pixels(point, width, height)
        r = 7
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 80, 40), outline=(255, 255, 255), width=2)
        draw.text((x + 8, y - 8), f"p{idx}", fill=(255, 80, 40))
    canvas.save(save_path)


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


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_path, args.revision, args.device, args.local_files_only)
    image = Image.open(args.image).convert("RGB")

    result = {
        "model_path": args.model_path,
        "revision": args.revision,
        "image_path": args.image,
        "query": args.query,
        "mode": args.mode,
        "detect": None,
        "point": None,
    }

    if args.mode in ("detect", "both"):
        detect_result = model.detect(image, args.query)
        result["detect"] = detect_result
        draw_boxes(image, detect_result.get("objects", []), output_dir / "detect.png")

    if args.mode in ("point", "both"):
        point_result = model.point(image, args.query)
        result["point"] = point_result
        draw_points(image, point_result.get("points", []), output_dir / "point.png")

    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
