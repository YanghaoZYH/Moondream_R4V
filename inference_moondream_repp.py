import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM

from main_repp import REPP_MVUE
from perturbation import Perturbation_Biasfield


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="vikhyatk/moondream2")
    parser.add_argument("--revision", type=str, default="2025-06-21")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="output_repp")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--bias-order", type=int, default=1)
    parser.add_argument("--iou-threshold", type=float, default=0.8)
    parser.add_argument("--count-particles", type=int, default=16)
    parser.add_argument("--count-mh-steps", type=int, default=20)
    parser.add_argument("--timeout-threshold", type=int, default=120)
    parser.add_argument("--grid-size", type=int, default=101)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_model(model_path, revision, device, local_files_only):
    device_map = {"": "cuda"} if device == "cuda" else {"": "cpu"}
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        trust_remote_code=True,
        device_map=device_map,
        local_files_only=local_files_only,
    )


def pil_to_chw_tensor(image):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def chw_tensor_to_pil(image_chw):
    image_chw = image_chw.detach().cpu().clamp(0, 255).to(torch.uint8)
    arr = image_chw.permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


def get_box_from_object(obj):
    box = obj.get("bbox") or obj.get("box") or obj.get("bounding_box")
    if box is not None:
        return [float(v) for v in box]
    if all(key in obj for key in ("x_min", "y_min", "x_max", "y_max")):
        return [float(obj["x_min"]), float(obj["y_min"]), float(obj["x_max"]), float(obj["y_max"])]
    return None


def normalize_box_to_pixels(box, width, height):
    x0, y0, x1, y1 = [float(v) for v in box]
    if all(0.0 <= v <= 1.0 for v in (x0, y0, x1, y1)):
        return [x0 * width, y0 * height, x1 * width, y1 * height]
    return [x0, y0, x1, y1]


def extract_first_box(detect_result, width, height):
    for obj in detect_result.get("objects", []):
        box = get_box_from_object(obj)
        if box is not None:
            return normalize_box_to_pixels(box, width, height)
    return None


def compute_iou(box_a, box_b):
    if box_a is None or box_b is None:
        return 0.0

    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def draw_box(image, box, color, label):
    if box is None:
        return
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = box
    draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
    draw.text((x0 + 4, y0 + 4), label, fill=color)


class MoondreamDetectionProblem:
    def __init__(self, model, image, query, perturbation, reference_box):
        self.model = model
        self.image = image
        self.query = query
        self.perturbation = perturbation
        self.reference_box = reference_box
        self.width, self.height = image.size
        self.frames = pil_to_chw_tensor(image).unsqueeze(0)
        self.perturbation.prepare(self.frames)
        self.last_eval = None

    def __call__(self, coeff_batch):
        if isinstance(coeff_batch, np.ndarray):
            coeff_batch = torch.from_numpy(coeff_batch)
        if not torch.is_tensor(coeff_batch):
            coeff_batch = torch.tensor(coeff_batch)
        coeff_batch = coeff_batch.to(
            device=self.perturbation.frames_prepared.device,
            dtype=self.perturbation.dtype,
        )
        if coeff_batch.ndim == 1:
            coeff_batch = coeff_batch.unsqueeze(0)

        perturbed = self.perturbation.transform_func(coeff_batch)
        losses = []
        eval_details = []
        for batch_idx in range(perturbed.shape[0]):
            pil_image = chw_tensor_to_pil(perturbed[batch_idx, 0])
            detect_result = self.model.detect(pil_image, self.query)
            pred_box = extract_first_box(detect_result, self.width, self.height)
            iou = compute_iou(self.reference_box, pred_box)
            loss = iou - 0.8
            losses.append(loss)
            eval_details.append(
                {
                    "iou": iou,
                    "pred_box": pred_box,
                    "detect": detect_result,
                }
            )
        self.last_eval = eval_details
        return torch.tensor(
            losses,
            device=coeff_batch.device,
            dtype=self.perturbation.dtype,
        )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.model_path, args.revision, args.device, args.local_files_only)
    image = Image.open(args.image).convert("RGB")
    width, height = image.size

    clean_detect = model.detect(image, args.query)
    reference_box = extract_first_box(clean_detect, width, height)
    if reference_box is None:
        raise RuntimeError("Clean detection did not return a usable bounding box.")

    perturbation = Perturbation_Biasfield(
        epsilon=args.epsilon,
        order=args.bias_order,
        clip_inputs=True,
        clip_min=0.0,
        clip_max=255.0,
        device=args.device,
        dtype=torch.float32,
    )

    problem = MoondreamDetectionProblem(
        model=model,
        image=image,
        query=args.query,
        perturbation=perturbation,
        reference_box=reference_box,
    )

    bounds = perturbation.bounds.detach().cpu().numpy()
    solver = REPP_MVUE(
        problem=problem,
        nb_var=perturbation.num_coeff,
        bounds=bounds,
        count_particles=args.count_particles,
        count_mh_steps=args.count_mh_steps,
        timeout_threshold=args.timeout_threshold,
        grid_sizes=args.grid_size,
        threshold=0.0,
        verify=True,
    )
    solver.solve()

    best_coeff = solver.best_x.detach().cpu() if torch.is_tensor(solver.best_x) else torch.tensor(solver.best_x)
    best_loss = float(problem(best_coeff.unsqueeze(0))[0].item())
    best_eval = problem.last_eval[0]
    best_iou = float(best_eval["iou"])
    success = best_iou < args.iou_threshold

    perturbed = problem.perturbation.transform_func(best_coeff.unsqueeze(0))[0, 0]
    adv_image = chw_tensor_to_pil(perturbed)
    clean_vis = image.copy()
    adv_vis = adv_image.copy()
    draw_box(clean_vis, reference_box, (54, 104, 191), "clean_ref")
    draw_box(adv_vis, reference_box, (54, 104, 191), "clean_ref")
    draw_box(adv_vis, best_eval["pred_box"], (255, 80, 40), "adv_pred")
    clean_vis.save(output_dir / "clean_box.png")
    adv_vis.save(output_dir / "adv_box.png")

    result = {
        "model_path": args.model_path,
        "revision": args.revision,
        "image_path": args.image,
        "query": args.query,
        "epsilon": args.epsilon,
        "bias_order": args.bias_order,
        "iou_threshold": args.iou_threshold,
        "clean": {
            "detect": clean_detect,
            "reference_box": reference_box,
        },
        "repp": {
            "success": success,
            "best_iou": best_iou,
            "best_loss": best_loss,
            "best_coeff": best_coeff.tolist(),
            "best_detect": best_eval["detect"],
            "best_pred_box": best_eval["pred_box"],
            "best_y": float(solver.best_y),
            "log_p_point": float(solver.log_p_point_star_true),
            "queries": int(solver.query) if hasattr(solver, "query") else None,
        },
    }

    with open(output_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
