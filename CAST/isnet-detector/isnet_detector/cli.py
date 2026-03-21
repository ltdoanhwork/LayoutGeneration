"""
CLI entry point: isnet-detect

Example:
    isnet-detect --input /path/to/frames --output /path/to/masks \\
                 --weights /path/to/isnetis.ckpt
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from . import SimpleISNetDetector


def main():
    parser = argparse.ArgumentParser(description="ISNet foreground detector")
    parser.add_argument("--input", required=True, help="Directory of input frames (jpg/png)")
    parser.add_argument("--output", required=True, help="Output directory for masks + summary.json")
    parser.add_argument("--weights", default=None, help="Path to isnetis.ckpt checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--merge-kernel", type=int, default=11)
    parser.add_argument("--dilate-iter", type=int, default=2)
    parser.add_argument("--erode-iter", type=int, default=1)
    parser.add_argument("--pre-filter-area", type=int, default=100)
    parser.add_argument("--adaptive-morph", action="store_true")
    parser.add_argument("--save-masks", action="store_true",
                        help="Save per-frame binary masks as PNG (for --warp-mask-dir in CAST)")
    parser.add_argument("--use-u2net", action="store_true",
                        help="Use U2Net to generate saved segmentation masks (--save-masks). Detection bboxes always use ISNet.")
    parser.add_argument("--seg-threshold", type=float, default=0.5,
                        help="Threshold for U2Net segmentation mask when --use-u2net (default: 0.5)")
    args = parser.parse_args()

    detector = SimpleISNetDetector(model_path=args.weights, device=args.device, use_u2net=args.use_u2net)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks_filtered"
    if args.save_masks:
        masks_dir.mkdir(exist_ok=True)

    frames = sorted(Path(args.input).glob("*.jpg")) + sorted(Path(args.input).glob("*.png"))
    frames = sorted(frames)

    print(f"Processing {len(frames)} frames → {output_dir}")

    summary = {
        "frames": [],
        "total_objects": 0,
        "params": vars(args),
    }

    for idx, frame_path in enumerate(frames, 1):
        objects, mask_binary = detector.detect_objects(
            str(frame_path),
            threshold=args.threshold,
            min_area=args.min_area,
            merge_kernel=args.merge_kernel,
            dilate_iter=args.dilate_iter,
            erode_iter=args.erode_iter,
            pre_filter_area=args.pre_filter_area,
            adaptive_morph=args.adaptive_morph,
        )
        print(f"[{idx}/{len(frames)}] {frame_path.name} → {len(objects)} objects")

        # Compute segmentation mask for visualization and saving
        if args.use_u2net:
            # U2Net: clean soft mask — computed once, used for both vis and save
            seg_soft = detector.get_mask(np.array(Image.open(str(frame_path)).convert('RGB')))
            vis_mask = (seg_soft[:, :, 0] > args.seg_threshold).astype(np.uint8) * 255
        else:
            vis_mask = mask_binary

        # Save visualization with mask overlay + ISNet detection bboxes
        vis_path = output_dir / f"{idx:04d}_{frame_path.stem}.jpg"
        detector.visualize(str(frame_path), objects, vis_mask, str(vis_path))

        # Save binary mask PNG (used as --warp-mask-dir in CAST)
        if args.save_masks:
            mask_out = masks_dir / frame_path.name
            cv2.imwrite(str(mask_out), vis_mask)

        summary["frames"].append({
            "name": frame_path.name,
            "num_objects": len(objects),
            "objects": [
                {
                    "bbox": list(o["bbox"]),
                    "pixel_area": o["pixel_area"],
                    "bbox_area": o["bbox_area"],
                    "size": list(o["size"]),
                    "confidence": o["confidence"],
                }
                for o in objects
            ],
        })
        summary["total_objects"] += len(objects)

    summary["avg_objects_per_frame"] = (
        summary["total_objects"] / len(frames) if frames else 0
    )

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Done. summary.json → {summary_path}")
    if args.save_masks:
        seg_source = "U2Net" if args.use_u2net else "ISNet"
        print(f"✅ Binary masks [{seg_source}] → {masks_dir}  (use as --warp-mask-dir in CAST)")


if __name__ == "__main__":
    main()
