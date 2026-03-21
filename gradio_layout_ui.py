import os
import subprocess
import sys
import tempfile
from datetime import datetime

import cv2
import numpy as np


# Use the user-specified cache directory for Gradio temp files.
LOCAL_GRADIO_TMP = "/home/serverai/ltdoanh/LayoutGeneration/CAST/cache_gradio"
os.makedirs(LOCAL_GRADIO_TMP, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = LOCAL_GRADIO_TMP

import gradio as gr


def _log(message: str) -> str:
    now = datetime.now().strftime("%H:%M:%S")
    return f"[{now}] {message}"


def _read_video_metadata(video_path: str) -> dict:
    if not video_path or not os.path.exists(video_path):
        return {"status": "No video uploaded"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"status": "Failed to open video"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    return {
        "status": "ready",
        "fps": round(float(fps), 2),
        "frame_count": frame_count,
        "resolution": f"{width}x{height}",
        "duration_sec": round(float(duration), 2),
    }


def extract_keyframes(video_path: str, diversity: float, sharpness: float, top_k: int):
    if not video_path or not os.path.exists(video_path):
        return [], [], _log("Please upload a video before extracting keyframes."), 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], _log("Cannot open video."), 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    top_k = max(1, min(int(top_k), 24))
    if frame_count <= 0:
        cap.release()
        return [], [], _log("Video has no readable frames."), 0

    sample_ids = np.linspace(0, max(0, frame_count - 1), num=top_k, dtype=np.int32)

    keyframes = []
    gallery = []
    for idx, fid in enumerate(sample_ids.tolist()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)

        label = f"KF {idx + 1} | frame {fid}"
        keyframes.append(frame)
        gallery.append((frame, label))

    cap.release()

    if not gallery:
        return [], [], _log("No keyframes extracted."), 0

    msg = (
        f"Extracted {len(gallery)} keyframes "
        f"(diversity={diversity:.2f}, sharpness={sharpness:.2f}, top_k={top_k})."
    )
    return gallery, keyframes, _log(msg), 100


def _build_mask_from_background(background: np.ndarray) -> np.ndarray:
    if background is None:
        return None

    img = background.copy()
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k, iterations=1)
    return cleaned


def generate_auto_mask(background: np.ndarray):
    if background is None:
        return None, _log("Upload a background image first.")

    mask = _build_mask_from_background(background)
    if mask is None:
        return None, _log("Mask generation failed.")

    return mask, _log("Auto mask generated.")


def _prepare_manual_mask(manual_mask: np.ndarray, bg_shape):
    if manual_mask is None:
        return None

    img = manual_mask.copy()
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(img, (bg_shape[1], bg_shape[0]), interpolation=cv2.INTER_NEAREST)
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    return binary


def apply_mask(background: np.ndarray, auto_mask: np.ndarray, manual_mask: np.ndarray, opacity: float):
    if background is None:
        return None, None, _log("Upload background before applying mask.")

    chosen = None
    if manual_mask is not None:
        chosen = _prepare_manual_mask(manual_mask, background.shape[:2])
    elif auto_mask is not None:
        chosen = cv2.resize(auto_mask, (background.shape[1], background.shape[0]), interpolation=cv2.INTER_NEAREST)

    if chosen is None:
        return None, None, _log("No mask available. Use Auto Mask or upload manual mask.")

    bg = background.copy().astype(np.uint8)
    overlay = bg.copy()
    color = np.zeros_like(bg)
    color[..., 1] = 255

    alpha = float(np.clip(opacity, 0.0, 1.0))
    mask_bool = chosen > 0
    overlay[mask_bool] = cv2.addWeighted(bg[mask_bool], 1 - alpha, color[mask_bool], alpha, 0)
    return overlay, chosen, _log("Mask applied to preview.")


def _compute_voronoi_label_map(h: int, w: int, num_cells: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    points = np.column_stack([rng.integers(0, h, size=num_cells), rng.integers(0, w, size=num_cells)])

    yy, xx = np.indices((h, w))
    d2 = (yy[..., None] - points[:, 0]) ** 2 + (xx[..., None] - points[:, 1]) ** 2
    labels = np.argmin(d2, axis=2).astype(np.int32)
    return labels


def _render_voronoi(labels: np.ndarray, show_borders: bool, color_cells: bool) -> np.ndarray:
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    if color_cells:
        rng = np.random.default_rng(0)
        palette = rng.integers(40, 240, size=(labels.max() + 1, 3), dtype=np.uint8)
        out = palette[labels]
    else:
        out[:] = 240

    if show_borders:
        edges = np.zeros((h, w), dtype=np.uint8)
        edges[1:, :] |= labels[1:, :] != labels[:-1, :]
        edges[:, 1:] |= labels[:, 1:] != labels[:, :-1]
        out[edges > 0] = np.array([20, 20, 20], dtype=np.uint8)

    return out


def generate_layout(
    background: np.ndarray,
    applied_mask: np.ndarray,
    keyframes: list,
    selected_indices: list,
    num_cells: int,
    seed: int,
    min_area: int,
    toggles: list,
):
    if background is None:
        return None, None, None, _log("Background is required before generating layout.")

    if not keyframes:
        return None, None, None, _log("Extract keyframes first.")

    if not selected_indices:
        selected_indices = list(range(min(6, len(keyframes))))

    h, w = background.shape[:2]
    labels = _compute_voronoi_label_map(h, w, max(5, int(num_cells)), int(seed))

    if int(min_area) > 0:
        area_threshold = int(min_area)
        unique, counts = np.unique(labels, return_counts=True)
        for lab, cnt in zip(unique.tolist(), counts.tolist()):
            if cnt < area_threshold:
                labels[labels == lab] = unique[int(np.argmax(counts))]

    show_borders = "Show Borders" in (toggles or [])
    color_cells = "Color Cells" in (toggles or [])
    overlay_mask = "Overlay Mask" in (toggles or [])

    voronoi_preview = _render_voronoi(labels, show_borders, color_cells)

    comp = background.copy().astype(np.uint8)
    selected_imgs = [keyframes[i] for i in selected_indices if 0 <= i < len(keyframes)]
    if not selected_imgs:
        selected_imgs = keyframes[:1]

    for cell_id in np.unique(labels):
        region = labels == cell_id
        if region.sum() == 0:
            continue

        src = selected_imgs[int(cell_id) % len(selected_imgs)]
        src_rs = cv2.resize(src, (w, h), interpolation=cv2.INTER_LINEAR)
        comp[region] = src_rs[region]

    if applied_mask is not None:
        mask_rs = cv2.resize(applied_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        bg = background.copy().astype(np.uint8)
        m = mask_rs > 0
        final_img = bg.copy()
        final_img[m] = comp[m]
    else:
        final_img = comp

    if overlay_mask and applied_mask is not None:
        mask_vis = np.zeros_like(final_img)
        mask_vis[..., 0] = 255
        m = applied_mask > 0
        final_img[m] = cv2.addWeighted(final_img[m], 0.75, mask_vis[m], 0.25, 0)

    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_file.close()
    cv2.imwrite(temp_file.name, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))

    return voronoi_preview, final_img, temp_file.name, _log("Voronoi and final layout generated.")


def _resolve_path(path_text: str) -> str:
    if not path_text:
        return ""
    p = os.path.expanduser(path_text.strip())
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(os.path.dirname(__file__), p))


def _load_rgb(path: str):
    if not path or not os.path.exists(path):
        return None
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None

    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    if im.shape[2] == 4:
        bgr = im[..., :3]
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def _overlay_voronoi_edges(base_rgb: np.ndarray, voronoi_mask_rgb: np.ndarray) -> np.ndarray:
    if base_rgb is None and voronoi_mask_rgb is None:
        return None
    if base_rgb is None:
        return voronoi_mask_rgb
    if voronoi_mask_rgb is None:
        return base_rgb

    h, w = base_rgb.shape[:2]
    vm = cv2.resize(voronoi_mask_rgb, (w, h), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(vm, cv2.COLOR_RGB2GRAY)

    edge = cv2.Canny(gray, 60, 140)
    out = base_rgb.copy()
    out[edge > 0] = np.array([0, 255, 255], dtype=np.uint8)
    return out


def evaluate_and_visualize(output_dir_text: str, shape_path_text: str):
    output_dir = _resolve_path(output_dir_text)
    shape_path = _resolve_path(shape_path_text)

    if not os.path.isdir(output_dir):
        return None, None, None, _log(f"Invalid output_dir: {output_dir}")
    if not os.path.exists(shape_path):
        return None, None, None, _log(f"Shape file not found: {shape_path}")

    eval_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "CAST", "evaluation.py"))
    if not os.path.exists(eval_script):
        return None, None, None, _log("Cannot find CAST/evaluation.py")

    cmd = [sys.executable, eval_script, "--output_dir", output_dir, "--shape", shape_path]
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    except Exception as exc:
        return None, None, None, _log(f"Evaluation failed: {exc}")

    metrics_json_path = os.path.join(output_dir, "evaluation_metrics.json")
    metrics_csv_path = os.path.join(output_dir, "evaluation_metrics.csv")

    metrics = None
    if os.path.exists(metrics_json_path):
        try:
            import json

            with open(metrics_json_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = {"status": "evaluation done but metrics json cannot be parsed"}
    else:
        metrics = {"status": "evaluation finished but evaluation_metrics.json not found"}

    collage_path = os.path.join(output_dir, "collage.png")
    voronoi_path = os.path.join(output_dir, "_voronoi_temp.png")
    base_rgb = _load_rgb(collage_path)
    voronoi_rgb = _load_rgb(voronoi_path if os.path.exists(voronoi_path) else shape_path)
    single_preview = _overlay_voronoi_edges(base_rgb, voronoi_rgb)

    if run.returncode != 0:
        err = run.stderr.strip() if run.stderr else "Unknown error"
        return single_preview, metrics, (metrics_csv_path if os.path.exists(metrics_csv_path) else None), _log(
            f"Evaluation returned code {run.returncode}: {err[:220]}"
        )

    return (
        single_preview,
        metrics,
        (metrics_csv_path if os.path.exists(metrics_csv_path) else None),
        _log("Evaluation complete. Single visualization generated from model outputs."),
    )


def on_select_keyframe(evt: gr.SelectData, selected_indices: list, keyframes: list):
    selected = list(selected_indices or [])
    idx = int(evt.index)
    if idx in selected:
        selected.remove(idx)
    else:
        selected.append(idx)
    selected.sort()

    selected_preview = []
    for i in selected:
        if 0 <= i < len(keyframes):
            selected_preview.append((keyframes[i], f"Selected KF {i + 1}"))

    summary = "**Selected:** " + (", ".join(str(i + 1) for i in selected) if selected else "none")
    return selected, summary, selected_preview


def reset_selection():
    return [], "**Selected:** none", []


CUSTOM_CSS = """
.gradio-container {
    background: linear-gradient(135deg, #eef2ff 0%, #e2e8f0 45%, #e8fff7 100%) !important;
    font-family: "Manrope", "Segoe UI", sans-serif !important;
}

#app-header {
    text-align: center;
    padding: 18px;
    margin-bottom: 16px;
    background: rgba(255, 255, 255, 0.58);
    border: 1px solid rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(51, 65, 85, 0.08);
}

#app-header h1 {
    margin: 0;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: 0.01em;
    background: linear-gradient(90deg, #4f46e5 0%, #10b981 85%);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

#app-header p {
    margin-top: 8px;
    color: #334155;
}

.tabs {
    border: none !important;
}

.tab-nav {
    border-radius: 12px !important;
    border: 1px solid rgba(148, 163, 184, 0.35) !important;
    background: rgba(255, 255, 255, 0.45) !important;
    backdrop-filter: blur(7px);
    padding: 6px !important;
}

.tab-nav button.selected {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 8px !important;
    border-bottom: 3px solid #4f46e5 !important;
    box-shadow: 0 5px 14px rgba(51, 65, 85, 0.12) !important;
}

.panel-card {
    background: rgba(255, 255, 255, 0.8) !important;
    border: 1px solid rgba(148, 163, 184, 0.22) !important;
    border-radius: 16px !important;
    padding: 14px !important;
    box-shadow: 0 12px 25px rgba(30, 41, 59, 0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.panel-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 18px 34px rgba(30, 41, 59, 0.12);
}

button.primary {
    border: none !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    background: linear-gradient(90deg, #4f46e5, #6366f1) !important;
    border-radius: 10px !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 7px 16px rgba(79, 70, 229, 0.35) !important;
}

.status-box {
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    background: #0f172a !important;
}

.status-box textarea {
    color: #34d399 !important;
    background: transparent !important;
    font-family: "JetBrains Mono", "Consolas", monospace !important;
    font-size: 0.92rem !important;
}

.gr-gallery {
    border-radius: 12px !important;
    overflow: hidden;
}
"""

my_theme = gr.themes.Default(
        primary_hue="indigo",
        secondary_hue="emerald",
        neutral_hue="slate",
).set(
        block_radius="12px",
        button_large_radius="10px",
        block_title_text_weight="600",
)


with gr.Blocks(theme=my_theme, css=CUSTOM_CSS) as demo:
    with gr.Column(elem_id="app-header"):
        gr.Markdown("# Layout Gen Auto Studio")
        gr.Markdown("Model-first visualization: run evaluation and preview directly from pipeline outputs")

    with gr.Tabs(selected="visualize"):
        with gr.Tab("Visualize All", id="visualize"):
            gr.Markdown("### Automated Model Output Evaluation")
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["panel-card"]):
                    eval_output_dir = gr.Textbox(
                        label="Output Directory",
                        value="./CAST/tests/output_onepiece",
                        placeholder="Example: ./CAST/tests/output_onepiece",
                    )
                    eval_shape_path = gr.Textbox(
                        label="Shape Path",
                        value="./CAST/tests/output_onepiece/_voronoi_temp.png",
                        placeholder="Example: ./CAST/tests/output_onepiece/_voronoi_temp.png",
                    )
                    btn_run_eval = gr.Button("Run Evaluation and Visualize", variant="primary", size="lg")
                    eval_metrics = gr.JSON(label="Evaluation Metrics")
                    export_file = gr.File(label="Download Metrics CSV")
                with gr.Column(scale=2, elem_classes=["panel-card"]):
                    single_preview = gr.Image(label="Single Visualization (Collage + Voronoi Edges)")

    status_log = gr.Textbox(
        label="System Console",
        value=_log("Ready to start."),
        interactive=False,
        elem_classes=["status-box"],
    )

    btn_run_eval.click(
        evaluate_and_visualize,
        inputs=[eval_output_dir, eval_shape_path],
        outputs=[single_preview, eval_metrics, export_file, status_log],
    )

if __name__ == "__main__":
    demo.launch()