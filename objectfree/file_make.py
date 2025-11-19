from ultralytics import YOLOWorld

# Ultralytics' YOLOWorld interface expects a .pt checkpoint that bundles both
# the architecture definition and weights. Passing an MMEngine-style `.pth`
# file (as released with the official YOLO-World repo) will raise an
# AttributeError because the underlying model stub is just a string path.
#
# Using the model name below lets Ultralytics download the matching checkpoint
# automatically. If you truly need to finetune from the MMEngine weights in
# `YOLO-World/weights/`, convert them to a `.pt` export first using the
# official conversion script (see README) and then point this script at the
# converted file.
model = YOLOWorld("yolov8s-worldv2.pt")

results = model.train(
	data="/home/serverai/ltdoanh/LayoutGeneration/yolo_splits/generic/data.yaml",
	epochs=100,
	imgsz=640,
)