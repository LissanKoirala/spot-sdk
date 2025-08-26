from ultralytics import YOLOE
import cv2

# Initialize a YOLOE model
model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

# Set text prompt to detect person and bus. You only need to do this once after you load the model.
names = ["gauge", "clock"]
model.set_classes(names, model.get_text_pe(names))

# Run detection on the given image
image_path = "meter-photos/photo_000222.jpg"
results = model.predict(image_path)

# BLUR THE IMAGE SO SHADOWS DISSAPEAR AND DONT GET COUNT AS LINES.

# Show results
results[0].show()

print(results[0])

# Crop the best detection box and save as output.jpg
r = results[0]
best = None
if hasattr(r, "boxes") and r.boxes is not None:
    for b in r.boxes:
        # Fallbacks in case attributes differ
        try:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        except Exception:
            continue
        try:
            conf = float(b.conf[0])
        except Exception:
            # If confidence not available, prioritize by area
            conf = (x2 - x1) * (y2 - y1)
        if best is None or conf > best[0]:
            best = (conf, x1, y1, x2, y2)

if best is not None:
    _, x1, y1, x2, y2 = best
    img = cv2.imread(image_path)
    if img is not None:
        h, w = img.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        crop = img[y1:y2, x1:x2]
        cv2.imwrite("output.jpg", crop)