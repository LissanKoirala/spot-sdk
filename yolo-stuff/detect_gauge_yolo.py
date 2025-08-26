
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser(description="YOLO gauge detector (single-class: gauge)")
    ap.add_argument("--weights", type=str, required=True, help="Path to YOLO weights (e.g., best.pt)")
    ap.add_argument("--source", type=str, required=True,
                    help="Camera index (e.g., 0), image/video file, or directory of images")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.6, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--device", type=str, default=None, help="cuda:0 or cpu (auto if None)")
    ap.add_argument("--view", action="store_true", help="Visualize detections")
    ap.add_argument("--save-crops", action="store_true", help="Save cropped gauge images when detected")
    ap.add_argument("--crop-dir", type=str, default="gauge_crops", help="Directory to save crops")
    return ap.parse_args()

def is_int(s: str):
    try:
        int(s); return True
    except:
        return False

def draw_box(img, x1, y1, x2, y2, conf):
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(img, f"gauge {conf:.2f}", (x1, max(20, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

def save_crop(img, x1, y1, x2, y2, outdir: Path, stem: str):
    outdir.mkdir(parents=True, exist_ok=True)
    crop = img[max(0,y1):y2, max(0,x1):x2]
    p = outdir / f"{stem}_{x1}-{y1}-{x2}-{y2}.jpg"
    cv2.imwrite(str(p), crop)

def handle_images(model, args: argparse.Namespace, paths):
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[WARN] Could not read {p}")
            continue
        results = model.predict(img, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)
        h, w = img.shape[:2]
        found = False
        best = None
        for r in results:
            if r.boxes is None: continue
            for b in r.boxes:
                conf = float(b.conf[0])
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                if best is None or conf > best[0]:
                    best = (conf, x1, y1, x2, y2)
        if best is not None:
            found = True
            conf,x1,y1,x2,y2 = best
            print(f"GAUGE_FOUND path={p} conf={conf:.3f} bbox={x1},{y1},{x2},{y2}")
            if args.save_crops:
                save_crop(img, x1,y1,x2,y2, Path(args.crop_dir), Path(p).stem)
            if args.view:
                draw_box(img, x1,y1,x2,y2, conf)
        else:
            print(f"NO_GAUGE path={p}")

        if args.view:
            cv2.imshow("gauge-detect", img)
            key = cv2.waitKey(0) & 0xFF
            if key == 27: break
    if args.view:
        cv2.destroyAllWindows()

def handle_video(model, args: argparse.Namespace, src):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[ERR] Cannot open video/camera source: {src}")
        return

    frame_id = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)
        found = False
        best = None
        for r in results:
            if r.boxes is None: continue
            for b in r.boxes:
                conf = float(b.conf[0])
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                if best is None or conf > best[0]:
                    best = (conf, x1, y1, x2, y2)
        if best is not None:
            found = True
            conf,x1,y1,x2,y2 = best
            print(f"GAUGE_FOUND frame={frame_id} conf={conf:.3f} bbox={x1},{y1},{x2},{y2}")
            if args.save_crops:
                save_crop(frame, x1,y1,x2,y2, Path(args.crop_dir), f"f{frame_id:06d}")
            if args.view:
                draw_box(frame, x1,y1,x2,y2, conf)
        else:
            print(f"NO_GAUGE frame={frame_id}")

        if args.view:
            cv2.imshow("gauge-detect", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break

    cap.release()
    if args.view:
        cv2.destroyAllWindows()
    dur = time.time() - t0
    print(f"[INFO] Processed {frame_id} frames in {dur:.2f}s ({frame_id/dur:.2f} FPS)")

def main():
    args = parse_args()
    model = YOLO(args.weights)

    src = args.source
    p = Path(src)
    if is_int(src):
        handle_video(model, args, int(src))
    elif p.is_dir():
        paths = sorted([pp for pp in p.iterdir() if pp.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}])
        handle_images(model, args, paths)
    else:
        # decide if it's an image or video by extension
        if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}:
            handle_images(model, args, [p])
        else:
            handle_video(model, args, src)

if __name__ == "__main__":
    main()
