import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


# -------------------------------
# TemplateMatcher (ORB + KCF tracker hybrid)
# -------------------------------
class TemplateMatcher:
    def __init__(self, template_gray, min_matches=15, ratio=0.55, min_inliers=8):
        self.template_gray = template_gray
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.kp_t, self.des_t = self.orb.detectAndCompute(template_gray, None)
        self.min_matches = min_matches
        self.ratio = ratio
        self.min_inliers = min_inliers

        self.tracker = None   # OpenCV tracker for continuous following
        self.last_bbox = None

    def _detect_with_orb(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Hf, Wf = gray.shape[:2]
        Ht, Wt = self.template_gray.shape[:2]

        kp_f, des_f = self.orb.detectAndCompute(gray, None)
        if des_f is None or len(kp_f) == 0:
            return None

        knn = self.bf.knnMatch(self.des_t, des_f, k=2)
        good = [m for m, n in knn if m.distance < self.ratio * n.distance]

        if len(good) >= max(self.min_matches, 4):
            src_pts = np.float32([self.kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            inliers = int(mask.sum()) if mask is not None else 0

            if H is not None and inliers >= self.min_inliers:
                corners = np.float32([[0, 0], [0, Ht], [Wt, Ht], [Wt, 0]]).reshape(-1, 1, 2)
                projected = cv2.perspectiveTransform(corners, H)
                x, y, w_box, h_box = cv2.boundingRect(projected)
                # clamp ROI
                x = max(0, min(x, Wf - 1))
                y = max(0, min(y, Hf - 1))
                w_box = max(1, min(w_box, Wf - x))
                h_box = max(1, min(h_box, Hf - y))
                return (x, y, w_box, h_box)
        return None

    def match(self, frame):
        # If we already have a tracker, try to update it
        if self.tracker is not None and self.last_bbox is not None:
            success, bbox = self.tracker.update(frame)
            if success:
                self.last_bbox = tuple(map(int, bbox))
                return self.last_bbox

        # Otherwise, fall back to ORB detection
        bbox = self._detect_with_orb(frame)
        if bbox is not None:
            # Initialize a new tracker from bbox
            self.tracker = cv2.legacy.TrackerKCF_create()
            self.tracker.init(frame, bbox)
            self.last_bbox = bbox
            return bbox

        return None


# -------------------------------
# YOLO person detector
# -------------------------------
def detect_people(frame, model):
    detections = model(frame, device="mps")[0]
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        class_id = data[5]
        if confidence >= 0.5 and class_id == 0:  # class 0 = person
            xmin, ymin, xmax, ymax = map(int, data[:4])
            yield [[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id]


# -------------------------------
# Reframe video: crop template and put at bottom
# -------------------------------
def reframe(video_path, template_path, output_path="output_shorts.mp4"):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Shorts target resolution (standard)
    shorts_w, shorts_h = 1080, 1920

    # Prepare video writer
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (shorts_w, shorts_h)
    )

    # Prepare template matcher
    template = cv2.imread(template_path)
    if template is None:
        raise RuntimeError(f"Cannot read template: {template_path}")
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_matcher = TemplateMatcher(template_gray)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tm = template_matcher.match(frame)
        if tm is not None:
            x, y, w, h = map(int, tm)
            cropped = frame[y:y+h, x:x+w]
            cropped_resized = cv2.resize(cropped, (width, height))
            stacked = np.vstack([frame, cropped_resized])
        else:
            stacked = np.vstack([frame, np.zeros_like(frame)])

        # ---- Resize to Shorts (9:16) ----
        # scale stacked to fit width
        scale = shorts_w / stacked.shape[1]
        new_w = shorts_w
        new_h = int(stacked.shape[0] * scale)
        resized = cv2.resize(stacked, (new_w, new_h))

        # pad vertically to shorts_h
        pad_top = max(0, (shorts_h - new_h) // 2)
        pad_bottom = shorts_h - new_h - pad_top
        final_frame = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, 0, 0,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        out.write(final_frame)

    cap.release()
    out.release()
    print(f"✅ Shorts-style 9:16 video saved at {output_path}")


# -------------------------------
# Main tracking with YOLO + ORB template
# -------------------------------
def track(video_path, template_path, preview=True):
    cap = cv2.VideoCapture(video_path)
    detector = YOLO("yolov8l.pt")
    tracker = DeepSort(max_age=10, embedder='clip_ViT-B/32', embedder_gpu=False)

    # Prepare template matcher
    template = cv2.imread(template_path)
    if template is None:
        raise RuntimeError(f"Cannot read template: {template_path}")
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_matcher = TemplateMatcher(template_gray)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = list(detect_people(frame, detector))

        # ORB+KCF template detection
        tm = template_matcher.match(frame)
        if tm is not None:
            x, y, w, h = map(int, tm)
            detections.append([[x, y, w, h], 0.95, 999])  # class_id=999 for template
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, "TEMPLATE", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        tracks = tracker.update_tracks(detections, frame=frame)

        for track_obj in tracks:
            if not track_obj.is_confirmed():
                continue
            ltrb = track_obj.to_ltrb()
            xmin, ymin, xmax, ymax = map(int, ltrb)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.putText(frame, f"ID {track_obj.track_id}", (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        if preview:
            cv2.imshow("Tracking", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------
# Main entry point with argparse
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO + DeepSORT + ORB template tracking")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("command", type=str, choices=["track", "reframe"],
                        help="Action to perform")
    parser.add_argument("--template", type=str, default=None,
                        help="Path to a template image (for ORB matching)")
    parser.add_argument("--preview", action="store_true", default=False,
                        help="Display the processed video in a window")
    parser.add_argument("--output", type=str, default="output_shorts.mp4",
                        help="Output video path (only for reframe)")

    args = parser.parse_args()

    if args.command == "track":
        if args.template is None:
            raise ValueError("You must provide --template when using 'track'")
        track(args.video_path, args.template, preview=args.preview)

    elif args.command == "reframe":
        if args.template is None:
            raise ValueError("You must provide --template when using 'reframe'")
        reframe(args.video_path, args.template, args.output)
