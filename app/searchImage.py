import cv2
import numpy as np

def make_csrt_tracker():
    # Robust creation across OpenCV builds (contrib vs legacy)
    try:
        return cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        try:
            return cv2.TrackerCSRT_create()
        except AttributeError:
            try:
                return cv2.legacy.TrackerKCF_create()
            except Exception:
                return cv2.TrackerKCF_create()

def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h

def robust_track_with_roi(video_path, template_path,
                          search_roi=None,  # (x,y,w,h) in frame coords, or None
                          interactive=False,  # use selectROI on first frame if True
                          min_matches=15, ratio=0.75, min_inliers=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: %s" % video_path)

    template = cv2.imread(template_path)
    if template is None:
        raise RuntimeError("Cannot read template: %s" % template_path)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    Ht, Wt = template_gray.shape[:2]

    # ORB + BF KNN for ratio test
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)  # good for small logos
    kp_t, des_t = orb.detectAndCompute(template_gray, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    tracker = None
    first_match = None
    frame_num = 0
    roi_fixed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Hf, Wf = gray.shape[:2]

        # One-time interactive ROI selection on first frame
        if interactive and not roi_fixed:
            r = cv2.selectROI("Select ROI (press ENTER)", frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Select ROI (press ENTER)")
            search_roi = tuple(map(int, r))  # (x,y,w,h)
            roi_fixed = True  # keep using same ROI across frames
        if search_roi is not None:
            rx, ry, rw, rh = clamp_roi(*map(int, search_roi), W=Wf, H=Hf)
        else:
            rx, ry, rw, rh = (0, 0, Wf, Hf)

        # If tracking works, keep updating
        if tracker is not None:
            ok, bbox = tracker.update(frame)
            if ok:
                x, y, w_box, h_box = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                cv2.putText(frame, f"Tracking #{frame_num}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Visualize ROI
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 1)
                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue
            else:
                tracker = None  # fall back to detection inside ROI

        # Detect inside ROI only
        roi_gray = gray[ry:ry + rh, rx:rx + rw]
        kp_f, des_f = orb.detectAndCompute(roi_gray, None)
        if des_f is None or des_t is None or len(kp_f) == 0 or len(kp_t) == 0:
            # Visualize ROI even if nothing detected
            disp = frame.copy()
            cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 1)
            cv2.putText(disp, "No features in ROI", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Tracking", disp)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        knn = bf.knnMatch(des_t, des_f, k=2)
        good = []
        for m, n in knn:
            if m.distance < ratio * n.distance:
                good.append(m)

        if len(good) >= max(min_matches, 4):
            src_pts = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            # Keypoints from ROI are relative to ROI; keep them relative for homography
            dst_pts_roi = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts_roi, cv2.RANSAC, 5.0)
            inliers = int(mask.sum()) if mask is not None else 0

            if H is not None and inliers >= min_inliers:
                corners = np.float32([[0, 0], [0, Ht], [Wt, Ht], [Wt, 0]]).reshape(-1, 1, 2)
                projected_roi = cv2.perspectiveTransform(corners, H)

                # Convert ROI-relative polygon to full-frame by adding offsets
                projected_full = projected_roi + np.array([[[rx, ry]]], dtype=np.float32)
                x, y, w_box, h_box = cv2.boundingRect(projected_full)

                # Clamp to frame
                x, y, w_box, h_box = clamp_roi(x, y, w_box, h_box, Wf, Hf)

                # Draw ROI and detection
                cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 1)
                cv2.polylines(frame, [np.int32(projected_full)], True, (0, 255, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

                if first_match is None:
                    first_match = (frame_num, (x, y, w_box, h_box))

                tracker = make_csrt_tracker()
                tracker.init(frame, (x, y, w_box, h_box))
            else:
                disp = frame.copy()
                cv2.rectangle(disp, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 1)
                cv2.putText(disp, "No robust homography", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Tracking", disp)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return first_match

if __name__ == "__main__":
    video_path = "input.mp4"
    template_path = "search_img.png"
    # Example: search only in a header band
    result = robust_track_with_roi(video_path, template_path,
                                   search_roi=(0, 0, 1920, 1080),
                                   interactive=False)
    print(result)
