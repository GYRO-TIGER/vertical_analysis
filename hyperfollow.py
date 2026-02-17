#!/usr/bin/env python3

from collections import defaultdict
import cv2
import imageio
import pickle
import argparse
import ffmpeg
import os
import tempfile
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import uuid

tempfile.tempdir = r"D:\churchil_queen_vertical\createdFiles"

def make_temp_path(suffix):
    # Use delete=False, return a path, and close immediately so Windows releases the handle
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=r"D:\churchil_queen_vertical\createdFiles")
    path = f.name
    f.close()
    return path

GREEN = (0, 255, 0)
WHITE = (255, 255, 255)


def detect_people(frame, model):
    # Apply the YOLOv8 detector to the frame and keep only people (class_id == 0)
    # mps here means it will use the hardware acceleration on macOS
    # Change it to cpu if you're on Linux or cuda if you have an Nvidia GPU
    detections = model(frame, device="cpu")[0]
    for data in detections.boxes.data.tolist():
        confidence = data[4]
        class_id = data[5]
        if confidence >= 0.5 and class_id == 0:
            xmin, ymin, xmax, ymax = int(data[0]), int(
                data[1]), int(data[2]), int(data[3])
            yield [[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id]


def bbox_center(bbox):
    return (int((bbox[0] + bbox[2]) // 2), int((bbox[1] + bbox[3]) // 2))


def filter_top_percent_tracks(track_durations, top_percent):
    # Calculate the number of tracks to keep (top N%)
    num_tracks_to_keep = int(len(track_durations) * top_percent)

    # Sort tracks by duration and keep the top 40%
    sorted_tracks = sorted(track_durations.items(),
                           key=lambda item: item[1], reverse=True)
    top_tracks = sorted_tracks[:num_tracks_to_keep]

    # Create a new dictionary with only the top tracks
    filtered_track_durations = {
        track_id: duration for track_id, duration in top_tracks}

    return filtered_track_durations


def find_subjects(frames, track_durations):

    subjects = []
    if len(track_durations) > 100:
        # The perentage of tracks to keep varies depnding on the crowdiness of the video
        track_durations = filter_top_percent_tracks(track_durations, 0.2)

    for frame in frames:
        longest_duration = 0
        subject_center = None
        # Don't even try to re-center frames with more than 8 people
        if len(frame) <= 8:
            for track in frame:
                track_id = track['track_id']
                duration = track_durations.get(track_id, 0)

                if duration > longest_duration:
                    longest_duration = duration
                    subject_center = bbox_center(
                        track['bbox'])

        subjects.append(subject_center)

    return subjects


def track(video_path, subjects_fn, preview):
    # Open the source file using OpenCV
    cap = cv2.VideoCapture(video_path)
    # Initialize the YOLOv8 detector
    # It will automatically download the model weights on the first run
    detector = YOLO("yolov8l.pt")
    # Also initialize the DeepSort tracker.
    # The embedder parameter specifies the model to use for feature extraction.
    # In our case we're going to use one of the pre-trained variants of a CLIP model.
    tracker = DeepSort(max_age=10, embedder='clip_ViT-B/32', embedder_gpu=False)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    detections = []
    tracks = []

    # For our subject detection logic we'll need to know the total durion of each track
    # Due to this, we can't do subject detection in "online", but we'll have to do a second pass
    # On the first pass we'll just accumulate tracks and their durations in this dictionary
    track_durations = defaultdict(int)
    tracks_per_frame = []
    subjects = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip every other frame to speed up processing
        if frame_count % 2 == 0:
            # Detect people
            detections = list(detect_people(frame, detector))
            tracks = tracker.update_tracks(detections, frame=frame)

        tracks_per_frame.append([])

        for track in tracks:
            # if the track is not confirmed, ignore it
            if not track.is_confirmed():
                continue

            # Update track durations and save some pre-frame info for the second pass
            track_durations[track.track_id] = track.age
            tracks_per_frame[-1].append({
                'track_id': track.track_id,
                'bbox': track.to_ltrb(),
            })

            # Draw the bounding box and the track id on the frame
            # And display a preview window to track progress
            track_id = track.track_id
            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = int(ltrb[0]), int(
                ltrb[1]), int(ltrb[2]), int(ltrb[3])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.rectangle(frame, (xmin, ymin - 20),
                          (xmin + 20, ymin), GREEN, -1)
            cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

        # Display the frame
        if preview:
            cv2.imshow('Processed Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # After all tracking is done, run subject detection and dump the result into an intermediary file
    subjects = find_subjects(
        tracks_per_frame, track_durations)
    pickle.dump(subjects, open(subjects_fn, 'wb'))

    cap.release()
    cv2.destroyAllWindows()


def ease_camera_towards_subject(current_pos, target_pos, damping_factor):
    # Calculate the distance vector between current position and target
    distance_vector = np.array(target_pos) - np.array(current_pos)

    # Apply damping to the distance vector
    eased_vector = distance_vector * damping_factor

    # Update the current position
    new_pos = np.array(current_pos) + eased_vector
    return tuple(new_pos.astype(int))


def center_subject_in_frame(frame, new_size, subject_position, last_position, damping_factor):
    original_height, original_width = frame.shape[:2]
    new_width, new_height = new_size

    # Calculate desired top-left corner for centered subject
    subject_center_x, subject_center_y = subject_position
    desired_x = max(0, min(original_width - new_width,
                    subject_center_x - new_width // 2))
    desired_y = max(0, min(original_height - new_height,
                    subject_center_y - new_height // 2))

    # Apply easing towards the subject
    new_x, new_y = ease_camera_towards_subject(
        last_position, (desired_x, desired_y), damping_factor)

    # Ensure the new position is within bounds
    new_x = max(0, min(new_x, original_width - new_width))
    new_y = max(0, min(new_y, original_height - new_height))

    # Crop the frame to the new dimensions
    cropped_frame = frame[new_y:new_y + new_height, new_x:new_x + new_width]

    return cropped_frame, (new_x, new_y), (new_x, new_y, new_width, new_height)


def round_to_multiple(number, multiple):
    return round(number / multiple) * multiple


def reframe(video_path, subjects_fn, preview):
    # We could parametrize this one too, but I'm just using this script for my
    # vertical IG videos so, 9:16 it is :P
    target_aspect_ratio = (9, 16)
    cap = cv2.VideoCapture(video_path)
    # Get the original video dimensions
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Determine the base dimension (shortest side)
    base_dimension = min(width, height)

    # Calculate target dimensions maintaining aspect ratio
    target_aspect_ratio_width, target_aspect_ratio_height = target_aspect_ratio
    if width < height:  # Landscape to portrait
        new_width = int(base_dimension)
        new_height = int(
            base_dimension * target_aspect_ratio_height / target_aspect_ratio_width)
    else:  # Portrait to landscape or same orientation
        new_height = int(base_dimension)
        new_width = int(base_dimension *
                        target_aspect_ratio_width / target_aspect_ratio_height)

    # Ensure new dimensions do not exceed original dimensions
    new_width = int(min(round_to_multiple(new_width, 16), width))
    new_height = int(min(round_to_multiple(new_height, 16), height))

    frame_center = (int(width // 2), int(height // 2))

   # Pre-create temp file paths and ensure no handles are open
    temp_audio_path = make_temp_path('.mp3')   # closed, ready for ffmpeg to write
    temp_video_path = make_temp_path('.mp4')   # closed, ready for imageio to write

    try:
        writer = imageio.get_writer(
            temp_video_path, fps=fps, format='mp4', codec='libx264', quality=10
        )

        subjects = pickle.load(open(subjects_fn, 'rb'))

        frame_count = 0
        last_crop_position = (0, 0)
        last_subject_position = (int(width // 2), int(height // 2))
        lost_subject_for = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # If no subject is found, just stick with the last position for a few seconds
            # hoping it will reappear. If not - ease back to the center
            if not subjects[frame_count]:
                subject = last_subject_position
                lost_subject_for += 1
            else:
                subject = subjects[frame_count]
                last_subject_position = subject
                lost_subject_for = 0

            # Drift back towards the center if the subject is lost for too long
            LOST_SUBJECT_THRESHOLD_SEC = 3
            if lost_subject_for > LOST_SUBJECT_THRESHOLD_SEC * fps:
                subject = frame_center

            # The last parameter is the damping factor
            # It determines how quickly the camera moves towards the subject
            # I found 0.1 to be a good overall value
            cropped_frame, last_crop_position, crop_bbox = center_subject_in_frame(
                frame, (new_width, new_height), subject, last_crop_position, 0.1
            )

            # Write the new frame to the output video
            writer.append_data(cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB))

            # Also create some markings on the original frame for the live preview
            cv2.rectangle(frame, (int(subject[0]) - 5, int(subject[1]) - 5),
                          (int(subject[0]) + 5, int(subject[1]) + 5), GREEN, 2)
            cv2.rectangle(frame, (crop_bbox[0], crop_bbox[1]), (crop_bbox[0] + crop_bbox[2],
                                                                crop_bbox[1] + crop_bbox[3]), GREEN, 2)

            if lost_subject_for > 0:
                cv2.putText(frame, f"Lost subject for {lost_subject_for / fps} seconds",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, WHITE, 2)

            # Display the live preview
            if preview:
                cv2.imshow('Processed Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1

        cap.release()
        writer.close()  # ensure file handle is released before ffmpeg reads it

        # Extract audio (ensure no player has the source file open)
        (
            ffmpeg
            .input(video_path)
            .output(temp_audio_path, q=0, map='a')
            .overwrite_output()
            .run(cmd=r"C:\ffmpeg\bin\ffmpeg.exe")
        )

        # Build unique reframed output path to avoid clashes/locks
        base, _ = os.path.splitext(video_path)
        output_path = f"{base}_reframed.mp4"
        if os.path.exists(output_path):
            # avoid Permission denied on locked existing file; write a new unique file
            output_path = f"{base}_reframed_{uuid.uuid4().hex[:8]}.mp4"

        # Mux video+audio (input files must be closed/unlocked)
        input_video_stream = ffmpeg.input(temp_video_path)
        input_audio_stream = ffmpeg.input(temp_audio_path)

        (
            ffmpeg
            .output(
                input_video_stream,
                input_audio_stream,
                output_path,
                codec='aac',
                vcodec='libx264',
                pix_fmt='yuv420p',
                vf='format=yuv420p',
                profile='main',
                level='4.0'
            )
            .overwrite_output()
            .run(cmd=r"C:\ffmpeg\bin\ffmpeg.exe")
        )

    finally:
        # Manually clean up temps created with delete=False
        for p in [temp_audio_path, temp_video_path]:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a GoPro video.")
    parser.add_argument('video_path', type=str,
                        help='Path to the GoPro video file')
    parser.add_argument('command', type=str,
                        help='Action to perform', choices=['track', 'reframe'])
    parser.add_argument('--preview', dest='preview', action='store_true',
                        help='Display the processed video in a window', default=False)
    args = parser.parse_args()
    subjects_fn = f'{args.video_path.split(".")[0]}_subjects.pickle'

    if args.command == 'track':
        track(args.video_path, subjects_fn, args.preview)
        print("\nTracking complete. Starting reframing...")
        reframe(args.video_path, subjects_fn, args.preview)
    elif args.command == 'reframe':
        reframe(args.video_path, subjects_fn, args.preview)
