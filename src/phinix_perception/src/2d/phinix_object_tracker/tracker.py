
import cv2
from ultralytics import YOLO
import supervision as sv
import argparse

# font and bounding-box macros
BOX_COLOR = (0, 255, 0) # This is GREEN  
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_COLOR = (0, 255, 0)
FONT_THICKNESS = 2
OUTPUT_FILENAME = "tracked_output.mp4"

def detect_objects(model, frame):
    return model(frame)

# USING SUPERVISION ----
def track_objects(frame, detection_results, model, tracker, use_yolo_track, tracker_config):
    if use_yolo_track:
        # Integrated
        return model.track(source=frame, tracker=tracker_config, persist=True)
    else:
        # SV
        return tracker.update(detection_results)


def video_scan(video_path, output_path, model, tracker, use_yolo_track, tracker_config):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 2 FUNCTION CALLS
        # first line detection - gives bounding boxes and labels
        # second function does the tracking

        detection_results = detect_objects(model,frame)
        results = track_objects(frame, detection_results, model, tracker, use_yolo_track, tracker_config)

        # Draw tracking boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                track_id = int(box.id[0]) if box.id is not None else -1  # Object tracking ID
                label = f"ID {track_id}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS) # added macros
                cv2.putText(frame, label, (x1, y1 - 10), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS) # added macros

        output_video.write(frame)

    cap.release()
    output_video.release()
    cv2.destroyAllWindows()

def main():
    
    # take model parameter as command line argument 
    parser = argparse.ArgumentParser(description="Object Detection and Tracking using YOLO or Supervision")
    parser.add_argument("--video", type=str, default="a.mp4", help="Path to input video file")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Path to YOLO model file")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Path to tracker config file (YOLO mode)")
    parser.add_argument("--use-yolo-track", action="store_true", help="Use YOLO's built-in tracking instead of Supervision ByteTrack")
    parser.add_argument("--output", type=str, default=OUTPUT_FILENAME, help="Path to save the output video")
    args = parser.parse_args()

    model = YOLO(args.model)
    tracker = None if args.use_yolo_track else sv.ByteTrack()
    
    video_scan(args.video, args.output, model, tracker, args.use_yolo_track, args.tracker)

if __name__ == "__main__":
    main()
