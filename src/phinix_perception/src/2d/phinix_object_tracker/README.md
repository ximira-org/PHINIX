- What does the Script do?
  
This Python script uses object tracking on a video file using YOLO by drawing bounding boxes and assigning IDs to detected objects, producing an output with an object-tracked video.

- What is it Using?
  
This script is using YOLOv8/v11, as well as Supervision's ByteTrack as another Tracking Method. OpenCV and argparse are also used for reading video frames and command line arguments respectively.

- How to Use it?
  
Install the required libraries and dependencies and run a command line argument involving a path to an input video and optionally a YOLO model and tracker config. By doing this, you can toggle between YOLO models like v8 and v11 and can switch between Integrated Ultralytics tracking and Supervision Bytetrack.

- Sample Usage Example
  
python XimiraTracking1.py --model yolov8s.pt --use-yolo-track --video to/video/path.mp4 --tracker bytetrack.yaml 
This command line uses a specific mp4 video input with a model (YOLOv8 small) and the integrated ByteTrack tracker.
