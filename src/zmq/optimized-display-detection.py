import cv2
import zmq
import signal
import sys
from ultralytics import YOLO
import numpy as np
import torch
import argparse
from pathlib import Path
import time
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class FramePacket:
    """Container for frame data with timestamp."""
    frames: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    timestamp: float

class DisplayDetectionManager:
    def __init__(self, model_path="yolov8n.pt", conf_thres=0.25):
        """Initialize display manager with optimized ZMQ subscriber and YOLO model."""
        self.running = True
        
        # Setup ZMQ subscriber with optimized settings
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        
        # Set socket options for better performance
        self.socket.setsockopt(zmq.CONFLATE, 1)  # Keep only most recent message
        self.socket.setsockopt(zmq.RCVHWM, 1)    # Set receive high water mark
        self.socket.setsockopt(zmq.RCVBUF, 1024 * 1024)  # 1MB receive buffer
        self.socket.setsockopt(zmq.LINGER, 0)    # Don't wait on close
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Set non-blocking receive with timeout
        self.socket.connect("tcp://localhost:5555")
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        
        # Initialize YOLO model
        print(f"Initializing YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        # self.model = YOLO("yolov8s.engine")
        self.conf_thres = conf_thres
        
        # Ensure CUDA is available and being used
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            # Pre-allocate CUDA tensor for faster inference
            self.dummy_inference()
        
        # Initialize timing metrics
        self.timing_metrics = {
            'zmq_receive': 0,
            'detection': 0,
            'drawing': 0,
            'total': 0,
            'processing_delay': 0  # New metric for end-to-end delay
        }
        self.frame_count = 0
        self.dropped_frames = 0
        self.last_metrics_print = time.time()
        self.last_frame_time = None
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def dummy_inference(self):
        """Run dummy inference to initialize CUDA memory."""
        dummy_input = torch.zeros((1, 3, 640, 640)).to(self.device)
        for _ in range(2):  # Warm up runs
            _ = self.model(dummy_input)
        torch.cuda.synchronize()
        print("CUDA memory initialized")

    def _signal_handler(self, signum, frame):
        """Handle cleanup on interrupt signal."""
        print("\nGracefully shutting down display...")
        self.running = False
        self.socket.close()
        self.context.term()

    def _add_overlay(self, frame, fps, exposure, colormap, median):
        """Add information overlay to frame."""
        y_offset = 20
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(frame, f"Exposure Comp: {exposure}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(frame, f"GPU: {self.device.upper()}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20

        # Add timing information
        if self.frame_count > 0:
            cv2.putText(frame, f"ZMQ Receive: {self.timing_metrics['zmq_receive']*1000:.1f}ms", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(frame, f"Detection: {self.timing_metrics['detection']*1000:.1f}ms", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(frame, f"Drawing: {self.timing_metrics['drawing']*1000:.1f}ms", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(frame, f"Total: {self.timing_metrics['total']*1000:.1f}ms", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(frame, f"Processing Delay: {self.timing_metrics['processing_delay']*1000:.1f}ms", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(frame, f"Dropped Frames: {self.dropped_frames}", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def _draw_detections(self, frame, results):
        """Draw detection boxes and labels on frame."""
        if results.boxes is not None:
            boxes = results.boxes.cpu().numpy()
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                cls = int(box.cls)
                
                if conf < self.conf_thres:
                    continue
                
                # Get class name
                class_name = self.model.names[cls]
                
                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"{class_name} {conf:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def _print_timing_metrics(self):
        """Print average timing metrics every second."""
        current_time = time.time()
        if current_time - self.last_metrics_print >= 1.0:
            print("\nTiming Metrics (averaged over last second):")
            print(f"ZMQ Receive: {self.timing_metrics['zmq_receive']*1000:.1f}ms")
            print(f"Detection: {self.timing_metrics['detection']*1000:.1f}ms")
            print(f"Drawing: {self.timing_metrics['drawing']*1000:.1f}ms")
            print(f"Total: {self.timing_metrics['total']*1000:.1f}ms")
            print(f"Processing Delay: {self.timing_metrics['processing_delay']*1000:.1f}ms")
            print(f"Dropped Frames: {self.dropped_frames}")
            self.last_metrics_print = current_time

    def display_frames(self):
        """Receive and display frames with optimized ZMQ handling."""
        try:
            print("Display and detection started. Press Ctrl+C to exit.")
            
            while self.running:
                # Poll for new message with timeout
                socks = dict(self.poller.poll(timeout=1))
                
                if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                    # Measure ZMQ receive time
                    zmq_start = time.time()
                    frames_data = self.socket.recv_pyobj()
                    zmq_time = time.time() - zmq_start
                    self.timing_metrics['zmq_receive'] = zmq_time
                    
                    frames = frames_data['frames']
                    fps = frames_data['fps']
                    exposure = frames_data['exposure']
                    colormap = frames_data['colormap']
                    median = frames_data['median']
                    
                    # Calculate processing delay from last frame
                    if self.last_frame_time is not None:
                        self.timing_metrics['processing_delay'] = time.time() - self.last_frame_time
                    self.last_frame_time = time.time()
                    
                    total_start = time.time()
                    
                    # Process and display frames
                    for name, frame in frames.items():
                        if name == "video":  # Perform detection only on RGB frame
                            # Time detection
                            detection_start = time.time()
                            results = self.model(frame, device=self.device)
                            torch.cuda.synchronize()  # Ensure GPU operations are complete
                            detection_time = time.time() - detection_start
                            self.timing_metrics['detection'] = detection_time
                            
                            # Time drawing
                            drawing_start = time.time()
                            self._draw_detections(frame, results[0])
                            self._add_overlay(frame, fps, exposure, colormap, median)
                            cv2.imshow(f"Detection_{name}", frame)
                            drawing_time = time.time() - drawing_start
                            self.timing_metrics['drawing'] = drawing_time
                        else:
                            # Display depth frame without detection
                            cv2.imshow(name, frame)
                    
                    total_time = time.time() - total_start
                    self.timing_metrics['total'] = total_time
                    
                    self.frame_count += 1
                    self._print_timing_metrics()
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    
        except KeyboardInterrupt:
            print("\nDisplay interrupted by user")
        except Exception as e:
            print(f"Error in display: {e}")
        finally:
            cv2.destroyAllWindows()
            self.socket.close()
            self.context.term()

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Object Detection Display")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                      help="Path to YOLO model (default: yolov8n.pt)")
    parser.add_argument("--conf", type=float, default=0.25,
                      help="Confidence threshold (default: 0.25)")
    args = parser.parse_args()
    
    display_manager = DisplayDetectionManager(args.model, args.conf)
    display_manager.display_frames()

if __name__ == "__main__":
    main()