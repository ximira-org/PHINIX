import depthai as dai
import numpy as np
import cv2
import argparse
import time
from datetime import timedelta
import signal
import sys
import zmq
import pickle

class OAKCameraManager:
    def __init__(self, resolution, fps):
        """Initialize camera manager with configuration."""
        self.resolution = resolution
        self.fps = fps
        self.device = None
        self.running = True
        self.colormaps = [cv2.COLORMAP_JET, cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_MAGMA, cv2.COLORMAP_TURBO]
        self.current_colormap = 0
        self.apply_median = True
        self.current_exposure_compensation = 6
        self.last_fps_print = time.time()
        self.frame_count = 0
        
        # Setup ZMQ publisher
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind("tcp://*:5555")
        
        # Setup signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle cleanup on interrupt signal."""
        print("\nGracefully shutting down...")
        self.running = False
        self.socket.close()
        self.context.term()
        cv2.destroyAllWindows()

    def _add_overlay(self, frame, fps):
        """Add information overlay to frame."""
        cv2.putText(frame, "Controls:", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "c: Change colormap", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "m: Toggle median filter", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Exp comp: {self.current_exposure_compensation}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def create_pipeline(self):
        """Create and configure the DepthAI pipeline with optimal settings."""
        pipeline = dai.Pipeline()
        
        # Create nodes
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        color_camera = pipeline.create(dai.node.ColorCamera)
        stereo_depth = pipeline.create(dai.node.StereoDepth)
        sync_node = pipeline.create(dai.node.Sync)
        xout = pipeline.create(dai.node.XLinkOut)
        control_in = pipeline.create(dai.node.XLinkIn)
        
        # Configure streams
        xout.setStreamName("xout")
        control_in.setStreamName('control')
        
        # Configure cameras
        for mono in [mono_left, mono_right]:
            mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            mono.setFps(self.fps)
            mono.initialControl.setAutoExposureEnable()
            mono.initialControl.setAutoExposureCompensation(self.current_exposure_compensation)
        mono_left.setCamera("left")
        mono_right.setCamera("right")
        
        color_camera.setCamera("color")
        color_camera.setFps(self.fps)
        color_camera.setResolution(self.resolution)
        color_camera.setInterleaved(False)
        color_camera.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        color_camera.initialControl.setAutoExposureEnable()
        color_camera.initialControl.setAutoExposureCompensation(self.current_exposure_compensation)
        color_camera.initialControl.setAutoWhiteBalanceMode(dai.CameraControl.AutoWhiteBalanceMode.AUTO)
        
        # Configure depth
        stereo_depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo_depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
        stereo_depth.initialConfig.setConfidenceThreshold(200)
        stereo_depth.setLeftRightCheck(True)
        stereo_depth.setSubpixel(True)
        
        # Configure sync
        sync_node.setSyncThreshold(timedelta(milliseconds=50))
        
        # Link nodes
        mono_left.out.link(stereo_depth.left)
        mono_right.out.link(stereo_depth.right)
        stereo_depth.disparity.link(sync_node.inputs["disparity"])
        stereo_depth.depth.link(sync_node.inputs["depth"])  # Add depth to sync
        color_camera.video.link(sync_node.inputs["video"])
        sync_node.out.link(xout.input)
        
        # Link controls
        control_in.out.link(mono_left.inputControl)
        control_in.out.link(mono_right.inputControl)
        control_in.out.link(color_camera.inputControl)
        
        return pipeline, stereo_depth

    def update_exposure(self, increment=True):
        """Update camera exposure with error handling."""
        try:
            if increment:
                self.current_exposure_compensation = min(12, self.current_exposure_compensation + 1)
            else:
                self.current_exposure_compensation = max(0, self.current_exposure_compensation - 1)
            
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            ctrl.setAutoExposureCompensation(self.current_exposure_compensation)
            
            if hasattr(self, 'control_queue'):
                self.control_queue.send(ctrl)
                print(f"Exposure compensation set to: {self.current_exposure_compensation}")
        except Exception as e:
            print(f"Error updating exposure: {e}")

    def process_frames(self):
        """Process camera frames and publish them via ZMQ."""
        try:
            pipeline, stereo_depth = self.create_pipeline()
            
            with dai.Device(pipeline) as self.device:
                queue = self.device.getOutputQueue("xout", maxSize=4, blocking=False)
                self.control_queue = self.device.getInputQueue('control')
                
                print("\nControls:")
                print("'=' or '+': Increase exposure")
                print("'-': Decrease exposure")
                print("'c': Change colormap")
                print("'m': Toggle median filter")
                print("'q': Quit")
                
                start_time = time.time()
                
                while self.running:
                    msg_group = queue.tryGet()
                    if msg_group is None:
                        continue
                    
                    self.frame_count += 1
                    current_time = time.time()
                    
                    if current_time - self.last_fps_print >= 1.0:
                        fps = self.frame_count / (current_time - start_time)
                        print(f"FPS: {fps:.1f}")
                        self.last_fps_print = current_time
                    
                    frames_dict = {}
                    for name, msg in msg_group:
                        frame = msg.getCvFrame()
                        
                        if name == "disparity":
                            # Store raw disparity frame
                            frames_dict["raw_disparity"] = frame.copy()
                            
                            # Process disparity for colormap display
                            disp_vis = frame.copy()
                            disp_vis = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX)
                            if self.apply_median:
                                disp_vis = cv2.medianBlur(disp_vis.astype(np.uint8), 5)
                            disp_vis = cv2.applyColorMap(disp_vis.astype(np.uint8), 
                                                    self.colormaps[self.current_colormap])
                            
                            frame = disp_vis
                        
                        elif name == "depth":
                            # Store and show raw depth frame (values in mm)
                            frames_dict["raw_depth"] = frame.copy()
                        
                        # Display frame locally
                        if name == "disparity":
                            self._add_overlay(frame, fps if 'fps' in locals() else 0)
                        cv2.imshow(f"Local_{name}", frame)
                        
                        frames_dict[name] = frame
                    
                    # Send frames via ZMQ
                    frames_data = {
                        'frames': frames_dict,
                        'fps': fps if 'fps' in locals() else 0,
                        'exposure': self.current_exposure_compensation,
                        'colormap': self.current_colormap,
                        'median': self.apply_median
                    }
                    self.socket.send_pyobj(frames_data)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1)
                    if key in [ord('q'), ord('Q')]:
                        break
                    elif key in [ord('+'), ord('=')]:
                        self.update_exposure(True)
                    elif key == ord('-'):
                        self.update_exposure(False)
                    elif key in [ord('c'), ord('C')]:
                        self.current_colormap = (self.current_colormap + 1) % len(self.colormaps)
                    elif key in [ord('m'), ord('M')]:
                        self.apply_median = not self.apply_median
                
        except Exception as e:
            print(f"Error in frame processing: {e}")
        finally:
            cv2.destroyAllWindows()
            if self.device is not None:
                self.device.close()
            self.socket.close()
            self.context.term()

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="OAK-D Camera Application")
    parser.add_argument("--resolution", type=str, default="THE_1080_P",
                      choices=["THE_720_P", "THE_1080_P", "THE_4_K", "THE_12_MP", "THE_13_MP"],
                      help="RGB camera resolution")
    parser.add_argument("--fps", type=int, default=30,
                      help="Camera FPS (default: 30)")
    args = parser.parse_args()
    
    resolution = getattr(dai.ColorCameraProperties.SensorResolution, args.resolution,
                        dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    
    camera_manager = OAKCameraManager(resolution, args.fps)
    camera_manager.process_frames()

if __name__ == "__main__":
    main()