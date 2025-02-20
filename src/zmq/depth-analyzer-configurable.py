import zmq
import numpy as np
import cv2
import signal
import argparse
import torch
import time
from pathlib import Path

class DepthGridAnalyzer:
    def __init__(self, grid_rows=3, grid_cols=3, depth_threshold=1000, min_points=1000):
        """Initialize depth analyzer with ZMQ subscriber.
        
        Args:
            grid_rows (int): Number of rows in analysis grid
            grid_cols (int): Number of columns in analysis grid
            depth_threshold (int): Maximum depth value to consider (in mm)
            min_points (int): Minimum points needed to mark a grid cell
        """
        self.running = True
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.depth_threshold = depth_threshold
        self.min_points = min_points
        
        # Setup ZMQ subscriber with polling
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect("tcp://localhost:5555")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # Setup polling
        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        
        # Initialize CUDA device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Grid size: {grid_rows}x{grid_cols}")
        print(f"Depth threshold: {depth_threshold}mm")
        print(f"Minimum points: {min_points}")
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle cleanup on interrupt signal."""
        print("\nShutting down depth analyzer...")
        self.running = False
        self.socket.close()
        self.context.term()
    
    def _analyze_grid(self, depth_frame):
        """Analyze depth frame using CUDA for grid calculations with batch processing."""
        t_start = time.perf_counter()
        
        # Convert uint16 depth frame to float32 tensor
        depth_tensor = torch.from_numpy(depth_frame.astype(np.float32)).to(self.device)
        
        # Get frame dimensions and calculate grid cell sizes
        height, width = depth_frame.shape
        grid_h = height // self.grid_rows
        grid_w = width // self.grid_cols
        
        # Reshape depth tensor into grid cells
        # Use unfold to create overlapping windows, then reshape to grid
        depth_grid = depth_tensor.unfold(0, grid_h, grid_h).unfold(1, grid_w, grid_w)
        depth_grid = depth_grid.reshape(self.grid_rows, self.grid_cols, -1)
        
        # Create mask for valid points (non-zero and below threshold)
        valid_points = torch.logical_and(depth_grid > 0, depth_grid < self.depth_threshold)
        
        # Count points in each grid cell
        points_below = torch.sum(valid_points, dim=2)
        
        # Create result grid
        grid_markers = np.full((self.grid_rows, self.grid_cols), '', dtype=str)
        grid_markers[points_below.cpu().numpy() > self.min_points] = 'X'
        
        grid_time = (time.perf_counter() - t_start) * 1000  # Convert to ms
        return grid_markers, (grid_h, grid_w), grid_time
    
    def _draw_grid(self, frame, grid_markers, grid_size):
        """Draw grid and markers on frame."""
        t_start = time.perf_counter()
        
        height, width = frame.shape[:2]
        grid_h, grid_w = grid_size
        
        # Draw horizontal lines
        for i in range(1, self.grid_rows):
            y = i * grid_h
            cv2.line(frame, (0, y), (width, y), (255, 255, 255), 1)
        
        # Draw vertical lines
        for i in range(1, self.grid_cols):
            x = i * grid_w
            cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 1)
        
        # Draw markers
        for i in range(self.grid_rows):
            for j in range(self.grid_cols):
                if grid_markers[i, j] == 'X':
                    x = j * grid_w + grid_w // 2
                    y = i * grid_h + grid_h // 2
                    cv2.putText(frame, 'X', (x-10, y+10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        vis_time = (time.perf_counter() - t_start) * 1000  # Convert to ms
        return frame, vis_time
    
    def process_depth(self):
        """Process depth frames and display grid analysis."""
        try:
            print("Depth grid analysis started. Press Ctrl+C to exit.")
            
            while self.running:
                try:
                    # Poll for data with timeout
                    t_recv_start = time.perf_counter()
                    socks = dict(self.poller.poll(timeout=1))  # 1ms timeout
                    if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                        frames_data = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
                        recv_time = (time.perf_counter() - t_recv_start) * 1000
                        depth_frame = frames_data['frames']['raw_depth']
                        disparity_frame = frames_data['frames']['disparity']
                    else:
                        continue
                    
                    # Analyze depth grid
                    grid_markers, grid_size, grid_time = self._analyze_grid(depth_frame)
                    
                    # Normalize depth frame for display
                    depth_display = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
                    depth_display = depth_display.astype(np.uint8)
                    depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                    
                    # Draw grid and markers
                    disparity_frame, vis_time = self._draw_grid(disparity_frame, grid_markers, grid_size)
                    
                    # Print timing information
                    print(f"\rReceive: {recv_time:.1f}ms | Grid Analysis: {grid_time:.1f}ms | Visualization: {vis_time:.1f}ms", end="")
                    
                    # Display frame
                    cv2.imshow("Depth Grid Analysis", disparity_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                except zmq.error.Again:
                    continue
                    
        except KeyboardInterrupt:
            print("\nDepth analysis interrupted by user")
        except Exception as e:
            print(f"Error in depth analysis: {e}")
        finally:
            cv2.destroyAllWindows()
            self.socket.close()
            self.context.term()

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Depth Grid Analysis")
    parser.add_argument("--rows", type=int, default=3,
                      help="Number of grid rows (default: 3)")
    parser.add_argument("--cols", type=int, default=3,
                      help="Number of grid columns (default: 3)")
    parser.add_argument("--threshold", type=int, default=1000,
                      help="Depth threshold in mm (default: 1000)")
    parser.add_argument("--min-points", type=int, default=1000,
                      help="Minimum points below threshold to mark X (default: 1000)")
    args = parser.parse_args()
    
    analyzer = DepthGridAnalyzer(args.rows, args.cols, args.threshold, args.min_points)
    analyzer.process_depth()

if __name__ == "__main__":
    main()