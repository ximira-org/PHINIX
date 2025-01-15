#!/usr/bin/env python3

import numpy as np
import pyopencl as cl
import time
import enum
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import MultiArrayDimension
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2 


RANGES = [4, 5, 4, 5, 4, 5] # feet
SIDE_FOV_CROP_RATIO = 0.0
NO_PTS_TO_BE_AN_OBSTACLE = 30
AVERAGE_CHANGE_THRESHOLD = -150 #mm

KERNEL_TEXT = """
__kernel void depthmap_sum_reduction(__global const float* depthmap, __local int* local_sum, __global int* group_sum, int N) 
{
   uint global_id = get_global_id(0);
   uint local_id = get_local_id(0);

   float max_depth = 0.2;

   // initailize L1 cache (local | shared) memory 
   local_sum[local_id] = 0;

   // Check in case global size kernels > actual N size
   if (global_id < N)
   {
      // Check for 2 depth values of the 2 halfs of the 
      local_sum[local_id] = (depthmap[global_id] < max_depth); 
   }

   // syncronize work-items in the work-group
   barrier(CLK_LOCAL_MEM_FENCE);

	// do sum reduction in shared memory
	// this loop now starts with offset(stride) = work-group size / 2 
	for (unsigned int offset = get_local_size(0) / 2; offset > 0; offset >>= 1)
   {
		if (local_id < offset) 
      {
			local_sum[local_id] += local_sum[local_id + offset];
		}
      // syncronize work-items in the work-group
      barrier(CLK_LOCAL_MEM_FENCE);
	}

	// write result for this block to global memory
   // the first index contains the total sum due to above sum reduction, store the sum in the partial/group sum global array
   if(local_id == 0) 
   {
      group_sum[get_group_id(0)] = local_sum[0];
   }
}
"""

# ====================== Modes ===========================
class DeviceMode(enum.Enum):
    # CPU Mode which calculate the sum using normal numpy operations
    CPU = 1
    # GPU Mode which uses OpenCL to calculate sum of the depthmap pixels
    GPU = 2
    # Map GPU Mode which uses OpenCL but with map/unmap algorithm to transfer memory between 
    GPU_MAP = 3 

# Grid mode to partition the depthmap into grids or use the full depthmap
class GridMode(enum.Enum):
    GRID = 1
    NO_GRID = 2
    
# ================== Parameters ===============================
# Select your GPU INDEX
GPU_INDEX = 0
# Select wether to use ROS or run as normal python script
USE_ROS = True

# Select mode of operation
# MODE = DeviceMode.GPU
MODE = DeviceMode.CPU
# MODE = DeviceMode.GPU_MAP
# MODE = DeviceMode.CPU

# Select to partition depthmap into grids or not
# GRID_MODE = GridMode.NO_GRID
GRID_MODE = GridMode.GRID
# =============================================================

TOPIC_NAME = "/phinix/depth/image_raw"
TOPIC_PHINIX_DISPARITY_IMG = "/phinix/disparity/image_raw"
TOPIC_DISP_VIS_IMG = "/phinix/vis/disparity"
TOPIC_OBSTACLE_DETS = "/phinix/obstacle_detector/detections"
TOPIC_NODE_STATES = "/phinix/node_states"

node_state_index = 0

#The number of cell depth updates to keep track of
NUM_CELL_DEPTH_UPDATES = 15

# ====================== OpenCL ==========================
# Depth map dimentions
WIDTH = 1920
HEIGHT = 1080
depthmap_np = np.random.rand(WIDTH, HEIGHT).astype(np.float32)
# Get the GPU Device and Platform
platforms = cl.get_platforms()
device = platforms[0].get_devices()[GPU_INDEX]
# Inference Context
context = cl.Context(devices=[device])
# Create the Command Queue and Compile the OpenCL Code
queue = cl.CommandQueue(context, device=device)
program_text = KERNEL_TEXT
program = cl.Program(context, program_text).build()
mf = cl.mem_flags
# Allocate Memory in GPU RAM for Depthmap
# ALLOC_HOST_PTR = pinned memory = higher copying performance .. https://downloads.ti.com/mctools/esd/docs/opencl/memory/buffers.html    
depthmap_gpu = cl.Buffer(context, mf.READ_ONLY | mf.ALLOC_HOST_PTR, size = depthmap_np.nbytes)
float_size = np.dtype(np.float32).itemsize # bytes
int_size = np.dtype(np.int32).itemsize # bytes
N = depthmap_np.size
global_size = N
# Local size is the size of the shared memory in the work group
local_size = device.max_work_group_size
# ceil global_size to be divisble by local size
global_size = global_size + (local_size - global_size % local_size) 
# Calculate number of work groups to be run on GPU
num_work_groups = global_size // local_size

# To update global/local and threads size (used in updating grid sizes)
def update_threads_size(depthmap):
    global N, global_size, local_size, num_work_groups, input_depthmap_nbytes
    # Global size
    N = depthmap.size
    global_size = N
    # Local size (work-items per group) & Num of Work Groups
    local_size = device.max_work_group_size
    # Allocate global size to be enough for all input depthmap
    global_size = global_size + (local_size - global_size % local_size) # ceil global_size to be divisble by local size
    # Number of work groups to allocate group sum memory
    num_work_groups = global_size // local_size
    # Number of bytes of depthmap, used for memory allocation
    input_depthmap_nbytes = depthmap.nbytes


# ============== Allocate Memory ==============
group_sums = cl.Buffer(context, mf.WRITE_ONLY, num_work_groups * int_size) # Allocate Sum memory workgroups so that we can store the sum of the work-group in it
group_sums_cpu = np.zeros((num_work_groups,), dtype=np.int32) # Same memory for CPU to copy the results back from GPU
local_sums = cl.LocalMemory(local_size * int_size) # Shared memory for each Work group

# ============ GPU Inference (Normal Mode) ==========
def inference(depthmap_numpy):
    print("NORMAL GPU")
    t1 = time.time()
    # copy data from numpy cpu to buffer gpu
    cl.enqueue_copy(queue, dest=depthmap_gpu, src=depthmap_numpy)
    # kernel 
    program.depthmap_sum_reduction(queue, (global_size, 1), (local_size, 1), depthmap_gpu, local_sums, group_sums,  np.int32(N))
    # copy group sum results to cpu memory
    cl.enqueue_copy(queue, dest=group_sums_cpu, src=group_sums, is_blocking=True)
    queue.finish()
    # it's better to sum it on CPU, it's faster than doing atomic operations on GPU
    gpu_sum = group_sums_cpu.sum()

    t2 = time.time()
    cpu_sum = (depthmap_numpy < 0.2).sum()
    t3 = time.time()
    print("Result = ", "CPU = ", cpu_sum, " .. GPU = ", gpu_sum, " .. Time CPU = ", round((t3-t2) * 1000,4), " ms .. Time GPU = ", round((t2-t1) * 1000,4), " ms")
    
    return gpu_sum

# ============ GPU Inference (Map/UnMap Mode) ==========
def inference_map_pinned_memory(depthmap_numpy):
    print("MAP GPU")
    t1 = time.time()
    # Give WRITE access to the CMEM region of memory to the CPU to write it's data into the depthmap device data
    (map_ptr, event) = cl.enqueue_map_buffer(queue=queue, buf=depthmap_gpu, flags=cl.map_flags.WRITE, offset=0, shape=depthmap_numpy.shape, dtype=depthmap_numpy.dtype)
    map_ptr[...] = depthmap_numpy
    del map_ptr # Get access back to GPU    
    # Run Kernel
    program.depthmap_sum_reduction(queue, (global_size, 1), (local_size, 1), depthmap_gpu, local_sums, group_sums,  np.int32(N))
    # Copy Group sum results 
    (map_ptr, event) = cl.enqueue_map_buffer(queue=queue, buf=group_sums, flags=cl.map_flags.READ, offset=0, shape=group_sums_cpu.shape, dtype=group_sums_cpu.dtype)
    group_sums_cpu[...] = map_ptr
    del map_ptr # Get access back to GPU
    
    queue.finish()
    gpu_sum = group_sums_cpu.sum()

    t2 = time.time()
    cpu_sum = (depthmap_numpy < 0.2).sum()
    t3 = time.time()
    print("Result = ", "CPU = ", cpu_sum, " .. GPU = ", gpu_sum, " .. Time CPU = ", round((t3-t2) * 1000,4), " ms .. Time GPU = ", round((t2-t1) * 1000,4), " ms")
    
    return gpu_sum

# ================ CPU Inference  ====================
def inference_cpu(depthmap_numpy):
    print("CPU")
    t1 = time.time()
    #range_in_mm = int(RANGE*0.3048*1000) #feet*0.3048*mm
    print("range is {} mm".format(range_in_mm))
    cpu_sum = ((depthmap_numpy < range_in_mm) & (depthmap_numpy != 0)).sum()
    t2 = time.time()
    print("Result = ", "CPU = ", cpu_sum, " .. Time CPU = ", round((t2-t1) * 1000,4))
    return cpu_sum





# ======================== ROS ===========================
    
class ObstacleDetectionNode(Node):
    def __init__(self):
        super().__init__("depthmap")
        self.subscriber = self.create_subscription(Image, TOPIC_NAME, self.callback, 10)        
        self.disp_subscriber = self.create_subscription(Image, TOPIC_PHINIX_DISPARITY_IMG, self.disp_callback, 10)
        self.node_state_subscriber = self.create_subscription(Int32MultiArray, TOPIC_NODE_STATES, self.node_state_callback, 10)         
        self.publisher = self.create_publisher(Int32MultiArray, TOPIC_OBSTACLE_DETS, 10)
        self.disp_vis_publisher_ = self.create_publisher(Image, TOPIC_DISP_VIS_IMG, 10)
        self.bridge = CvBridge()
        self.timer_period = 0.01  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.disp_img = np.zeros((720, 1280, 3)).astype(np.uint8)
        self.top_left_cell_coords = None
        self.bottom_left_cell_coords = None
        self.top_center_cell_coords = None
        self.bottom_center_cell_coords = None
        self.top_right_cell_coords = None
        self.bottom_right_cell_coords = None
        self.obstacle_presence_list = []
        #Keep track of average cell depth on a per cell basis
        self.cell_depth_averages = [[0] * 6] * NUM_CELL_DEPTH_UPDATES
        #The last index i stored a cell depth at
        self.cell_depth_index = 0

        # Am I active in the node manager
        self.node_active = False

    def callback(self, depthmap_ros_image: Image):
        #early exit if this node is not enabled in node manager
        if self.node_active == False:
            return
        
        # convert ros image to numpy array depthmap
        self.depthmap = self.bridge.imgmsg_to_cv2(depthmap_ros_image, "16UC1")
        result = self.handle_inference_modes(self.depthmap, side_fov_crop_ratio=SIDE_FOV_CROP_RATIO)
        # convert to numpy int32
        result = list(np.array(result).astype(np.int32))
        # convert to int32
        result = [int(r) for r in result]

        msg = Int32MultiArray()
        msg.data = result
        dim_0 = MultiArrayDimension()
        dim_0.label = "width"
        dim_0.size = 3
        dim_0.stride = 3
        
        dim_1 = MultiArrayDimension()
        dim_1.label = "height"
        dim_1.size = 2
        dim_1.stride = 2
        
        msg.layout.dim = [dim_0, dim_1]
        print("========== Result = ", result, " ================")
        self.publisher.publish(msg)        
    
    def disp_callback(self, disp_ros_image: Image):
        self.disp_img = self.bridge.imgmsg_to_cv2(disp_ros_image, "bgr8")

    # Imp Note: TODO Jagadish: This logic needs to be changed to topic syncing
    def timer_callback(self):
        #early exit if this node is not enabled in node manager
        if self.node_active == False:
            return
        
        cells = [self.top_left_cell_coords, self.bottom_left_cell_coords, 
                self.top_center_cell_coords, self.bottom_center_cell_coords,
                self.top_right_cell_coords, self.bottom_right_cell_coords]

        for (cell, obs_present) in zip(cells, self.obstacle_presence_list):
            if cell is not None:
                self.disp_img = cv2.rectangle(self.disp_img, (cell[1][0], cell[0][0]), 
                                                    (cell[1][1], cell[0][1]), (0,200,255), 7)
                if obs_present == 1:
                    marker_x = int((cell[1][0] + cell[1][1])/2)
                    marker_y = int((cell[0][0] + cell[0][1])/2)
                    cv2.drawMarker(self.disp_img, (marker_x, marker_y), color=[0, 0, 255], thickness=3, 
                                    markerType= cv2.MARKER_TILTED_CROSS, line_type=cv2.LINE_AA,
                                    markerSize=15)
                    cv2.circle(self.disp_img, (marker_x, marker_y), radius=15, color=[255,0,255], thickness=2)

        msg = self.bridge.cv2_to_imgmsg(self.disp_img, "bgr8")
        self.disp_vis_publisher_.publish(msg)
    
    #Set node_active to true if node manager so
    def node_state_callback(self, node_states: Int32MultiArray):
        self.node_active = node_states.data[node_state_index] == 1
            
    def handle_inference_modes(self, depthmap, side_fov_crop_ratio):
        # =========================== inference on GPU ===========================
        if GRID_MODE == GridMode.GRID:
            return self.inference_grid(depthmap, side_fov_crop_ratio=side_fov_crop_ratio) # for grid depth map
        else:    
            if MODE == DeviceMode.GPU:
                return inference(depthmap) # for normal memory transfer on full depth map
            elif MODE == DeviceMode.GPU_MAP:
                return inference_map_pinned_memory(depthmap) # for pinned memory transfer on full depthmap
            elif MODE == DeviceMode.CPU:
                return inference_cpu(depthmap, side_fov_crop_ratio=side_fov_crop_ratio)
    
    def inference_grid(self, depthmap_grid, side_fov_crop_ratio=0.05):
        print("GRID ")
        # clear the dummy data stored in both input & output buffers (depth map & group sum) from the previous run
        zero_depthmap = np.zeros_like(depthmap_np).flatten()
        zero_group_sum = np.zeros_like(group_sums_cpu, dtype=np.int32).flatten()
        cl._enqueue_write_buffer(queue, depthmap_gpu, zero_depthmap, is_blocking=True)
        cl._enqueue_write_buffer(queue, group_sums, zero_group_sum, is_blocking=True)
        
        # # Split the grid into cells
        # top_left = depthmap_grid[:depthmap_grid.shape[0]//3, :depthmap_grid.shape[1]//3]
        # top_center = depthmap_grid[depthmap_grid.shape[0]//3:2*depthmap_grid.shape[1]//3, :depthmap_grid.shape[1]//3]
        # top_right = depthmap_grid[2*depthmap_grid.shape[0]//3:, :depthmap_grid.shape[1]//3]

        # center_left   = depthmap_grid[:depthmap_grid.shape[0]//3, depthmap_grid.shape[1]//3:2*depthmap_grid.shape[1]//3]
        # center_center = depthmap_grid[depthmap_grid.shape[0]//3:2*depthmap_grid.shape[0]//3, depthmap_grid.shape[1]//3:2*depthmap_grid.shape[1]//3]
        # center_right  = depthmap_grid[2*depthmap_grid.shape[0]//3:, depthmap_grid.shape[1]//3:2*depthmap_grid.shape[1]//3]

        # bottom_left = depthmap_grid[:depthmap_grid.shape[0]//3, 2*depthmap_grid.shape[1]//3:]
        # bottom_center = depthmap_grid[depthmap_grid.shape[0]//3:2*depthmap_grid.shape[1]//3, 2*depthmap_grid.shape[1]//3:]
        # bottom_right = depthmap_grid[2*depthmap_grid.shape[0]//3:, 2*depthmap_grid.shape[1]//3:]

        crop_fov_in_cols = int(depthmap_grid.shape[1]*side_fov_crop_ratio)
        print("cropping {} on each side : ", crop_fov_in_cols)
        depthmap_cropped = depthmap_grid.copy()
        depthmap_cropped = depthmap_cropped[:, crop_fov_in_cols: depthmap_cropped.shape[1]-crop_fov_in_cols]
        # Split the grid into cells
        # NOTE: the cells are processed and stored column wise 
        # e.g for left column (top_left, center_left and bottom left) is the order
        self.top_left_cell_coords = np.array([[0, depthmap_cropped.shape[0]//2],
                                    [0, depthmap_cropped.shape[1]//3]])
        self.bottom_left_cell_coords= np.array([[depthmap_cropped.shape[0]//2, depthmap_cropped.shape[0]],
                                    [0, depthmap_cropped.shape[1]//3]])
        self.top_center_cell_coords = np.array([[0,depthmap_cropped.shape[0]//2],
                                    [depthmap_cropped.shape[1]//3,2*depthmap_cropped.shape[1]//3]])
        self.bottom_center_cell_coords = np.array([[depthmap_cropped.shape[0]//2, depthmap_cropped.shape[0]],
                                    [depthmap_cropped.shape[1]//3,2*depthmap_cropped.shape[1]//3]])
        self.top_right_cell_coords = np.array([[0,depthmap_cropped.shape[0]//2],
                                    [2*depthmap_cropped.shape[1]//3,depthmap_cropped.shape[1]]])
        self.bottom_right_cell_coords = np.array([[depthmap_cropped.shape[0]//2, depthmap_cropped.shape[0]],
                                    [2*depthmap_cropped.shape[1]//3,depthmap_cropped.shape[1]]])

        top_left = depthmap_cropped[self.top_left_cell_coords[0][0] : self.top_left_cell_coords[0][1], 
                                    self.top_left_cell_coords[1][0]:self.top_left_cell_coords[1][1]]
        bottom_left = depthmap_cropped[self.bottom_left_cell_coords[0][0] : self.bottom_left_cell_coords[0][1],
                                    self.bottom_left_cell_coords[1][0]:self.bottom_left_cell_coords[1][1]]


        top_center   = depthmap_cropped[self.top_center_cell_coords[0][0] : self.top_center_cell_coords[0][1], 
                                    self.top_center_cell_coords[1][0] : self.top_center_cell_coords[1][1]]
        bottom_center  = depthmap_cropped[self.bottom_center_cell_coords[0][0] : self.bottom_center_cell_coords[0][1],
                                    self.bottom_center_cell_coords[1][0] : self.bottom_center_cell_coords[1][1]]

        top_right = depthmap_cropped[self.top_right_cell_coords[0][0] : self.top_right_cell_coords[0][1], 
                                    self.top_right_cell_coords[1][0] : self.top_right_cell_coords[1][1]]
        bottom_right = depthmap_cropped[self.bottom_right_cell_coords[0][0] : self.bottom_right_cell_coords[0][1], 
                                    self.bottom_right_cell_coords[1][0] : self.bottom_right_cell_coords[1][1]]

        grid_np = [top_left, bottom_left, top_center, bottom_center, top_right, bottom_right]

        current_cell_depth_averages = [0] * len(grid_np)

        self.obstacle_presence_list = []
        for index, cell in enumerate(grid_np):
            cell = cell.flatten()
            
            #add the average depth of the cell to the list of cell depths
            current_cell_depth_averages[index] = np.average(cell)

            update_threads_size(cell)

            # =========== select memory transfer mode ===========
            if MODE == DeviceMode.GPU:
                cell_sum = inference(cell)
            elif MODE == DeviceMode.GPU_MAP:
                cell_sum = inference_map_pinned_memory(cell)
            elif MODE == DeviceMode.CPU:
                #cell_sum = inference_cpu(cell)
                
                #if the cell depth average is significatly less that it was NUM_DEPTH_UPDATES ago, then it is an obstacle
                current_cell_depth_average = current_cell_depth_averages[index]
                prev_cell_depth_average = self.cell_depth_averages[self.cell_depth_index - NUM_CELL_DEPTH_UPDATES][index]
                #self.get_logger().info(str(current_cell_depth_average - prev_cell_depth_average))
                if current_cell_depth_average - prev_cell_depth_average < AVERAGE_CHANGE_THRESHOLD and current_cell_depth_average < RANGES[index]*0.3048*1000:
                    cell_sum = NO_PTS_TO_BE_AN_OBSTACLE
                else:
                    cell_sum = 0
            if cell_sum >= NO_PTS_TO_BE_AN_OBSTACLE:
                self.obstacle_presence_list.append(1)
            else:
                self.obstacle_presence_list.append(0)
        
        self.cell_depth_averages[self.cell_depth_index] = current_cell_depth_averages
        self.cell_depth_index = (self.cell_depth_index + 1) % NUM_CELL_DEPTH_UPDATES

        print("[top_left, bottom_left, top_center, bottom_center, top_right, bottom_right]")
        print(self.obstacle_presence_list)
        print("*" * 40)
        return self.obstacle_presence_list

def run_ros():
    rclpy.init()
    node = ObstacleDetectionNode()
    rclpy.spin(node)
    rclpy.shutdown()

def run_normal_script():
    for i in range(10):
        depthmap_np = np.random.rand(1920, 1080).astype(np.float32)
        handle_inference_modes(depthmap_np)    

def main():
    if USE_ROS:
        run_ros()
    else:  
        run_normal_script()

if __name__ == '__main__':
    main()