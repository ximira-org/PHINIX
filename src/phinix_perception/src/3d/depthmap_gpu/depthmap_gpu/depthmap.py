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
MODE = DeviceMode.GPU
# MODE = DeviceMode.GPU_MAP
# MODE = DeviceMode.CPU

# Select to partition depthmap into grids or not
# GRID_MODE = GridMode.NO_GRID
GRID_MODE = GridMode.GRID
# =============================================================

bridge = CvBridge()
TOPIC_NAME = "depthmap_topic"
PUBLISH_TOPIC_NAME = "depthmap_result"
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
    cpu_sum = (depthmap_numpy < 0.2).sum()
    t2 = time.time()
    print("Result = ", "CPU = ", cpu_sum, " .. Time CPU = ", round((t2-t1) * 1000,4))
    return cpu_sum

def inference_grid(depthmap_grid):
    print("GRID ")
    # clear the dummy data stored in both input & output buffers (depth map & group sum) from the previous run
    zero_depthmap = np.zeros_like(depthmap_np).flatten()
    zero_group_sum = np.zeros_like(group_sums_cpu, dtype=np.int32).flatten()
    cl._enqueue_write_buffer(queue, depthmap_gpu, zero_depthmap, is_blocking=True)
    cl._enqueue_write_buffer(queue, group_sums, zero_group_sum, is_blocking=True)
    
    # Split the grid into cells
    top_left = depthmap_grid[:WIDTH//3, :HEIGHT//3]
    top_center = depthmap_grid[WIDTH//3:2*WIDTH//3, :HEIGHT//3]
    top_right = depthmap_grid[2*WIDTH//3:, :HEIGHT//3]

    center_left   = depthmap_grid[:WIDTH//3, HEIGHT//3:2*HEIGHT//3]
    center_center = depthmap_grid[WIDTH//3:2*WIDTH//3, HEIGHT//3:2*HEIGHT//3]
    center_right  = depthmap_grid[2*WIDTH//3:, HEIGHT//3:2*HEIGHT//3]

    bottom_left = depthmap_grid[:WIDTH//3, 2*HEIGHT//3:]
    bottom_center = depthmap_grid[WIDTH//3:2*WIDTH//3, 2*HEIGHT//3:]
    bottom_right = depthmap_grid[2*WIDTH//3:, 2*HEIGHT//3:]

    grid_np = [top_left, top_center, top_right, center_left, center_center, center_right, bottom_left, bottom_center, bottom_right]
    grid_sum = []
    for cell in grid_np:
        cell = cell.flatten()
        update_threads_size(cell)

        # =========== select memory transfer mode ===========
        if MODE == DeviceMode.GPU:
            cell_sum = inference(cell)
        elif MODE == DeviceMode.GPU_MAP:
            cell_sum = inference_map_pinned_memory(cell)
        elif MODE == DeviceMode.CPU:
            cell_sum = inference_cpu(cell)

        grid_sum.append(cell_sum)
    
    print("[top_left, top_center, top_right, center_left, center_center, center_right, bottom_left, bottom_center, bottom_right]")
    print(grid_sum)
    print("*" * 40)
    return grid_sum
            
def handle_inference_modes(depthmap):
    # =========================== inference on GPU ===========================
    if GRID_MODE == GridMode.GRID:
        return inference_grid(depthmap) # for grid depth map
    else:    
        if MODE == DeviceMode.GPU:
            return inference(depthmap) # for normal memory transfer on full depth map
        elif MODE == DeviceMode.GPU_MAP:
            return inference_map_pinned_memory(depthmap) # for pinned memory transfer on full depthmap
        elif MODE == DeviceMode.CPU:
            return inference_cpu(depthmap)


# ======================== ROS ===========================
    
class DepthmapNode(Node):
    def __init__(self):
        super().__init__("depthmap")
        self.subscriber = self.create_subscription(Image, TOPIC_NAME, self.callback, 10)        
        self.publisher = self.create_publisher(Int32MultiArray, PUBLISH_TOPIC_NAME, 10)

    def callback(self, depthmap_ros_image: Image):
        # convert ros image to numpy array depthmap
        depthmap = bridge.imgmsg_to_cv2(depthmap_ros_image, "32FC1")
        result = handle_inference_modes(depthmap)
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
        dim_1.size = 3
        dim_1.stride = 3
        
        msg.layout.dim = [dim_0, dim_1]
        print("========== Result = ", result, " ================")
        self.publisher.publish(msg)        


def run_ros():
    rclpy.init()
    node = DepthmapNode()
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