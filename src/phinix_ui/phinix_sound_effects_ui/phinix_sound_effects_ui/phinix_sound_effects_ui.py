import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import time
import pygame
from ament_index_python.packages import get_package_share_directory

# Define the topic for the depth image from Phinix
TOPIC_PHINIX_DEPTH_IMG = "/phinix/depth/image_raw"
# Define the topic for the sidewalk detection from Phinix
TOPIC_SIDEWALK_DETECTION_IMG = "/phinix/vis/sidewalk"
TOPIC_NODE_STATES = "/phinix/node_states"

TOPIC_WAKEWORD = "/phinix/wakeword"

# Define names for the different sounds
SOUND_NAMES = ["high_sound", "mid_sound", "low_sound"]

# Define the number of rows and columns for the obs det grid
NUM_ROWS = 3
NUM_COLUMNS = 10
NUM_CELLS = NUM_ROWS * NUM_COLUMNS

# Define the maximum depths for each row
MAX_DEPTHS = [1350, 1000, 1350]

# Define the minimum depth, at which the volume of the cell will be 1
MIN_DEPTH = 100

MAX_HOLE_DEPTH = 400
HOLE_THRESHOLD = 30

'''
Path Detection checks 3 points for each side, y offset from the bottom of the screen and x offset from the center.
Each point represents a level of urgency. If sidewalk is not detected a the point, the sound will be played.
'''
PATH_DET_LEFT_Y = [.8, .8, .8]
PATH_DET_LEFT_X = [.3, .38, .45]
PATH_DET_RIGHT_Y = [.8, .8, .8]
PATH_DET_RIGHT_X = [.55, .62, .7]

path_detection_node_state_index = 1

# Function to linearly interpolate between two values
def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolate on the scale given by a to b, using t as the point on that scale."""
    return (1 - t) * a + t * b

# Function to calculate the inverse linear interpolation
def inv_lerp(a: float, b: float, v: float) -> float:
    """Inverse Linear Interpolation, get the fraction between a and b on which v resides."""
    return (v - a) / (b - a)

class PHINIXSoundEffectsUI(Node):

    def __init__(self):
        super().__init__('phinix_sound_effects_ui')
        self.get_logger().info("Begin SFX UI")

        self.prevDepthCallbackTime = time.time()

        self.wake_word_sub = self.create_subscription(String, TOPIC_WAKEWORD, self.wakeword_callback, 10)

        # Create a subscriber for the depth image topic
        self.depth_subscriber = self.create_subscription(Image, TOPIC_PHINIX_DEPTH_IMG, self.depth_callback, 10)
        #Subscribe to the sidewalk detection topic
        self.sidewalk_detection_subscriber = self.create_subscription(Image, TOPIC_SIDEWALK_DETECTION_IMG, self.sidewalk_detection_callback, 10)
        self.node_state_subscriber = self.create_subscription(
            Int32MultiArray, 
            TOPIC_NODE_STATES, 
            self.node_state_callback, 
            10)
        self.bridge = CvBridge()
        
        # Initialize state with empty sound channels and objects
        self.obs_det_sounds = [None] * NUM_ROWS
        self.obs_det_channels = [None] * NUM_ROWS

        self.cell_depth_averages = [0] * NUM_CELLS

        # Initialize pygame and set the number of channels
        pygame.mixer.pre_init(buffer=16)
        pygame.mixer.init()
        # One cahnnel for each row, plus one for holes, and one for Path Det
        pygame.mixer.set_num_channels(NUM_ROWS + 2)

        # Get the location of the sound effects
        pkg_share_dir = get_package_share_directory("phinix_sound_effects_ui")

        # Connect the obstacle detection sounds with their sound channels
        r = 0
        while r < NUM_ROWS:
            sound_name = 'sounds/' + SOUND_NAMES[r] + '.wav'
            sound_file_path = os.path.join(pkg_share_dir, sound_name)
            self.obs_det_sounds[r] = pygame.mixer.Sound(sound_file_path)
            self.obs_det_channels[r] = pygame.mixer.Channel(r)
            self.obs_det_channels[r].set_volume(0.0)
            self.obs_det_channels[r].play(self.obs_det_sounds[r], loops=-1) 
            r += 1
        
        # Set up the hole sound
        self.hole_sound_name = 'sounds/hole_sound.wav'
        self.hole_sound_file_path = os.path.join(pkg_share_dir, self.hole_sound_name)
        self.hole_sound = pygame.mixer.Sound(self.hole_sound_file_path)
        self.hole_channel = pygame.mixer.Channel(NUM_ROWS)
        self.hole_channel.set_volume(0.0)
        self.hole_channel.play(self.hole_sound, loops=-1)

        # Path Det Sound
        self.path_det_sound_name = 'sounds/path_det_sound.wav'
        self.path_det_sound_file_path = os.path.join(pkg_share_dir, self.path_det_sound_name)
        self.path_det_sound = pygame.mixer.Sound(self.path_det_sound_file_path)
        self.path_det_channel = pygame.mixer.Channel(NUM_ROWS + 1)
        self.path_det_channel.set_volume(0.0)
        self.path_det_channel.play(self.path_det_sound, loops=-1)

        self.path_detection_active = False


    def wakeword_callback(self, wakeword: String):
        word = wakeword.data
        self.get_logger().info("Hear Wakeword")

        if word == "set_ground_level":
            self.get_logger().info("Set the ground level")
            MAX_DEPTHS[NUM_ROWS - 1] = self.cell_depth_averages[(NUM_ROWS * NUM_COLUMNS) - NUM_COLUMNS // 2]

    def depth_callback(self, depth_ros_image: Image):
        #deltaTime = time.time() - self.prevDepthCallbackTime
        #self.prevDepthCallbackTime = time.time()
        #self.get_logger().info(f"Depth callback time: {deltaTime}")
        
        #startTime = time.time()
        # Convert ROS image to OpenCV image
        self.depth_img = self.bridge.imgmsg_to_cv2(depth_ros_image, "16UC1")
        #endTime = time.time()
        #self.get_logger().info(f"Depth image conversion time: {endTime - startTime}")

        #calculate_average_start_time = time.time()
        # Initialize a list to store the average depth of each cell
        self.cell_depth_averages = [0] * NUM_CELLS
        # Get the dimensions of the depth image
        depth_map_height = self.depth_img.shape[0]
        depth_map_width = self.depth_img.shape[1]

        step = 10
        r = 0
        while r < depth_map_height:
            c = 0
            while c < depth_map_width:
                cell_r = r * NUM_ROWS // depth_map_height
                cell_c = c * NUM_COLUMNS // depth_map_width
                self.cell_depth_averages[cell_r * NUM_COLUMNS + cell_c] += self.depth_img[r][c]
                c += step
            r += step
        #calculate_averages_end_time = time.time()
        #self.get_logger().info(f"Calculate average depth time: {calculate_averages_end_time - calculate_average_start_time}")
        #normalize_and_volume_start_time = time.time()
        # Normalize the depth values and adjust the volume of each channel
        r = 0
        while r < NUM_ROWS:
            left_obs_volume = 0
            right_obs_volume = 0
            left_hole_volume = 0
            right_hole_volume = 0
            c = 0
            while c < NUM_COLUMNS:
                cell_index = r * NUM_COLUMNS + c
                
                # Calculate the average depth for the cell
                self.cell_depth_averages[cell_index] /= ((depth_map_height * depth_map_width) / NUM_CELLS) / (step * step)
                
                # Calculate the volume based on the depth
                obs_depth_percent = inv_lerp(MAX_DEPTHS[r], MIN_DEPTH, self.cell_depth_averages[cell_index])
                obs_volume = max(0, min(1, obs_depth_percent)) / NUM_COLUMNS * 2
                
                # Adjust left and right volume based on column position
                left_obs_volume += obs_volume * (1 - (c / (NUM_COLUMNS - 1)))
                right_obs_volume += obs_volume * (c / (NUM_COLUMNS - 1))

                '''
                if r == NUM_ROWS - 1:
                    hole_depth_percent = inv_lerp(HOLE_THRESHOLD + MAX_DEPTHS[r] + MAX_HOLE_DEPTH, HOLE_THRESHOLD + MAX_DEPTHS[r], self.cell_depth_averages[cell_index])
                    hole_volume = max(0, min(1, - hole_depth_percent)) / NUM_COLUMNS * 2
                    #self.get_logger().info(f"hole_volume {hole_volume}")

                    left_hole_volume += hole_volume * (1 - (c / (NUM_COLUMNS - 1)))
                    right_hole_volume += hole_volume * (c / (NUM_COLUMNS - 1))
                '''

                c += 1
            
            # Set volume for the current channel
            self.obs_det_channels[r].set_volume(left_obs_volume, right_obs_volume)
            #self.hole_channel.set_volume(left_hole_volume, right_hole_volume)
            r += 1
    
    def sidewalk_detection_callback(self, sidewalk_detection_ros_image: Image):
        if not self.path_detection_active:
            self.path_det_channel.set_volume(0, 0)
            return
        
        sidewalk_detection_img = self.bridge.imgmsg_to_cv2(sidewalk_detection_ros_image, "bgr8")


        path_det_left_volume = self.path_det_check_side(sidewalk_detection_img, PATH_DET_LEFT_X, PATH_DET_LEFT_Y)
        path_det_right_volume = self.path_det_check_side(sidewalk_detection_img, PATH_DET_RIGHT_X, PATH_DET_RIGHT_Y)

        self.path_det_channel.set_volume(path_det_left_volume, path_det_right_volume)
    
    def path_det_check_side(self, img, x_offsets, y_offsets):
        # Get the dimensions of the image
        sidewalk_img_height = img.shape[0]
        sidewalk_img_width = img.shape[1]

        #intensity of path det alarm
        path_det_alarm_intensity = 0

        #loop through the x offsets
        for i in range(len(x_offsets)):
            x = int(sidewalk_img_width * x_offsets[i])
            y = int(sidewalk_img_height * y_offsets[i])
            if not (img[y][x] == [0,255,0]).all():
                path_det_alarm_intensity = i + 1

        return path_det_alarm_intensity / len(x_offsets)

    #Set node_active to true if node manager so
    def node_state_callback(self, node_states: Int32MultiArray):
        self.path_detection_active = node_states.data[path_detection_node_state_index] == 1


def main(args=None):
    rclpy.init(args=args)
    sound_node = PHINIXSoundEffectsUI()
    rclpy.spin(sound_node)
    sound_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
