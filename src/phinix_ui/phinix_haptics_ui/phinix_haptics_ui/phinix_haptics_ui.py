import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from bluepy.btle import Peripheral, DefaultDelegate, ADDR_TYPE_RANDOM
import time
from phinix_haptics_ui.phinix_haptics_obstacle_detection import obstacle_detection
from phinix_haptics_ui.phinix_haptics_path_detection import path_detection
from phinix_haptics_ui.phinix_haptics_cardinal_direction import cardinal_direction
from phinix_haptics_ui.phinix_haptics_battery_level import battery_level
from phinix_haptics_ui.phinix_haptics_confirmation import confirmation_buzz
from phinix_haptics_ui.phinix_haptics_face_recognition import face_recognition
from phinix_haptics_ui.phinix_haptics_buzzer_manager import format_buzz_command


CHARACTERISTIC_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # Bluetooth UART service UUID
DEVICE_NAME = "Ximira_Phinix_Bracelet"  # Replace with your device's name

TOPIC_WAKEWORD = "/phinix/wakeword"
# Define the topic for the depth image from Phinix
TOPIC_PHINIX_DEPTH_IMG = "/phinix/depth/image_raw"
# Define the topic for the sidewalk detection from Phinix
TOPIC_SIDEWALK_DETECTION_IMG = "/phinix/vis/sidewalk"

# Define the number of rows and columns for the grid
NUM_ROWS = 2
NUM_COLUMNS = 3
NUM_OBS_DET_CELLS = NUM_ROWS * NUM_COLUMNS
'''[Left Top, Left Bottom, Center Top, Center Bottom, Right Top, Right Bottom]'''
OBS_DET_CELL_BUZZER_INDEXES = [1, 3, 0, 4, 7, 5]

#Total number of hapic actuators on the bracelet
NUM_BUZZERS = 8

# Define the maximum depths for each row
MAX_DEPTHS = [1350, 1000, 1350]

# Define the minimum depth, at which the power of the haptic vibration will be 100%
MIN_DEPTH = 100

MIN_POWER = 0
MAX_POWER = 9

MAX_HOLE_DEPTH = 400
HOLE_THRESHOLD = 30

'''
Path Detection checks 3 points for each side, y offset from the bottom of the screen and x offset from the center.
Each point represents a level of urgency. If sidewalk is not detected a the point, the actuator will be vibrated.
'''
PATH_DET_LEFT_Y = [.8, .8, .8]
PATH_DET_LEFT_X = [.3, .38, .45]
PATH_DET_RIGHT_Y = [.8, .8, .8]
PATH_DET_RIGHT_X = [.55, .62, .7]

# Function to linearly interpolate between two values
def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolate on the scale given by a to b, using t as the point on that scale."""
    return (1 - t) * a + t * b

# Function to calculate the inverse linear interpolation
def inv_lerp(a: float, b: float, v: float) -> float:
    """Inverse Linear Interpolation, get the fraction between a and b on which v resides."""
    return (v - a) / (b - a)



class PHINIXHapticsUI(Node):

    def __init__(self):
        super().__init__('phinix_haptics_ui')
        
        # Create a subscriber for the depth image topic
        self.depth_subscriber = self.create_subscription(Image, TOPIC_PHINIX_DEPTH_IMG, self.depth_callback, 10)
        #Subscribe to the sidewalk detection topic
        self.sidewalk_detection_subscriber = self.create_subscription(Image, TOPIC_SIDEWALK_DETECTION_IMG, self.sidewalk_detection_callback, 10)
        
        self.bridge = CvBridge()

        self.device_address = "E9:B4:7A:77:33:EC"
        #self.device_address = "ED:D6:73:B4:F2:DF" #read_mac_address_from_file(self)
        
        self.get_logger().info(self.device_address)

        self.connect()

        #create a timer for publishing obs det power levels
        self.obs_det_power_level_publish_timer = self.create_timer(.1, self.send_obs_det_power_levels_to_haptics)

        #for logging time between deoth callbacks
        self.prev_depth_callback_time = time.time()

        # Initialize a list to store the average depth of each cell
        self.cell_depth_averages = [0] * NUM_OBS_DET_CELLS

        self.obs_det_power_levels = [0] * NUM_OBS_DET_CELLS

        # Booleans for if there is a visible hole in each cell of the bottom row
        self.holes_detected = [False] * NUM_COLUMNS

        #Turn all the buzzers we use for obs det on all the way.
        i = 0
        while i < NUM_OBS_DET_CELLS:
            command = format_buzz_command(OBS_DET_CELL_BUZZER_INDEXES[i], 0, 255, 0, 0)
            self.characteristic.write(bytearray(command, "utf-8"))
            i += 1
            time.sleep(.2)
        
        command = format_buzz_command(2, 0, 255, 100, 200)
        self.characteristic.write(bytearray(command, "utf-8"))
        time.sleep(.2)
        command = format_buzz_command(6, 0, 255, 100, 200)
        self.characteristic.write(bytearray(command, "utf-8"))
        time.sleep(.2)
        
        #power levels for sidewalk detection
        self.path_det_left_power = 0
        self.path_det_right_power = 0

        #create a timer for publishing sidewalk detection power levels
        self.path_det_power_level_publish_timer = self.create_timer(.07, self.send_path_det_power_levels_to_haptics)
    
    def connect(self):
        self.get_logger().info(f"Attempting to connect to haptic band: {self.device_address}")

        haptic_band = Peripheral()
        haptic_band.connect(self.device_address, ADDR_TYPE_RANDOM)

        self.characteristic = haptic_band.getCharacteristics(uuid=CHARACTERISTIC_UUID)[0]

        self.get_logger().info(f"Connected to device: {self.device_address}")
    
    def depth_callback(self, depth_ros_image: Image):
        deltaTime = time.time() - self.prev_depth_callback_time
        self.prev_depth_callback_time = time.time()
        #self.get_logger().info(f"Depth callback time: {deltaTime}")

        # Convert ROS image to OpenCV image
        self.depth_img = self.bridge.imgmsg_to_cv2(depth_ros_image, "16UC1")

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
        
        # Normalize the depth values and make a buzzer command for each level
        i = 0
        self.obs_det_power_levels = [0] * NUM_OBS_DET_CELLS
        r = 0
        while r < NUM_ROWS:
            c = 0
            while c < NUM_COLUMNS:
                cell_index = r * NUM_COLUMNS + c
                self.cell_depth_averages[cell_index] /= ((depth_map_height * depth_map_width) / NUM_OBS_DET_CELLS) / (step * step)
                percent = 1 - (self.cell_depth_averages[cell_index] - MIN_DEPTH) / (MAX_DEPTHS[r] - MIN_DEPTH)
                percent = max(0, min(1, percent))
                power = int(MIN_POWER + (MAX_POWER - MIN_POWER) * percent)
                self.obs_det_power_levels[cell_index] = power

                '''
                #Detect and warn the user about holes in the ground in front of them
                if r == NUM_ROWS - 1:
                    hole_depth_percent = inv_lerp(HOLE_THRESHOLD + MAX_DEPTHS[r] + MAX_HOLE_DEPTH, HOLE_THRESHOLD + MAX_DEPTHS[r], self.cell_depth_averages[cell_index])
                    hole_depth_percent = max(0, min(1, - hole_depth_percent))
                    hole_power = int(MIN_POWER + (MAX_POWER - MIN_POWER) * hole_depth_percent)
                    self.obs_det_power_levels[cell_index] = hole_power
                    #if there is a hole and it has net been detected yet, switch patterns
                    if hole_power > 0 and self.holes_detected[c] == False:
                        self.holes_detected[c] = True
                        command = format_buzz_command(OBS_DET_CELL_BUZZER_INDEXES[cell_index], 0, 255, 0, 60)
                        self.characteristic.write(bytearray(command, "utf-8"))
                    elif hole_power == 0 and self.holes_detected[c] == True:
                        self.holes_detected[c] = False
                        command = format_buzz_command(OBS_DET_CELL_BUZZER_INDEXES[cell_index], 0, 255, 0, 0)
                        self.characteristic.write(bytearray(command, "utf-8"))
                '''
                
                c += 1
            r += 1
    
    def sidewalk_detection_callback(self, sidewalk_detection_ros_image: Image):
        sidewalk_detection_img = self.bridge.imgmsg_to_cv2(sidewalk_detection_ros_image, "bgr8")


        self.path_det_left_power = self.path_det_check_side(sidewalk_detection_img, PATH_DET_LEFT_X, PATH_DET_LEFT_Y, True)
        self.path_det_right_power = self.path_det_check_side(sidewalk_detection_img, PATH_DET_RIGHT_X, PATH_DET_RIGHT_Y, False)

        #self.path_det_channel.set_volume(path_det_left_volume, path_det_right_volume)
    
    def path_det_check_side(self, img, x_offsets, y_offsets, is_left):
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

        return int(path_det_alarm_intensity / len(x_offsets) * MAX_POWER)
    
    def send_path_det_power_levels_to_haptics(self):
        command = "!L"
        command += "2" + str(self.path_det_left_power)
        command += "6" + str(self.path_det_right_power)
        command += ";"
        #command = "!L69;"
        self.get_logger().info(command)
        self.characteristic.write(bytearray(command, "utf-8"))
    
    def send_obs_det_power_levels_to_haptics(self):
        command = "!L"
        if len(self.obs_det_power_levels) == 0:
            return
        
        i = 0
        while i < len(self.obs_det_power_levels):
            command += str(OBS_DET_CELL_BUZZER_INDEXES[i])
            command += str(self.obs_det_power_levels[i])
            i += 1
        
        command += ";"
        self.get_logger().info(command)
        self.characteristic.write(bytearray(command, "utf-8"))

        '''
        self.commands = [
            "!o111000000",
            "!o000010000",
            "!o000000011",
            "!p10",
            "!p01",
            "!p22",
            "!p33",
            "!cNN",
            "!cSW",
            "!cEE",
            "!b3",
            "!b1",
            "!y",
            "!f0",
            "!f2",
            "!f4",
        ]
        self.command_index = 0
        self.connect()
        self.send_data()
        '''

    '''
    def obs_det_callback(self, obs_det_msg: Int32MultiArray):
        obs_det = obs_det_msg.data
        self.current_obs_det_state = obs_det

    def connect(self):
    
    
    
    def send_data(self):

        

        self.timer = self.create_timer(UI_PERIOD, self.ui_callback)
        
        #print(time.time())
        while False:
            
            time.sleep(3)

        while True:
            current_command = self.commands[self.command_index]
            command_list = []

            self.get_logger().info(f"current_command: {current_command}")

            if current_command[1] == 'o':
                self.get_logger().info(f"command invokes obstacle detection")
                command_list = obstacle_detection(current_command)
            elif current_command[1] == 'p':
                self.get_logger().info(f"command invokes path detection")
                command_list = path_detection(current_command)
            elif current_command[1] == 'c':
                self.get_logger().info(f"command invokes cardinal direction")
                command_list = cardinal_direction(current_command)
            elif current_command[1] == 'b':
                self.get_logger().info(f"command invokes battery level")
                command_list = battery_level(current_command)
            elif current_command[1] == 'y':
                self.get_logger().info(f"command invokes haptics")
                command_list = confirmation_buzz()
            elif current_command[1] == 'f':
                self.get_logger().info(f"command invokes face recognition")
                command_list = face_recognition(current_command)

            for command in command_list:
                #print(command)
                characteristic.write(bytearray(command, "utf-8"))
                self.get_logger().info(f"Sent command: {command} to device")

            self.command_index = (self.command_index + 1) % len(self.commands)
            
            time.sleep(3)
    
    
    def ui_callback(self):
        #current_command = self.commands[self.command_index]
        command_list = []
        #command_list = obstacle_detection(current_command)
        #print("ui callback")
        #if there is new information in the obs detection array
        if obs_det_arrays_equal(self.prev_obs_det_ui_state, self.current_obs_det_state) == False:
            command_list = obstacle_detection(self.current_obs_det_state)
            self.prev_obs_det_ui_state = self.current_obs_det_state
        
        for command in command_list:
            #print(command)
            self.characteristic.write(bytearray(command, "utf-8"))
            self.get_logger().info(f"Sent command: {command} to device")

        self.command_index = (self.command_index + 1) % len(self.commands)
    '''

'''
def read_mac_address_from_file(self):
    file_path = "settings/mac_address_file.txt"

    with open(file_path, 'r') as file:
        mac_address = file.readline().strip()  # Assuming MAC address is on the first line
        return mac_address

def obs_det_arrays_equal(old_array, new_array):
    equal = True
    i = 0
    while i < len(old_array):
        if old_array[i] != new_array[i]:
            equal = False
            break
        i += 1
    return equal
'''

def main(args=None):
    rclpy.init(args=args)
    haptics_node = PHINIXHapticsUI()
    rclpy.spin(haptics_node)
    haptics_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main() 
