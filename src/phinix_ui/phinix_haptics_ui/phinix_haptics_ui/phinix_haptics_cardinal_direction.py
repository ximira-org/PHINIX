from phinix_haptics_ui.phinix_haptics_buzzer_manager import format_buzz_command

# Counter-clockwise orientation of cardinal directions
cardinal_direction_buzzer_channels = ["NN", "NW", "WW", "SW", "SS", "SE", "EE", "NE"]

def get_index(target, array):
    for i, item in enumerate(array):
        if target == item:
            return i
    return -1  # Return -1 if the target string is not found in the array

# Buzz the channel that corresponds to the direction
def cardinal_direction(packet_buffer):
    print("CardinalDirection")
    direction = str(packet_buffer[2]) + str(packet_buffer[3])
    print(direction)
    index = get_index(direction, cardinal_direction_buzzer_channels)
    print(index)
    if index != -1:
        channel = index + 1  # Add 1 to convert from 0-based index to 1-based channel number
        return_list = []
        return_list.append(format_buzz_command(channel, 0, 255, 0, 250))
        return_list.append(format_buzz_command(channel, 3000, 0, 0, 0))
        return return_list
