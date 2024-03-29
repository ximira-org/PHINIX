from phinix_haptics_ui.phinix_haptics_buzzer_manager import format_buzz_command

def play_path_detection_side(channel, urgency):
    urgency_int = int(urgency)
    if urgency_int == 0:
        return []
    
    strength = 255 // (4 - urgency_int)
    return_list = []
    return_list.append(format_buzz_command(channel, 0, strength, 0, 250))
    return_list.append(format_buzz_command(channel, 1500, 0, 0, 0))
    return return_list

def path_detection(packet_buffer):
    return_list = []
    # The packet buffer has format !pnn where each n is a 0-3 that corresponds to an urgency level for a side of the path detection
    urgency_1 = packet_buffer[2]
    urgency_2 = packet_buffer[3]
    
    path_command_list = play_path_detection_side(1, urgency_1)
    for command in path_command_list:
        return_list.append(command)
    
    path_command_list = play_path_detection_side(5, urgency_2)
    for command in path_command_list:
        return_list.append(command)
    
    return return_list
