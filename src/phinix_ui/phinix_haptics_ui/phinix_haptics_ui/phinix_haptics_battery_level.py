from phinix_haptics_ui.phinix_haptics_buzzer_manager import format_buzz_command

def battery_level(packet_buffer):
    level = int(packet_buffer[2])
    start_time = 0
    return_list = []
    
    for i in range(level):
        return_list.append(format_buzz_command(0, start_time, 157, 0, 0))
        start_time += 200
        return_list.append(format_buzz_command(0, start_time, 0, 0, 0))
        start_time += 200
    
    return return_list
