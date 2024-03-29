import numpy as np
from phinix_haptics_ui.phinix_haptics_buzzer_manager import format_buzz_command

def od_play_buzzer_column(top, middle, bottom, channel, time):
    channel_count = top + middle + bottom
    print(channel_count)
    if channel_count == 0:
        return []
    
    time_per_buzz = round(time / channel_count)
    start_time = 0
    return_list = []

    if top:
        return_list.append(format_buzz_command(channel, start_time, 255, 255, 0))
        start_time += time_per_buzz

    if middle:
        return_list.append(format_buzz_command(channel, start_time, 153, 153, 0))
        start_time += time_per_buzz

    if bottom:
        return_list.append(format_buzz_command(channel, start_time, 55, 55, 0))
        start_time += time_per_buzz

    if channel_count > 0:
        return_list.append(format_buzz_command(channel, start_time, 0, 0, 0))

    return return_list

def od_detect_buzzer_column(obs_presence_list, top_index, middle_index, bottom_index, channel, time):
    top = obs_presence_list[top_index] == 1
    middle = obs_presence_list[middle_index] == 1
    bottom = obs_presence_list[bottom_index] == 1
    return od_play_buzzer_column(top, middle, bottom, channel, time)

def obstacle_detection(obs_presence_list):
    #print(obs_presence_list)
    return_list = []
    # Left column
    column_commands = od_detect_buzzer_column(obs_presence_list, 0, 1, 2, 5, 2000)
    for command in column_commands:
        return_list.append(command)
    # Middle column
    column_commands = od_detect_buzzer_column(obs_presence_list, 3, 4, 5, 0, 2000)
    for command in column_commands:
        return_list.append(command)
    # Right column
    column_commands = od_detect_buzzer_column(obs_presence_list, 6, 7, 8, 3, 2000)
    for command in column_commands:
        return_list.append(command)
        
    return return_list

