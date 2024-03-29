from phinix_haptics_ui.phinix_haptics_buzzer_manager import format_buzz_command

#buzz the buzzer for each person index one after another, in a circle around the wrist.
#so if it is person zero, buzz only the frist buzzer, if it is person one, buzz the first and second buzzer, etc.
def face_recognition(packet_buffer):
    person = int(packet_buffer[2])
    index = 0
    return_list = []
    start_time = 0
    while(index <= person):
        return_list.append(format_buzz_command(index, start_time, 157, 0, 0))
        start_time += 200
        return_list.append(format_buzz_command(index, start_time, 0, 0, 0))
        index += 1
    return return_list