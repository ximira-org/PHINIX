from phinix_haptics_ui.phinix_haptics_buzzer_manager import format_buzz_command

# Simple confirmation buzz of the main buzzer for a quarter second
def confirmation_buzz():
    print("ConfirmationBuzz")
    return_list = []
    return_list.append(format_buzz_command(0, 0, 155, 0, 0))
    return_list.append(format_buzz_command(0, 250, 0, 0, 0))
    return return_list