def format_buzz_command(channel, start_time, crest, trough, wavelength):
    # Format the command as a string
    command_str = f"!{channel}:{start_time}:{crest}:{trough}:{wavelength}::;"
    return command_str
