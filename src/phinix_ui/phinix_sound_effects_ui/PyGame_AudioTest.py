import pygame
import keyboard

num_channels = 5
i = 0

channels_playing = [None] * num_channels
channels = [None] * num_channels
sounds = [None] * num_channels

pygame.mixer.init()
print(pygame.mixer.get_init())
print(pygame.mixer.get_num_channels())

while i < num_channels:
    channels_playing[i] = False
    sounds[i] = pygame.mixer.Sound("audio_files/sound" + str(i) + ".wav")
    channels[i] = pygame.mixer.Channel(i)
    i += 1


while True:
    if keyboard.is_pressed('esc'):  # Exit if 'Esc' is pressed
        pygame.mixer.quit()
        break
    if keyboard.is_pressed('left'):
        print("left!!!")
    i = 0
    while i < num_channels:
        if keyboard.is_pressed(str(i)) and channels_playing[i] == False:
            print("pressed:" + str(i))
            channels_playing[i] = True
            channels[i].play(sounds[i], -1)
        if keyboard.is_pressed(str(i)) == False and channels_playing[i]:
            print("released:" + str(i))
            channels_playing[i] = False
            channels[i].stop()
        i += 1