import os
import mido
from mido import MidiFile, MidiTrack, Message

# we've got 0 for piano and 40 for violin 
piano_program_number = 0
violin_program_number = 40
'''
https://mido.readthedocs.io/en/latest/messages/index.html
Notes on Mido from it's documentation:
Mido numbers channels 0-16 rather than 1-16. Add and subtract 1 when communicating with user.

Messages: mido.Message('note_on', note = 100, velocity = 3, time = 6.2)

All attributes will default to 0. Velocity defaults to 64 and data defaults to ()
Mido always makes a copy of messages
'''
# dictionary of notes to MIDI representation (ie numbers)
NOTE_TO_MIDI = {
    'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
}

# convert notes to their midi representation using our dictionary
def note_to_midi(note_name):
    # get note name
    note_name = note_name[:-1]  
    # get octave (might need some work)
    octave = int(note_name[-1]) 
    # return the note accounting for the octave number
    return NOTE_TO_MIDI[note_name] + (12 * (octave + 1))

# good news, it looks like all of them are type ones
path = "C:/Users/16173/CS4100/Harmony Project/cs4100-project/piano-violin-data"
# in the event a non type 1 file is given, use just this list
list_of_type_ones = []
def get_type_one_only ():
    # loop through all the midi files
    for filename in os.listdir(path):
        # full path for file
        full_path = os.path.join(path, filename)
        mid = MidiFile(full_path, clip = True)
        # only want the type 1s
        if(mid.type == 1):
            # print file name aka number
            print(filename)
            # add the file name not the full path to the list
            list_of_type_ones.append(filename)
            # separate melodies and harmonies
            separate_melody_and_harmony(filename)

# commented out because we don't need it anymore
# get_type_one_only()
# print(len(list_of_type_ones)) 

# get the channel numbers for melody and harmony parts    
def separate_melody(filename):
    mid = MidiFile(filename, clip = True)

    # ISOLATE INSTRUMENT CHANNEL
    # iterate through all the tracks in the file
    for i, track in enumerate(mid.tracks):
        # iterate through the messages in the track looking for the instrument
        for msg in track:
            # piano (thus harmony) channel
            if msg.type == "program_change" and msg.program == piano_program_number:
                print("piano harmony track: ", msg.channel)       
            # violin (thus melody) channel
            if msg.type == "program_change" and msg.program == violin_program_number:
                print("violin melody track: ", msg.channel)        

def print_midi_info(filename):
    # clip set to true in case we open a file with notes over 127 velocity (max for a note in a midi file)
    mid = MidiFile(filename, clip = True)

    separate_melody(filename)
    #print(mid)

    # # meta information about the MIDI file
    # for msg in mid:
    #     print(msg)

# printing this as an example
print_midi_info('piano-violin-data\Mono-Melodies-0001.mid')
# it appears every file is type 1 and the violin is always in track 3
 
# creating a midi file