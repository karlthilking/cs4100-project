from typing import List, Tuple
import pretty_midi as pm
from data_processor import DataProcessor
from hmm import HMM
from music21 import stream, note, instrument, tempo
import random
import os
import numpy as np

# decode from hash to MIDI information
def decode_note(note_hash: int) -> Tuple[int, int, float]:
    dur_bin      =  note_hash        & 0x3          
    octave_bin   = (note_hash >> 2)  & 0x3
    velo_bin     = (note_hash >> 4)  & 0x3        
    pitch_class  = (note_hash >> 6)  & 0xF
    
    # TODO: could be tweaked
    OCTAVE_BIN_TO_OCT = [2, 3, 4, 5]

    VEL_BIN_TO_VEL = [
        24,   
        64,   
        96,  
        112, 
    ]
    
    DUR_BIN_TO_SEC_PIANO = [
        0.10,  
        0.22,  
        0.45,  
        0.80,  
    ]

    octave = OCTAVE_BIN_TO_OCT[octave_bin]
    pitch = (octave + 1) * 12 + pitch_class
    velocity = VEL_BIN_TO_VEL[velo_bin]
    duration = DUR_BIN_TO_SEC_PIANO[dur_bin]

    return pitch, velocity, duration

# take in a sequence and turn it into a midi file
def sequence_to_midi(state_sequence: List[int], out_path: str):
    midi = pm.PrettyMIDI()
    # set instrument to piano
    inst = pm.Instrument(program=0)

    current_time = 0.0
    for note_hash in state_sequence:
        pitch, vel, dur = decode_note(note_hash)
        start = current_time
        end = start + dur
        inst.notes.append(pm.Note(vel, pitch, start, end))
        current_time = end
        
    midi.instruments.append(inst)
    midi.write(out_path)

# create sheet music from sequence
def sequence_to_sheet(state_sequence: List[int], out_path: str):
    s = stream.Stream()
    s.append(instrument.Piano())

    s.append(tempo.MetronomeMark(number=120))
    DUR_BIN_TO_QL = [0.25, 0.5, 1.0, 2.0]

    for note_hash in state_sequence:
        pitch, vel, dur = decode_note(note_hash)

        n = note.Note(pitch)
        dur_bin = note_hash & 0x3  
        n.quarterLength = DUR_BIN_TO_QL[dur_bin]
        
        s.append(n)

    s.write("musicxml", fp=out_path)

# generate the harmony part with our HMM
def generate_harmony(
    observations,
    midi_out: str,
    sheet_out: str,
):
    hmm = HMM()
    best_path, best_log_prob = hmm.viterbi(observations)

    sequence_to_midi(best_path, midi_out)
    sequence_to_sheet(best_path, sheet_out)

    return best_path, best_log_prob

def main():
    # Load testing melodies
    dp_test = DataProcessor(train=False)
    dp_test.init_note_sequences()

    song_ix = np.random.randint(dp_test.num_songs)
    violin_seq = dp_test.violin_sequences[song_ix]

    # save for analysis
    fp = f"samples/song_{song_ix}/{song_ix}.txt"
    os.mkdir(f"samples/song_{song_ix}")
    with open(fp, "w") as f:
        f.write(fp)

    print("Generating harmony with melody #: ", song_ix)

    observations = [DataProcessor.hash_note(n, is_piano=False) for n in violin_seq]

    midi_path = f"samples/song_{song_ix}/harmony_{song_ix}.mid"
    sheet_path = f"samples/song_{song_ix}/harmony_{song_ix}.musicxml"
    
    best_path, best_log_prob = generate_harmony(
        observations=observations,
        midi_out=midi_path,
        sheet_out=sheet_path,
    )
    
    print("Best log-prob:", best_log_prob)
    
if __name__ == "__main__":
    main()
