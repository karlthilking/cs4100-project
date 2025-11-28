from typing import List, Dict, Tuple
import pretty_midi as pm
from data_processor import DataProcessor
from hmm_v2 import HMM
from music21 import stream, note, instrument, tempo
import random

# decode from hash to MIDI information
def decode_note(note_hash: int) -> Tuple[int, int, float]:
    dur_bin      =  note_hash        & 0x3          
    octave_bin   = (note_hash >> 2)  & 0x3        
    velo_bin     = (note_hash >> 4)  & 0x7        
    pitch_class  = (note_hash >> 7)  & 0xF 

    # TODO: could be tweaked
    OCTAVE_BIN_TO_OCT = [2, 3, 4, 5]  # you can tweak these if needed

    # use midpoints as representative values
    VEL_BIN_TO_VEL = [
        8,  
        24, 
        40,  
        56,  
        72,  
        88,
        104,
        120,
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
    
    for note_hash in state_sequence:
        pitch, vel, dur = decode_note(note_hash)

        n = note.Note(pitch)
        n.duration.seconds = dur
        s.append(n)

    s.write("musicxml", fp=out_path)

# generate the harmony part with our HMM
def generate_harmony(
    states,
    start_prob,
    trans_prob,
    emit_prob,
    observations,
    midi_out: str,
    sheet_out: str,
):
    hmm = HMM(states, start_prob, trans_prob, emit_prob)
    best_path, best_log_prob = hmm.viterbi(observations)

    sequence_to_midi(best_path, midi_out)
    sequence_to_sheet(best_path, sheet_out)

    return best_path, best_log_prob

def main():
    # Train HMM on train set
    dp_train = DataProcessor(train=True)
    dp_train.init_note_sequences()
    dp_train.init_hmm()
    T_prob, O_prob, pi_prob, states = dp_train.get_hmm_params()

    # Load testing melodies
    dp_test = DataProcessor(train=False)
    dp_test.init_note_sequences()

    # violin_sequences are
    violin_seqs = dp_test._DataProcessor__violin_sequences 

    # pick a random test melody
    non_empty_indices = [i for i, seq in enumerate(violin_seqs) if len(seq) > 0]
    test_idx = random.choice(non_empty_indices)

    raw_melody = violin_seqs[test_idx]
    observations = [DataProcessor.hash_note(n, is_piano=False) for n in raw_melody]

    midi_path = "harmony_final.mid"
    sheet_path = "harmony_final.musicxml"
    
    best_path, best_log_prob = generate_harmony(
        states=list(states),
        start_prob=pi_prob,
        trans_prob=T_prob,
        emit_prob=O_prob,
        observations=observations,
        midi_out=midi_path,
        sheet_out=sheet_path,
    )
    
    print("Test melody idx:", test_idx)
    print("Best log-prob:", best_log_prob)
    
if __name__ == "__main__":
    main()
