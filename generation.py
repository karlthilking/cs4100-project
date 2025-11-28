from typing import List, Dict, Tuple
import pretty_midi as pm
from data_processor import DataProcessor
from hmm import HMM
from music21 import stream, note, instrument, tempo
import random
import pickle
import os
import numpy as np


def load_hmm_params(path: str = "HMM_params"):
    # load arrays saved by DataProcessor.save_hmm_params
    T_arr = np.load(os.path.join(path, "T.npy"))
    O_arr = np.load(os.path.join(path, "O.npy"))
    pi_arr = np.load(os.path.join(path, "pi.npy"))
    states = np.load(os.path.join(path, "states.npy")).tolist()
    obs = np.load(os.path.join(path, "obs.npy")).tolist()

    state_list = list(states)
    obs_list = list(obs)

    # convert arrays back to dict-of-dicts for HMM
    T_dict: Dict[int, Dict[int, float]] = {}
    O_dict: Dict[int, Dict[int, float]] = {}
    pi_dict: Dict[int, float] = {}

    # transitions: T[s][s'] = prob
    for i, s in enumerate(state_list):
        row = T_arr[i]
        T_dict[s] = {}
        for j, s2 in enumerate(state_list):
            val = float(row[j])
            if val > 0.0:
                T_dict[s][s2] = val

    # emissions: O[s][o] = prob
    for i, s in enumerate(state_list):
        row = O_arr[i]
        O_dict[s] = {}
        for k, o in enumerate(obs_list):
            val = float(row[k])
            if val > 0.0:
                O_dict[s][o] = val

    # start probs: pi[s] = prob
    for i, s in enumerate(state_list):
        val = float(pi_arr[i])
        if val > 0.0:
            pi_dict[s] = val

    return T_dict, O_dict, pi_dict, state_list, obs_list


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
    # Load trained HMM 
    T_prob, O_prob, pi_prob, states, obs = load_hmm_params()

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
