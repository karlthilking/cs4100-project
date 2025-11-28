from typing import List, Dict, Tuple
import pretty_midi as pm
from data_processor import DataProcessor
from hmm_v2 import HMM
from music21 import stream, note, instrument, tempo
import random

# decode from hash to MIDI information
def decode_note(note_hash: int) -> Tuple[int, int, float]:
    pitch = note_hash & 0x7F
    velocity = (note_hash >> 7) & 0x7F
    dur_bin = (note_hash >> 14) & 0x7 

    piano_bin_durations = [
        0.08,   
        0.125,  
        0.18,   
        0.25,  
        0.35,   
        0.5,    
        1.2,    
        2.0,    
    ]

    duration = piano_bin_durations[dur_bin]
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
    dp = DataProcessor()
    dp.init_note_sequences()

    # pick a random violin melody as a test
    non_empty_indices = [i for i, seq in enumerate(dp.violin_sequences) if seq]
    test_idx = random.choice(non_empty_indices)

    raw_melody = dp.violin_sequences[test_idx]
    observations = [DataProcessor.hash_note(n, is_piano=False) for n in raw_melody]

    # build training sets on all melodies, except the testing one
    train_piano = [seq for i, seq in enumerate(dp.piano_sequences) if i != test_idx]
    train_violin = [seq for i, seq in enumerate(dp.violin_sequences) if i != test_idx]

    # HMM trained on N-1 songs
    dp._DataProcessor__piano_sequences = train_piano
    dp._DataProcessor__violin_sequences = train_violin
    dp.init_hmm()
    T_prob, O_prob, pi_prob, states = dp.get_hmm_params()

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

if __name__ == "__main__":
    main()