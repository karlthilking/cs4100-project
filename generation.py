from typing import List, Dict, Tuple
import pretty_midi as pm
from data_processor import DataProcessor
from hmm_v2 import HMM
from music21 import stream, note, instrument, tempo

# decode from hash to MIDI information
def decode_note(note_hash: int) -> Tuple[int, int, float, float]:
    pitch = note_hash & 0x7F
    velocity = (note_hash >> 7) & 0x7F
    duration = ((note_hash >> 15) & 0x7FFF) / 100.0
    start = ((note_hash >> 29) & 0x3FFFF) / 100.0
    return pitch, velocity, duration, start

# take in a sequence and turn it into a midi file
def sequence_to_midi(state_sequence: List[int], out_path: str):
    midi = pm.PrettyMIDI()
    # set instrument to piano
    inst = pm.Instrument(program=0)

    for note_hash in state_sequence:
        pitch, vel, dur, start = decode_note(note_hash)
        inst.notes.append(pm.Note(vel, pitch, start, start + dur))

    midi.instruments.append(inst)
    midi.write(out_path)

# create sheet music from sequence
def sequence_to_sheet(state_sequence: List[int], out_path: str):
    s = stream.Stream()
    s.append(instrument.Piano())
    # take into account tempo

    for note_hash in state_sequence:
        pitch, vel, dur, start = decode_note(note_hash)

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

    dp.init_midi_objects()
    dp.init_note_sequences()

    # initially set observations to none
    observations = None
    # for every sequence in the violin/melody
    for seq in dp.violin_sequences:
        # if there is a sequence
        if seq:
            # set our observation to this sequence
            observations = seq
            break

    # HMM
    T_prob, O_prob, pi_prob, states, interval_count = dp.get_hmm()

    # Generate harmony and music sheet
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
