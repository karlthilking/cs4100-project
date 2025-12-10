from typing import List, Dict, Tuple
import os
import random
import numpy as np
import pretty_midi as pm
import matplotlib.pyplot as plt
import seaborn as sns
from music21 import stream, note, instrument, tempo
from data_processor import DataProcessor
from hmm import HMM
from collections import Counter


def decode_note(note_hash: int) -> Tuple[int, int, float]:
    """
    Decode a hashed note representation into its components: pitch, velocity, duration.
    Each field is unpacked from specific bit positions of the note_hash and mapped back to real MIDI values
    """
    dur_bin      =  note_hash        & 0x3
    octave_bin   = (note_hash >> 2)  & 0x3
    velo_bin     = (note_hash >> 4)  & 0x3
    pitch_class  = (note_hash >> 6)  & 0xF

    OCTAVE_BIN_TO_OCT = [2, 3, 4, 5]
    VEL_BIN_TO_VEL = [24, 64, 96, 112]
    DUR_BIN_TO_SEC_PIANO = [0.10, 0.22, 0.45, 0.80]

    octave = OCTAVE_BIN_TO_OCT[octave_bin]
    pitch = (octave + 1) * 12 + pitch_class
    velocity = VEL_BIN_TO_VEL[velo_bin]
    duration = DUR_BIN_TO_SEC_PIANO[dur_bin]

    return pitch, velocity, duration

def sequence_to_midi(state_sequence: List[int], out_path: str):
    """
    Convert a sequence of hashed HMM states into a MIDI file using decoded pitch, velocity, and duration values
    """
    midi = pm.PrettyMIDI()
    inst = pm.Instrument(program=0)

    current_time = 0.0
    for note_hash in state_sequence:
        pitch, vel, dur = decode_note(note_hash)
        start, end = current_time, current_time + dur
        inst.notes.append(pm.Note(vel, pitch, start, end))
        current_time = end

    midi.instruments.append(inst)
    midi.write(out_path)

def sequence_to_sheet(state_sequence: List[int], out_path: str):
    """
     Convert a sequence of hashed states into a MusicXML score by decoding pitch and duration
     """
    s = stream.Stream()
    s.append(instrument.Piano())
    s.append(tempo.MetronomeMark(number=120))

    DUR_BIN_TO_QL = [0.25, 0.5, 1.0, 2.0]

    for note_hash in state_sequence:
        pitch, _, _ = decode_note(note_hash)
        dur_bin = note_hash & 0x3

        n = note.Note(pitch)
        n.quarterLength = DUR_BIN_TO_QL[dur_bin]
        s.append(n)

    s.write("musicxml", fp=out_path)

def get_pitch_from_hash(h: int) -> int:
    """
    Extract the MIDI pitch from a hashed note representation
    """
    return decode_note(h)[0]

def generate_harmony(
    observations: List[int],
    midi_out: str,
    sheet_out: str,
):
    """
    Run Viterbi decoding on a melody to generate the most likely harmony, then export it as both MIDI and MusicXML
    """
    hmm = HMM(path="HMM_params")
    best_path, best_log_prob = hmm.viterbi(observations)

    sequence_to_midi(best_path, midi_out)
    sequence_to_sheet(best_path, sheet_out)

    return best_path, best_log_prob

def run_analysis(dp_test, test_idx, observations, best_path, folder):
    """
    Compare the generated harmony to the original accompaniment for one test melody and create some analysis plots
    """
    violin_seqs = dp_test.violin_sequences
    piano_seqs = dp_test.piano_sequences

    raw_melody = violin_seqs[test_idx]
    raw_piano = piano_seqs[test_idx]

    melody_pitches = [get_pitch_from_hash(h) for h in observations]
    generated_harmony_pitches = [get_pitch_from_hash(h) for h in best_path]
    true_harmony_pitches = [int(n[0]) for n in raw_piano]

    min_len = min(len(melody_pitches),
                  len(true_harmony_pitches),
                  len(generated_harmony_pitches))

    melody = melody_pitches[:min_len]
    true_h = true_harmony_pitches[:min_len]
    gen_h = generated_harmony_pitches[:min_len]

    # Melody vs True Harmony
    plt.figure(figsize=(12, 4))
    plt.plot(melody, label="Violin Melody", color="darkblue")
    plt.plot(true_h, label="Original Piano Accompaniment Harmony", color="darkgreen")
    plt.title("Melody vs Original Harmony")
    plt.xlabel("Note index")
    plt.ylabel("Pitch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "melody_vs_true.png"), dpi=300)
    plt.close()

    # Melody vs Generated harmony
    plt.figure(figsize=(12, 4))
    plt.plot(melody, label="Violin Melody", color="darkblue")
    plt.plot(gen_h, label="Generated Piano Accompaniment Harmony", color="darkgreen")
    plt.title("Melody vs Generated Harmony")
    plt.xlabel("Note index")
    plt.ylabel("Pitch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "melody_vs_generated.png"), dpi=300)
    plt.close()

    # Melody pitch vs Harmony pitch (True vs Generated)
    min_pitch = min(melody + true_h + gen_h)
    max_pitch = max(melody + true_h + gen_h)
    plt.figure(figsize=(6, 6))
    plt.scatter(melody, true_h, s=15, alpha=0.6, label="Original Harmony")
    plt.scatter(melody, gen_h, s=15, alpha=0.6, label="Generated Harmony")
    plt.plot([min_pitch, max_pitch], [min_pitch, max_pitch],
             linestyle="--", linewidth=1, label="Unison line")
    plt.xlabel("Melody pitch (MIDI)")
    plt.ylabel("Harmony pitch (MIDI)")
    plt.title("Melody vs Harmony Pitch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "melody_vs_harmony_scatter.png"), dpi=300)
    plt.close()

    # Interval (harmony - melody) distributions
    true_intervals = [h - m for m, h in zip(melody, true_h)]
    gen_intervals  = [h - m for m, h in zip(melody, gen_h)]
    plt.figure(figsize=(10, 4))
    bins = np.arange(-24, 25)
    sns.histplot(true_intervals, bins=bins, stat="probability",
                 element="step", fill=False, label="Original Harmony")
    sns.histplot(gen_intervals, bins=bins, stat="probability",
                 element="step", fill=False, label="Generated Harmony")
    plt.xlabel("Interval (Harmony - Melody, semitones)")
    plt.ylabel("Probability")
    plt.title("Interval Distribution: Melody to Harmony")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "interval_distribution.png"), dpi=300)
    plt.close()


def main():
    """
    Select a random test melody, generate its harmony using the HMM,
    save MIDI/score outputs and analysis plots, and write run metadata
    """
    dp_test = DataProcessor(train=False)
    dp_test.init_note_sequences()

    violin_seqs = dp_test.violin_sequences

    non_empty_indices = [i for i, seq in enumerate(violin_seqs) if len(seq) > 0]

    # Pick a random non-empty melody from the dataset
    test_idx = random.choice(non_empty_indices)
    print("Randomly selected test melody idx:", test_idx)

    folder = f"run_{test_idx:04d}"
    os.makedirs(folder, exist_ok=True)

    raw_melody = violin_seqs[test_idx]

    # Hash melody notes into observation symbols for the HMM
    observations = [DataProcessor.hash_note(n, is_piano=False) for n in raw_melody]

    midi_path = os.path.join(folder, "harmony.mid")
    sheet_path = os.path.join(folder, "harmony.musicxml")

    best_path, best_log_prob = generate_harmony(
        observations=observations,
        midi_out=midi_path,
        sheet_out=sheet_path,
    )

    # Save run metadata
    with open(os.path.join(folder, "metadata.txt"), "w") as f:
        f.write(f"test_idx = {test_idx}\n")
        f.write(f"melody_length = {len(observations)}\n")
        f.write(f"best_log_prob = {best_log_prob}\n")

    print("Saved output to:", folder)

    run_analysis(
        dp_test=dp_test,
        test_idx=test_idx,
        observations=observations,
        best_path=best_path,
        folder=folder,
    )


if __name__ == '__main__':
    main()

