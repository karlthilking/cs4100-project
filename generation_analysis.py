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


def load_hmm_params(path: str = "HMM_params"):
    T_arr = np.load(os.path.join(path, "T.npy"))
    O_arr = np.load(os.path.join(path, "O.npy"))
    pi_arr = np.load(os.path.join(path, "pi.npy"))
    states = np.load(os.path.join(path, "states.npy")).tolist()
    obs = np.load(os.path.join(path, "obs.npy")).tolist()

    state_list = list(states)
    obs_list = list(obs)

    T_dict, O_dict, pi_dict = {}, {}, {}

    for i, s in enumerate(state_list):
        row = T_arr[i]
        T_dict[s] = {state_list[j]: float(val) for j, val in enumerate(row) if val > 0}

    for i, s in enumerate(state_list):
        row = O_arr[i]
        O_dict[s] = {obs_list[k]: float(val) for k, val in enumerate(row) if val > 0}

    for i, s in enumerate(state_list):
        if pi_arr[i] > 0:
            pi_dict[s] = float(pi_arr[i])

    return T_dict, O_dict, pi_dict, state_list, obs_list



def decode_note(note_hash: int) -> Tuple[int, int, float]:
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
    return decode_note(h)[0]


def generate_harmony(
    observations: List[int],
    midi_out: str,
    sheet_out: str,
):
    hmm = HMM(path="HMM_params")
    best_path, best_log_prob = hmm.viterbi(observations)

    sequence_to_midi(best_path, midi_out)
    sequence_to_sheet(best_path, sheet_out)

    return best_path, best_log_prob



def run_analysis(dp_test, test_idx, observations, best_path, folder):
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
    plt.plot(melody, label="Violin Melody)", color="darkblue")
    plt.plot(gen_h, label="Generated Piano Accompaniment Harmony", color="darkgreen")
    plt.title("Melody vs Generated Harmony")
    plt.xlabel("Note index")
    plt.ylabel("Pitch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "melody_vs_generated.png"), dpi=300)
    plt.close()

    T_arr = np.load("HMM_params/T.npy")
    plt.figure(figsize=(10, 8))
    sns.heatmap(T_arr, xticklabels=False, yticklabels=False)
    plt.title("HMM Transition Probability Matrix")
    plt.xlabel("Next state")
    plt.ylabel("Previous state")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "transition_matrix.png"), dpi=300)
    plt.close()


def main():
    dp_test = DataProcessor(train=False)
    dp_test.init_note_sequences()

    violin_seqs = dp_test.violin_sequences

    non_empty_indices = [i for i, seq in enumerate(violin_seqs) if len(seq) > 0]
    test_idx = random.choice(non_empty_indices)

    print("Randomly selected test melody idx:", test_idx)

    folder = f"run_{test_idx:04d}"
    os.makedirs(folder, exist_ok=True)

    raw_melody = violin_seqs[test_idx]
    observations = [DataProcessor.hash_note(n, is_piano=False) for n in raw_melody]

    midi_path = os.path.join(folder, "harmony.mid")
    sheet_path = os.path.join(folder, "harmony.musicxml")

    best_path, best_log_prob = generate_harmony(
        observations=observations,
        midi_out=midi_path,
        sheet_out=sheet_path,
    )

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

