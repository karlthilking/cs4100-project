import matplotlib.pyplot as plt
from data_processor import DataProcessor
from hmm import HMM
import seaborn as sns
import numpy as np
from generation import load_hmm_params, decode_note


def get_pitch_from_hash(h: int) -> int:
    pitch, _, _ = decode_note(h)
    return pitch

def main():
    # Load trained HMM params
    T_prob, O_prob, pi_prob, states, obs = load_hmm_params()
    states = list(states)

    # Load testing melodies from test-data
    dp_test = DataProcessor(train=False)
    dp_test.init_note_sequences()

    violin_seqs = dp_test.violin_sequences

    with open("selected_test_idx.txt", "r") as f:
        test_idx = int(f.read().strip())

    print("Loaded selected test melody idx:", test_idx)
    raw_melody = violin_seqs[test_idx]

    observations = [DataProcessor.hash_note(n, is_piano=False) for n in raw_melody]
    print("Melody length:", len(observations))

    # Build HMM from data
    hmm = HMM(
        states=states,
        start_prob=pi_prob,
        trans_prob=T_prob,
        emit_prob=O_prob,
    )

    best_path, best_log_prob = hmm.viterbi(observations)

    print("Final best log-prob:", best_log_prob)

    melody_pitches = [get_pitch_from_hash(h) for h in observations]
    harmony_pitches = [get_pitch_from_hash(h) for h in best_path]

    plt.figure(figsize=(12, 4))
    plt.plot(melody_pitches, label="Melody - violin pitches)", color="darkblue")
    plt.plot(harmony_pitches, label="Harmony - piano pitches)", color="darkgreen")
    plt.title("Melody vs Generated Harmony")
    plt.xlabel("Note index")
    plt.ylabel("Pitch")
    plt.legend()
    plt.savefig("analysis_melody_vs_harmony.png", dpi=300)
    plt.show()

    # Heatmap transition matrix
    state_list = list(states)
    N = len(state_list)
    mat = np.zeros((N, N))

    for i, s1 in enumerate(state_list):
        for j, s2 in enumerate(state_list):
            mat[i, j] = T_prob.get(s1, {}).get(s2, 0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(mat, xticklabels=False, yticklabels=False)
    plt.title("HMM Transition Probability Matrix")
    plt.xlabel("Next state")
    plt.ylabel("Previous state")
    plt.savefig("analysis_transition_matrix.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
