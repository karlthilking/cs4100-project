import math
import numpy as np
from typing import List, Dict, Tuple, Any
import pickle
import os
from data_processor import *

LOG_ZERO = float("-inf")


class HMM:
    """
    HMM:
    - states: list of state labels (chords)
    - start_prob: P(state at t=0)
    - trans_prob: P(next_state | current_state)
    - emit_prob: P(observation | state)
    """

    def __init__(
            self,
            states: List[Any],
            start_prob: Dict[Any, float],
            trans_prob: Dict[Any, Dict[Any, float]],
            emit_prob: Dict[Any, Dict[Any, float]],
    ):
        self.states = states
        self.start_prob = start_prob
        self.trans_prob = trans_prob
        self.emit_prob = emit_prob

    def _safe_log(self, p: float) -> float:
        """Return log(p)"""
        if p <= 0.0:
            return LOG_ZERO
        return math.log(p)

    def interval_consonance_reward(self, melody_note, harmony_note):
      melody_pitch = melody_note & 0x7F 
      harmony_pitch = harmony_note & 0x7F
      interval = abs(melody_pitch - harmony_note) % 12
      perfect_intervals = {0, 5, 7, 12}
      imperfect_intervals = {3, 4, 8, 9}
      dissonant_intervals = {1, 2, 6, 10, 11}
      if interval in perfect_intervals:
        return 0.7
      elif interval in imperfect_intervals:
        return 0.5
      else:
        return 0.3

    def viterbi(self, observations):
        """
        observations = the melody notes
        states = the harmony/chords we want to guess

        Returns:
            best_path -> most likely note sequence
            best_log_prob -> log-probability of that sequence
        """

        T = len(observations)  # number of time steps (melody notes)
        states = self.states  # all possible hidden states (notes)

        # V[t][state] = best log-probability of any path that ends in state at time t
        V = []

        # back[t][state] = which previous state gives that best path
        back = []

        # 1. Initialization (t = 0)
        # For the first note, the best score is: log P(start in state s) + log P(observation | state s)
        first_obs = observations[0]
        V0 = {}
        back0 = {}

        for s in states:
            # how well this chord explains the first melody note
            p_start = self.start_prob.get(s, 1e-12)
            p_emit = self.emit_prob.get(s, {}).get(first_obs, 1e-12)
            
            # convert to log probabilities
            V0[s] = self._safe_log(p_start) + self._safe_log(p_emit)

            # no previous state at the first time step
            back0[s] = None

        V.append(V0)
        back.append(back0)

        # 2. Recursion (t = 1 to T-1)
        # For every future note, we try all possible previous chords and pick the best-scoring path.
        # best_score = previous best score + log P(transition prev_s -> s) + log P(emission of current note | s)
        # Get the best score and which prev state gave that score
        for t in tqdm(range(1, T)):
            obs = observations[t]  # current melody note
            V_t = {}
            back_t = {}

            for s in states:  # s = current chord
                best_score = LOG_ZERO
                best_prev_state = None

                # try every possible previous chord
                for prev_s in states:
                    # transition probability: prev chord -> current chord
                    p_trans = self.trans_prob.get(prev_s, {}).get(s, 0.0)

                    # emission probability: how well current chord explains current note
                    p_emit = self.emit_prob.get(s, {}).get(obs, 1e-12)

                    # total score for ending in state s via prev_s
                    score = (
                            V[t - 1][prev_s]
                            + self._safe_log(p_trans)
                            + self._safe_log(p_emit)
                    )

                    # keep the best-scoring previous state
                    if score > best_score:
                        best_score = score
                        best_prev_state = prev_s

                # store best score and best previous note 
                V_t[s] = best_score
                back_t[s] = best_prev_state

            V.append(V_t)
            back.append(back_t)

        # 3. Termination: Pick the best-scoring final note at time T-1.
        last_time = T - 1
        best_last_state = None
        best_log_prob = LOG_ZERO

        for s in states:
            score = V[last_time][s]
            if score > best_log_prob:
                best_log_prob = score
                best_last_state = s

        # 4. Backtracking to get full best path
        # Starting from the best final chord, walk backwards using the stored backpointers to recover
        # the entire musical sequence.
        best_path = [None] * T
        best_path[last_time] = best_last_state

        # follow the arrows backwards in time
        for t in range(last_time, 0, -1):
            best_path[t - 1] = back[t][best_path[t]]

        return best_path, best_log_prob

if __name__ == '__main__':
    pass