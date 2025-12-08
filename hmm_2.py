import numpy as np
from numpy import ndarray
from typing import List, Dict, Any 
from data_processor import DataProcessor as DP
from tqdm import tqdm

class HMM:
  def __init__(self, path='HMM_params'):
    self.__T = np.load(f'{path}/T.npy')
    self.__O = np.load(f'{path}/O.npy')
    self.__pi = np.load(f'{path}/pi.npy')
    self.__states = np.load(f'{path}/states.npy')
    self.__obs = np.load(f'{path}/obs.npy')
    self.__states_map = {s: i for i, s in enumerate(self.__states)} # maps state id to index in T, O, pi
    self.__obs_map = {o: i for i, o in enumerate(self.__obs)} # maps obs id to index in entry of O
    self.__log_T = np.log(self.__T)
    self.__log_O = np.log(self.__O)
    self.__log_pi = np.log(self.__pi)

  def interval_consonance_reward(self, melody_note, harmony_note):
    melody_pitch = (melody_note  >> 6) & 0xF
    harmony_pitch = (harmony_note >> 6) & 0xF
    interval = abs(melody_pitch - harmony_pitch) % 12

    perfect_intervals = {0, 5, 7, 12}
    imperfect_intervals = {3, 4, 8, 9}
    dissonant_intervals = {1, 2, 6, 10, 11}
          
    if interval in perfect_intervals:
      return 0.7
    elif interval in imperfect_intervals:
      return 0.5
    else:
      return 0.3

  def harmony_octave_penalty(self, prev_s, s):
    '''
    Implements small/medium punishment for the harmony jumping between octaves 
    ie. encouraging the harmony to be much smoother
    '''
    # get the pitches including the octave
    prev_pitch = ((prev_s >> 6) & 0xF) + 12 * ((prev_s >> 2) & 0x3)
    curr_pitch = ((s      >> 6) & 0xF) + 12 * ((s      >> 2) & 0x3)
    diff = abs(prev_pitch - curr_pitch)

    # no penalty for within an octave
    if diff == 0:
      return 0.0
    # small penalty if within 2 octaves
    elif diff == 1:
      return -0.5
    # larger penalty greater than 2 octave differences
    else:
      return -1.0
        

  def duration_reward(self, melody_note, harmony_note):
    '''
    Rewards the harmony for matching the melody duration according to their bins
    NOTE: the binned durations for melodies and harmonies differ
    '''
    melody_duration = (melody_note  >> 0) & 0x3
    harmony_duration = (harmony_note >> 0) & 0x3
        
    # rewards for harmony note matching the duration of the melody (in bin)
    if melody_duration == harmony_duration:
      return 1
    else: 
      return 0    

  def viterbi(self, observations):
    # initialize variables
    T = len(observations); N = len(self.__states)
    V = np.full((T, N), -np.inf, dtype=np.float64)
    back = np.zeros((T, N), dtype=np.int32)
    obs_ix = np.array([self.__obs_map[o] for o in observations])

    # t = 0
    V[0, :] = self.__log_pi + self.__log_O[:, obs_ix[0]]

    # t = 1 to T
    for t in tqdm(range(1, T), total=T):
      temp = V[t - 1, :].reshape(-1, 1) + self.__log_T
      back[t, :] = np.argmax(temp, axis=0)
      base_score = temp[back[t, :], np.arange(N)]

      # get the current melody note
      melody_note = observations[t]

      # duration reward
      dur_reward = np.array([
        self.duration_reward(melody_note, self.__states[s])
        for s in range(N)
      ])

      # interval reward multiplier
      interval_reward_multiplier = np.array([
        self.interval_consonance_reward(melody_note, self.__states[s])
        for s in range(N)
      ])

      # jump penalty 
      jump_penalty = np.array([
        self.harmony_octave_penalty(
          self.__states[back[t, s]],     
          self.__states[s]       
        )
        for s in range(N)
      ])

      # combine rewards into score
      score = (
        base_score
        + self.__log_O[:, obs_ix[t]]
        + dur_reward
        + jump_penalty
      )

      score *= interval_reward_multiplier

      V[t, :] = score
    
    # termination
    last_ix = np.argmax(V[-1, :])
    best_log_prob = V[-1, last_ix]

    path = np.zeros(T, dtype=np.int32)
    path[T - 1] = last_ix

    # backtracking to find best path
    for t in range(T - 2, -1, -1):
      path[t] = back[t + 1, path[t + 1]]
    
    best_path = self.__states[path]
    return best_path, best_log_prob


if __name__ == '__main__':
  hmm = HMM()
  dp = DP(train=False)
  dp.init_note_sequences()

  song_ix = np.random.randint(dp.num_songs)
  print(f'Song {song_ix} being used')

  violin_sequence = dp.violin_sequences[np.random.randint(dp.num_songs)]
  obs = [DP.hash_note(n, False) for n in violin_sequence]

  best_path, best_log_prob = hmm.viterbi(obs)

  print(best_log_prob)
  print(best_path)
