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
      V[t, :] = temp[back[t, :], np.arange(N)] + self.__log_O[:, obs_ix[t]]
    
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
