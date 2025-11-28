import pretty_midi as pm
from pathlib import Path
from typing import List, Dict, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from numpy import ndarray
from tqdm import tqdm
import pickle
import os

class DataProcessor:
  '''
  Handles complete workflow from raw midi files to constructing HMM
  Example workflow:
    - Initialize PrettyMIDI objects from midi_files in given path
    - Create invidivual piano, violin note sequences for each midi object.
    - Use piano, violin sequences to obtain all transitions, observations, initial states, states.
    - Convert transitions, observations, initial states to probability distributions.
  '''
  def __init__(self, train: bool = True):
    self.__train = train
    self.__num_songs = 6650 if self.__train else 1137 
    self.__midi_path = 'train-data' if self.__train else 'test-data'
    self.__midi_files = list(Path(self.__midi_path).glob('*.mid'))[:self.__num_songs]
    self.__piano_sequences: List[ndarray] = []
    self.__violin_sequences: List[ndarray] = [] 
    self.__T: ndarray 
    self.__O: ndarray 
    self.__pi: ndarray 
    self.__states: List[int] 
    self.__obs: List[int]

  @property
  def num_songs(self) -> int:
    return self.__num_songs

  @property
  def midi_files(self) -> List[Path]:
    return self.__midi_files
  
  @property
  def piano_sequences(self) -> List[ndarray]:
    return self.__piano_sequences

  @property
  def violin_sequences(self) -> List[ndarray]:
    return self.__violin_sequences
  
  @staticmethod 
  def bin_duration_piano(dur: float) -> int:
    if dur < 0.15:
      return 0
    elif dur < 0.29:
      return 1
    elif dur < 0.62:
      return 2
    else:
      return 3

  @staticmethod
  def bin_duration_violin(dur: float) -> int:
    if dur < 0.11:
      return 0
    elif dur < 0.21:
      return 1
    elif dur < 0.39:
      return 2
    else:
      return 3

  @staticmethod
  def bin_velocity(velo: int) -> int:
    if velo <= 33:
      return 0
    elif velo <= 64:
      return 1
    elif velo <= 96:
      return 2
    else:
      return 3

  @staticmethod
  def bin_octave_piano(octave: int) -> int:
    if octave <= 2:
      return 0
    elif octave <= 3:
      return 1
    elif octave <= 4:
      return 2
    else:
      return 3
  
  @staticmethod
  def bin_octave_violin(octave: int) -> int:
    if octave <= 3:
      return 0
    elif octave <= 4:
      return 1
    elif octave <= 5:
      return 2
    else:
      return 3

  @staticmethod
  def hash_note(note: List[Any], is_piano: bool) -> int:
    """
    Encode note as a tuple of pitch, velocity, duration, start
    into an integer storing pitch class, octave, velocity, and duration.

    Features:
    Pitch class: integer between 0 and 11
    Octave bin: integer between 0 and 3 
    Velocity: integer between 0 and 7
    Duration: integer between 0 and 3
    """
    pitch_class = int(note[0]) % 12 
    octave = int(note[0]) // 12 - 1
    octave_bin = DataProcessor.bin_octave_piano(octave) if is_piano else DataProcessor.bin_octave_violin(octave)
    velo_bin = DataProcessor.bin_velocity(int(note[1]))
    dur_bin = DataProcessor.bin_duration_piano(note[2]) if is_piano else DataProcessor.bin_duration_violin(note[2])
    return (dur_bin & 0x3) | (octave_bin & 0x3) << 2 | (velo_bin & 0x3) << 4 | (pitch_class & 0xF) << 6 

  def init_note_sequences(self):
    '''
    Initializes the piano sequences and violin sequences member variables as lists of lists containing the individual piano
    and violin notes from each midi file. 

    Converts midi files into pm.PrettyMIDI objects in order to extract information about the instrument and instrument notes 
    (pitch, velocity, duration, start timestamp). 
    
    Processes midi files in parallel to speed up the process of obtaining note sequences for each midi file.
    '''
    with ProcessPoolExecutor() as executor:
      futures = [executor.submit(pm.PrettyMIDI, midi) for midi in self.__midi_files]
      for f in tqdm(as_completed(futures), total=len(futures)):
        for instrument in f.result().instruments:
          if instrument.program == 0:
            self.__piano_sequences.append(np.array([[n.pitch, n.velocity, n.get_duration(), n.start] for n in instrument.notes]))
          elif instrument.program == 40:
            self.__violin_sequences.append(np.array([[n.pitch, n.velocity, n.get_duration(), n.start] for n in instrument.notes]))

  @staticmethod
  def convert_prob_distribution(entry: Dict[int, int]):
    '''
    Converts a dictionary entry where each value counts the occurence of either a transition or an observation
    to a valid probability distribution.
    '''
    denom = np.sum([c for c in entry.values()])
    for k in entry.keys():
      entry[k] /= denom 
    
  def init_hmm(self):
    '''
    Initializes the hidden markov model parameters of transition probabilities, emission probabilites, and initial state
    probabilities as well lists containing all individual, unique states and observations.
    '''
    states = set()
    pi = {}
    T = {}
    O = {}
    # loop through each piano, violin sequence pair
    for piano_seq, violin_seq in tqdm(zip(self.__piano_sequences, self.__violin_sequences), total=len(self.__piano_sequences)):
      # initialize piano, violin note indices
      i, j = 0, 0

      # initialize state 
      s = DataProcessor.hash_note(piano_seq[0], True) 

      # add initial state to states set and initial probability distribution
      states.add(s); pi[s] = pi.get(s, 0) + 1
            
      # loop through piano notes and violin notes while neither have been exhausted
      while i < len(piano_seq) - 1 and j < len(violin_seq):
        # next state
        s_prime = DataProcessor.hash_note(piano_seq[i + 1], True)

        # add s_prime to states set
        states.add(s_prime)

        # count transition from s to s_prime
        T[s] = T.get(s, {}); T[s][s_prime] = T[s].get(s_prime, 0) + 1

        # calculate current state start and end timestampe
        s_start = piano_seq[i][3]
        s_end = s_start + piano_seq[i][2]

        # if current violin note proceeds the current piano chord skip
        if violin_seq[j][3] < s_start:
          while j < len(violin_seq) and violin_seq[j][3] < s_start:
            j += 1
        # else if current violin note follows the current piano chord
        # try to find an earlier violin note that is observed from s
        elif violin_seq[j][3] > s_start:
          while j > 0 and violin_seq[j - 1][3] > s_start:
            j -= 1

        # while current violin note overlaps the current piano note
        while j < len(violin_seq) and s_start < violin_seq[j][3] < s_end:
          # get hashed obs
          o = DataProcessor.hash_note(violin_seq[j], False)

          # initialize key s in obs matrix if it does not exist, then increment or initialize observation o from s in obs matrix
          O[s] = O.get(s, {}); O[s][o] = O[s].get(o, 0) + 1

          # go to next violin note
          j += 1
        
        # set current state to s_prime and increment i
        s = s_prime; i += 1

      # process remaining states after observations have been exhausted to account for all transitions 
      while i < len(piano_seq) - 1:
        s_prime = DataProcessor.hash_note(piano_seq[i + 1], True); states.add(s_prime)
        T[s] = T.get(s, {}); T[s][s_prime] = T[s].get(s_prime, 0) + 1
        i += 1
        s = s_prime

    # convert occurences to probability distributions
    for k in T.keys():
      DataProcessor.convert_prob_distribution(T[k])
    for k in O.keys():
      DataProcessor.convert_prob_distribution(O[k])
    DataProcessor.convert_prob_distribution(pi)
    
    # converting all dictionaries to numpy arrays
    states = sorted(states)
    states_map = {s: i for i, s in enumerate(states)}

    num_states = len(states)
    T_arr = np.full((num_states, num_states), 1e-12)
    for s in T.keys():
      i = states_map[s]
      for s_prime, prob in T[s].items():
        j = states_map[s_prime]
        T_arr[i, j] = prob
    
    all_obs = set()
    for s in O.keys():
      for o in O[s].keys():
        all_obs.add(o)

    obs = sorted(all_obs)
    obs_map = {o: i for i, o in enumerate(obs)}

    num_obs = len(obs)
    O_arr = np.full((num_states, num_obs), 1e-12)
    for s in O.keys():
      i = states_map[s]
      for o, prob in O[s].items():
        j = obs_map[o]
        O_arr[i, j] = prob
    
    pi_arr = np.full(num_states, 1e-12)
    for s, prob in pi.items():
      i = states_map[s]
      pi_arr[i] = prob

    self.__T = T_arr; self.__O = O_arr; self.__pi = pi_arr; self.__states = states; self.__obs = obs 
  
  def save_hmm_params(self):
    np.save('HMM_params/T.npy', self.__T)
    np.save('HMM_params/O.npy', self.__O)
    np.save('HMM_params/pi.npy', self.__pi)
    np.save('HMM_params/states.npy', self.__states)
    np.save('HMM_params/obs.npy', self.__obs)

if __name__ == '__main__':
  dp = DataProcessor()
  dp.init_note_sequences()
  dp.init_hmm()
  dp.save_hmm_params()