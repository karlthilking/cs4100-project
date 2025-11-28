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
  def __init__(self, midi_path='piano-violin-data', num_songs=7823):
    self.__num_songs = num_songs
    self.__midi_files = list(Path(midi_path).glob('*.mid'))[:num_songs]
    self.__midi_objects: List[pm.PrettyMIDI] 
    self.__piano_sequences: List[ndarray] = []
    self.__violin_sequences: List[ndarray] = [] 
    self.__state_space: int = 0x1FFFF
    self.__obs_space: int = 0x1FFFF
    self.__T: Dict[int, Dict[int, float]]
    self.__O: Dict[int, Dict[int, float]]
    self.__pi: Dict[int, float]
    self.__states: set[int]

  @property
  def midi_files(self) -> List[Path]:
    return self.__midi_files
  
  @property
  def midi_objects(self) -> List[pm.PrettyMIDI]:
    return self.__midi_objects
  
  @property
  def piano_sequences(self) -> List[ndarray]:
    return self.__piano_sequences

  @property
  def violin_sequences(self) -> List[ndarray]:
    return self.__violin_sequences
  
  @property
  def state_space(self) -> int:
    return self.__state_space
  
  @property
  def obs_space(self) -> int:
    return self.__obs_space

  @staticmethod
  def bin_duration_piano(dur: float) -> int:
    '''
    Input:
    Duration of note (seconds) as a float
    Returns:
    Integer in (0, 7) representing the bin that the duration falls in
    '''
    if dur <= 0.10:
      return 0
    elif dur <= 0.15:
      return 1
    elif dur <= 0.21:
      return 2
    elif dur <= 0.29:
      return 3
    elif dur <= 0.42:
      return 4
    elif dur <= 0.62:
      return 5
    elif dur <= 1.82:
      return 6
    else:
      return 7

  @staticmethod
  def bin_duration_violin(dur: float) -> int:
    '''
    Input:
    Duration of note (seconds) as a float
    Returns:
    Integer in (0, 7) representing the bin that the duration falls in
    '''
    if dur <= 0.08:
      return 0
    elif dur <= 0.11:
      return 1 
    elif dur <= 0.16:
      return 2
    elif dur <= 0.21:
      return 3 
    elif dur <= 0.26:
      return 4 
    elif dur <= 0.39:
      return 5
    elif dur <= 1.19:
      return 6 
    else: 
      return 7
 
  @staticmethod
  def hash_note(note: List[Any], is_piano: bool) -> int:
    '''
    Input:
    Note as a list object: [pitch (int), velocity (int), duration (float), start (float)]
    Returns:
    Single integer (17 bit) representing pitch, velocity, and duration 
    '''
    pitch = int(note[0]) & 0x7F; velocity = int(note[1]) & 0x7F
    duration = DataProcessor.bin_duration_piano(note[2]) if is_piano else DataProcessor.bin_duration_violin(note[2])
    return pitch | velocity << 7 | (duration & 7) << 14

  def init_note_sequences(self):
    '''
    Initialize the class members self.piano_sequences and self.violin_sequences.
    Process midi objects (songs) asynchronously and to obtain each individual piano, violin sequence pair.
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
    denom = np.sum([c for c in entry.values()])
    for k in entry.keys():
      entry[k] /= denom 
    
  def init_hmm(self):
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

          # initialize key s in obs matrix if dne, increment or initialize observation o from s in obs matrix
          O[s] = O.get(s, {}); O[s][o] = O[s].get(o, 0) + 1

          # go to next violin note
          j += 1
        
        # set current state to s_prime and increment i
        s = s_prime; i += 1

      # process remaining states (piano notes) if all violin notes have been exhausted (first loop terminates)
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
    self.__T = T; self.__O = O; self.__pi = pi; self.__states = states 
  
  def get_hmm_params(self) -> Tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, float]], Dict[int, float], set[int]]:
    return self.__T, self.__O, self.__pi, self.__states
  
  def save_hmm_params(self):
    path = f'config_{self.__num_songs}'
    os.mkdir(path)
    with open(f'{path}/T_{self.__num_songs}.pickle', 'wb') as handle:
      pickle.dump(self.__T, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{path}/O_{self.__num_songs}.pickle', 'wb') as handle:
      pickle.dump(self.__O, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{path}/pi_{self.__num_songs}.pickle', 'wb') as handle:
      pickle.dump(self.__pi, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{path}/states_{self.__num_songs}.pickle', 'wb') as handle:
      pickle.dump(self.__states, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
  dp = DataProcessor()
  dp.init_note_sequences()
  dp.init_hmm()
  dp.save_hmm_params()