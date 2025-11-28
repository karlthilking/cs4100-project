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
  def __init__(self, midi_path='train-data', num_songs=6650):
    self.__num_songs = num_songs
    self.__midi_files = list(Path(midi_path).glob('*.mid'))[:num_songs]
    self.__midi_objects: List[pm.PrettyMIDI] 
    self.__piano_sequences: List[ndarray] = []
    self.__violin_sequences: List[ndarray] = [] 
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
    if velo <= 16: # ppp
      return 0
    elif velo <= 33: # pp
      return 1
    elif velo <= 49: # p
      return 2
    elif velo <= 64: # mp
      return 3
    elif velo <= 80: # mf
      return 4
    elif velo <= 96: # f
      return 5
    elif velo <= 112: #ff
      return 6
    else: # fff
      return 7

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
    '''
    Encode note as a tuple of pitch, velocity, duration, start
    into an integer storing pitch class, octave, velocity, and duration
    Pitch class: 0 to 11
    Octave bin: 0 to 3 
    Velocity: 0 to 7
    Duration: 0 to 3
    '''
    pitch_class = int(note[0]) % 12 
    octave = int(note[0]) // 12 - 1
    octave_bin = DataProcessor.bin_octave_piano(octave) if is_piano else DataProcessor.bin_octave_violin(octave)
    velo_bin = DataProcessor.bin_velocity(int(note[1]))
    dur_bin = DataProcessor.bin_duration_piano(note[2]) if is_piano else DataProcessor.bin_duration_violin(note[2])
    return (dur_bin & 0x3) | (octave_bin & 0x3) << 2 | (velo_bin & 0x7) << 4 | (pitch_class & 0xF) << 7

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
    path = f'HMM_params'
    os.mkdir(path)
    with open(f'{path}/T.pickle', 'wb') as handle:
      pickle.dump(self.__T, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{path}/O.pickle', 'wb') as handle:
      pickle.dump(self.__O, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{path}/pi.pickle', 'wb') as handle:
      pickle.dump(self.__pi, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'{path}/states.pickle', 'wb') as handle:
      pickle.dump(self.__states, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
  dp = DataProcessor()
  dp.init_note_sequences()
  dp.init_hmm()
  dp.save_hmm_params()