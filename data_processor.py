import pretty_midi as pm
import pandas as pd
from pathlib import Path
from typing import List, Union, Optional, Dict, Tuple, Any, Counter
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
import seaborn as sns
import time
from tqdm import tqdm
from IPython.display import display

class DataProcessor:
  def __init__(self, midi_path='piano-violin-data', num_songs=7823):
    self.__midi_files = list(Path(midi_path).glob('*.mid'))[:num_songs]
    self.__midi_objects: List[pm.PrettyMIDI] 
    self.__piano_sequences: List[List[int]]
    self.__violin_sequences: List[List[int]]

  @property
  def midi_files(self) -> List[Path]:
    return self.__midi_files
  
  @property
  def midi_objects(self) -> List[pm.PrettyMIDI]:
    return self.__midi_objects
  
  @property
  def piano_sequences(self) -> List[List[int]]:
    return self.__piano_sequences

  @property
  def violin_sequences(self) -> List[List[int]]:
    return self.__violin_sequences

  @staticmethod
  def initial_hash(note: List[Any]) -> int:
    '''
    Hash note represented as a list of the form: [pitch, velocity, duration, float] into an integer
    Pitch: integer in (0, 127) -> 7 bits 
    Velocity: integer in (0, 127) -> 7 bits 
    Duration: float in (0, 296.72) converted to int in (0, 29672) -> 15 bits
    Start: float in (0, 2144.73) converted to int in (0, 214473) -> 18 bits
    '''
    duration = int(note[2] * 100); start = int(note[3] * 100)
    return (note[0] & 0x7F) | (note[1] & 0x7F) << 7 | (duration & 0x7FFF) << 15 | (start & 0x3FFFF) << 29
  
  @staticmethod
  def final_hash(note: int) -> int:
    pitch = note 

  @staticmethod 
  def get_notes(midi: pm.PrettyMIDI) -> Tuple[List[int], List[int]]:
    '''
    Input: individual midi object
    Returns: list of piano notes, list of violin notes

    For each note in the midi object, notes of the form [pitch, velocity, duration, is start note, is piano]
    are hashed into a single integer and added to the respective sequence (piano notes or violin notes).
    '''
    piano_notes, violin_notes = [], []
    for instrument in midi.instruments:
      name = pm.program_to_instrument_name(instrument.program)
      if 'piano' in name.lower():
        for i, n in enumerate(instrument.notes):
          if i == 0:
            piano_notes.append(DataProcessor.initial_hash([n.pitch, n.velocity, n.get_duration(), True, True]))
          else:
            piano_notes.append(DataProcessor.initial_hash([n.pitch, n.velocity, n.get_duration(), False, True]))
      elif name.lower() == 'violin':
        for i, n in enumerate(instrument.notes):
          if i == 0:
            violin_notes.append(DataProcessor.initial_hash([n.pitch, n.velocity, n.get_duration(), True, False]))
          else:
            violin_notes.append(DataProcessor.initial_hash([n.pitch, n.velocity, n.get_duration(), False, False]))
    return piano_notes, violin_notes

  @staticmethod
  def process_midi_file(midi_file: Path) -> pm.PrettyMIDI:
    # convert midi_file to pretty_midi midi object
    try:
      midi = pm.PrettyMIDI(str(midi_file))
      return midi
    except:
      return pm.PrettyMIDI(None)
  
  # intialize the class member self.midi_objects by processing all midi files into pretty_midi midi objects
  def init_midi_objects(self, num_threads=16):
    midi_objects: List[pm.PrettyMIDI] = [pm.PrettyMIDI(None)] * len(self.__midi_files)
    with ThreadPoolExecutor(num_threads) as executor:
      futures_indexed = {executor.submit(self.process_midi_file, midi_file): ix for ix, midi_file in enumerate(self.__midi_files)}
      for f in tqdm(as_completed(futures_indexed), total=len(futures_indexed)):
        ix = futures_indexed[f]
        midi_objects[ix] = f.result()
    self.__midi_objects = midi_objects

  def init_note_sequences(self, num_threads=16):
    '''
    Initialize the class members self.piano_sequences and self.violin_sequences.
    Process midi objects (songs) asynchronously and to obtain each individual piano, violin sequence pair.
    '''
    piano_sequences, violin_sequences = [[] for _ in range(len(self.__midi_files))], [[] for _ in range(len(self.__midi_files))]
    with ThreadPoolExecutor(num_threads) as executor:
      futures_indexed = {executor.submit(DataProcessor.get_notes, midi): ix for ix, midi in enumerate(self.__midi_objects)}
      for f in as_completed(futures_indexed):
        ix = futures_indexed[f]
        piano_seq, violin_seq = f.result()
        piano_sequences[ix] = piano_seq; violin_sequences[ix] = violin_seq
    self.__piano_sequences = piano_sequences
    self.__violin_sequences = violin_sequences

  @staticmethod
  def prob_distribution(x: ndarray[Any, Any]) -> List[float]:
    '''
    Convert list of counts (i.e number of transitions from s to all s_prime) into probability distribution
    '''
    return [n for n in x] / np.sum(x)
  
  def get_hmm(self):
    states = set()
    T = {}
    O = {}
    pi = np.zeros(self.__state_space)
    # loop through each piano, violin sequence pair
    for piano_seq, violin_seq in tqdm(zip(self.__piano_sequences, self.__violin_sequences), total=len(self.__piano_sequences)):
      # initialize piano, violin note indices
      i, j = 0, 0

      # current state (piano note)
      s = piano_seq[i] 

      # add initial state to state set and initial state matrix
      states.add(s); pi[s] += 1 

      # loop through piano notes and violin notes while neither have been exhausted
      while i < len(piano_seq) - 1 and j < len(violin_seq):
        # next state
        s_prime = piano_seq[i + 1]

        # count transition from s to s_prime
        T[s][s_prime] += 1; states.add(s_prime)

        # calculate start and end time of current state
        s_start = (s >> 28) / 100
        s_end = s_start + (s >> 14 & 0x3FFF) / 100

        # go to next violin note while current violin note proceeds current state
        while (violin_seq[j] >> 28) / 100 < s_start:
          j += 1
        
        # go to previous violin note while current violin note follows current state
        while (violin_seq[j] >> 28) / 100 > s_start:
          j -= 1
        
        # current observation (violin note)
        o = violin_seq[j]

        # while current violin note overlaps the current piano note
        while s_start < (o >> 28) / 100 < s_end:
          # count observation o from state s
          O[s][o] += 1
          j += 1
          o = violin_seq[j]
        
        # set current state to s_prime and increment i
        s = s_prime; i += 1

      # process remaining states (piano notes) if all violin notes have been exhausted (first loop terminates)
      while i < len(piano_seq) - 1:
        s = piano_seq[i]
        s_prime = piano_seq[i]; states.add(s_prime)
        T[s][s_prime] += 1
      
    # return transition matrix, observation matrix, initial state counts and set of states
    T = [DataProcessor.prob_distribution(t) for t in T]
    O = [DataProcessor.prob_distribution(t) for t in O]
    pi = DataProcessor.prob_distribution(pi)
    return T, O, pi, states

if __name__ == '__main__':
  start_time = time.time()
  dp = DataProcessor()
  dp.init_midi_objects()
  midis = dp.midi_objects

  starts = []
  for midi in midis:
    for instrument in midi.instruments:
      name = pm.program_to_instrument_name(instrument.program)
      starts.extend([n.start for n in instrument.notes])
  
  print(f'Max start: {np.max(starts)}')