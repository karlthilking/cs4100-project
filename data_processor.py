import pretty_midi as pm
import pandas as pd
from pandas import DataFrame
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
import pickle

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
    self.__midi_files = list(Path(midi_path).glob('*.mid'))[:num_songs]
    self.__midi_objects: List[pm.PrettyMIDI] 
    self.__piano_sequences: List[List[List[Any]]]
    self.__violin_sequences: List[List[List[Any]]]
    self.__state_space: int = 0x1FFFF
    self.__obs_space: int = 0x1FFFF

  @property
  def midi_files(self) -> List[Path]:
    return self.__midi_files
  
  @property
  def midi_objects(self) -> List[pm.PrettyMIDI]:
    return self.__midi_objects
  
  @property
  def piano_sequences(self) -> List[List[List[Any]]]:
    return self.__piano_sequences

  @property
  def violin_sequences(self) -> List[List[List[Any]]]:
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
    duration = DataProcessor.bin_duration_piano(note[2]) if is_piano else DataProcessor.bin_duration_violin(note[2])
    return (note[0] & 0x7F) | (note[1] & 0x7F) << 7 | (duration & 7) << 14

  @staticmethod
  def get_notes(midi: pm.PrettyMIDI) -> Tuple[List[List[Any]], List[List[Any]]]:
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
        piano_notes = [[n.pitch, n.velocity, n.get_duration(), n.start] for n in instrument.notes]
      elif name.lower() == 'violin':
        violin_notes = [[n.pitch, n.velocity, n.get_duration(), n.start] for n in instrument.notes]
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
      for f in tqdm(as_completed(futures_indexed), total=len(futures_indexed)):
        ix = futures_indexed[f]
        piano_seq, violin_seq = f.result()
        piano_sequences[ix] = piano_seq; violin_sequences[ix] = violin_seq
    self.__piano_sequences = piano_sequences
    self.__violin_sequences = violin_sequences

  @staticmethod
  def prob_distribution(x: ndarray[Any, Any]) -> ndarray:
    '''
    Convert list of counts (i.e number of transitions from s to all s_prime) into probability distribution
    '''
    return np.array([n for n in x]) / np.sum(x)
  
  @staticmethod
  def table_to_prob_distribution(table: Dict[int, ndarray], num_threads=16) -> Dict[int, ndarray]:
    res = {}
    with ThreadPoolExecutor(num_threads) as exec:
      futures = {exec.submit(DataProcessor.prob_distribution, entry): i for i, entry in table.items()}
      for f in tqdm(as_completed(futures), total=len(futures)):
        i = futures[f]
        res[i] = f.result()
    return res

  def get_hmm(self) -> Tuple[set[int], ndarray, Dict[int, ndarray], Dict[int, ndarray]]:
    states = set()
    pi = np.zeros(self.__state_space)
    T = {}
    O = {}
    # loop through each piano, violin sequence pair
    for piano_seq, violin_seq in tqdm(zip(self.__piano_sequences, self.__violin_sequences), total=len(self.__piano_sequences)):
      # initialize piano, violin note indices
      i, j = 0, 0

      # initialize state 
      s = DataProcessor.hash_note(piano_seq[0], True) 

      # add initial state to state set and initial state matrix
      states.add(s); pi[s] += 1 

      # loop through piano notes and violin notes while neither have been exhausted
      while i < len(piano_seq) - 1 and j < len(violin_seq):
        # next state
        s_prime = DataProcessor.hash_note(piano_seq[i + 1], True) 

        # add s_prime to states set
        states.add(s_prime)

        # initialize key s in transition matrix if it does not exist
        if s not in T:
          T[s] = np.zeros(self.__state_space)

        # count transitions form s to s_prime
        T[s][s_prime] += 1

        # calculate start and end time of current state
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

          # initialize O[s] if s not in observation matrix
          if s not in O:
            O[s] = np.zeros(self.__obs_space)
          
          # increment emission of obs o from state s
          O[s][o] += 1
          j += 1
        
        # set current state to s_prime and increment i
        s = s_prime; i += 1

      # process remaining states (piano notes) if all violin notes have been exhausted (first loop terminates)
      while i < len(piano_seq) - 1:
        s = DataProcessor.hash_note(piano_seq[i], True)
        s_prime = DataProcessor.hash_note(piano_seq[i + 1], True); states.add(s_prime)
        if s not in T:
          T[s] = np.zeros(self.__state_space)
        T[s][s_prime] += 1
        i += 1
      
    T = DataProcessor.table_to_prob_distribution(T) 
    O = DataProcessor.table_to_prob_distribution(O)
    pi = DataProcessor.prob_distribution(pi)
    return states, pi, T, O

if __name__ == '__main__':
  pass