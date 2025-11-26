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
from IPython.display import display
import multiprocessing

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
    Pitch: integer in (0, 127) -> 7 bits (0x7F)
    Velocity: integer in (0, 127) -> 7 bits (0x7F)
    Duration: float in (0, 296.72) converted to int in (0, 29672) -> 15 bits (0x7FFF)
    Start: float in (0, 2144.73) converted to int in (0, 214473) -> 18 bits (0x3FFFF)
    '''
    duration = min(0x7FFF, int(note[2] * 100)); start = min(0x3FFFF, int(note[3] * 100))
    return (note[0] & 0x7F) | (note[1] & 0x7F) << 7 | (duration & 0x7FFF) << 15 | (start & 0x3FFFF) << 29
  
  @staticmethod
  def bin_duration_piano(dur: float) -> int:
    '''
    Bin piano note duration into category represented by integer in (0, 7)
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
    Bin violin note duration into category represented by integer in (0, 7)
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
  def final_hash(note: int, is_piano: bool) -> int:
    dur = (note >> 14 & 0x7FFF) / 100
    dur_bin = DataProcessor.bin_duration_piano(dur) if is_piano else DataProcessor.bin_duration_violin(dur)
    return (note & 0x3FFF) | (dur_bin & 7) << 14

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
    pi = [] 
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
  dp = DataProcessor()
  dp.init_midi_objects()

  piano_starts, violin_starts = [], []
  piano_durs, violin_durs = [], []
  for midi in dp.midi_objects:
    for instrument in midi.instruments:
        name = pm.program_to_instrument_name(instrument.program)
        if name.lower() == 'violin':
          violin_starts.extend([n.start for n in instrument.notes])
          violin_durs.extend([n.get_duration() for n in instrument.notes])
        else:
          piano_starts.extend([n.start for n in instrument.notes])
          piano_durs.extend([n.get_duration() for n in instrument.notes])
    
    piano_start_stats = [np.percentile(piano_starts, i) for i in range(14, 100, 14)]
    piano_durs_stats = [np.percentile(piano_durs, i) for i in range(14, 100, 14)]

    violin_start_stats = [np.percentile(violin_starts, i) for i in range(14, 100, 14)]
    violin_durs_stats = [np.percentile(violin_durs, i) for i in range(14, 100, 14)]

    piano_starts_stats = [f'{x:.2f}' for x in piano_start_stats]
    piano_durs_stats = [f'{x:.2f}' for x in piano_durs_stats]
    violin_start_stats = [f'{x:.2f}' for x in violin_start_stats]
    violin_durs_stats = [f'{x:.2f}' for x in violin_durs_stats]

    pd.set_option('display.max_columns', None)
    df = pd.DataFrame(
      [piano_start_stats, piano_durs_stats, violin_start_stats, violin_durs_stats],
      index=['piano start timestamps', 'piano duration timestamps', 'violin start timestamps', 'violin duration timestamps'],
      columns=['14th pct', '28th pct', '42nd pct', '56th pct', '70th pct', '84th pct', '98th pct']
    )
    display(df)