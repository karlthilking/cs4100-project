import pretty_midi as pm
import pandas as pd 
from pathlib import Path
from IPython.display import display
from typing import List, Union, Optional, Dict, Tuple, Any
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

class DataProcessor:
  def __init__(self, midi_path='piano-violin-data', num_songs=2500):
    self.midi_files = list(Path(midi_path).glob('*.mid'))[:num_songs]
    self.midi_objects: List[pm.PrettyMIDI]
    self.all_piano_notes: List[int]
    self.all_violin_notes: List[int]

  @staticmethod
  def hash_note(note: List[Any]) -> int:
    duration = min(255, int(note[2] * 100))
    return (note[0] & 0x7F) | (note[1] & 0x7F) << 7 | (duration & 0xFF) << 14 | (note[3] & 1) << 22 | (note[4] & 1) << 23

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
            piano_notes.append([n.pitch, n.velocity, n.get_duration(), True, True])
          else:
            piano_notes.append([n.pitch, n.velocity, n.get_duration(), False, True])
      elif name.lower() == 'violin':
        for i, n in enumerate(instrument.notes):
          if i == 0:
            violin_notes.append([n.pitch, n.velocity, n.get_duration(), True, False])
          else:
            violin_notes.append([n.pitch, n.velocity, n.get_duration(), False, False])
    return piano_notes, violin_notes

  @staticmethod
  def process_midi_file(midi_file: Path) -> Union[pm.PrettyMIDI, None]:
    try:
      midi = pm.PrettyMIDI(str(midi_file))
      return midi
    except:
      return None 
    
  def load_midi_objects(self, num_threads=16):
    midi_objects = []
    with ThreadPoolExecutor(num_threads) as executor:
      futures = executor.map(self.process_midi_file, self.midi_files)
      midi_objects = [f for f in futures if f]
    self.midi_objects = midi_objects

  def load_piano_sequence(self):
    if not self.midi_objects:
      print('Piano sequences cannot be loaded yet')
      return
    raw_piano_notes = []
    for midi in tqdm(self.midi_objects):
      raw_piano_notes.extend(self.get_piano_notes(midi))
    all_piano_notes = []
    for note in raw_piano_notes:
      all_piano_notes.append(self.hash_note(note))
    self.all_piano_notes = all_piano_notes
  
  def load_violin_sequence(self):
    if not self.midi_objects:
      print('Violin sequences cannot be loaded yet')
      return
    

  