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
  def __init__(self, midi_path='piano-violin-data'):
    self.midi_files = list(Path(midi_path).glob('*.mid'))[:5000]
    self.midi_objects = None

  @staticmethod
  def hash_note(note: List[Any]) -> int:
    '''
    Input: list of pitch (int), velocity (int), duration (float), is start (0 or 1), is piano (0 or 1)

    Pitch: integer in (0, 127) -> 7 bits 
    Velocity: integer in (0, 127) -> 7 bits 
    Duration: float in (0, 2.xx) -> multiply by 100 for integer that include 2 decimal points -> integer in (0, ~256) -> 8 bits
    Start: 1 if start note else 0 -> 1 bit 
    Instrument: 1 if piano (harmony) else if violin 0 (melody) -> 1 bit

    Returns: 30 bit integer representing pitch, velocity, duration, start, and instrument
    '''
    duration = min(255, int(note[2] * 100))
    return (note[0] & 0x7F) | (note[1] & 0x7F) << 7 | (duration & 0xFF) << 14 | (note[3] & 1) << 22 | (note[4] & 1) << 23

  @staticmethod
  def get_piano_notes(midi: pm.PrettyMIDI) -> List[Any]:
    '''
    Return list of notes of the form [p, v, d, is_s, is_p] where:
    p: pitch -> integer(0, 127)
    v: velocity -> integer(0, 127)
    d: duration -> float
    is_s: is the note a start note -> 1 if so 0 otherwise
    is_p: is the instrument a piano -> 1 if so 0 otherwise 
    '''
    piano_notes = []
    for instrument in midi.instruments:
      name = pm.program_to_instrument_name(instrument.program)
      if 'piano' not in name:
        continue 
      for i, n in enumerate(instrument.notes):
        if i == 0:
          piano_notes.append([n.pitch, n.velocity, n.get_duration(), 1, 1])
        else:
          piano_notes.append([n.pitch, n.velocity, n.get_duration(), 0, 1])
    return piano_notes
  
  @staticmethod
  def get_violin_notes(midi: pm.PrettyMIDI) -> List[Any]:
    violin_notes = []
    for instrument in midi.instruments: 
      name = pm.program_to_instrument_name(instrument.program)
      if name.lower() != 'violin':
        continue 
      for i, n in enumerate(instrument.notes):
        if i == 0:
          violin_notes.append([n.pitch, n.velocity, n.get_duration(), 1, 0])
        else:
          violin_notes.append([n.pitch, n.velocity, n.get_duration(), 0, 0])
    return violin_notes

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


  