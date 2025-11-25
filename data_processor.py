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
    self.midi_objects = self.get_midi_objects() 

  def get_piano_notes(self, midi_file: Path) -> List[Any]:
    midi = pm.PrettyMIDI(midi_file)
    piano_notes = []
    for instrument in midi.instruments:
      name = pm.program_to_instrument_name(instrument.program)
      if 'piano' not in name.lower():
        continue
      for note in instrument.notes:
        piano_notes.append([note.start, note.end - note.start, note.pitch, note.velocity])
    return sorted(piano_notes, key=lambda x: x[0])
  
  def get_violin_notes(self, midi_file: Path) -> List[Any]:
    midi = pm.PrettyMIDI(midi_file)
    violin_notes = []
    for instrument in midi.instruments:
      name = pm.program_to_instrument_name(instrument.program)
      if name.lower() != 'violin':
        continue 
      for note in instrument.notes:
        violin_notes.append([note.start, note.end - note.start, note.pitch, note.velocity])
    return sorted(violin_notes, key=lambda x: x[0])

  @staticmethod
  def process_midi_file(midi_file: Path) -> Union[pm.PrettyMIDI, None]:
    try:
      midi = pm.PrettyMIDI(str(midi_file))
      return midi
    except:
      return None 
    
  def get_midi_objects(self, num_threads=16) -> List[pm.PrettyMIDI]:
    midi_objects = []
    with ThreadPoolExecutor(num_threads) as executor:
      futures = executor.map(self.process_midi_file, self.midi_files)
      midi_objects = [f for f in futures if f]
    return midi_objects

if __name__ == "__main__":
  dp = DataProcessor()
  print(len(dp.midi_files))
  print(len(dp.midi_objects))
  