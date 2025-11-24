from datasets import load_dataset
import io
import pretty_midi as pm
from collections import defaultdict
from typing import Dict, List
import numpy as np

class DataProcessor:
  def __init__(self):
    self.ds = load_dataset('roszcz/maestro-sustain-v2', split='train')
    self.all_notes: List[Dict] = self.ds['notes']
    self.num_songs = len(self.all_notes)
  
  def separate_melody_harmony(self, songs: List[Dict]):
    pass
      

if __name__ == "__main__":
  data_processor = DataProcessor()
  print(len(data_processor.ds))
  print(len(data_processor.all_notes))
  print(data_processor.num_songs)