from datasets import load_dataset
import pretty_midi as pm
from typing import Dict, Optional, Union
import note_seq
import json
import time

ds = load_dataset("roszcz/lakh-lmd-full")
train_ds = ds['train']

print(f"{len(train_ds)} total songs in lakh datatset")

def parse_midi(file_path: str) -> Dict:
  dict = {
    'path' : file_path,
  }

  try:
    stream = pm.PrettyMIDI(file_path)
  except (EOFError, KeyError, ValueError):
    return
  
  try:
    dict['instrument_names'] = [i.name.strip() for i in stream.instruments]
  except ValueError:
    return
  
  dict['notes'] = note_seq.midi_file_to_note_sequence(stream)
  return dict

def parse_filename(filename: str) -> Union[str, None]:
  try:
    data = json.loads(filename)
    return data['filenames'][0]
  except KeyError:
    return

filepaths = []
for i in range(50):
  info = train_ds[i]['source']
  filename = parse_filename(info)
  filepaths.append(filename)

for filename in filepaths:
  print(f"Filepath: {filename}")