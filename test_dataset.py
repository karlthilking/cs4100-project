from datasets import load_dataset
import pretty_midi as pm
from typing import Dict, Optional, Union
import numpy as np
from tqdm import tqdm
import note_seq
import json
import time

ds = load_dataset("roszcz/lakh-lmd-full")
train_ds = ds['train']

print(f"{len(train_ds)} songs in lakh-lmd-full")

def parse_filename(filename: str) -> Union[str, None]:
  try:
    data = json.loads(filename)
    return data['filenames'][0]
  except KeyError:
    return

for i in tqdm(range(250)):
  song = train_ds[i]
  filename = parse_filename(song['source'])

  pitches = song['notes']['pitch']
  durations = song['notes']['duration']
  velocities = song['notes']['velocity']

  rand = np.random.randint(0, min(len(pitches), len(durations), len(velocities)))
  print(f"(Song {i + 1})\nFilename: {filename}")
  print(f"Example note: duration = {durations[rand]}, pitch = {pitches[rand]}, velocity = {velocities[rand]}\n")
  