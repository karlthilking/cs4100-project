import numpy as np
from datasets import load_dataset
from typing import Dict, defaultdict, Counter
from tqdm import tqdm
import time

ds = load_dataset("roszcz/lakh-lmd-full")
train_ds = ds['train']

def hash_function(duration, pitch, velocity):
  duration = round(duration * 100)
  hash_val = (pitch & 0x7F) | ((velocity & 0x7F) << 7) | ((duration & 0xFFFF) << 14)
  return hash_val

def create_sequences(num_songs=1000):
  hash_sequence = []
  tuple_sequence = []

  for i in tqdm(len(train_ds)):
    notes = train_ds[i]['notes']
    durations = notes['duration']
    pitches = notes['pitch']
    velocities = notes['velocity']
    for d, p, v in zip(durations, pitches, velocities):
      hash_sequence.append(hash_function(d, p, v))
      tuple_sequence.append((d, p, v))
  
  return hash_sequence, tuple_sequence

def transition_matrix(sequence):
  T = defaultdict(Counter)
  for i in tqdm(range(len(sequence) - 1)):
    s = sequence[i]
    s_prime = sequence[i + 1]
    T[s][s_prime] += 1
  return T

if __name__ == "__main__":
  hash_seq, tuple_seq = create_sequences(5000)

  start_time = time.time()
  hash_matrix = transition_matrix(hash_seq)
  end_time = time.time()
  total_time = round(end_time - start_time)
  print(f"Hash Performance: {total_time} seconds")

  start_time = time.time()
  tuple_matrix = transition_matrix(tuple_seq)
  end_time = time.time()
  total_time = round(end_time - start_time)
  print(f"Tuple Performance: {total_time} seconds")