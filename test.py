from datasets import load_dataset
import numpy as np
import time

dataset = load_dataset("roszcz/lakh-lmd-full", split="train[:10000]")

num_songs = 0;
total_notes = 0;
start_time = time.time();

for song in dataset:
  durations = song['notes']['duration']
  pitches = song['notes']['pitch']
  velocities = song['notes']['velocity']
  for duration, pitch, velocity in zip(durations, pitches, velocities):
    print(f"Duration: {duration}, Pitch: {pitch}, Velocity: {velocity}")
    total_notes += 1
  num_songs += 1

end_time = time.time();
total_time = end_time - start_time
print(f"Total songs: {num_songs}, Total notes: {total_notes}, Total time: {total_time - (total_time % 0.01)}s")