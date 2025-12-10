# AI Harmony Generation (CS4100 F2025)

## Overview
Our project's purpose was to develop an AI System capable of generating harmonies to complement a provided melody. In order to accomplish this purpose, we collected ~7000 MIDI files containing piano and violin mono-melodies in order to develop a Hidden Markov Model that would capture note patterns, piano-violin note relationships, and more. We then applied the Viterbi algorithm for decoding a piano note sequence to accompany a violin sequence provided as input. For generation, we were able to produce a music sheet and midi file using the piano output sequence. Generating these outputs was made convenient due to the Python interfaces provided by music21 and pretty_midi.

## Project Structure
**data_processor.py**:
  - Responsible for converting original midi files obtained from the Tegridy-MIDI-Dataset to note sequence lists with a tuple containing each note's information.
  - Produced the parameters of our Hidden Markov Model (transition probabilities, emission probabilities, initial state probabilities, etc.) by iterating through piano and violin sequences corresponding to each individual song contained in our training data (5650 songs).

**hmm.py**
  - Implements the Viterbi algortihm that takes a violin melody as input and identifies the most likely accompanying piano sequence according to the parameters of our Hidden Markov Model.

**generation_analysis.py**
  - Decrypts the path of piano notes provided by Viterbi decoding and produces the resulting song (as a midi file) and complementary music sheet.
