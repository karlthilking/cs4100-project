Topic: Generating harmonies for a given melody

Training Dataset: https://huggingface.co/datasets/roszcz/lakh-lmd-full/viewer/default/train?row=16

Academic Resources:
  - https://www.cis.upenn.edu/~cis2620/notes/Example-Viterbi-DNA.pdf
  - https://www2.isye.gatech.edu/~yxie77/ece587/viterbi_algorithm.pdf
  - https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_HiddenMarkovModel.html
  - https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_Viterbi.html
  - https://www.audiolabs-erlangen.de/resources/MIR/FMP/C1/C1S2_MIDI.html
  - https://www.audiolabs-erlangen.de/resources/MIR/FMP/B/B_PythonAudio.html 
  

Tools & Libraries:
  - numpy: hmm calculations
  - Music-specific Python Libraries:
    - MusicXML: textual data format for storing music files
    - music21: can parse MusicXML files, represents music data in a list (noted as being slow)
    - PrettyMIDI: processes MIDI files

Implementation:
  - implement hmm where hidden states represent chord progressions and observations are melody notes
  - use viterbi algorithm to find most likely chord sequence

Implementation Decisions:
  - Choosing uniform format for representing musical information
