Topic: Generating Harmonies for a Given Melody

Training Dataset: https://huggingface.co/datasets/roszcz/lakh-lmd-full/viewer/default/train?row=16

Academic Resources:
  - https://www.cis.upenn.edu/~cis2620/notes/Example-Viterbi-DNA.pdf
  - https://www2.isye.gatech.edu/~yxie77/ece587/viterbi_algorithm.pdf
  - https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_HiddenMarkovModel.html
  - https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_Viterbi.html
  - https://www.audiolabs-erlangen.de/resources/MIR/FMP/C1/C1S2_MIDI.html
  - https://www.audiolabs-erlangen.de/resources/MIR/FMP/B/B_PythonAudio.html 

MIDI File Resources:
  - Key Signature: https://midiprog.com/midi-key-signature/
  - Setup Messages: https://midiprog.com/midi-setup-messages/
  

Tools & Libraries:
  - numpy: HMM calculations
  - Music-specific Python Libraries:
    - MusicXML: textual data format for storing music files
    - music21: can parse MusicXML files, represents music data in a list (noted as being slow)
    - PrettyMIDI: processes MIDI files
  - datasets: load training data
  - tqdm
  - pandas
  - mnnlearn
  - mido: for working with MIDI files
  - note_seq

  - LMMS Studio (free version): creating our own MIDI files and melodies to use
    - https://www.image-line.com/fl-studio/download
  - Keyboard: record notes for MIDI file

Implementation:
  - implement Hidden Markov Model where hidden states represent chord progressions and observations are melody notes
    - transition probabilities represent how likely it is to move from one note to another
    - emission probabilities represent how likely a (harmony) note is observed given a (melody) note
  - use Viterbi Algorithm to find most likely note sequence

Implementation Decisions:
  - choose uniform format for representing musical information
  - 
