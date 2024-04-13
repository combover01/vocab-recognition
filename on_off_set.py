import numpy as np
from scipy.io import wavfile
#Calculate the Short-Time Energy of the signal.
def short_time_energy(signal, window_size, hop_size):

    energy = np.array([
        np.sum(np.abs(signal[i:i+window_size])**2)
        for i in range(0, len(signal) - window_size, hop_size)
    ])
    return energy / np.max(energy)  # Normalize energy

# Detect onsets and offsets in the audio signal.
def detect_first_onset_offset(signal, sample_rate, window_size, hop_size, threshold):
    # Normalize signal
    signal = signal / np.max(np.abs(signal))
    
    # Calculate Short-Time Energy
    ste = short_time_energy(signal, window_size, hop_size)
    
    first_onset = None
    first_offset = None
    is_onset = False
    
    # Detect the first onset and its corresponding offset
    for i in range(1, len(ste)):
        if ste[i] > threshold and ste[i-1] <= threshold and not is_onset:
            first_onset = i * hop_size / sample_rate
            is_onset = True
        elif is_onset and ste[i] < threshold and ste[i-1] >= threshold:
            first_offset = i * hop_size / sample_rate
            break  # Stop after finding the first offset
    
    # If an onset was found but no offset, consider the end of the signal as the offset
    if is_onset and first_offset is None:
        first_offset = len(signal) / sample_rate
    
    return first_onset, first_offset

window_size = 1500
hop_size = 512
threshold = 0.02
input_file_path = 'wav_file.wav' #change the name to our wav file
sample_rate, audio = wavfile.read(input_file_path)

onsets, offsets = detect_first_onset_offset(audio, sample_rate, window_size, hop_size, threshold)
onset_sample = int(onsets * int(sample_rate))
offset_sample = int(offsets * int(sample_rate))

audio_segment = audio[onset_sample:offset_sample]
output_file_path = 'new_wav_file.wav'
wavfile.write(output_file_path, sample_rate, audio_segment)
