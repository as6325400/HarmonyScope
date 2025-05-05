import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_waveform(y, sr, output_path):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_spectrogram(y, sr, output_path):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.title('Spectrogram (dB)')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_chroma(y, sr, output_path):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', sr=sr)
    plt.title('Chroma Feature')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_file(wav_path, output_dir):
    y, sr = librosa.load(wav_path)
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    plot_waveform(y, sr, os.path.join(output_dir, f"{base_name}_waveform.png"))
    plot_spectrogram(y, sr, os.path.join(output_dir, f"{base_name}_spectrogram.png"))
    plot_chroma(y, sr, os.path.join(output_dir, f"{base_name}_chroma.png"))
    print(f"Plots saved for {base_name}")

if __name__ == "__main__":
    input_wav = "src/HarmonyScope/data/C.wav"
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    process_file(input_wav, output_dir)
