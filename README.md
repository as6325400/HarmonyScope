[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/as6325400/HarmonyScope)
# HarmonyScope

> Real-time chord detection and analysis for musicians 🎶

## 🚀 Installation


### System Dependencies

To ensure the package works properly, especially for microphone-related features, you need to have the `PortAudio` library installed on your system.

| Platform         | Command                                                         |
|------------------|-----------------------------------------------------------------|
| macOS            | `brew install portaudio`                                        |
| Ubuntu/Debian    | `sudo apt install libportaudio2`                                |
| Windows          | Usually works out of the box. If issues occur, refer to [sounddevice documentation](https://python-sounddevice.readthedocs.io/). |

Make sure to install the system dependency **before** running `pip install harmonyscope`.

### Install via pip

```bash
pip install harmonyscope
```


## Demo: Live Chord Detection

After installation, simply run:
```bash
mic_analyze
```
It will open an interactive selector to choose your microphone device, and start live chord detection in your terminal.

Demo preview:

![Demo GIF](plots/realtime_demo.gif)

## Example 1: C Major Chord Analysis

### Waveform
![Waveform](plots/C_waveform.png)

### Spectrogram
![Spectrogram](plots/C_spectrogram.png)

### Chroma
![Chroma](plots/C_chroma.png)


## Example 2: Piano C Major Chord Analysis

### Waveform
![Waveform](plots/piano_c-major_waveform.png)

### Spectrogram
![Spectrogram](plots/piano_c-major_spectrogram.png)

### Chroma
![Chroma](plots/piano_c-major_chroma.png)
