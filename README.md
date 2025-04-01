# Audio Processing and Direction of Arrival (DOA) Estimation

This project captures and processes audio signals using a four-microphone array to estimate the Direction of Arrival (DOA) of sound sources using the MUSIC algorithm.

## Features
- Real-time audio capture and processing.
- Normalization and noise filtering.
- Estimation of sound source direction using MUSIC algorithm.
- Calculation of Signal-to-Noise Ratio (SNR).
- Visualization of results including spectrograms and correlation matrices.
- Waterfall PSD analysis for frequency domain representation.

## Requirements
- Python 3.x
- `sounddevice`
- `numpy`
- `scipy`
- `matplotlib`
- `pathlib`
- `dataclasses`

## Usage
1. Install dependencies:  
   `
   pip install sounddevice numpy scipy matplotlib
   `
2. Run the main script to start capturing and analyzing audio:
   `
   python main.py
   `
3. View real-time plots and analysis results.

## Configuration
- The script allows customization of sampling rate, microphone spacing, and buffer duration.
- Parameters can be modified in the `Config` class.

## Visualization
- The script generates real-time radar plots for sound localization.
- Spectrograms and MUSIC spectrum plots help analyze sound characteristics.
