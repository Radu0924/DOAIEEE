import sounddevice as sd
from scipy.io.wavfile import write, read
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import datetime
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
from scipy.signal import spectrogram

class Config:
    FS = 44100  # Sampling frequency
    R = 0.1  # Microphone array radius (m)
    MIC_ANGLES = np.array([-45, -25, 25, 45])  # Microphone angles
    WAVELENGTH = 343 / FS  # Wavelength (sound speed / frequency)
    BUFFER_DURATION = 5  # Buffer duration for live capture
    NPERSEG = int(2 ** np.ceil(np.log2(FS // 200)))  # Window size for spectrogram
    NOVERLAP = NPERSEG // 2  # Spectrogram overlap (50%)
    PLOT_FREQ_LIMIT = 5000  # Upper frequency limit for display


@dataclass
class AudioAnalysisResult:
    """Class for storing audio analysis results"""
    distance: float
    angle: float
    snr: float
    spectrum: np.ndarray
    rms: np.ndarray


class AudioProcessor:
    """Class for audio signal processing"""

    def __init__(self):
        self.config = Config()

    def normalize_audio(self, data: np.ndarray, target_level: float = 0.9) -> np.ndarray:
        """Normalizes audio signal to specified level"""
        data = data.astype(np.float32)
        max_vals = np.max(np.abs(data), axis=1, keepdims=True)
        max_vals[max_vals == 0] = 1.0  # Avoid division by zero
        scaling_factors = target_level / max_vals
        return data * scaling_factors

    def estimate_covariance_matrix(self, data: np.ndarray) -> np.ndarray:
        """Estimates covariance matrix"""
        mean_centered = data - np.mean(data, axis=1, keepdims=True)
        return (mean_centered @ mean_centered.T) / (data.shape[1] - 1)

    def calculate_steering_vector(self, angle: float) -> np.ndarray:
        """Calculates steering vector for MUSIC"""
        return np.exp(-1j * 2 * np.pi * self.config.R * np.cos(np.deg2rad(angle - self.config.MIC_ANGLES)) / self.config.WAVELENGTH)

    def music_algorithm(self, cov_matrix: np.ndarray, num_sources: int = 1) -> Tuple[float, np.ndarray]:
        """MUSIC algorithm implementation for localization"""
        eigvals, eigvecs = eigh(cov_matrix)
        idx = np.argsort(eigvals)[::-1]
        noise_subspace = eigvecs[:, idx[num_sources:]]

        angles = np.linspace(-45, 45, 180)
        spectrum = []

        for angle in angles:
            steering_vector = self.calculate_steering_vector(angle)
            projection = np.abs(1 / (steering_vector.conj() @ noise_subspace @ noise_subspace.conj().T @ steering_vector))
            spectrum.append(projection)

        spectrum = np.array(spectrum)
        peak_angle = angles[np.argmax(spectrum)]
        return peak_angle, spectrum

    def calculate_waterfall_psd(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates waterfall PSD for sum of all 4 microphones"""
        summed_signal = np.sum(data, axis=0)
        freqs, times, Sxx = spectrogram(summed_signal, fs=self.config.FS, window='hann',
                                      nperseg=self.config.NPERSEG, noverlap=self.config.NOVERLAP)
        psd = 10 * np.log10(Sxx + 1e-12)  # Add small offset to avoid log(0)
        return freqs, times, psd

    def process_frame(self, frame: np.ndarray) -> Optional[AudioAnalysisResult]:
        """Processes an audio frame"""
        frame = frame.T
        frame = self.normalize_audio(frame)
        rms = np.sqrt(np.mean(frame ** 2, axis=1))
        peak_to_peak = np.max(frame, axis=1) - np.min(frame, axis=1)

        noise_threshold = np.percentile(rms, 95)
        if np.max(rms) < noise_threshold * 0.5:
            return None

        cov_matrix = self.estimate_covariance_matrix(frame)
        angle, spectrum = self.music_algorithm(cov_matrix)

        signal_range = np.max(peak_to_peak)
        noise_range = np.percentile(peak_to_peak, 10)
        snr = 20 * np.log10(signal_range / noise_range) if noise_range != 0 else 0
        distance = np.abs(np.tan(np.deg2rad(angle))) * 1.0

        return AudioAnalysisResult(distance, angle, snr, spectrum, rms)

    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyzes an audio file and returns results"""
        try:
            fs, data = read(file_path)
            data = data.astype(np.float32)
            channels_data = self.normalize_audio(data.T)

            # Calculate metrics
            rms = np.sqrt(np.mean(channels_data ** 2, axis=1))
            cov_matrix = self.estimate_covariance_matrix(channels_data)
            std_dev = np.sqrt(np.diag(cov_matrix))
            std_dev[std_dev == 0] = 1e-12
            corr_matrix = cov_matrix / np.outer(std_dev, std_dev)

            # Calculate PSD waterfall
            freqs, times, psd_waterfall = self.calculate_waterfall_psd(channels_data)

            # Calculate parameters
            angle, music_spectrum = self.music_algorithm(cov_matrix)
            peak_to_peak = np.ptp(channels_data, axis=1)
            snr = 20 * np.log10(np.max(peak_to_peak) / np.percentile(peak_to_peak, 10))
            distance = np.abs(np.tan(np.deg2rad(angle))) * 1.0

            # Statistical calculations
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)

            # Calculate dominant frequencies
            peak_freqs = [freqs[np.argmax(10 ** (psd_db / 20))] for psd_db in psd_waterfall.T]

            return {
                'file_name': os.path.basename(file_path),
                'rms': rms,
                'corr_matrix': corr_matrix,
                'angle': angle,
                'distance': distance,
                'music_spectrum': music_spectrum,
                'snr': snr,
                'freq': freqs,
                'psd_waterfall': psd_waterfall,
                'times': times,
                'peak_freqs': peak_freqs,
                'rms_mean': rms_mean,
                'rms_std': rms_std,
                'fs': fs,
                'n_samples': data.shape[0]
            }
        except Exception as e:
            print(f"Processing error: {e}")
            return {}


class Visualizer:
    """Class for visualizing results"""

    @staticmethod
    def format_value(value):
        """Formats values for display"""
        return np.format_float_positional(value, precision=4, unique=False, fractional=False, trim='k')

    def plot_radar(self, result: AudioAnalysisResult, capture_time: float):
        """Generates radar visualization for live capture"""
        plt.clf()

        # Radar subplot
        ax1 = plt.subplot(1, 2, 1, projection='polar')
        ax1.set_theta_zero_location('N')
        ax1.set_theta_direction(-1)
        ax1.set_thetamin(-45)
        ax1.set_thetamax(45)
        ax1.set_rlim(0, 5)

        angle_rad = np.deg2rad(result.angle)
        ax1.scatter(angle_rad, result.distance, c='red', s=100)

        capture_dt = datetime.datetime.fromtimestamp(capture_time)
        current_date = capture_dt.strftime("%d-%m-%Y")
        plt.figtext(0.01, 0.01, f"Date: {current_date}", fontsize=10, color="blue")
        ax1.set_title(f'Sound source: {result.angle:.1f}° | Distance: {result.distance:.2f} m')

        # Volume subplot
        ax2 = plt.subplot(1, 2, 2)
        bars = ax2.bar(range(1, 5), result.rms, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_xticks(range(1, 5))
        ax2.set_xlabel('Microphone')
        ax2.set_ylabel('RMS Value')
        ax2.set_title('Signal level on microphones')

        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.,
                     height,
                     f'{self.format_value(height)}',
                     ha='center',
                     va='bottom')

        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.pause(0.1)

    def plot_analysis(self, results: Dict[str, Any]):
        """Generates complete analysis plot for file"""
        plt.figure(figsize=(18, 10), constrained_layout=True)
        plt.suptitle(f"Recording analysis: {results['file_name']}", y=0.98)

        # Radar plot
        ax = plt.subplot(2, 3, 1, polar=True)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_thetamin(-45)
        ax.set_thetamax(45)
        ax.set_rlim(0, 5)
        ax.plot(np.deg2rad(results['angle']), results['distance'], 'ro', markersize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.title(f'Localization\n{results["angle"]:.1f}°, {results["distance"]:.2f}m', pad=20)

        # Correlation matrix
        plt.subplot(2, 3, 2)
        plt.imshow(results['corr_matrix'], cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation coefficient')
        plt.xticks(range(4), ['M1', 'M2', 'M3', 'M4'])
        plt.yticks(range(4), ['M1', 'M2', 'M3', 'M4'])

        for i in range(4):
            for j in range(4):
                plt.text(j, i, f"{results['corr_matrix'][i, j]:.2f}",
                         ha='center', va='center',
                         color='w' if abs(results['corr_matrix'][i, j]) > 0.5 else 'k')
        plt.grid(False)
        plt.title('Correlation matrix', pad=15)

        # MUSIC spectrum
        plt.subplot(2, 3, 3)
        angles = np.linspace(-45, 45, 180)
        plt.plot(angles, results['music_spectrum'], lw=2)
        plt.axvline(results['angle'], color='r', linestyle='--', label='Detected angle')

        peak_val = np.max(results['music_spectrum'])
        plt.scatter(results['angle'], peak_val, color='red', zorder=5)
        plt.annotate(f'{results["angle"]:.1f}°\n{self.format_value(peak_val)}',
                     (results['angle'], peak_val),
                     textcoords='offset points',
                     xytext=(15, 0),
                     ha='left',
                     arrowprops=dict(arrowstyle='->'))

        plt.xlabel('Angle (degrees)', labelpad=10)
        plt.ylabel('MUSIC Amplitude', labelpad=10)
        plt.title('MUSIC DOA Spectrum', pad=15)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Waterfall periodogram
        plt.subplot(2, 3, 4)
        plt.pcolormesh(results['times'], results['freq'], results['psd_waterfall'], shading='gouraud')
        plt.colorbar(label='Power (dB)')
        plt.xlabel('Time (s)', labelpad=10)
        plt.ylabel('Frequency (Hz)', labelpad=10)
        plt.title('Waterfall Periodogram', pad=15)
        plt.ylim(0, Config.PLOT_FREQ_LIMIT)

        # Microphone volumes
        plt.subplot(2, 3, 5)
        bars = plt.bar(range(1, 5), results['rms'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.xticks(range(1, 5))

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.,
                     height,
                     f'{self.format_value(height)}',
                     ha='center',
                     va='bottom')

        plt.xlabel('Microphone', labelpad=10)
        plt.ylabel('RMS Value', labelpad=10)
        plt.title('Signal level on microphones', pad=15)
        plt.grid(True, axis='y', alpha=0.3)

        # Info panel
        plt.subplot(2, 3, 6)
        plt.axis('off')

        # Calculate dominant frequencies dynamically
        psd_waterfall = results['psd_waterfall']

        # Calculate average spectrum to find dominant frequencies
        mean_spectrum = np.mean(psd_waterfall, axis=1)

        # Calculate dynamic threshold (mean power + standard deviation)
        threshold = np.mean(mean_spectrum) + np.std(mean_spectrum)

        # Identify spectral peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(mean_spectrum, height=threshold, distance=5)

        # Get frequencies and powers associated with peaks
        peak_freqs = results['freq'][peaks]
        peak_powers = mean_spectrum[peaks]

        # Sort by power (descending)
        sorted_indices = np.argsort(-peak_powers)
        peak_freqs = peak_freqs[sorted_indices]
        peak_powers = peak_powers[sorted_indices]

        # Limit to top 5 (or fewer if less than 5 exist)
        num_peaks = min(5, len(peak_freqs))
        peak_freqs = peak_freqs[:num_peaks]
        peak_powers = peak_powers[:num_peaks]

        # Format for display
        if len(peak_freqs) > 0:
            peak_freq_str = ", ".join(f"{peak_freqs[i]:.0f} Hz ({peak_powers[i]:.1f} dB)" for i in range(num_peaks))
        else:
            peak_freq_str = "No dominant frequencies detected"

        info_text = (
                f"• SNR: {results['snr']:.1f} dB\n"
                f"• Estimated distance: {results['distance']:.2f} m\n"
                f"• Detected angle: {results['angle']:.1f}°\n"
                f"• Dominant frequencies:\n  " + peak_freq_str.replace(", ", "\n  ") + "\n"
                f"• Average RMS: {self.format_value(results['rms_mean'])} ± {self.format_value(results['rms_std'])}\n"
                f"• Sampling frequency: {results['fs'] / 1000:.1f} kHz\n"
                f"• Recording duration: {results['n_samples'] / results['fs']:.2f}s"
        )

        plt.text(-0.5, 0.5, info_text,
                 fontfamily='monospace',
                 fontsize=10,
                 verticalalignment='center')

        plt.show()

    def format_value(self, value):
        """Formats value based on magnitude"""
        if value < 0.001:
            return f"{value * 1000000:.2f} µ"
        elif value < 1:
            return f"{value * 1000:.2f} m"
        else:
            return f"{value:.2f}"


class AudioCapture:
    """Class for live audio capture"""

    def __init__(self, save_directory: str):
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        self.processor = AudioProcessor()
        self.visualizer = Visualizer()

    def generate_filename(self, current_date: str) -> str:
        """Generates unique filename based on date"""
        pattern = r"^CS(\d+)-" + re.escape(current_date) + r"\.wav$"
        max_num = 0

        for file in self.save_directory.glob(f"*-{current_date}.wav"):
            match = re.match(pattern, file.name)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)

        return f"CS{max_num + 1}-{current_date}.wav"

    def capture_audio(self, buffer_duration: int, device_index: int, fs: int = Config.FS, channels: int = 4):
        """Captures and processes live audio"""
        try:
            plt.ion()
            silence_time = 0

            while silence_time < 15:
                record_voice = sd.rec(int(buffer_duration * fs),
                                      samplerate=fs,
                                      channels=channels,
                                      device=device_index)
                sd.wait()
                capture_time = time.time()
                capture_dt = datetime.datetime.fromtimestamp(capture_time)
                current_date = capture_dt.strftime("%d-%m-%Y")

                filename = self.generate_filename(current_date)
                file_path = self.save_directory / filename
                write(file_path, fs, record_voice)
                print(f"Recording saved as: {filename}")

                result = self.processor.process_frame(record_voice)
                if result:
                    print("\nSound detected!")
                    print(f"Distance: {result.distance:.2f} m")
                    print(f"Angle: {result.angle:.1f}°")
                    print(f"SNR: {result.snr:.1f} dB")
                    print("Microphone volumes:")
                    for i, vol in enumerate(result.rms):
                        print(f"Microphone {i + 1}: {self.visualizer.format_value(vol)}")
                    self.visualizer.plot_radar(result, capture_time)
                    silence_time = 0
                else:
                    silence_time += buffer_duration
                    print("No sound detected.")
        except KeyboardInterrupt:
            print("\nProgram stopped.")
            plt.close()


class AudioAnalysisApp:
    """Main application class"""

    def __init__(self):
        self.base_dir = Path("E:\\Sounds")
        self.processor = AudioProcessor()
        self.visualizer = Visualizer()

    def list_folders(self) -> List[Tuple[str, Path]]:
        """Lists available folders for analysis"""
        folders = []
        for name in ["LiveCapture", "Recordings"]:
            path = self.base_dir / name
            if path.exists():
                folders.append((name, path))
        return folders

    def list_wav_files(self, folder_path: Path) -> List[Path]:
        """Lists WAV files in folder"""
        return sorted(folder_path.glob("*.wav"))

    def run_live_capture_mode(self):
        """Runs live capture mode"""
        save_directory = self.base_dir / "LiveCapture"
        capture = AudioCapture(save_directory)

        buffer_duration = int(input("Select buffer duration (1-10 seconds): "))

        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            print(f"{idx}: {device['name']} (input channels: {device['max_input_channels']})")

        device_index = int(input("Enter desired input device index: "))

        capture.capture_audio(buffer_duration, device_index)

    def run_file_analysis_mode(self):
        """Runs file analysis mode"""
        folders = self.list_folders()

        if not folders:
            print("No folders available for analysis.")
            return

        print("Available folders:")
        for i, (name, path) in enumerate(folders, 1):
            print(f"{i}. {name}")
            wav_files = self.list_wav_files(path)
            for f in wav_files[:3]:
                print(f"   - {f.name}")
            if len(wav_files) > 3:
                print(f"   ... and {len(wav_files) - 3} more files")

        folder_idx = int(input("Select folder number: ")) - 1
        folder_path = folders[folder_idx][1]

        wav_files = self.list_wav_files(folder_path)
        if not wav_files:
            print("Folder is empty.")
            return

        print("Available files:")
        for i, f in enumerate(wav_files, 1):
            print(f"{i}. {f.name}")
        file_idx = int(input("Select file number: ")) - 1
        file_path = wav_files[file_idx]

        results = self.processor.analyze_file(str(file_path))
        if results:
            self.visualizer.plot_analysis(results)

    def run(self):
        """Main function to run the application"""
        print("Select mode:")
        print("1. Live capture")
        print("2. Recording analysis")
        mode = input("> ").strip()
        
        if mode == "1":
            print("Selected mode: Live capture")
            self.run_live_capture_mode()
        elif mode == "2":
            self.run_file_analysis_mode()
        else:
            print("Invalid option.")


if __name__ == "__main__":
    app = AudioAnalysisApp()
    app.run()
