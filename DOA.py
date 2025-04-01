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


# Configurații
class Config:
    FS = 44100  # Frecvența de eșantionare
    R = 0.1  # Raza configurației microfonului (m)
    MIC_ANGLES = np.array([-45, -25, 25, 45])  # Unghiurile microfoanelor
    WAVELENGTH = 343 / FS  # Lungimea de undă (viteza_sunet / frecvența)
    BUFFER_DURATION = 5  # Durata buffer-ului pentru captura live
    NPERSEG = int(2 ** np.ceil(np.log2(FS // 200)))  # Dimensiunea ferestrei pentru spectrogramă
    NOVERLAP = NPERSEG // 2  # Overlap pentru spectrogramă (50%)
    PLOT_FREQ_LIMIT = 5000  # Limita superioară pentru afișarea frecvențelor


@dataclass
class AudioAnalysisResult:
    """Clasa pentru stocarea rezultatelor analizei audio."""
    distance: float
    angle: float
    snr: float
    spectrum: np.ndarray
    rms: np.ndarray


class AudioProcessor:
    """Clasa pentru procesarea semnalelor audio."""

    def __init__(self):
        self.config = Config()

    def normalize_audio(self, data: np.ndarray, target_level: float = 0.9) -> np.ndarray:
        """Normalizează semnalul audio la un nivel specificat."""
        data = data.astype(np.float32)
        max_vals = np.max(np.abs(data), axis=1, keepdims=True)
        max_vals[max_vals == 0] = 1.0  # Evită împărțirea la zero
        scaling_factors = target_level / max_vals
        return data * scaling_factors

    def estimate_covariance_matrix(self, data: np.ndarray) -> np.ndarray:
        """Estimează matricea de covarianță."""
        mean_centered = data - np.mean(data, axis=1, keepdims=True)
        return (mean_centered @ mean_centered.T) / (data.shape[1] - 1)

    def calculate_steering_vector(self, angle: float) -> np.ndarray:
        """Calculează vectorul de direcție pentru MUSIC."""
        return np.exp(-1j * 2 * np.pi * self.config.R * np.cos(np.deg2rad(angle - self.config.MIC_ANGLES)) / self.config.WAVELENGTH)

    def music_algorithm(self, cov_matrix: np.ndarray, num_sources: int = 1) -> Tuple[float, np.ndarray]:
        """Implementare algoritm MUSIC pentru localizare."""
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
        """Calculează periodograma waterfall pentru suma celor 4 microfoane."""
        summed_signal = np.sum(data, axis=0)
        freqs, times, Sxx = spectrogram(summed_signal, fs=self.config.FS, window='hann',
                                        nperseg=self.config.NPERSEG, noverlap=self.config.NOVERLAP)
        psd = 10 * np.log10(Sxx + 1e-12)  # Adăugăm un offset mic pentru a evita log(0)
        return freqs, times, psd

    def process_frame(self, frame: np.ndarray) -> Optional[AudioAnalysisResult]:
        """Procesează un cadru audio."""
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
        """Analizează un fișier audio și returnează rezultatele."""
        try:
            fs, data = read(file_path)
            data = data.astype(np.float32)
            channels_data = self.normalize_audio(data.T)

            # Calcul metrici
            rms = np.sqrt(np.mean(channels_data ** 2, axis=1))
            cov_matrix = self.estimate_covariance_matrix(channels_data)
            std_dev = np.sqrt(np.diag(cov_matrix))
            std_dev[std_dev == 0] = 1e-12
            corr_matrix = cov_matrix / np.outer(std_dev, std_dev)

            # Calcul PSD waterfall
            freqs, times, psd_waterfall = self.calculate_waterfall_psd(channels_data)

            # Calcul parametri
            angle, music_spectrum = self.music_algorithm(cov_matrix)
            peak_to_peak = np.ptp(channels_data, axis=1)
            snr = 20 * np.log10(np.max(peak_to_peak) / np.percentile(peak_to_peak, 10))
            distance = np.abs(np.tan(np.deg2rad(angle))) * 1.0

            # Calcule statistice
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)

            # Calcul frecvențe dominante
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
            print(f"Eroare la procesare: {e}")
            return {}


class Visualizer:
    """Clasa pentru vizualizarea rezultatelor."""

    @staticmethod
    def format_value(value):
        """Formatează valori pentru afișare."""
        return np.format_float_positional(value, precision=4, unique=False, fractional=False, trim='k')

    def plot_radar(self, result: AudioAnalysisResult, capture_time: float):
        """Generează vizualizarea radar pentru captură live."""
        plt.clf()

        # Subplot radar
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
        plt.figtext(0.01, 0.01, f"Data: {current_date}", fontsize=10, color="blue")
        ax1.set_title(f'Sursa sunet: {result.angle:.1f}° | Distanță: {result.distance:.2f} m')

        # Subplot volume
        ax2 = plt.subplot(1, 2, 2)
        bars = ax2.bar(range(1, 5), result.rms, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax2.set_xticks(range(1, 5))
        ax2.set_xlabel('Microfon')
        ax2.set_ylabel('Valoare RMS')
        ax2.set_title('Nivel semnal pe microfoane')

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
        """Generează graficul complet pentru analiza fișierului."""
        plt.figure(figsize=(18, 10), constrained_layout=True)
        plt.suptitle(f"Analiză înregistrare: {results['file_name']}", y=0.98)

        # Radar plot
        ax = plt.subplot(2, 3, 1, polar=True)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_thetamin(-45)
        ax.set_thetamax(45)
        ax.set_rlim(0, 5)
        ax.plot(np.deg2rad(results['angle']), results['distance'], 'ro', markersize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.title(f'Localizare\n{results["angle"]:.1f}°, {results["distance"]:.2f}m', pad=20)

        # Matrice corelație
        plt.subplot(2, 3, 2)
        plt.imshow(results['corr_matrix'], cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Coeficient de corelație')
        plt.xticks(range(4), ['M1', 'M2', 'M3', 'M4'])
        plt.yticks(range(4), ['M1', 'M2', 'M3', 'M4'])

        for i in range(4):
            for j in range(4):
                plt.text(j, i, f"{results['corr_matrix'][i, j]:.2f}",
                         ha='center', va='center',
                         color='w' if abs(results['corr_matrix'][i, j]) > 0.5 else 'k')
        plt.grid(False)
        plt.title('Matrice de corelație', pad=15)

        # Spectru MUSIC
        plt.subplot(2, 3, 3)
        angles = np.linspace(-45, 45, 180)
        plt.plot(angles, results['music_spectrum'], lw=2)
        plt.axvline(results['angle'], color='r', linestyle='--', label='Unghi detectat')

        peak_val = np.max(results['music_spectrum'])
        plt.scatter(results['angle'], peak_val, color='red', zorder=5)
        plt.annotate(f'{results["angle"]:.1f}°\n{self.format_value(peak_val)}',
                     (results['angle'], peak_val),
                     textcoords='offset points',
                     xytext=(15, 0),
                     ha='left',
                     arrowprops=dict(arrowstyle='->'))

        plt.xlabel('Unghi (grade)', labelpad=10)
        plt.ylabel('Amplitudine MUSIC', labelpad=10)
        plt.title('Spectru MUSIC DOA', pad=15)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Periodogramă waterfall
        plt.subplot(2, 3, 4)
        plt.pcolormesh(results['times'], results['freq'], results['psd_waterfall'], shading='gouraud')
        plt.colorbar(label='Putere (dB)')
        plt.xlabel('Timp (s)', labelpad=10)
        plt.ylabel('Frecvență (Hz)', labelpad=10)
        plt.title('Periodogramă Waterfall', pad=15)
        plt.ylim(0, Config.PLOT_FREQ_LIMIT)

        # Volume microfoane
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

        plt.xlabel('Microfon', labelpad=10)
        plt.ylabel('Valoare RMS', labelpad=10)
        plt.title('Nivel semnal pe microfoane', pad=15)
        plt.grid(True, axis='y', alpha=0.3)

        # Panou informativ
        plt.subplot(2, 3, 6)
        plt.axis('off')

        # Calculează frecvențele dominante în mod dinamic
        psd_waterfall = results['psd_waterfall']

        # Calcul mediu spectru pentru a găsi frecvențele dominante
        mean_spectrum = np.mean(psd_waterfall, axis=1)

        # Calculează pragul dinamic (media puterii + deviația standard)
        threshold = np.mean(mean_spectrum) + np.std(mean_spectrum)

        # Identifică vârfurile spectrale
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(mean_spectrum, height=threshold, distance=5)

        # Obține frecvențele și puterile asociate vârfurilor
        peak_freqs = results['freq'][peaks]
        peak_powers = mean_spectrum[peaks]

        # Sortează după putere (descrescător)
        sorted_indices = np.argsort(-peak_powers)
        peak_freqs = peak_freqs[sorted_indices]
        peak_powers = peak_powers[sorted_indices]

        # Limitează la primele 5 (sau mai puține dacă nu există 5)
        num_peaks = min(5, len(peak_freqs))
        peak_freqs = peak_freqs[:num_peaks]
        peak_powers = peak_powers[:num_peaks]

        # Formatează pentru afișare
        if len(peak_freqs) > 0:
            peak_freq_str = ", ".join(f"{peak_freqs[i]:.0f} Hz ({peak_powers[i]:.1f} dB)" for i in range(num_peaks))
        else:
            peak_freq_str = "Nicio frecvență dominantă detectată"

        info_text = (
                f"• SNR: {results['snr']:.1f} dB\n"
                f"• Distanță estimată: {results['distance']:.2f} m\n"
                f"• Unghi detectat: {results['angle']:.1f}°\n"
                f"• Frecvențe dominante:\n  " + peak_freq_str.replace(", ", "\n  ") + "\n"
                                                                                      f"• RMS mediu: {self.format_value(results['rms_mean'])} ± {self.format_value(results['rms_std'])}\n"
                                                                                      f"• Frecvență eșantionare: {results['fs'] / 1000:.1f} kHz\n"
                                                                                      f"• Durată înregistrare: {results['n_samples'] / results['fs']:.2f}s"
        )

        plt.text(-0.5, 0.5, info_text,
                 fontfamily='monospace',
                 fontsize=10,
                 verticalalignment='center')

        plt.show()

    def format_value(self, value):
        """Formatează valoarea în funcție de mărime."""
        if value < 0.001:
            return f"{value * 1000000:.2f} µ"
        elif value < 1:
            return f"{value * 1000:.2f} m"
        else:
            return f"{value:.2f}"
class AudioCapture:
    """Clasa pentru captura audio live."""

    def __init__(self, save_directory: str):
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(parents=True, exist_ok=True)
        self.processor = AudioProcessor()
        self.visualizer = Visualizer()

    def generate_filename(self, current_date: str) -> str:
        """Generează un nume de fișier unic bazat pe dată."""
        pattern = r"^CS(\d+)-" + re.escape(current_date) + r"\.wav$"
        max_num = 0

        for file in self.save_directory.glob(f"*-{current_date}.wav"):
            match = re.match(pattern, file.name)
            if match:
                num = int(match.group(1))
                max_num = max(max_num, num)

        return f"CS{max_num + 1}-{current_date}.wav"

    def capture_audio(self, buffer_duration: int, device_index: int, fs: int = Config.FS, channels: int = 4):
        """Captează și procesează audio live."""
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
                print(f"Înregistrare salvată ca: {filename}")

                result = self.processor.process_frame(record_voice)
                if result:
                    print("\nSunet detectat!")
                    print(f"Distanță: {result.distance:.2f} m")
                    print(f"Unghi: {result.angle:.1f}°")
                    print(f"SNR: {result.snr:.1f} dB")
                    print("Volume microfoane:")
                    for i, vol in enumerate(result.rms):
                        print(f"Microfon {i + 1}: {self.visualizer.format_value(vol)}")
                    self.visualizer.plot_radar(result, capture_time)
                    silence_time = 0
                else:
                    silence_time += buffer_duration
                    print("Niciun sunet detectat.")
        except KeyboardInterrupt:
            print("\nProgramul a fost oprit.")
            plt.close()


class AudioAnalysisApp:
    """Clasa principală a aplicației."""

    def __init__(self):
        self.base_dir = Path("E:\\Sunete")
        self.processor = AudioProcessor()
        self.visualizer = Visualizer()

    def list_folders(self) -> List[Tuple[str, Path]]:
        """Listează folderele disponibile pentru analiză."""
        folders = []
        for name in ["CapturaLive", "Înregistrări"]:
            path = self.base_dir / name
            if path.exists():
                folders.append((name, path))
        return folders

    def list_wav_files(self, folder_path: Path) -> List[Path]:
        """Listează fișierele WAV din folder."""
        return sorted(folder_path.glob("*.wav"))

    def run_live_capture_mode(self):
        """Rulează modul de captură live."""
        save_directory = self.base_dir / "CapturaLive"
        capture = AudioCapture(save_directory)

        buffer_duration = int(input("Selectați durata buffer-ului (1-10 secunde): "))

        devices = sd.query_devices()
        for idx, device in enumerate(devices):
            print(f"{idx}: {device['name']} (canale de input: {device['max_input_channels']})")

        device_index = int(input("Introduceți indexul dispozitivului de input dorit: "))

        capture.capture_audio(buffer_duration, device_index)

    def run_file_analysis_mode(self):
        """Rulează modul de analiză a fișierelor."""
        folders = self.list_folders()

        if not folders:
            print("Niciun folder disponibil pentru analiză.")
            return

        print("Foldere disponibile:")
        for i, (name, path) in enumerate(folders, 1):
            print(f"{i}. {name}")
            wav_files = self.list_wav_files(path)
            for f in wav_files[:3]:
                print(f"   - {f.name}")
            if len(wav_files) > 3:
                print(f"   ... și încă {len(wav_files) - 3} fișiere")

        folder_idx = int(input("Selectați numărul folderului: ")) - 1
        folder_path = folders[folder_idx][1]

        wav_files = self.list_wav_files(folder_path)
        if not wav_files:
            print("Folderul este gol.")
            return

        print("Fișiere disponibile:")
        for i, f in enumerate(wav_files, 1):
            print(f"{i}. {f.name}")
        file_idx = int(input("Selectați numărul fișierului: ")) - 1
        file_path = wav_files[file_idx]

        results = self.processor.analyze_file(str(file_path))
        if results:
            self.visualizer.plot_analysis(results)

    def run(self):
        """Funcția principală pentru rularea aplicației."""
        print("Selectați modul:")
        print("1. Captură live")
        print("2. Analiza înregistrare")
        mode = input("> ").strip()

        if mode == "1":
            print("Modul selectat: Captură live")
            self.run_live_capture_mode()
        elif mode == "2":
            self.run_file_analysis_mode()
        else:
            print("Opțiune invalidă.")


if __name__ == "__main__":
    app = AudioAnalysisApp()
    app.run()
