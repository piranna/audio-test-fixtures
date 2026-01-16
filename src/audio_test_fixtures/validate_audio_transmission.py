#!/usr/bin/env python3
"""
Valida la transmisión de audio comparando archivo de referencia con archivo decodificado.

Comprueba:
- Metadatos WAV (sample rate, channels, bits)
- Duración y tamaño de archivos
- Análisis espectral por segmentos (FFT)
- Frecuencias dominantes vs esperadas
- Métricas de calidad (SNR, accuracy de frecuencias)

Uso:
    python3 validate_audio_transmission.py reference.wav decoded.wav [--tolerance 5.0] [--verbose]

Exit codes:
    0 - Validación exitosa (todas las métricas dentro de umbrales)
    1 - Validación fallida (frecuencias incorrectas o calidad baja)
    2 - Error de archivo o formato
"""

import numpy as np
import wave
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Notas esperadas del vocal_scale_10s.wav (debe coincidir con generate_vocal_scale.py)
# Lista base de notas (para mapeo de nombres y fallback)
VOCAL_NOTES = [
    ("E2",  82.41),   # Bajo profundo
    ("F2",  87.31),
    ("G2",  98.00),
    ("A2", 110.00),
    ("B2", 123.47),
    ("C3", 130.81),
    ("D3", 146.83),
    ("E3", 164.81),
    ("F3", 174.61),
    ("G3", 196.00),
    ("A3", 220.00),
    ("B3", 246.94),
    ("C4", 261.63),
    ("D4", 293.66),
    ("E4", 329.63),
    ("F4", 349.23),
    ("G4", 392.00),
    ("A4", 440.00),
    ("B4", 493.88),
    ("C5", 523.25),
    ("D5", 587.33),
    ("E5", 659.25),
    ("F5", 698.46),
    ("G5", 783.99),
    ("A5", 880.00),
    ("B5", 987.77),
    ("C6", 1046.50),
]

# Configuración de análisis
NOTE_DURATION = 10.0 / len(VOCAL_NOTES)  # ~0.37s por nota
SILENCE_DURATION = 0.02  # 20ms entre notas
TONE_DURATION = NOTE_DURATION - SILENCE_DURATION  # ~0.35s

# Umbrales de validación
MIN_SNR_DB = 10.0  # SNR mínimo aceptable (dB)
MIN_FREQUENCY_ACCURACY = 0.80  # 80% de frecuencias correctas
MAX_MEAN_FREQ_ERROR_HZ = 10.0  # Error promedio máximo en Hz

@dataclass
class WavMetadata:
    """Metadatos de un archivo WAV"""
    channels: int
    sample_width: int
    frame_rate: int
    n_frames: int
    duration: float
    file_size: int

    def __str__(self):
        return (f"Canales: {self.channels}, "
                f"Bits: {self.sample_width * 8}, "
                f"Sample rate: {self.frame_rate} Hz, "
                f"Frames: {self.n_frames}, "
                f"Duración: {self.duration:.6f}s, "
                f"Tamaño: {self.file_size / 1024:.1f} KB")

@dataclass
class FrequencyMatch:
    """Resultado de comparación de una frecuencia"""
    expected_note: str
    expected_freq: float
    detected_freq: float
    error_hz: float
    error_percent: float
    is_match: bool
    segment_time: float

@dataclass
class ValidationResult:
    """Resultado completo de la validación"""
    metadata_match: bool
    duration_match: bool
    frequency_matches: List[FrequencyMatch]
    frequency_accuracy: float
    mean_freq_error_hz: float
    snr_db: float
    passed: bool
    message: str

def read_wav_metadata(filepath: str) -> WavMetadata:
    """Lee metadatos de un archivo WAV"""
    import os

    with wave.open(filepath, 'r') as f:
        channels = f.getnchannels()
        sample_width = f.getsampwidth()
        frame_rate = f.getframerate()
        n_frames = f.getnframes()
        duration = n_frames / frame_rate
        file_size = os.path.getsize(filepath)

    return WavMetadata(channels, sample_width, frame_rate, n_frames, duration, file_size)

def read_wav_data(filepath: str) -> Tuple[np.ndarray, int]:
    """
    Lee datos de audio de un archivo WAV.

    Returns:
        (audio_data, sample_rate) donde audio_data es normalizado a [-1, 1]
    """
    with wave.open(filepath, 'r') as f:
        sample_rate = f.getframerate()
        n_frames = f.getnframes()
        sample_width = f.getsampwidth()

        # Leer datos binarios
        raw_data = f.readframes(n_frames)

        # Convertir según sample_width
        if sample_width == 1:  # 8-bit unsigned
            audio = np.frombuffer(raw_data, dtype=np.uint8)
            audio = (audio.astype(np.float32) - 128) / 128.0
        elif sample_width == 2:  # 16-bit signed
            audio = np.frombuffer(raw_data, dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0
        else:
            raise ValueError(f"Sample width {sample_width} no soportado")

        # Si es estéreo, convertir a mono
        if f.getnchannels() == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)

    return audio, sample_rate


def find_dominant_frequency(audio_segment: np.ndarray, sample_rate: int,
                           min_freq: float = 50.0, max_freq: float = 1200.0) -> float:
    """
    Encuentra la frecuencia dominante en un segmento de audio usando FFT.

    Args:
        audio_segment: Segmento de audio normalizado
        sample_rate: Tasa de muestreo
        min_freq: Frecuencia mínima a considerar (Hz)
        max_freq: Frecuencia máxima a considerar (Hz)

    Returns:
        Frecuencia dominante en Hz
    """
    # Aplicar ventana de Hann para reducir efectos de borde
    window = np.hanning(len(audio_segment))
    windowed = audio_segment * window

    # FFT
    fft_result = np.fft.rfft(windowed)
    fft_freqs = np.fft.rfftfreq(len(windowed), 1.0 / sample_rate)
    fft_magnitude = np.abs(fft_result)

    # Filtrar rango de frecuencias de interés
    freq_mask = (fft_freqs >= min_freq) & (fft_freqs <= max_freq)
    valid_freqs = fft_freqs[freq_mask]
    valid_magnitudes = fft_magnitude[freq_mask]

    if len(valid_magnitudes) == 0:
        return 0.0

    # Encontrar pico dominante
    peak_idx = np.argmax(valid_magnitudes)
    dominant_freq = valid_freqs[peak_idx]

    return dominant_freq


def nearest_note_name(freq: float) -> str:
    """
    Encuentra el nombre de la nota más cercana de la lista base.
    """
    closest = min(VOCAL_NOTES, key=lambda nf: abs(nf[1] - freq))
    return closest[0]


def detect_note_segments(audio: np.ndarray, sample_rate: int,
                         energy_threshold: float = 0.05,
                         min_note_ms: float = 100.0,
                         min_silence_ms: float = 10.0) -> List[Tuple[int, int]]:
    """
    Detecta segmentos de notas por energía (envelope + umbral).

    Returns:
        Lista de (start_sample, end_sample) por cada nota detectada
    """
    # Envelope suavizada (media móvil ~5ms)
    win = max(1, int(0.005 * sample_rate))
    kernel = np.ones(win, dtype=np.float32) / win
    env = np.convolve(np.abs(audio).astype(np.float32), kernel, mode='same')

    # Umbral relativo si energy_threshold es None o 0
    if energy_threshold is None or energy_threshold <= 0:
        energy_threshold = 0.1 * float(np.max(env))

    is_tone = env >= energy_threshold
    segments = []
    in_seg = False
    start = 0
    min_note_samples = int((min_note_ms / 1000.0) * sample_rate)
    min_silence_samples = int((min_silence_ms / 1000.0) * sample_rate)

    i = 0
    last_end = None
    while i < len(is_tone):
        if not in_seg and is_tone[i]:
            # inicio de segmento
            # si gap pequeño respecto al anterior, unir
            if last_end is not None and (i - last_end) <= min_silence_samples and segments:
                # unir con el anterior (reanudar)
                start = segments[-1][0]
                segments.pop()
            else:
                start = i
            in_seg = True
        elif in_seg and not is_tone[i]:
            # fin de segmento
            end = i
            last_end = end
            in_seg = False
            if (end - start) >= min_note_samples:
                segments.append((start, end))
        i += 1

    # si terminó en tono, cerrar último
    if in_seg:
        end = len(is_tone) - 1
        if (end - start) >= min_note_samples:
            segments.append((start, end))

    return segments


def expected_notes_from_reference(reference_audio: np.ndarray, sample_rate: int,
                                  segments: List[Tuple[int, int]]) -> List[Tuple[str, float]]:
    """
    Deriva frecuencias esperadas analizando el audio de referencia en cada segmento.
    """
    expected = []
    for (s, e) in segments:
        # evitar fades: recortar 10ms a cada lado si es posible
        pad = int(0.010 * sample_rate)
        ss = s + pad if (e - s) > 2 * pad else s
        ee = e - pad if (e - s) > 2 * pad else e
        seg = reference_audio[ss:ee]
        if len(seg) <= 0:
            continue
        freq = find_dominant_frequency(seg, sample_rate)
        note = nearest_note_name(freq)
        expected.append((note, float(freq)))
    return expected

def calculate_snr(reference: np.ndarray, decoded: np.ndarray) -> float:
    """
    Calcula SNR (Signal-to-Noise Ratio) entre señal de referencia y decodificada.

    SNR = 10 * log10(P_signal / P_noise)
    donde P_noise = potencia del error (diferencia)

    Args:
        reference: Audio de referencia normalizado
        decoded: Audio decodificado normalizado (mismo tamaño que reference)

    Returns:
        SNR en dB
    """
    # Asegurar mismo tamaño
    min_len = min(len(reference), len(decoded))
    ref = reference[:min_len]
    dec = decoded[:min_len]

    # Calcular potencias
    signal_power = np.mean(ref ** 2)
    error = ref - dec
    noise_power = np.mean(error ** 2)

    if noise_power < 1e-10:  # Evitar división por cero
        return 100.0  # SNR muy alto

    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def compare_metadata(ref_meta: WavMetadata, dec_meta: WavMetadata, verbose: bool = False) -> bool:
    """Compara metadatos de dos archivos WAV"""

    if verbose:
        print("\n" + "="*60)
        print("COMPARACIÓN DE METADATOS")
        print("="*60)
        print(f"\nReferencia: {ref_meta}")
        print(f"Decodificado: {dec_meta}")

    issues = []

    if ref_meta.channels != dec_meta.channels:
        issues.append(f"Canales diferentes: {ref_meta.channels} vs {dec_meta.channels}")

    if ref_meta.sample_width != dec_meta.sample_width:
        issues.append(f"Sample width diferente: {ref_meta.sample_width} vs {dec_meta.sample_width}")

    if ref_meta.frame_rate != dec_meta.frame_rate:
        issues.append(f"Sample rate diferente: {ref_meta.frame_rate} vs {dec_meta.frame_rate}")

    # Tolerancia de 1% en duración (codec puede ajustar)
    duration_diff_pct = abs(ref_meta.duration - dec_meta.duration) / ref_meta.duration * 100
    if duration_diff_pct > 1.0:
        issues.append(f"Duración diferente: {ref_meta.duration:.3f}s vs {dec_meta.duration:.3f}s ({duration_diff_pct:.1f}% diff)")

    if issues:
        if verbose:
            print("\n❌ PROBLEMAS DETECTADOS:")
            for issue in issues:
                print(f"  - {issue}")
        return False

    if verbose:
        print("\n✓ Metadatos compatibles")

    return True


def analyze_frequencies_by_segments(decoded_audio: np.ndarray,
                                    sample_rate: int,
                                    expected_notes: List[Tuple[str, float]],
                                    segments: List[Tuple[int, int]],
                                    tolerance_hz: float,
                                    verbose: bool = False) -> List[FrequencyMatch]:
    """
    Analiza frecuencias dominantes en segmentos temporales y las compara.

    Args:
        reference_audio: Audio de referencia
        decoded_audio: Audio decodificado
        sample_rate: Tasa de muestreo
        tolerance_hz: Tolerancia en Hz para considerar frecuencia correcta
        verbose: Imprimir detalles

    Returns:
        Lista de FrequencyMatch con resultados de cada segmento
    """
    matches = []

    if verbose:
        print("\n" + "="*60)
        print("ANÁLISIS DE FRECUENCIAS (FFT)")
        print("="*60)
        print(f"\nTolerancia: ±{tolerance_hz} Hz")
        print()
        print("Nota  Esperada   Detectada  Error(Hz)  Error(%)  Estado")
        print("-" * 60)

    for i, ((note_name, expected_freq), (start_sample, end_sample)) in enumerate(zip(expected_notes, segments)):
        segment_center_time = (start_sample + end_sample) / (2.0 * sample_rate)

        # Validar rango
        if end_sample > len(decoded_audio):
            if verbose:
                print(f"{note_name:4s}  {expected_freq:7.2f}   ---        ---        ---       TRUNCADO")
            continue

        # Extraer segmento de audio del decodificado (evitar fades al borde)
        pad = int(0.010 * sample_rate)
        ss = start_sample + \
            pad if (end_sample - start_sample) > 2 * pad else start_sample
        ee = end_sample - \
            pad if (end_sample - start_sample) > 2 * pad else end_sample
        segment = decoded_audio[ss:ee]

        # Detectar frecuencia dominante
        detected_freq = find_dominant_frequency(segment, sample_rate)

        # Calcular error
        error_hz = detected_freq - expected_freq
        error_percent = abs(error_hz) / expected_freq * 100
        is_match = abs(error_hz) <= tolerance_hz

        match = FrequencyMatch(
            expected_note=note_name,
            expected_freq=expected_freq,
            detected_freq=detected_freq,
            error_hz=error_hz,
            error_percent=error_percent,
            is_match=is_match,
            segment_time=segment_center_time
        )
        matches.append(match)

        if verbose:
            status = "✓ OK" if is_match else "✗ FAIL"
            print(f"{note_name:4s}  {expected_freq:7.2f}   {detected_freq:7.2f}   "
                  f"{error_hz:+6.2f}     {error_percent:5.1f}%   {status}")

    return matches

def calculate_metrics(matches: List[FrequencyMatch], snr_db: float, verbose: bool = False) -> Tuple[float, float]:
    """
    Calcula métricas agregadas de calidad.

    Returns:
        (frequency_accuracy, mean_freq_error_hz)
    """
    if not matches:
        return 0.0, float('inf')

    correct_count = sum(1 for m in matches if m.is_match)
    frequency_accuracy = correct_count / len(matches)

    errors = [abs(m.error_hz) for m in matches]
    mean_freq_error_hz = np.mean(errors)

    if verbose:
        print("\n" + "="*60)
        print("MÉTRICAS DE CALIDAD")
        print("="*60)
        print(f"\nFrecuencias correctas: {correct_count}/{len(matches)} ({frequency_accuracy*100:.1f}%)")
        print(f"Error promedio: {mean_freq_error_hz:.2f} Hz")
        print(f"Error máximo: {max(errors):.2f} Hz")
        print(f"Error mínimo: {min(errors):.2f} Hz")
        print(f"SNR: {snr_db:.2f} dB")

    return frequency_accuracy, mean_freq_error_hz


def validate_transmission(reference_path: str, decoded_path: str,
                          tolerance_hz: float = 5.0, verbose: bool = False,
                          expected_notes_file: Optional[str] = None,
                          energy_threshold: float = 0.05,
                          min_note_ms: float = 100.0,
                          min_silence_ms: float = 10.0) -> ValidationResult:
    """
    Valida la transmisión de audio comparando referencia con decodificado.

    Args:
        reference_path: Ruta al archivo WAV de referencia
        decoded_path: Ruta al archivo WAV decodificado
        tolerance_hz: Tolerancia en Hz para frecuencias
        verbose: Imprimir detalles

    Returns:
        ValidationResult con todos los resultados
    """
    try:
        # 1. Leer metadatos
        if verbose:
            print(f"Leyendo archivos...")
            print(f"  Referencia: {reference_path}")
            print(f"  Decodificado: {decoded_path}")

        ref_meta = read_wav_metadata(reference_path)
        dec_meta = read_wav_metadata(decoded_path)

        # 2. Comparar metadatos
        metadata_match = compare_metadata(ref_meta, dec_meta, verbose)

        # 3. Leer datos de audio
        ref_audio, ref_sr = read_wav_data(reference_path)
        dec_audio, dec_sr = read_wav_data(decoded_path)

        # 4. Validar duración
        duration_diff = abs(ref_meta.duration - dec_meta.duration)
        duration_match = duration_diff < 0.1  # 100ms de tolerancia

        # 5. Calcular SNR
        snr_db = calculate_snr(ref_audio, dec_audio)

        # 6. Derivar notas esperadas (auto) o cargar desde archivo
        expected_notes: List[Tuple[str, float]]
        segments: List[Tuple[int, int]]

        if expected_notes_file:
            # Cargar notas esperadas desde CSV simple: nombre,frecuencia
            try:
                loaded = []
                with open(expected_notes_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) == 1:
                            freq = float(parts[0])
                            loaded.append((nearest_note_name(freq), freq))
                        elif len(parts) >= 2:
                            name, freq = parts[0], float(parts[1])
                            loaded.append((name, freq))
                expected_notes = loaded
                # Segmentar referencia en N notas equidistantes
                n = len(expected_notes)
                note_duration_samples = int(len(ref_audio) / n)
                segments = [(i * note_duration_samples,
                             min((i + 1) * note_duration_samples, len(ref_audio))) for i in range(n)]
            except Exception as e:
                raise RuntimeError(f"Error leyendo expected_notes_file: {e}")
        else:
            segments = detect_note_segments(ref_audio, ref_sr,
                                            energy_threshold=energy_threshold,
                                            min_note_ms=min_note_ms,
                                            min_silence_ms=min_silence_ms)
            expected_notes = expected_notes_from_reference(
                ref_audio, ref_sr, segments)

        if verbose:
            print(f"\nDetectadas {len(segments)} notas en referencia")

        # 7. Analizar frecuencias por segmentos
        frequency_matches = analyze_frequencies_by_segments(dec_audio, dec_sr,
                                                            expected_notes, segments,
                                                            tolerance_hz, verbose)

        # 8. Calcular métricas
        frequency_accuracy, mean_freq_error_hz = calculate_metrics(frequency_matches, snr_db, verbose)

        # 9. Determinar si pasó la validación
        passed = (
            metadata_match and
            duration_match and
            frequency_accuracy >= MIN_FREQUENCY_ACCURACY and
            mean_freq_error_hz <= MAX_MEAN_FREQ_ERROR_HZ and
            snr_db >= MIN_SNR_DB
        )

        # 10. Generar mensaje
        if passed:
            message = "✓ VALIDACIÓN EXITOSA - Audio transmitido correctamente"
        else:
            issues = []
            if not metadata_match:
                issues.append("metadatos incompatibles")
            if not duration_match:
                issues.append(f"duración diferente ({duration_diff:.2f}s)")
            if frequency_accuracy < MIN_FREQUENCY_ACCURACY:
                issues.append(f"precisión baja ({frequency_accuracy*100:.1f}% < {MIN_FREQUENCY_ACCURACY*100:.0f}%)")
            if mean_freq_error_hz > MAX_MEAN_FREQ_ERROR_HZ:
                issues.append(f"error alto ({mean_freq_error_hz:.1f} Hz > {MAX_MEAN_FREQ_ERROR_HZ} Hz)")
            if snr_db < MIN_SNR_DB:
                issues.append(f"SNR bajo ({snr_db:.1f} dB < {MIN_SNR_DB} dB)")

            message = "✗ VALIDACIÓN FALLIDA - " + ", ".join(issues)

        return ValidationResult(
            metadata_match=metadata_match,
            duration_match=duration_match,
            frequency_matches=frequency_matches,
            frequency_accuracy=frequency_accuracy,
            mean_freq_error_hz=mean_freq_error_hz,
            snr_db=snr_db,
            passed=passed,
            message=message
        )

    except Exception as e:
        return ValidationResult(
            metadata_match=False,
            duration_match=False,
            frequency_matches=[],
            frequency_accuracy=0.0,
            mean_freq_error_hz=float('inf'),
            snr_db=-float('inf'),
            passed=False,
            message=f"✗ ERROR - {str(e)}"
        )

def main():
    parser = argparse.ArgumentParser(
        description="Valida transmisión de audio comparando referencia con decodificado",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Validación básica
  %(prog)s reference.wav decoded.wav

  # Con tolerancia personalizada y verbose
  %(prog)s reference.wav decoded.wav --tolerance 10.0 --verbose

  # Para tests automatizados (exit code indica éxito/fallo)
  %(prog)s ref.wav dec.wav && echo "Test pasó" || echo "Test falló"
"""
    )

    parser.add_argument('reference', help='Archivo WAV de referencia')
    parser.add_argument('decoded', help='Archivo WAV decodificado/transmitido')
    parser.add_argument('--tolerance', type=float, default=5.0,
                       help='Tolerancia en Hz para frecuencias (default: 5.0)')
    parser.add_argument('--expected-notes-file', type=str, default=None,
                        help='Ruta a archivo CSV de notas esperadas (nombre,freq). Si se provee, se usa en lugar de auto-detección.')
    parser.add_argument('--energy-threshold', type=float, default=0.05,
                        help='Umbral de energía para detectar notas en referencia (default: 0.05)')
    parser.add_argument('--min-note-ms', type=float, default=100.0,
                        help='Duración mínima de nota para segmentación (ms, default: 100.0)')
    parser.add_argument('--min-silence-ms', type=float, default=10.0,
                        help='Silencio mínimo entre notas para segmentación (ms, default: 10.0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Imprimir detalles del análisis')

    args = parser.parse_args()

    # Ejecutar validación
    result = validate_transmission(
        args.reference,
        args.decoded,
        tolerance_hz=args.tolerance,
        verbose=args.verbose,
        expected_notes_file=args.expected_notes_file,
        energy_threshold=args.energy_threshold,
        min_note_ms=args.min_note_ms,
        min_silence_ms=args.min_silence_ms,
    )

    # Imprimir resultado
    if args.verbose:
        print("\n" + "="*60)
        print("RESULTADO FINAL")
        print("="*60)

    print(f"\n{result.message}")

    if not args.verbose and not result.passed:
        print(f"  Precisión: {result.frequency_accuracy*100:.1f}%")
        print(f"  Error medio: {result.mean_freq_error_hz:.2f} Hz")
        print(f"  SNR: {result.snr_db:.2f} dB")
        print("\nUsa --verbose para más detalles")

    # Exit code para tests automatizados
    sys.exit(0 if result.passed else 1)

if __name__ == "__main__":
    main()
