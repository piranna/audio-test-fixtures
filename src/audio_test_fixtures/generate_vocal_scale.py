#!/usr/bin/env python3
"""
Genera un archivo WAV de prueba con escala musical ascendente para tests automatizados.

Características:
- Formato parametrizable: PCM 8/16-bit, mono/estéreo, sample rate configurable
- Duración parametrizable
- Contenido: Notas sinusoidales puras en el rango vocal humano
- Rango vocal seleccionable (presets) o por límites en Hz
- Uso: Fixture para verificar coherencia de audio codificado/transmitido/decodificado con NCC

Análisis de frecuencias esperado:
- Cada nota tiene duración fija y frecuencia conocida
- Al decodificar, las frecuencias dominantes deben coincidir con las originales
"""

import argparse
import numpy as np
import wave
import sys

# Notas musicales base en el rango vocal humano (frecuencias en Hz)
# Rango completo: E2 (82 Hz) hasta C6 (1047 Hz)
BASE_VOCAL_NOTES = [
    ("E2",  82.41),   # Bajo profundo
    ("F2",  87.31),
    ("G2",  98.00),
    ("A2", 110.00),   # Inicio barítono
    ("B2", 123.47),
    ("C3", 130.81),   # Inicio tenor
    ("D3", 146.83),
    ("E3", 164.81),
    ("F3", 174.61),   # Inicio alto
    ("G3", 196.00),
    ("A3", 220.00),
    ("B3", 246.94),
    ("C4", 261.63),   # Inicio soprano / Do central
    ("D4", 293.66),
    ("E4", 329.63),
    ("F4", 349.23),
    ("G4", 392.00),
    ("A4", 440.00),   # La de referencia
    ("B4", 493.88),
    ("C5", 523.25),
    ("D5", 587.33),
    ("E5", 659.25),
    ("F5", 698.46),
    ("G5", 783.99),
    ("A5", 880.00),
    ("B5", 987.77),
    ("C6", 1046.50),  # Soprano agudo
]

PRESET_RANGES = {
    # Presets aproximados por frecuencia fundamental
    "full": (82.0, 1050.0),
    "conversation": (100.0, 400.0),
    "bass": (82.0, 196.0),
    "baritone": (110.0, 246.0),
    "tenor": (130.0, 392.0),
    "alto": (174.0, 440.0),
    "soprano": (261.0, 1050.0),
}


def select_notes(
    range_name: str,
    min_freq: float | None = None,
    max_freq: float | None = None
):
    """
    Selecciona subconjunto de notas según preset o límites en Hz.
    """
    if range_name:
        if range_name not in PRESET_RANGES:
            print(
                f"✗ Rango '{range_name}' no reconocido. Presets: {', '.join(PRESET_RANGES.keys())}")
            sys.exit(2)
        min_freq, max_freq = PRESET_RANGES[range_name]

    if min_freq and max_freq and min_freq > max_freq:
        print("✗ min_freq debe ser <= max_freq")
        sys.exit(2)

    notes = [(n, f) for (n, f) in BASE_VOCAL_NOTES
             if (min_freq is None or f >= min_freq) and (max_freq is None or f <= max_freq)]

    if not notes:
        print("✗ Filtro de rango dejó 0 notas. Ajusta presets o min/max Hz.")
        sys.exit(2)

    return notes


def generate_tone(frequency, duration, sample_rate, amplitude=0.3):
    """
    Genera una onda sinusoidal pura.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = amplitude * np.sin(2 * np.pi * frequency * t)
    return tone

def add_fade(audio, fade_samples):
    """
    Añade fade-in y fade-out para evitar clics.
    """
    if fade_samples <= 0:
        return audio
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out
    return audio


def write_wav(output_file, audio_float, channels, bit_depth, sample_rate, normalize_factor=1.0):
    """
    Escribe audio float [-1,1] a WAV con configuración dada.
    """
    # Normalizar al factor especificado del rango digital
    peak = np.max(np.abs(audio_float))
    if peak > 0:
        audio_float = audio_float / peak * normalize_factor

    # Convertir a PCM
    if bit_depth == 16:
        audio_pcm = (audio_float * 32767).astype(np.int16)
        sampwidth = 2
    elif bit_depth == 8:
        # PCM 8-bit unsigned
        audio_pcm = (audio_float * 127 + 128).clip(0, 255).astype(np.uint8)
        sampwidth = 1
    else:
        print("✗ bit_depth soportado: 8 o 16")
        sys.exit(2)

    # Canales
    if channels == 2:
        audio_pcm = np.column_stack([audio_pcm, audio_pcm]).ravel()
    elif channels != 1:
        print("✗ channels soportado: 1 o 2")
        sys.exit(2)

    with wave.open(output_file, 'w') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_pcm.tobytes())


def verify_wav(wav_file, expected_channels, expected_bit_depth, expected_sample_rate, expected_duration):
    """
    Verifica que el WAV generado tenga el formato correcto.
    """
    print(f"\nVerificando {wav_file}...")
    with wave.open(wav_file, 'r') as f:
        channels = f.getnchannels()
        sampwidth = f.getsampwidth()
        framerate = f.getframerate()
        nframes = f.getnframes()
        duration = nframes / framerate

        print(f"  Canales: {channels} (esperado: {expected_channels})")
        print(f"  Bits: {sampwidth * 8} (esperado: {expected_bit_depth})")
        print(
            f"  Sample rate: {framerate} Hz (esperado: {expected_sample_rate})")
        print(f"  Frames: {nframes}")
        print(f"  Duración: {duration:.6f} s (esperado: {expected_duration})")

        all_ok = (
            channels == expected_channels and
            (sampwidth * 8) == expected_bit_depth and
            framerate == expected_sample_rate and
            abs(duration - expected_duration) < 0.001  # tolerancia de 1ms
        )

        if all_ok:
            print("  ✓ Formato correcto")
        else:
            print("  ✗ Formato incorrecto")
            sys.exit(1)


def generate_vocal_scale_wav(output_file,
                             sample_rate=16000,
                             duration=10.0,
                             channels=1,
                             bit_depth=16,
                             range_name='full',
                             min_freq: float | None = None,
                             max_freq: float | None = None,
                             silence_duration=0.02,
                             tone_amplitude=0.3,
                             fade_ms=5.0,
                             normalize_factor=1.0):
    """
    Genera archivo WAV con escala vocal ascendente.
    """
    notes = select_notes(range_name, min_freq, max_freq)
    total_samples = int(sample_rate * duration)
    num_notes = len(notes)

    # Calcular duración por nota (con pequeño silencio entre notas)
    note_duration = duration / num_notes
    tone_duration = max(0.0, note_duration - silence_duration)

    # Generar audio completo
    audio = np.zeros(total_samples, dtype=np.float32)

    print(
        f"Generando escala vocal de {num_notes} notas en {duration} segundos...")
    print(
        f"Rango seleccionado: {range_name} (min={min_freq or PRESET_RANGES.get(range_name, (None, None))[0]} Hz, max={max_freq or PRESET_RANGES.get(range_name, (None, None))[1]} Hz)")
    print(f"Duración por nota: {tone_duration:.3f}s + {silence_duration:.3f}s silencio")
    print()
    print("Nota  Frecuencia  Inicio    Fin")
    print("-" * 40)

    current_sample = 0
    fade_samples = int((fade_ms / 1000.0) * sample_rate)

    for note_name, frequency in notes:
        # Generar tono
        tone = generate_tone(frequency, tone_duration,
                             sample_rate, amplitude=tone_amplitude)
        tone = add_fade(tone, fade_samples)

        # Insertar en posición correcta
        tone_samples = len(tone)
        start_sample = current_sample
        end_sample = min(current_sample + tone_samples, total_samples)
        actual_samples = end_sample - start_sample

        audio[start_sample:end_sample] = tone[:actual_samples]

        # Info para debug
        start_time = start_sample / sample_rate
        end_time = end_sample / sample_rate
        print(f"{note_name:4s}  {frequency:7.2f} Hz  {start_time:6.3f}s  {end_time:6.3f}s")

        # Avanzar (tono + silencio)
        current_sample += int((tone_duration + silence_duration) * sample_rate)

    # Escribir archivo WAV con la configuración
    write_wav(output_file, audio, channels, bit_depth,
              sample_rate, normalize_factor)

    print()
    print(f"✓ Archivo generado: {output_file}")
    ch_str = 'mono' if channels == 1 else 'estéreo'
    print(f"  Formato: PCM {bit_depth}-bit {ch_str} {sample_rate} Hz")
    print(f"  Duración: {duration} segundos exactos")
    byte_width = 2 if bit_depth == 16 else 1
    print(
        f"  Tamaño (aprox.): {total_samples * byte_width * channels} bytes ({(total_samples * byte_width * channels) / 1024:.1f} KB)")
    print()
    print("Uso para tests:")
    print("  1. Codificar con NCC")
    print("  2. Transmitir por UDP/BLE")
    print("  3. Decodificar")
    print("  4. Analizar FFT para verificar frecuencias esperadas")

    # Verificación rápida
    verify_wav(output_file, channels, bit_depth, sample_rate, duration)


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Genera un WAV de escala vocal parametrizable",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Default: rango completo, 16kHz, 10s, mono 16-bit
  %(prog)s vocal_scale_10s.wav

  # Conversación (100-400 Hz), 12s, estéreo 16-bit @ 16kHz
  %(prog)s conv.wav --range conversation --duration 12 --channels 2 --bit-depth 16 --sample-rate 16000

  # Rango personalizado 120-500 Hz y sample rate 8kHz
  %(prog)s custom.wav --min-freq 120 --max-freq 500 --sample-rate 8000
"""
    )
    p.add_argument('output', nargs='?',
                   default='vocal_scale_10s.wav', help='Archivo WAV de salida')
    p.add_argument('--sample-rate', type=int, default=16000,
                   help='Sample rate en Hz (default: 16000)')
    p.add_argument('--duration', type=float, default=10.0,
                   help='Duración en segundos (default: 10.0)')
    p.add_argument('--channels', type=int,
                   choices=[1, 2], default=1, help='Canales: 1=mono, 2=estéreo (default: 1)')
    p.add_argument('--bit-depth', type=int,
                   choices=[8, 16], default=16, help='Profundidad de bits: 8 o 16 (default: 16)')
    p.add_argument('--range', dest='range_name', choices=list(PRESET_RANGES.keys()),
                   default='full', help='Preset de rango vocal (default: full)')
    p.add_argument('--min-freq', type=float, default=None,
                   help='Frecuencia mínima en Hz (override del preset)')
    p.add_argument('--max-freq', type=float, default=None,
                   help='Frecuencia máxima en Hz (override del preset)')
    p.add_argument('--silence', type=float, default=0.02,
                   help='Silencio entre notas en s (default: 0.02)')
    p.add_argument('--amplitude', type=float, default=0.3,
                   help='Amplitud del tono [0-1] (default: 0.3)')
    p.add_argument('--fade-ms', type=float, default=5.0,
                   help='Fade-in/out en ms (default: 5.0)')
    p.add_argument('--normalize', type=float, default=1.0,
                   help='Factor de normalización [0-1] donde 1.0=0dB, 0.9=−0.9dB, etc. (default: 1.0)')
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    generate_vocal_scale_wav(
        output_file=args.output,
        sample_rate=args.sample_rate,
        duration=args.duration,
        channels=args.channels,
        bit_depth=args.bit_depth,
        range_name=args.range_name,
        min_freq=args.min_freq,
        max_freq=args.max_freq,
        silence_duration=args.silence,
        tone_amplitude=args.amplitude,
        fade_ms=args.fade_ms,
        normalize_factor=args.normalize,
    )

    print("\nNotas para análisis automatizado:")
    print("  - Usar FFT con ventana centrada en cada tono")
    print("  - Buscar pico dominante en cada segmento temporal")
    print("  - Comparar frecuencias detectadas con el rango/notas generadas")
    print("  - Ajustar tolerancia según codec (±2..±10 Hz)")


if __name__ == "__main__":
    main()
