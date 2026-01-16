#!/usr/bin/env python3
"""
Genera un archivo WAV de prueba con escala musical ascendente para tests automatizados.

Características:
- Formato: RIFF WAVE PCM 16-bit mono 16000 Hz
- Duración: 10 segundos exactos
- Contenido: Notas sinusoidales puras en el rango vocal humano (82 Hz - 1047 Hz)
- Uso: Fixture para verificar coherencia de audio codificado/transmitido/decodificado con NCC

Análisis de frecuencias esperado:
- Cada nota tiene duración fija y frecuencia conocida
- Al decodificar, las frecuencias dominantes deben coincidir con las originales
"""

import numpy as np
import wave
import struct
import sys

# Configuración WAV
SAMPLE_RATE = 16000  # Hz
DURATION = 10.0      # segundos
CHANNELS = 1         # mono
SAMPLE_WIDTH = 2     # 16-bit = 2 bytes

# Notas musicales en el rango vocal humano (frecuencias en Hz)
# Rango completo: E2 (82 Hz) hasta C6 (1047 Hz)
VOCAL_NOTES = [
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

def generate_tone(frequency, duration, sample_rate, amplitude=0.3):
    """
    Genera una onda sinusoidal pura.
    
    Args:
        frequency: Frecuencia en Hz
        duration: Duración en segundos
        sample_rate: Tasa de muestreo en Hz
        amplitude: Amplitud normalizada (0.0 a 1.0)
    
    Returns:
        numpy array con muestras de audio
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    tone = amplitude * np.sin(2 * np.pi * frequency * t)
    return tone

def add_fade(audio, fade_samples):
    """
    Añade fade-in y fade-out para evitar clics.
    
    Args:
        audio: numpy array con audio
        fade_samples: número de muestras para fade
    
    Returns:
        audio con fades aplicados
    """
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    
    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out
    
    return audio

def generate_vocal_scale_wav(output_file):
    """
    Genera archivo WAV con escala vocal ascendente.
    """
    total_samples = int(SAMPLE_RATE * DURATION)
    num_notes = len(VOCAL_NOTES)
    
    # Calcular duración por nota (con pequeño silencio entre notas)
    note_duration = DURATION / num_notes
    silence_duration = 0.02  # 20ms de silencio entre notas
    tone_duration = note_duration - silence_duration
    
    # Generar audio completo
    audio = np.zeros(total_samples, dtype=np.float32)
    
    print(f"Generando escala vocal de {num_notes} notas en {DURATION} segundos...")
    print(f"Duración por nota: {tone_duration:.3f}s + {silence_duration:.3f}s silencio")
    print()
    print("Nota  Frecuencia  Inicio    Fin")
    print("-" * 40)
    
    current_sample = 0
    
    for note_name, frequency in VOCAL_NOTES:
        # Generar tono
        tone = generate_tone(frequency, tone_duration, SAMPLE_RATE)
        
        # Añadir fade corto para evitar clics
        fade_samples = int(0.005 * SAMPLE_RATE)  # 5ms
        tone = add_fade(tone, fade_samples)
        
        # Insertar en posición correcta
        tone_samples = len(tone)
        start_sample = current_sample
        end_sample = min(current_sample + tone_samples, total_samples)
        actual_samples = end_sample - start_sample
        
        audio[start_sample:end_sample] = tone[:actual_samples]
        
        # Info para debug
        start_time = start_sample / SAMPLE_RATE
        end_time = end_sample / SAMPLE_RATE
        print(f"{note_name:4s}  {frequency:7.2f} Hz  {start_time:6.3f}s  {end_time:6.3f}s")
        
        # Avanzar (tono + silencio)
        current_sample += int((tone_duration + silence_duration) * SAMPLE_RATE)
    
    # Normalizar a rango de 16-bit con margen
    audio = audio / np.max(np.abs(audio)) * 0.9  # 90% del rango para evitar clipping
    
    # Convertir a 16-bit PCM
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Escribir archivo WAV
    with wave.open(output_file, 'w') as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_int16.tobytes())
    
    print()
    print(f"✓ Archivo generado: {output_file}")
    print(f"  Formato: PCM 16-bit mono {SAMPLE_RATE} Hz")
    print(f"  Duración: {DURATION} segundos exactos")
    print(f"  Tamaño: {total_samples * SAMPLE_WIDTH} bytes ({total_samples * SAMPLE_WIDTH / 1024:.1f} KB)")
    print()
    print("Uso para tests:")
    print("  1. Codificar con NCC")
    print("  2. Transmitir por UDP/BLE")
    print("  3. Decodificar")
    print("  4. Analizar FFT para verificar frecuencias esperadas")

def verify_wav(wav_file):
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
        
        print(f"  Canales: {channels} (esperado: {CHANNELS})")
        print(f"  Bits: {sampwidth * 8} (esperado: {SAMPLE_WIDTH * 8})")
        print(f"  Sample rate: {framerate} Hz (esperado: {SAMPLE_RATE})")
        print(f"  Frames: {nframes}")
        print(f"  Duración: {duration:.6f} s (esperado: {DURATION})")
        
        all_ok = (
            channels == CHANNELS and
            sampwidth == SAMPLE_WIDTH and
            framerate == SAMPLE_RATE and
            abs(duration - DURATION) < 0.001  # tolerancia de 1ms
        )
        
        if all_ok:
            print("  ✓ Formato correcto")
        else:
            print("  ✗ Formato incorrecto")
            sys.exit(1)

if __name__ == "__main__":
    output_file = "vocal_scale_10s.wav"
    
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    
    generate_vocal_scale_wav(output_file)
    verify_wav(output_file)
    
    print("\nNotas para análisis automatizado:")
    print("  - Usar FFT con ventana de análisis de ~0.3s")
    print("  - Buscar pico dominante en cada segmento temporal")
    print("  - Comparar frecuencias detectadas con VOCAL_NOTES")
    print("  - Tolerancia recomendada: ±2 Hz (debido a cuantización FFT)")
