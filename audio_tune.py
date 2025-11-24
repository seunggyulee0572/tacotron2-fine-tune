import librosa
import soundfile as sf
import os
from pathlib import Path

def resample_audio_files(input_dir, output_dir, target_sr=22050):
    """모든 오디오 파일을 target_sr로 리샘플링"""
    os.makedirs(output_dir, exist_ok=True)
    
    audio_files = list(Path(input_dir).rglob('*.wav'))

    for audio_file in audio_files:
        try:
            # 오디오 로드
            audio, sr = librosa.load(str(audio_file), sr=None)
            
            # 리샘플링
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                print(f"Resampled {audio_file.name}: {sr} -> {target_sr} Hz")
            
            # 저장
            output_path = Path(output_dir) / audio_file.name
            sf.write(str(output_path), audio, target_sr)
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")

# 사용 예시
input_directory = "/Users/zest/Downloads/tacotron2/datasets/son/audio"  # 원본 오디오 경로
output_directory = "/Users/zest/Downloads/tacotron2/datasets/son/audio_22050"  # 변환된 오디오 저장 경로

resample_audio_files(input_directory, output_directory)