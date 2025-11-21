import sys
import os
from pydub import AudioSegment
from mutagen import File as MutagenFile

def get_media_info(file_path):
    if not os.path.isfile(file_path):
        print("File not found")
        return

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ['.mp3', '.wav']:
        print("Only mp3 and wav files are supported.")
        return

    try:
        audio = AudioSegment.from_file(file_path)
        duration_sec = len(audio) / 1000
        print(f"Duration: {duration_sec:.2f} seconds")

        # Отримуємо метадані
        metadata = MutagenFile(file_path, easy=True)
        if metadata is not None:
            print("Metadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        else:
            print("Metadata not found")

    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python media_info.py <file_path>")
    else:
        get_media_info(sys.argv[1])
