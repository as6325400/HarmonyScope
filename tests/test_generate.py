import os
from src.HarmonyScope import generate

def test_generate_many_chords(tmp_path):
    chords = [
        "C", "D", "E", "F", "G", "A", "B",
        "Cm", "Dm", "Em", "Fm", "Gm", "Am", "Bm",
        "Cdim", "Ddim", "Edim", "Fdim", "Gdim", "Adim", "Bdim"
    ]
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    generate.create_wav_files(chords, output_dir)
    
    for chord in chords:
        file_path = output_dir / f"{chord}.wav"
        assert file_path.exists(), f"{file_path} does not exist"
        assert os.path.getsize(file_path) > 0, f"{file_path} is empty"
