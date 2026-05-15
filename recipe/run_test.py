import subprocess
import tempfile
from pathlib import Path

import numpy as np
from videoio import read_video_params, videoread, videosave


def test_ffmpeg_runtime():
    encoders = subprocess.check_output(["ffprobe", "-encoders"], text=True)
    assert "libx264" in encoders


def test_rgb_round_trip():
    frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    frames[0, :, :, 0] = 255
    frames[1, :, :, 1] = 128
    frames[2, :, :, 2] = 64

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "rgb.mp4"
        videosave(path, frames, preset="ultrafast", fps=2)

        params = read_video_params(path)
        assert params["width"] == 8
        assert params["height"] == 8

        decoded = videoread(path)
        assert decoded.shape == frames.shape
        assert decoded.dtype == np.uint8


if __name__ == "__main__":
    test_ffmpeg_runtime()
    test_rgb_round_trip()
