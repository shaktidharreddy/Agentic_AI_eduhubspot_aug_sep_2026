import base64
import hashlib


def video_to_text(video_path, text_path, chunk_size=57 * 4096):
    """Stream a video into a Base64 text file (76-character lines) without loading it all into memory."""
    sha = hashlib.sha256()
    with open(video_path, "rb") as src, open(text_path, "wb") as dst:
        while chunk := src.read(chunk_size):  # a multiple of 57 bytes = whole 76-character lines
            sha.update(chunk)
            dst.write(base64.encodebytes(chunk))
    print(f"Saved {text_path}\nSHA-256 of the video: {sha.hexdigest()}")


video_to_text(r"C:\path\to\video.mp4", r"C:\path\to\encoded_video.txt")
