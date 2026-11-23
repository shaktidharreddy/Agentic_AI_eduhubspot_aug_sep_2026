import base64
import hashlib

_B64 = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
_DROP = bytes(b for b in range(256) if b not in _B64)


def text_to_video(text_path, output_video_path, block_size=4 * 1024 * 1024):
    """Restore a video from a Base64 text file in blocks. Tolerates line breaks, spaces and a BOM."""
    sha, leftover = hashlib.sha256(), b""
    with open(text_path, "rb") as src, open(output_video_path, "wb") as dst:
        while block := src.read(block_size):
            data = leftover + block.translate(None, _DROP)  # keep only Base64 characters
            cut = len(data) - len(data) % 4                  # decode whole 4-character groups only
            raw = base64.b64decode(data[:cut])
            sha.update(raw)
            dst.write(raw)
            leftover = data[cut:]
        if leftover:                                         # tolerate a missing '=' padding at the very end
            raw = base64.b64decode(leftover + b"=" * (-len(leftover) % 4))
            sha.update(raw)
            dst.write(raw)
    print(f"Restored {output_video_path}\nSHA-256 of the restored video: {sha.hexdigest()}")


text_to_video(r"C:\path\to\encoded_video.txt", r"C:\path\to\restored_video.mp4")
