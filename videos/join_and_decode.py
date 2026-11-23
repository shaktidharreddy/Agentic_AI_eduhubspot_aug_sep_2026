# Restoring later: put the 5 files in one folder and run python join_and_decode.py restored.mp4. It should print Matches the published SHA-256: YES.

import base64
import glob
import hashlib
import os
import re
import sys

_B64 = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
_DROP = bytes(b for b in range(256) if b not in _B64)


def join_and_decode(output_video_path, parts_dir=None, block_size=4 * 1024 * 1024):
    """Decode every '*.partN-of-M.txt' file in `parts_dir` (in numeric order) back into the original video."""
    parts_dir = parts_dir or os.path.dirname(os.path.abspath(__file__))
    parts = sorted(
        glob.glob(os.path.join(parts_dir, "*.part*-of-*.txt")),
        key=lambda p: int(re.search(r"\.part(\d+)-of-", p).group(1)),
    )
    if not parts:
        raise SystemExit(f"No '*.partN-of-M.txt' files found in {parts_dir}")
    sha, leftover = hashlib.sha256(), b""
    with open(output_video_path, "wb") as dst:
        for part in parts:
            with open(part, "rb") as src:
                while block := src.read(block_size):
                    data = leftover + block.translate(None, _DROP)  # keep only Base64 characters
                    cut = len(data) - len(data) % 4                  # decode whole 4-character groups only
                    raw = base64.b64decode(data[:cut])
                    sha.update(raw)
                    dst.write(raw)
                    leftover = data[cut:]
        if leftover:
            raw = base64.b64decode(leftover + b"=" * (-len(leftover) % 4))
            sha.update(raw)
            dst.write(raw)
    digest = sha.hexdigest()
    print(f"Restored {output_video_path} from {len(parts)} parts\nSHA-256: {digest}")
    for sha_file in glob.glob(os.path.join(parts_dir, "*.sha256.txt")):  # optional integrity check
        expected = open(sha_file, encoding="utf-8").read().split()[0].lower()
        print("Matches the published SHA-256: " + ("YES" if expected == digest else "NO - the restored file is not identical"))


if __name__ == "__main__":
    # usage: python join_and_decode.py <output_video.mp4> [folder_with_the_parts]
    join_and_decode(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
