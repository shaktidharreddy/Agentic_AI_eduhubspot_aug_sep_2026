import math
import os
import sys


def split_text(text_path, parts, out_dir=None):
    """Split a line-wrapped Base64 text file into `parts` files, breaking only between lines."""
    out_dir = out_dir or os.path.dirname(os.path.abspath(text_path))
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(text_path)
    if base.lower().endswith(".txt"):
        base = base[:-4]
    target = math.ceil(os.path.getsize(text_path) / parts)  # aim for equal-sized parts
    names, written, index, dst = [], 0, 0, None
    with open(text_path, "rb") as src:
        for line in src:
            if dst is None or (written >= target and index < parts):
                if dst:
                    dst.close()
                index += 1
                name = os.path.join(out_dir, f"{base}.part{index}-of-{parts}.txt")
                dst = open(name, "wb")
                names.append(name)
                written = 0
            dst.write(line)
            written += len(line)
    if dst:
        dst.close()
    for name in names:
        print(f"{os.path.getsize(name) / 1048576:6.2f} MiB  {name}")
    return names


if __name__ == "__main__":
    # usage: python split_text.py <encoded_text_file> <number_of_parts> [output_folder]
    split_text(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else None)
