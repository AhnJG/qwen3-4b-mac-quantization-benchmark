"""
tools/ballast.py

Hold `--gb` gigabytes of *resident, incompressible* memory to create real,
sustained system memory pressure. Run as a background process; SIGTERM/SIGINT
releases the memory and exits.

Two things have to be true for the ballast to actually pressure the system,
and the first version got the second one wrong:

  1. Incompressible: macOS compresses inactive memory before swapping it, so
     zero/patterned pages create no pressure. We fill with os.urandom (max
     entropy -> incompressible) and keep a rotating window freshly randomized.

  2. Resident: pages that are never re-touched go "inactive" and get compressed
     or swapped out, so the resident footprint silently shrinks to almost
     nothing. The first version only re-touched a fixed 64MB window (an
     off-by-modulo bug), so ~99% of the ballast evaporated after the initial
     fill. This version touches ONE byte of every page each cycle (a cheap
     strided read via numpy) to keep the whole buffer referenced/resident.

Usage:
    python tools/ballast.py --gb 8
"""

import argparse
import os
import signal
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gb", type=float, required=True)
    parser.add_argument("--refresh-interval", type=float, default=1.0)
    args = parser.parse_args()

    stop = False

    def _handle_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    n_bytes = int(args.gb * 1024**3)
    if n_bytes == 0:
        print("[ballast] 0 GB requested; nothing to hold.", flush=True)
        return

    fill_chunk = 64 * 1024 * 1024          # 64MB chunks for the initial fill
    refresh_chunk = 256 * 1024 * 1024      # rewrite this much fresh randomness each cycle
    page = 16 * 1024                       # macOS page size on Apple Silicon

    buf = bytearray(n_bytes)
    print(f"[ballast] allocating {args.gb} GB, filling with random data...", flush=True)
    pos = 0
    while pos < n_bytes and not stop:
        end = min(pos + fill_chunk, n_bytes)
        buf[pos:end] = os.urandom(end - pos)
        pos = end

    print(f"[ballast] filled. PID={os.getpid()}. Holding until terminated.", flush=True)
    sys.stdout.flush()

    # strided view: one uint8 per page, summed each cycle to force residency
    touch_view = np.frombuffer(buf, dtype=np.uint8)[::page]
    write_pos = 0
    sink = 0
    while not stop:
        # (a) keep the WHOLE buffer resident by referencing every page
        sink += int(touch_view.sum())
        # (b) keep a rotating window incompressible / non-dedupable
        end = min(write_pos + refresh_chunk, n_bytes)
        buf[write_pos:end] = os.urandom(end - write_pos)
        write_pos = end % n_bytes
        # sleep in small slices so SIGTERM is responsive
        slept = 0.0
        while slept < args.refresh_interval and not stop:
            time.sleep(min(0.2, args.refresh_interval - slept))
            slept += 0.2

    print(f"[ballast] releasing (touch checksum {sink & 0xffff}).", flush=True)


if __name__ == "__main__":
    main()
