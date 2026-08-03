"""
tools/ballast.py

Allocate and hold `--gb` gigabytes of memory, refreshing a slice of it
periodically with random data so macOS's memory compressor can't
trivially compress it away (which would defeat the point of creating
real memory pressure -- compressed ballast wouldn't force the swapping
we're trying to study).

Run as a background process; SIGTERM/SIGINT releases the memory and exits.

Usage:
    python tools/ballast.py --gb 8
"""

import argparse
import os
import signal
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gb", type=float, required=True)
    parser.add_argument("--refresh-interval", type=float, default=10.0)
    args = parser.parse_args()

    stop = False

    def _handle_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    n_bytes = int(args.gb * 1024**3)
    chunk = 64 * 1024 * 1024  # 64MB chunks, so filling doesn't stall on one huge write
    buf = bytearray(n_bytes)

    print(f"[ballast] allocating {args.gb} GB, filling with random data...", flush=True)
    pos = 0
    while pos < n_bytes and not stop:
        end = min(pos + chunk, n_bytes)
        buf[pos:end] = os.urandom(end - pos)
        pos = end

    print(f"[ballast] filled. PID={os.getpid()}. Holding until terminated.", flush=True)
    sys.stdout.flush()

    refresh_chunk = min(chunk, n_bytes) if n_bytes > 0 else 0
    while not stop:
        time.sleep(args.refresh_interval)
        if stop or refresh_chunk == 0:
            break
        start = int(time.time() * 1000) % max(1, n_bytes - refresh_chunk)
        buf[start:start + refresh_chunk] = os.urandom(refresh_chunk)

    print("[ballast] releasing.", flush=True)


if __name__ == "__main__":
    main()
