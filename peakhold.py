#!/usr/bin/env python3
"""Long peak-hold dwell: catch bursty transmitters that a fast sweep misses."""
import argparse, time
import numpy as np
import iio

URI = "ip:192.168.3.1"
FS = 2048.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=float, required=True, help="center MHz")
    ap.add_argument("--gain", type=float, default=0.0, help="manual RX gain dB")
    ap.add_argument("--secs", type=float, default=20.0, help="dwell seconds")
    ap.add_argument("--fs", type=float, default=30.72, help="MSPS")
    ap.add_argument("--n", type=int, default=32768, help="samples per buffer")
    a = ap.parse_args()

    ctx = iio.Context(URI)
    phy = ctx.find_device("ad9361-phy")
    rxd = ctx.find_device("cf-ad9361-lpc")

    rxch = phy.find_channel("voltage0", False)
    rxch.attrs["gain_control_mode"].value = "manual"
    rxch.attrs["hardwaregain"].value = str(a.gain)
    rxch.attrs["sampling_frequency"].value = str(int(a.fs * 1e6))
    rxch.attrs["rf_bandwidth"].value = str(int(a.fs * 1e6 * 0.8))
    phy.find_channel("altvoltage0", True).attrs["frequency"].value = str(int(a.freq * 1e6))

    for c in ("voltage0", "voltage1"):
        rxd.find_channel(c).enabled = True
    buf = iio.Buffer(rxd, a.n, False)

    t0 = time.time()
    nbuf = 0
    total = 0
    clipped = 0
    peak_env = 0.0
    hist = np.zeros(64, dtype=np.int64)     # envelope histogram in dBFS bins
    burst_buffers = 0

    print("# dwell %.1f s at %.3f MHz, gain %.1f dB, fs %.2f MSPS"
          % (a.secs, a.freq, a.gain, a.fs))
    while time.time() - t0 < a.secs:
        buf.refill()
        raw = np.frombuffer(buf.read(), dtype=np.int16)
        I = raw[0::2].astype(np.float32)
        Q = raw[1::2].astype(np.float32)
        env = np.sqrt(I * I + Q * Q) / FS
        c = int(np.sum((np.abs(I) >= 2047) | (np.abs(Q) >= 2047)))
        clipped += c
        total += len(I)
        nbuf += 1
        m = float(env.max())
        if m > peak_env:
            peak_env = m
        # a "burst buffer" = any buffer whose peak is >20 dB above its own median
        med = float(np.median(env)) + 1e-9
        if m / med > 10.0:
            burst_buffers += 1
        db = 20 * np.log10(env + 1e-9)
        idx = np.clip(((db + 120) / 2).astype(int), 0, 63)
        hist += np.bincount(idx, minlength=64)

    el = time.time() - t0
    print("# buffers: %d   samples: %d   elapsed: %.1f s   duty observed: %.1f%%"
          % (nbuf, total, el, 100.0 * total / (a.fs * 1e6 * el)))
    print()
    print("peak envelope:   %.1f dBFS" % (20 * np.log10(peak_env + 1e-12),))
    print("clipped samples: %d / %d  (%.4f%%)" % (clipped, total, 100.0 * clipped / max(total, 1)))
    print("burst buffers:   %d / %d  (peak >20 dB over local median)" % (burst_buffers, nbuf))
    print()
    cum = np.cumsum(hist[::-1])[::-1]
    print("envelope exceedance (what fraction of time above each level):")
    for i in range(63, -1, -1):
        if cum[i] == 0:
            continue
        lvl = i * 2 - 120
        frac = 100.0 * cum[i] / total
        if frac < 1e-4:
            continue
        if lvl % 10 == 0 and lvl >= -80:
            print("   above %4d dBFS : %9.5f%% of samples" % (lvl, frac))
    print()
    hdr = -20 * np.log10(peak_env + 1e-12)
    print("# headroom from observed peak to full scale: %.1f dB" % hdr)
    print("# suggested gain for 6 dB safety margin: %.1f dB" % (a.gain + hdr - 6.0))


if __name__ == "__main__":
    main()
