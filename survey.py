#!/usr/bin/env python3
"""Low-gain spectrum survey: find the strong emitters before choosing capture gain."""
import argparse
import numpy as np
import iio

URI = "ip:192.168.3.1"
FULL_SCALE = 2048.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, required=True, help="start freq MHz")
    ap.add_argument("--stop", type=float, required=True, help="stop freq MHz")
    ap.add_argument("--gain", type=float, default=0.0, help="manual RX gain dB")
    ap.add_argument("--fs", type=float, default=30.72, help="sample rate MSPS")
    ap.add_argument("--n", type=int, default=32768, help="samples per step")
    ap.add_argument("--reps", type=int, default=4, help="captures per step")
    a = ap.parse_args()

    ctx = iio.Context(URI)
    phy = ctx.find_device("ad9361-phy")
    rxd = ctx.find_device("cf-ad9361-lpc")

    rxch = phy.find_channel("voltage0", False)
    rxch.attrs["gain_control_mode"].value = "manual"
    rxch.attrs["hardwaregain"].value = str(a.gain)
    rxch.attrs["sampling_frequency"].value = str(int(a.fs * 1e6))
    rxch.attrs["rf_bandwidth"].value = str(int(a.fs * 1e6 * 0.8))
    lo = phy.find_channel("altvoltage0", True)

    for c in ("voltage0", "voltage1"):
        rxd.find_channel(c).enabled = True
    buf = iio.Buffer(rxd, a.n, False)

    step = a.fs * 0.8
    freqs = np.arange(a.start, a.stop + step, step)
    print("# gain=%.1f dB  fs=%.2f MSPS  %d samples x%d per step"
          % (a.gain, a.fs, a.n, a.reps))
    print("#  center MHz    peak MHz   pk dBFS  floor dBFS   SNR dB  clip%")

    results = []
    for f_mhz in freqs:
        lo.attrs["frequency"].value = str(int(f_mhz * 1e6))
        best = None
        clipped_total = 0
        n_total = 0
        for r in range(a.reps):
            buf.refill()
            raw = np.frombuffer(buf.read(), dtype=np.int16)
            if r == 0:
                continue
            I = raw[0::2].astype(np.float32)
            Q = raw[1::2].astype(np.float32)
            clipped_total += int(np.sum((np.abs(I) >= 2047) | (np.abs(Q) >= 2047)))
            n_total += len(I)
            iq = (I + 1j * Q) / FULL_SCALE
            w = np.hanning(len(iq))
            S = np.fft.fftshift(np.abs(np.fft.fft(iq * w)) ** 2) / (len(iq) ** 2)
            best = S if best is None else np.maximum(best, S)
        fr = np.fft.fftshift(np.fft.fftfreq(len(best), 1.0 / a.fs))
        pk = int(np.argmax(best))
        pk_db = 10 * np.log10(best[pk] + 1e-20)
        floor_db = 10 * np.log10(np.median(best) + 1e-20)
        clip_pct = 100.0 * clipped_total / max(n_total, 1)
        print("  %11.2f  %10.3f  %8.1f  %10.1f  %7.1f  %5.2f"
              % (f_mhz, f_mhz + fr[pk], pk_db, floor_db, pk_db - floor_db, clip_pct))
        results.append((f_mhz + fr[pk], pk_db, clip_pct))

    print()
    results.sort(key=lambda r: -r[1])
    print("# strongest emitters found:")
    for f, p, c in results[:8]:
        print("   %10.3f MHz   %7.1f dBFS   clip %.2f%%" % (f, p, c))
    if results:
        hottest = results[0][1]
        print()
        print("# loudest peak at gain %.1f dB: %.1f dBFS" % (a.gain, hottest))
        print("# headroom to full scale: %.1f dB" % (-hottest,))


if __name__ == "__main__":
    main()
