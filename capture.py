#!/usr/bin/env python3
"""
Capture IQ from a PlutoSDR with fixed gain, per-buffer clip rejection,
and a JSON metadata sidecar so the data is still interpretable later.

  ./capture.py --freq 2402 --gain 20 --secs 5 --out walk1
    -> walk1.iq   (int16 interleaved I,Q)
    -> walk1.json (settings + envelope stats + clip accounting)
"""
import argparse, json, os, sys, time, datetime
import numpy as np
import iio

URI = "ip:192.168.3.1"
AXIS_FULL = 2048.0
RAIL = 2047


def db(x):
    return 20.0 * np.log10(np.maximum(x, 1e-12) / AXIS_FULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=float, required=True, help="center MHz")
    ap.add_argument("--gain", type=float, required=True, help="manual RX gain dB")
    ap.add_argument("--secs", type=float, default=5.0, help="capture seconds")
    ap.add_argument("--fs", type=float, default=5.0,
                    help="sample rate MSPS (default 5.0 = max the Pluto sustains over USB2)")
    ap.add_argument("--bw", type=float, default=None, help="RF bandwidth MHz (default 0.8*fs)")
    ap.add_argument("--n", type=int, default=32768, help="samples per buffer")
    ap.add_argument("--out", required=True, help="output basename (no extension)")
    ap.add_argument("--reject-clipped", action="store_true",
                    help="discard any buffer containing railed samples instead of writing it")
    ap.add_argument("--uri", default=URI)
    a = ap.parse_args()

    bw = a.bw if a.bw else a.fs * 0.8
    iq_path = a.out + ".iq"
    js_path = a.out + ".json"
    if os.path.exists(iq_path):
        sys.exit("refusing to overwrite existing %s" % iq_path)

    ctx = iio.Context(a.uri)
    phy = ctx.find_device("ad9361-phy")
    rxd = ctx.find_device("cf-ad9361-lpc")

    rxch = phy.find_channel("voltage0", False)
    rxch.attrs["gain_control_mode"].value = "manual"
    rxch.attrs["hardwaregain"].value = str(a.gain)
    rxch.attrs["sampling_frequency"].value = str(int(a.fs * 1e6))
    rxch.attrs["rf_bandwidth"].value = str(int(bw * 1e6))
    phy.find_channel("altvoltage0", True).attrs["frequency"].value = str(int(a.freq * 1e6))

    # read back what the hardware actually accepted (it quantises requests)
    actual = {
        "gain_db": float(rxch.attrs["hardwaregain"].value.split()[0]),
        "gain_control_mode": rxch.attrs["gain_control_mode"].value,
        "sample_rate_hz": int(rxch.attrs["sampling_frequency"].value),
        "rf_bandwidth_hz": int(rxch.attrs["rf_bandwidth"].value),
        "center_freq_hz": int(phy.find_channel("altvoltage0", True).attrs["frequency"].value),
    }
    hw = {k: ctx.attrs[k] for k in ("hw_model", "hw_serial", "fw_version") if k in ctx.attrs}

    for c in ("voltage0", "voltage1"):
        rxd.find_channel(c).enabled = True
    buf = iio.Buffer(rxd, a.n, False)

    buf.refill(); buf.read()          # discard first: LO/gain still settling

    hist = np.zeros(2049, dtype=np.int64)   # envelope histogram, 0..2048+
    peak = 0.0
    kept = rejected = 0
    clipped_written = 0      # railed samples inside data actually written
    clipped_observed = 0     # railed samples seen, including rejected buffers
    written = 0
    t0 = time.time()
    t_start_iso = datetime.datetime.now().astimezone().isoformat()

    with open(iq_path, "wb") as f:
        while time.time() - t0 < a.secs:
            buf.refill()
            blk = buf.read()
            raw = np.frombuffer(blk, dtype=np.int16)
            I = raw[0::2].astype(np.float32)
            Q = raw[1::2].astype(np.float32)
            nclip = int(np.count_nonzero((np.abs(I) >= RAIL) | (np.abs(Q) >= RAIL)))

            clipped_observed += nclip
            if nclip and a.reject_clipped:
                rejected += 1
                continue

            f.write(blk)
            written += len(blk)
            kept += 1
            clipped_written += nclip
            env = np.sqrt(I * I + Q * Q)
            m = float(env.max())
            if m > peak:
                peak = m
            hist += np.bincount(np.minimum(env.astype(int), 2048), minlength=2049)

    elapsed = time.time() - t0
    total_samp = int(hist.sum())
    # samples the radio actually handed us, including ones we rejected
    delivered_samp = total_samp + rejected * a.n
    delivered_rate = delivered_samp / elapsed if elapsed > 0 else 0.0
    requested_rate = float(actual["sample_rate_hz"])
    capture_ratio = delivered_rate / requested_rate if requested_rate else 0.0
    contiguous = capture_ratio > 0.95

    def pct(p):
        if total_samp == 0:
            return None
        c = np.cumsum(hist)
        return float(np.searchsorted(c, total_samp * p / 100.0))

    meta = {
        "file": os.path.basename(iq_path),
        "format": "int16 interleaved I,Q (little-endian)",
        "started": t_start_iso,
        "duration_s": round(elapsed, 3),
        "requested": {"freq_mhz": a.freq, "gain_db": a.gain, "fs_msps": a.fs, "bw_mhz": bw},
        "actual": actual,
        "hardware": hw,
        "samples": total_samp,
        "bytes": written,
        "throughput": {
            "requested_rate_hz": int(requested_rate),
            "delivered_rate_hz": int(delivered_rate),
            "capture_ratio": round(capture_ratio, 4),
            "contiguous": bool(contiguous),
            "_note": ("contiguous=false means the link could not carry the requested rate; "
                      "each buffer is internally contiguous but there are GAPS BETWEEN "
                      "buffers. Per-buffer FFTs remain valid; do not treat the file as a "
                      "continuous time series."),
        },
        "buffers": {"kept": kept, "rejected": rejected, "size": a.n,
                    "reject_clipped": bool(a.reject_clipped)},
        "clipping": {
            "clipped_samples_written": clipped_written,
            "clipped_samples_observed": clipped_observed,
            "clip_pct_written": round(100.0 * clipped_written / max(total_samp, 1), 6),
            "clip_pct_observed": round(
                100.0 * clipped_observed / max(total_samp + rejected * a.n, 1), 6),
            "clean": clipped_written == 0,
        },
        "envelope": {
            "_note": "dBFS referenced to one axis (2048). Envelope may legitimately "
                     "reach +3.0 dBFS (the corner, 2048*sqrt(2)) without clipping.",
            "peak_raw": peak, "peak_dbfs": round(float(db(peak)), 2),
            "p50_dbfs": round(float(db(pct(50) or 1)), 2),
            "p99_dbfs": round(float(db(pct(99) or 1)), 2),
            "p99_9_dbfs": round(float(db(pct(99.9) or 1)), 2),
            "headroom_db": round(float(-db(peak)), 2),
        },
    }
    with open(js_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("wrote %s  (%.1f MB, %d samples)" % (iq_path, written / 1e6, total_samp))
    print("wrote %s" % js_path)
    print()
    print("  gain           : %.1f dB (manual)" % actual["gain_db"])
    print("  center         : %.3f MHz" % (actual["center_freq_hz"] / 1e6))
    print("  sample rate    : %.3f MSPS" % (actual["sample_rate_hz"] / 1e6))
    print("  buffers kept   : %d    rejected: %d" % (kept, rejected))
    print("  clipped (written): %d  (%s)" % (clipped_written,
          "CLEAN" if clipped_written == 0 else "CONTAMINATED"))
    if rejected:
        print("  clipped (rejected buffers): %d  - not written to disk" % clipped_observed)
    print("  peak envelope  : %.1f dBFS   headroom %.1f dB"
          % (meta["envelope"]["peak_dbfs"], meta["envelope"]["headroom_db"]))
    print("  p50 / p99 / p99.9: %.1f / %.1f / %.1f dBFS"
          % (meta["envelope"]["p50_dbfs"], meta["envelope"]["p99_dbfs"],
             meta["envelope"]["p99_9_dbfs"]))
    print("  delivered rate : %.2f MSPS of %.2f requested (%.1f%%)"
          % (delivered_rate / 1e6, requested_rate / 1e6, 100 * capture_ratio))
    if not contiguous:
        print()
        print("  " + "!" * 66)
        print("  WARNING: the link delivered only %.1f%% of the requested rate."
              % (100 * capture_ratio))
        print("  This file has GAPS BETWEEN BUFFERS and is NOT a continuous")
        print("  time series. Per-buffer FFTs are fine; demodulation, packet")
        print("  timing and duty-cycle analysis will be WRONG.")
        print("  Lower --fs (Pluto over USB2 sustains about 5 MSPS).")
        print("  " + "!" * 66)


if __name__ == "__main__":
    main()
