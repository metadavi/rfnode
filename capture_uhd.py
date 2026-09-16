#!/usr/bin/env python3
"""
Capture IQ from a USRP B210 with fixed gain, clip detection, overrun accounting
and a JSON metadata sidecar.

  ./capture_uhd.py --freq 2402 --gain 30 --rate 61.44 --secs 10 --out run1
  ./capture_uhd.py --freq 2402 --gain 30 --rate 61.44 --secs 20 --out /dev/shm/run2
  ./capture_uhd.py --freq 2402 --gain 30 --channels 0,1 --out dual   # coherent pair

Note on clipping: UHD presents the B210's 12-bit ADC as a filtered 16-bit sc16
stream, so "railed" here means DIGITAL saturation at +/-32767, downstream of the
converter. Keep real headroom; do not run at the edge.
"""
import argparse, json, os, sys, time, datetime
import numpy as np
import uhd

RAIL = 32767
FULL = 32768.0


def dbfs(x):
    return 20.0 * np.log10(max(float(x), 1e-9) / FULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=float, required=True, help="center MHz")
    ap.add_argument("--gain", type=float, required=True, help="RX gain dB (0-76)")
    ap.add_argument("--rate", type=float, default=20.0, help="sample rate MSPS")
    ap.add_argument("--bw", type=float, default=None, help="analog bandwidth MHz")
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("--channels", default="0", help="0 or 0,1 for coherent pair")
    ap.add_argument("--out", required=True, help="output basename")
    ap.add_argument("--nsamps", type=int, default=32768, help="samples per recv buffer")
    ap.add_argument("--reject-clipped", action="store_true")
    ap.add_argument("--args", default="", help="UHD device args")
    ap.add_argument("--stride", type=int, default=16,
                    help="analyse every Nth sample for envelope stats "
                         "(clip detection is always on every sample)")
    a = ap.parse_args()

    chans = [int(c) for c in a.channels.split(",")]
    bw = a.bw if a.bw else min(a.rate, 56.0)

    usrp = uhd.usrp.MultiUSRP(a.args)
    for c in chans:
        usrp.set_rx_rate(a.rate * 1e6, c)
        usrp.set_rx_freq(uhd.types.TuneRequest(a.freq * 1e6), c)
        usrp.set_rx_gain(a.gain, c)
        try:
            usrp.set_rx_bandwidth(bw * 1e6, c)
        except Exception:
            pass

    actual = {
        "sample_rate_hz": float(usrp.get_rx_rate(chans[0])),
        "center_freq_hz": float(usrp.get_rx_freq(chans[0])),
        "gain_db": float(usrp.get_rx_gain(chans[0])),
        "bandwidth_hz": float(usrp.get_rx_bandwidth(chans[0])),
        "antenna": usrp.get_rx_antenna(chans[0]),
        "channels": chans,
    }
    info = usrp.get_usrp_rx_info(chans[0])
    hw = {k: str(info[k]) for k in ("mboard_id", "mboard_serial", "rx_subdev_name")
          if k in info}

    sa = uhd.usrp.StreamArgs("sc16", "sc16")
    sa.channels = chans
    streamer = usrp.get_rx_stream(sa)
    md = uhd.types.RXMetadata()
    buf = np.zeros((len(chans), a.nsamps), dtype=np.int32)

    paths = [("%s.ch%d.iq" % (a.out, c)) if len(chans) > 1 else (a.out + ".iq")
             for c in chans]
    for p in paths:
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s" % p)
    files = [open(p, "wb") for p in paths]

    hist = np.zeros(1025, dtype=np.int64)
    peak = 0.0
    axis_peak = 0
    total_samples = 0        # TRUE sample count (hist only holds the decimated subset)
    kept = rejected = overruns = late = 0
    clipped_written = clipped_observed = 0
    written = 0

    streamer.issue_stream_cmd(
        uhd.types.StreamCMD(uhd.types.StreamMode.start_cont))
    t_iso = datetime.datetime.now().astimezone().isoformat()
    t0 = time.time()
    try:
        while time.time() - t0 < a.secs:
            n = streamer.recv(buf, md, 3.0)
            ec = md.error_code
            if ec == uhd.types.RXMetadataErrorCode.overflow:
                overruns += 1
                continue
            if ec == uhd.types.RXMetadataErrorCode.late:
                late += 1
                continue
            if ec != uhd.types.RXMetadataErrorCode.none or n == 0:
                continue

            iq0 = buf[0, :n].view(np.int16)

            # full-rate clip check: one pass over the raw int16 view, covers I and Q
            absmax = int(np.abs(iq0).max())
            if absmax >= RAIL:
                nclip = int(np.count_nonzero(np.abs(iq0) >= RAIL))
            else:
                nclip = 0
            clipped_observed += nclip
            if nclip and a.reject_clipped:
                rejected += 1
                continue

            # write BEFORE analysing so the stream is the priority
            for i, f in enumerate(files):
                f.write(buf[i, :n].tobytes())
                written += n * 4
            kept += 1
            total_samples += n
            clipped_written += nclip
            if absmax > axis_peak:
                axis_peak = absmax

            # envelope stats on a decimated subset (cheap, statistically fine)
            Is = iq0[0::2][::a.stride].astype(np.int64)
            Qs = iq0[1::2][::a.stride].astype(np.int64)
            if Is.size:
                pw = Is * Is + Qs * Qs
                m = float(np.sqrt(float(pw.max())))
                if m > peak:
                    peak = m
                hist += np.bincount(
                    np.minimum((np.sqrt(pw.astype(np.float64)) / 32).astype(int), 1024),
                    minlength=1025)
    finally:
        streamer.issue_stream_cmd(
            uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))
        for f in files:
            f.close()

    elapsed = time.time() - t0
    total = total_samples                 # true samples written per channel
    n_stats = int(hist.sum())             # decimated count, for percentiles only
    delivered = (total + rejected * a.nsamps) / elapsed if elapsed else 0.0
    ratio = delivered / actual["sample_rate_hz"] if actual["sample_rate_hz"] else 0
    contiguous = ratio > 0.95 and overruns == 0

    c = np.cumsum(hist)

    def pk(p):
        return float(np.searchsorted(c, n_stats * p / 100.0) * 32) if n_stats else 0.0

    meta = {
        "files": [os.path.basename(p) for p in paths],
        "format": "int16 interleaved I,Q (little-endian), one file per channel",
        "device": "USRP B210",
        "started": t_iso,
        "duration_s": round(elapsed, 3),
        "requested": {"freq_mhz": a.freq, "gain_db": a.gain,
                      "rate_msps": a.rate, "bw_mhz": bw},
        "actual": actual,
        "hardware": hw,
        "samples_per_channel": total,
        "samples_analysed_for_stats": n_stats,
        "bytes_total": written,
        "throughput": {
            "requested_rate_hz": int(actual["sample_rate_hz"]),
            "delivered_rate_hz": int(delivered),
            "capture_ratio": round(ratio, 4),
            "overruns": overruns,
            "late_packets": late,
            "contiguous": bool(contiguous),
            "_note": "contiguous=false or overruns>0 means samples were LOST; "
                     "the file is not a continuous time series.",
        },
        "buffers": {"kept": kept, "rejected": rejected, "size": a.nsamps,
                    "reject_clipped": bool(a.reject_clipped)},
        "clipping": {
            "clipped_samples_written": clipped_written,
            "clipped_samples_observed": clipped_observed,
            "clean": clipped_written == 0,
            "_note": "rail = +/-32767 (digital saturation of the sc16 stream, "
                     "downstream of the 12-bit ADC)",
        },
        "envelope": {
            "_note": "peak_axis_raw is exact (every sample). Envelope percentiles "
                     "and peak_raw come from every Nth sample (see analysis_stride).",
            "analysis_stride": a.stride,
            "peak_axis_raw": int(axis_peak),
            "peak_axis_dbfs": round(dbfs(axis_peak), 2),
            "peak_raw": peak,
            "peak_dbfs": round(dbfs(peak), 2),
            "p50_dbfs": round(dbfs(pk(50)), 2),
            "p99_dbfs": round(dbfs(pk(99)), 2),
            "p99_9_dbfs": round(dbfs(pk(99.9)), 2),
            "headroom_db": round(-dbfs(peak), 2),
        },
    }
    with open(a.out + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    mb = written / 1e6
    print("wrote %s  (%.1f MB total, %d samples/channel)"
          % (", ".join(os.path.basename(p) for p in paths), mb, total))
    print("  gain / freq    : %.1f dB  /  %.4f MHz"
          % (actual["gain_db"], actual["center_freq_hz"] / 1e6))
    print("  rate           : %.3f MSPS requested, %.3f delivered (%.1f%%)"
          % (actual["sample_rate_hz"] / 1e6, delivered / 1e6, 100 * ratio))
    print("  disk rate      : %.1f MB/s" % (mb / elapsed if elapsed else 0))
    print("  overruns       : %d    late: %d" % (overruns, late))
    print("  buffers kept   : %d    rejected: %d" % (kept, rejected))
    print("  clipped written: %d  (%s)"
          % (clipped_written, "CLEAN" if clipped_written == 0 else "CONTAMINATED"))
    print("  peak envelope  : %.1f dBFS   headroom %.1f dB"
          % (meta["envelope"]["peak_dbfs"], meta["envelope"]["headroom_db"]))
    print("  peak axis      : %d (%.1f dBFS, exact over every sample)"
          % (axis_peak, dbfs(axis_peak)))
    if not contiguous:
        print()
        print("  " + "!" * 62)
        print("  WARNING: %d overruns, %.1f%% of requested rate delivered."
              % (overruns, 100 * ratio))
        print("  Samples were LOST. Not a continuous time series.")
        print("  Lower --rate, or write to /dev/shm (RAM) instead of disk.")
        print("  " + "!" * 62)


if __name__ == "__main__":
    main()
