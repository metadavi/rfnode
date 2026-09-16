#!/usr/bin/env python3
"""
Low-gain spectrum survey on a USRP B210, with offset tuning.

The DC offset / LO-leakage spike sits permanently at 0 Hz baseband, i.e. at
whatever frequency the LO is tuned to. Rather than try to cancel it, we simply
never analyse the part of the spectrum it lives in:

  * a guard band of +/- --guard MHz around DC is excluded from analysis
  * the band edges are excluded too (the analog filter rolls off there)
  * steps overlap enough that the region blanked in one tile is covered by
    its neighbours, so nothing is actually missed

Hardware DC correction is also enabled, as cheap insurance for the residue.
"""
import argparse
import numpy as np
import uhd

RAIL = 32767
FULL = 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, required=True, help="start MHz")
    ap.add_argument("--stop", type=float, required=True, help="stop MHz")
    ap.add_argument("--gain", type=float, default=20.0, help="RX gain dB (0-76)")
    ap.add_argument("--rate", type=float, default=61.44, help="MSPS")
    ap.add_argument("--n", type=int, default=32768, help="FFT / samples per step")
    ap.add_argument("--reps", type=int, default=3, help="captures per step")
    ap.add_argument("--guard", type=float, default=2.0,
                    help="MHz around DC to exclude (offset tuning guard band)")
    ap.add_argument("--edge", type=float, default=0.45,
                    help="fraction of rate to keep each side (filter rolloff)")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--no-offset-tune", action="store_true",
                    help="disable the DC guard band")
    ap.add_argument("--no-dc-correction", action="store_true",
                    help="disable UHD hardware DC cancellation")
    a = ap.parse_args()

    usrp = uhd.usrp.MultiUSRP("")
    usrp.set_rx_rate(a.rate * 1e6, 0)
    usrp.set_rx_gain(a.gain, 0)
    try:
        usrp.set_rx_bandwidth(min(a.rate, 56.0) * 1e6, 0)
    except Exception:
        pass
    try:
        usrp.set_rx_dc_offset(not a.no_dc_correction, 0)
    except Exception:
        pass
    rate = usrp.get_rx_rate(0)

    sa = uhd.usrp.StreamArgs("sc16", "sc16")
    sa.channels = [0]
    st = usrp.get_rx_stream(sa)
    md = uhd.types.RXMetadata()
    buf = np.zeros((1, a.n), dtype=np.int32)

    guard_hz = 0.0 if a.no_offset_tune else a.guard * 1e6
    edge_hz = a.edge * rate
    # step small enough that each tile's blanked DC region is covered by a neighbour
    step = (edge_hz - guard_hz) * 0.9 / 1e6
    freqs = np.arange(a.start, a.stop + step, step)

    print("# B210 survey  gain=%.1f dB  rate=%.2f MSPS  %d steps" % (a.gain, rate / 1e6, len(freqs)))
    if guard_hz:
        print("# offset tuning ON: excluding +/-%.1f MHz around DC and beyond +/-%.1f MHz"
              % (guard_hz / 1e6, edge_hz / 1e6))
    else:
        print("# offset tuning OFF: DC artifacts will appear at each tile centre")
    print("# hardware DC cancellation: %s" % ("OFF" if a.no_dc_correction else "ON"))
    print("#  LO MHz      peak MHz   pk dBFS  floor dBFS   SNR dB   DC dBFS  clip%")

    results = []
    for f_mhz in freqs:
        usrp.set_rx_freq(uhd.types.TuneRequest(f_mhz * 1e6), 0)
        lo = usrp.get_rx_freq(0)
        best = None
        clipped = ntot = errs = 0
        cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        cmd.num_samps = a.n * (a.reps + 2)
        cmd.stream_now = True
        st.issue_stream_cmd(cmd)
        got = 0
        want = a.reps + 2
        for _ in range(want):
            n = st.recv(buf, md, 3.0)
            if md.error_code != uhd.types.RXMetadataErrorCode.none or n != a.n:
                errs += 1
                continue
            got += 1
            if got == 1:
                continue                      # discard: LO settling
            iq = buf[0, :n].view(np.int16)
            clipped += int(np.count_nonzero(np.abs(iq) >= RAIL))
            ntot += n
            z = (iq[0::2].astype(np.float32) + 1j * iq[1::2].astype(np.float32)) / FULL
            w = np.hanning(len(z))
            S = np.fft.fftshift(np.abs(np.fft.fft(z * w)) ** 2) / (len(z) ** 2)
            best = S if best is None else np.maximum(best, S)
        # drain any residue of this burst so the next step starts clean
        for _ in range(8):
            if st.recv(buf, md, 0.05) == 0:
                break

        if best is None:
            print("  %9.2f   -- no valid data (%d recv errors)" % (lo / 1e6, errs))
            continue

        fr = np.fft.fftshift(np.fft.fftfreq(len(best), 1.0 / rate))
        dc_bin = int(np.argmin(np.abs(fr)))
        dc_db = 10 * np.log10(best[dc_bin] + 1e-20)

        valid = (np.abs(fr) > guard_hz) & (np.abs(fr) < edge_hz)
        if not valid.any():
            continue
        Sv = np.where(valid, best, 0.0)
        pk = int(np.argmax(Sv))
        pk_db = 10 * np.log10(best[pk] + 1e-20)
        fl_db = 10 * np.log10(np.median(best[valid]) + 1e-20)
        cp = 100.0 * clipped / max(ntot * 2, 1)
        abs_pk = (lo + fr[pk]) / 1e6
        print("  %9.2f  %10.3f  %8.1f  %10.1f  %7.1f  %8.1f  %5.2f"
              % (lo / 1e6, abs_pk, pk_db, fl_db, pk_db - fl_db, dc_db, cp))
        results.append((abs_pk, pk_db, cp))

    print()
    results.sort(key=lambda r: -r[1])
    print("# strongest emitters (DC-excluded):")
    shown = []
    for f, p, c in results:
        if len(shown) >= a.top:
            break
        if all(abs(f - s) > 1.0 for s in shown):
            shown.append(f)
            print("   %10.3f MHz   %7.1f dBFS   clip %.2f%%" % (f, p, c))
    if results:
        print()
        print("# loudest %.1f dBFS at gain %.1f dB" % (results[0][1], a.gain))
        print("# NOTE: a peak seen at the same absolute frequency from two different")
        print("#       LO settings is real. Re-run with a different --rate to test.")


if __name__ == "__main__":
    main()
