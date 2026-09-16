#!/usr/bin/env python3
"""Long peak-hold dwell on a USRP B210: catch bursty transmitters a sweep misses."""
import argparse, time
import numpy as np
import uhd

RAIL = 32767
FULL = 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=float, required=True, help="center MHz")
    ap.add_argument("--gain", type=float, default=20.0)
    ap.add_argument("--secs", type=float, default=25.0)
    ap.add_argument("--rate", type=float, default=61.44, help="MSPS")
    ap.add_argument("--n", type=int, default=32768)
    ap.add_argument("--stride", type=int, default=8)
    a = ap.parse_args()

    usrp = uhd.usrp.MultiUSRP("")
    usrp.set_rx_rate(a.rate * 1e6, 0)
    usrp.set_rx_gain(a.gain, 0)
    usrp.set_rx_freq(uhd.types.TuneRequest(a.freq * 1e6), 0)
    try:
        usrp.set_rx_bandwidth(min(a.rate, 56.0) * 1e6, 0)
    except Exception:
        pass
    rate = usrp.get_rx_rate(0)

    sa = uhd.usrp.StreamArgs("sc16", "sc16")
    sa.channels = [0]
    st = usrp.get_rx_stream(sa)
    md = uhd.types.RXMetadata()
    buf = np.zeros((1, a.n), dtype=np.int32)

    hist = np.zeros(1025, dtype=np.int64)
    peak = 0
    clipped = 0
    total = 0
    nbuf = 0
    bursts = 0
    overruns = 0

    print("# B210 dwell %.1f s at %.3f MHz, gain %.1f dB, rate %.2f MSPS"
          % (a.secs, a.freq, a.gain, rate / 1e6))
    st.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.start_cont))
    t0 = time.time()
    try:
        while time.time() - t0 < a.secs:
            n = st.recv(buf, md, 2.0)
            if md.error_code == uhd.types.RXMetadataErrorCode.overflow:
                overruns += 1
                continue
            if md.error_code != uhd.types.RXMetadataErrorCode.none or n == 0:
                continue
            iq = buf[0, :n].view(np.int16)
            am = int(np.abs(iq).max())
            if am >= RAIL:
                clipped += int(np.count_nonzero(np.abs(iq) >= RAIL))
            if am > peak:
                peak = am
            total += n
            nbuf += 1
            I = iq[0::2][::a.stride].astype(np.int64)
            Q = iq[1::2][::a.stride].astype(np.int64)
            env = np.sqrt((I * I + Q * Q).astype(np.float64))
            med = float(np.median(env)) + 1e-9
            if float(env.max()) / med > 10.0:
                bursts += 1
            hist += np.bincount(np.minimum((env / 32).astype(int), 1024), minlength=1025)
    finally:
        st.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))

    el = time.time() - t0
    ns = int(hist.sum())
    print("# buffers %d  samples %d  elapsed %.1f s  duty %.1f%%  overruns %d"
          % (nbuf, total, el, 100.0 * total / (rate * el), overruns))
    print()
    print("peak axis value : %d  (%.1f dBFS)" % (peak, 20 * np.log10(peak / FULL + 1e-12)))
    print("clipped samples : %d / %d  (%.5f%%)" % (clipped, total * 2,
                                                   100.0 * clipped / max(total * 2, 1)))
    print("burst buffers   : %d / %d  (peak >20 dB over local median)" % (bursts, nbuf))
    print()
    cum = np.cumsum(hist[::-1])[::-1]
    print("envelope exceedance:")
    for i in range(1024, -1, -1):
        if cum[i] == 0 or ns == 0:
            continue
        lvl = 20 * np.log10(max(i * 32, 1) / FULL)
        frac = 100.0 * cum[i] / ns
        if frac < 1e-4 or int(round(lvl)) % 10 != 0:
            continue
        print("   above %6.1f dBFS : %9.5f%%" % (lvl, frac))
    print()
    hd = -20 * np.log10(peak / FULL + 1e-12)
    print("# headroom from observed peak: %.1f dB" % hd)
    print("# suggested gain for 6 dB margin: %.1f dB" % (a.gain + hd - 6.0))


if __name__ == "__main__":
    main()
