# rfnode

RF spectrum sampling tools for a Raspberry Pi 5 with an **ADALM-PLUTO** and a
**USRP B210**, built for spectrum survey work and for collecting clean IQ
datasets suitable for machine learning.

Every number in this README was measured on the hardware, not read off a
datasheet. Several of them contradict the datasheet, which is the point.

## Tools

| | Pluto (libiio) | B210 (UHD) |
|---|---|---|
| Survey a frequency range | `survey.py` | `survey_uhd.py` |
| Long dwell on one frequency | `peakhold.py` | `peakhold_uhd.py` |
| Record IQ with metadata | `capture.py` | `capture_uhd.py` |

Intended workflow: **survey** to find what is on the air, **dwell** to learn how
strong it gets over time (including bursts), then **capture** with a gain chosen
from those measurements.

## Measured limits

These matter more than the advertised specs.

| | ADALM-PLUTO | USRP B210 |
|---|---|---|
| Tuning range | 70 MHz – 6 GHz | **50 MHz – 6 GHz** |
| ADC | 12-bit | 12-bit |
| RX channels | 1 | **2, phase-coherent** |
| Host link | USB 2.0 | USB 3.0 |
| Rate the chip reports | 61.44 MSPS | 61.44 MSPS |
| **Rate actually sustained** | **~5 MSPS** | **61.44 MSPS** |

**The Pluto's advertised 61.44 MSPS is not achievable over USB 2.0.** The radio
digitises at whatever rate you ask for, but the link cannot carry it and samples
are silently dropped:

```
requested   delivered   verdict
  2.50        2.50      continuous
  5.00        5.00      continuous
 10.00        5.58      DROPPING 44%
 30.72        5.56      DROPPING 82%
```

A file captured at 30.72 MSPS on a Pluto contains contiguous *buffers* with
unknown gaps *between* them. Per-buffer FFTs stay valid; demodulation, packet
timing and duty-cycle analysis do not. `capture.py` defaults to 5 MSPS and warns
loudly if the link cannot keep up.

### Storage is the B210 bottleneck, not the host

The Pi 5 sustains the B210 at its full 61.44 MSPS (245.8 MB/s) with zero
overruns. Writing it is the hard part:

| Destination | Sustained | Max rate |
|---|---|---|
| SD card | 52 MB/s | ~13 MSPS (measured usable to ~20 MSPS) |
| `/dev/shm` (RAM) | 245.8 MB/s | **61.44 MSPS**, bounded by RAM size |
| NVMe over PCIe | not yet tested | expected sufficient |

With `/dev/shm` raised to 6 GB, a full-rate capture runs **~26 seconds**.

## Data quality

The tools exist because naive captures produce data that quietly misrepresents
the environment.

**Clipping.** A sample at the ADC rail has had its true value destroyed, and
flat-topped waveforms generate harmonics and intermodulation that were never on
the air. A model trained on clipped data learns the receiver, not the world.
Both capture tools count railed samples per buffer and can reject contaminated
buffers before they reach disk (`--reject-clipped`).

**DC offset / LO leakage.** Direct-conversion receivers produce a permanent
spike at 0 Hz baseband — that is, at whatever frequency you tuned to. Without
mitigation it is reported as a strong emitter at every step of a sweep:

```
# both protections disabled:
  LO MHz      peak MHz   pk dBFS   SNR dB
   600.00     600.002     -60.2     39.7    <== artifact
   624.88     624.885     -61.3     39.4    <== artifact
   ... all 18 steps ...
```

`survey_uhd.py` applies offset tuning by default: a guard band around DC is
excluded from analysis, band edges are dropped where the analog filter rolls
off, and steps overlap so the blanked region of one tile is covered by its
neighbours. Hardware DC cancellation is enabled as well. Hardware cancellation
alone improves the spike by ~29 dB; the guard band removes it from
consideration entirely.

**How to tell an artifact from a signal:** retune the LO. A real transmitter
stays at the same absolute frequency. A DC artifact follows the LO. Spurs often
move when you change sample rate.

## Metadata

Every capture writes a JSON sidecar next to the IQ file recording requested and
**actual** hardware settings (they differ — a request for 2402.000 MHz tunes to
2401.999999 MHz), delivered sample rate, overrun and clipping counts, and
envelope statistics. IQ files without recorded settings are nearly useless
later, because you cannot tell whether a difference between two captures
reflects the environment or your own gain knob.

## Usage

```sh
# what is on the air between 600 and 1000 MHz?
python3 survey_uhd.py --start 600 --stop 1000 --gain 20

# how strong does 461.3 MHz get over 25 seconds, including bursts?
python3 peakhold_uhd.py --freq 461.3 --gain 20 --secs 25

# record 10 s, rejecting any buffer containing clipped samples
python3 capture_uhd.py --freq 2402 --gain 20 --rate 20 --secs 10 \
    --out run1 --reject-clipped

# full rate to RAM (bounded by /dev/shm size)
python3 capture_uhd.py --freq 2402 --gain 20 --rate 61.44 --secs 20 \
    --out /dev/shm/burst1
```

Pluto tools take `--freq`/`--gain` the same way and default to a rate the USB 2.0
link can actually sustain.

## Requirements

Raspberry Pi OS Trixie (64-bit) or any Debian 13 system.

```sh
# Pluto
sudo apt install libiio-utils python3-libiio libad9361-0 python3-ad9361

# B210
sudo apt install uhd-host libuhd4.8.0 python3-uhd
sudo uhd_images_downloader -t b2xx
```

On Debian the UHD images land in a versioned path the library does not search.
If `uhd_find_devices` reports missing `usrp_b200_fw.hex`:

```sh
echo "UHD_IMAGES_DIR=/usr/share/uhd/4.8.0/images" | sudo tee -a /etc/environment
```

Also `python3-numpy`, and `gnuradio` with `libgnuradio-iio3.10.12` if you want
flowgraphs. Note that GNU Radio 3.10 has no dedicated PlutoSDR block — use
**FMComms2/3/4 Source** with URI `ip:192.168.3.1`.

## Never commit IQ data

A 20-second capture at 61.44 MSPS is 4.9 GB. The `.gitignore` excludes raw
sample formats and capture directories. Git keeps history forever, so a repo
that once contained a large file stays bloated after deletion.

## License

GPL-3.0. UHD is GPL-3+, and these tools import it.
