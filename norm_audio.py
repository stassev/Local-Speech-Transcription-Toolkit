#!/usr/bin/env python3
# Copyright (C) 2026 Svetlin Tassev. SPDX-License-Identifier: GPL-3.0.
"""Single-gain EBU R128 loudness normalization for speech recordings.

This tool performs no denoising, filtering, compression, silence removal, VAD,
or dynamic gain. It measures integrated
loudness using pyloudnorm's ITU-R BS.1770-4 implementation, applies one
constant gain, then exports a mono WAV at the requested sample rate.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_RATE = 44100
DEFAULT_TARGET_LUFS = -23.0
DUAL_MONO_CORRECTION_DB = 10.0 * math.log10(2.0)


def require_ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if not binary:
        raise RuntimeError("ffmpeg is required; install it with pacman -S ffmpeg")
    return binary


def run_checked(command: list[str]) -> None:
    print("RUN: " + shlex.join([str(part) for part in command]), flush=True)
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        detail = result.stderr.strip()[-4000:]
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}\n{detail}")


def default_output(source: Path) -> Path:
    return source.with_name(source.stem + "_norm.wav")


def process(args: argparse.Namespace) -> Path:
    import numpy as np
    import pyloudnorm as pyln
    import soundfile as sf

    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    output = Path(args.out).expanduser().resolve() if args.out else default_output(source)
    if output == source:
        raise ValueError("Refusing to overwrite the input file; choose a different -o/--out path")
    if args.sample_rate < 8000 or args.sample_rate > 192000:
        raise ValueError("--sample-rate must be between 8000 and 192000 Hz")
    if not -70.0 <= args.target_lufs <= 0.0:
        raise ValueError("--target-lufs must be between -70 and 0 LUFS")
    if args.peak_ceiling_dbfs is not None and not -20.0 <= args.peak_ceiling_dbfs <= 0.0:
        raise ValueError("--peak-ceiling-dbfs must be between -20 and 0 dBFS")
    if args.subtype not in {"PCM_16", "PCM_24", "FLOAT"}:
        raise ValueError("--subtype must be PCM_16, PCM_24, or FLOAT")

    ffmpeg = require_ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="norm-audio-") as scratch:
        scratch_dir = Path(scratch)
        # Resample for the user's desired WAV rate but preserve the original
        # channel count during loudness measurement, as Audacity does when an
        # effect is applied to a stereo track before a mono export.
        measured_input = scratch_dir / "input_at_output_rate_float.wav"
        run_checked([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source), "-map", "0:a:0", "-vn",
            "-ar", str(args.sample_rate), "-c:a", "pcm_f32le", str(measured_input),
        ])
        audio, sample_rate = sf.read(measured_input, dtype="float32", always_2d=False)
        if audio.size == 0:
            raise ValueError("Input contains no audio samples")
        if audio.ndim not in (1, 2):
            raise ValueError(f"Unsupported decoded audio shape: {audio.shape}")
        if audio.ndim == 2 and audio.shape[1] > 2:
            raise ValueError(
                f"This script supports mono/stereo input, but decoded {audio.shape[1]} channels. "
                "Downmix explicitly first to avoid an ambiguous multichannel loudness layout."
            )
        duration = audio.shape[0] / float(sample_rate)
        if duration < 0.4:
            raise ValueError("Audio is shorter than the 400 ms minimum analysis block used by BS.1770")

        meter = pyln.Meter(int(sample_rate))
        try:
            measured_lufs = float(meter.integrated_loudness(audio))
        except Exception as exc:
            raise RuntimeError(f"Could not measure integrated loudness: {exc}") from exc

        if not math.isfinite(measured_lufs):
            # Fully silent input has no meaningful LUFS. Keep it silent rather
            # than attempting an infinite gain increase.
            print("Input is silent or below the loudness meter's measurable floor; leaving samples unchanged.", flush=True)
            normalized = np.asarray(audio, dtype=np.float32)
            gain_db = 0.0
        else:
            target_effective = float(args.target_lufs)
            is_mono = audio.ndim == 1 or audio.shape[1] == 1
            if args.dual_mono and is_mono:
                # Audacity's default 'Treat mono as dual mono' counts the mono
                # signal as two identical channels for loudness matching. This
                # means the underlying single-channel waveform is 3.0103 dB
                # lower than a non-dual-mono LUFS meter would target.
                target_effective -= DUAL_MONO_CORRECTION_DB
            gain_db = target_effective - measured_lufs
            if args.peak_ceiling_dbfs is not None:
                sample_peak = float(np.max(np.abs(audio)))
                if sample_peak > 0:
                    peak_dbfs = 20.0 * math.log10(sample_peak)
                    gain_db = min(gain_db, float(args.peak_ceiling_dbfs) - peak_dbfs)
            normalized = np.asarray(audio * (10.0 ** (gain_db / 20.0)), dtype=np.float32)

        peak = float(np.max(np.abs(normalized))) if normalized.size else 0.0
        if args.subtype != "FLOAT" and peak > 1.0:
            peak_dbfs = 20.0 * math.log10(peak)
            print(
                f"WARNING: global gain predicts a {peak_dbfs:+.2f} dBFS sample peak; "
                f"{args.subtype} export can clip. Use --peak-ceiling-dbfs -1 to lower the entire signal "
                "without dynamic compression, or --subtype FLOAT to preserve over-range samples.",
                file=sys.stderr, flush=True,
            )

        normalized_multichannel = scratch_dir / "normalized_float.wav"
        sf.write(normalized_multichannel, normalized, int(sample_rate), subtype="FLOAT", format="WAV")

        # Export as mono at 44.1 kHz by default, like the user's Audacity export.
        fd, temp_out_name = tempfile.mkstemp(
            prefix=f".{output.stem}.", suffix=".tmp.wav", dir=str(output.parent)
        )
        os.close(fd)
        temp_out = Path(temp_out_name)
        try:
            run_checked([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(normalized_multichannel), "-map", "0:a:0", "-vn",
                "-ac", "1", "-ar", str(args.sample_rate), "-c:a",
                {"PCM_16": "pcm_s16le", "PCM_24": "pcm_s24le", "FLOAT": "pcm_f32le"}[args.subtype],
                str(temp_out),
            ])
            if not temp_out.is_file() or temp_out.stat().st_size < 2048:
                raise RuntimeError("FFmpeg did not produce a valid output WAV")
            info = sf.info(temp_out)
            if info.samplerate != args.sample_rate or info.channels != 1:
                raise RuntimeError(
                    f"Output validation failed: expected mono/{args.sample_rate} Hz, "
                    f"got {info.channels} channel(s)/{info.samplerate} Hz"
                )
            os.replace(temp_out, output)
        finally:
            temp_out.unlink(missing_ok=True)

    print(f"Input: {source}", flush=True)
    print(f"Output: {output}", flush=True)
    print(f"Output format: mono, {args.sample_rate} Hz, {args.subtype} WAV", flush=True)
    if math.isfinite(measured_lufs):
        print(f"Measured integrated loudness: {measured_lufs:.2f} LUFS", flush=True)
        if args.dual_mono and (audio.ndim == 1 or (audio.ndim == 2 and audio.shape[1] == 1)):
            print("Mono dual-mono compensation: enabled (Audacity default behavior)", flush=True)
        print(f"Applied global gain: {gain_db:+.2f} dB", flush=True)
    print("Processing is a single constant gain only; no compression, denoising, or dynamic normalization was applied.", flush=True)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize integrated loudness like Audacity's Loudness Normalization, then export mono WAV."
    )
    parser.add_argument("input", help="Input audio file readable by FFmpeg")
    parser.add_argument("-o", "--out", help="Output WAV path (default: beside input as <stem>_norm.wav)")
    parser.add_argument("--target-lufs", type=float, default=DEFAULT_TARGET_LUFS,
                        help="Target integrated loudness in LUFS (default: -23, Audacity default)")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_RATE,
                        help="Output sample rate in Hz (default: 44100)")
    parser.add_argument("--subtype", choices=("PCM_16", "PCM_24", "FLOAT"), default="PCM_16",
                        help="Output WAV sample format (default: PCM_16)")
    parser.add_argument("--no-dual-mono", dest="dual_mono", action="store_false",
                        help="Do not use Audacity's default mono-as-dual-mono compensation")
    parser.set_defaults(dual_mono=True)
    parser.add_argument("--peak-ceiling-dbfs", type=float, default=None,
                        help="Optional safety ceiling: lower the whole waveform gain if its sample peak exceeds this dBFS (no limiter)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        process(args)
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"norm_audio: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
