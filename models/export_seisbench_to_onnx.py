#!/usr/bin/env python3
#
# Copyright (C) by ETHZ/SED
# GNU Affero General Public License v3 or later (see the LICENSE file).
"""
Export a SeisBench phase-picking model to ONNX with the seiscomp.picker.*
metadata that Seiscomp::Processing::OnnxSession (the DL1C/DL3C/SDL1C/SDL3C
picker interfaces of scdlrtpicker) requires.

Supported models (see MODELS): PhaseNet, PhaseNetLight, EQTransformer,
OBSTransformer, EQCCTP, EQCCTS, BasicPhaseAE, Skynet, EQTP. All are
3-component, so this always produces a DL3C / SDL3C model; there is no
pretrained single-vertical-component weight set for the DL1C / SDL1C
interfaces. COMPATIBILITY.md has the per-model window/phase/normalization
table and the list of SeisBench models that do NOT fit the plugin's
contract.

Usage:
    python3 export_seisbench_to_onnx.py --model phasenet --dataset geofon -o phasenet_geofon.onnx
    python3 export_seisbench_to_onnx.py --model eqcctp -o eqcctp.onnx           # --dataset defaults per model
    python3 export_seisbench_to_onnx.py --model skynet --dataset original_multiphase \\
            --phase-map "P=Pn,Pg S=Sn,Sg" -o skynet_multiphase.onnx
    python3 export_seisbench_to_onnx.py --model eqtp -o eqtp.onnx               # P + first-motion polarity

--phase-map folds sub-phase output rows into the P / S rows the plugin
expects (element-wise max). SeisComP's picker framework carries only one
P and one S phase per trigger, and a P pick's phase code is fixed by
scautopick config, so sub-phases cannot be exposed individually anyway.

EQTP (Peng et al. 2025) additionally exports first-motion polarity
(Polarity_U, Polarity_D); those rows are auto-detected and written as
seiscomp.picker.polarityLabels -- see --polarity-labels for a model using
different row names.

Requires: seisbench, onnx, onnxscript (torch comes in as a seisbench dep)
    pip install seisbench onnx onnxscript
Optional: onnxruntime, only used for the self-check below
    pip install onnxruntime

Read the printed diagnostics before trusting the exported file -- they
are there to catch a mismatch between what this script assumes about
the SeisBench API and what your installed version actually does.
"""
import argparse
import os
import re
import sys

import numpy as np
import torch
import onnx
import seisbench.models as sbm


# name -> SeisBench class name. getattr keeps older seisbench installs
# working (they just offer fewer --model choices).
MODELS = {
    name: getattr(sbm, cls_name)
    for name, cls_name in {
        "phasenet": "PhaseNet",
        "phasenetlight": "PhaseNetLight",
        "eqtransformer": "EQTransformer",
        "obstransformer": "OBSTransformer",
        "eqcctp": "EQCCTP",
        "eqccts": "EQCCTS",
        "basicphaseae": "BasicPhaseAE",
        "skynet": "Skynet",
        "eqtp": "EQTP",
    }.items()
    if hasattr(sbm, cls_name)
}

# --dataset default per model when not given on the command line.
DEFAULT_DATASET = {
    "eqcctp": "original",
    "eqccts": "original",
    "obstransformer": "obst2024",
    "skynet": "original",
    "eqtp": "ncedc",
}

# Raw output-row names recognized as first-motion polarity without
# needing --polarity-labels, in (up, down) order. EQTP (Peng et al.
# 2025) is the only SeisBench model that currently exports these.
KNOWN_POLARITY_LABELS = ("Polarity_U", "Polarity_D")


class SingleTensorWrapper(torch.nn.Module):
    """Normalises a SeisBench model's forward() to a single output tensor
    of shape (batch, numPhases, samples). Some models (EQTransformer,
    OBSTransformer) return a tuple (detection, P, S) instead of one
    stacked tensor; PhaseNet, EQCCT, ... already return a single tensor.

    row_groups, if given, additionally merges output rows: row_groups[i]
    is the list of raw-output row indices feeding final row i, combined
    by element-wise max -- used by --phase-map to fold e.g. Pn,Pg -> P.
    """

    def __init__(self, model, row_groups=None):
        super().__init__()
        self.model = model
        self.row_groups = row_groups

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = torch.stack(list(out), dim=1)
        if self.row_groups is not None:
            out = torch.stack(
                [out[:, g].amax(dim=1) for g in self.row_groups], dim=1
            )
        return out


def parse_phase_map(spec):
    """"P=Pn,Pg S=Sn,Sg" -> {"P": ["Pn", "Pg"], "S": ["Sn", "Sg"]}."""
    groups = {}
    for token in spec.split():
        if "=" not in token:
            sys.exit(f"ERROR: --phase-map: '{token}' is not TARGET=src1,src2,...")
        target, srcs = token.split("=", 1)
        groups[target] = [s for s in srcs.split(",") if s]
    return groups


def apply_phase_map(raw_labels, groups):
    """Given the model's raw output-row names and a phase map, return
    (row_groups, out_labels): the row index lists to hand
    SingleTensorWrapper and the resulting phase names, keeping raw
    order and replacing each group by its target at the position of its
    first member."""
    idx = {name: i for i, name in enumerate(raw_labels)}
    src_to_target = {}
    for target, srcs in groups.items():
        for s in srcs:
            if s not in idx:
                sys.exit(f"ERROR: --phase-map {target}={','.join(srcs)}: "
                         f"'{s}' is not a model output row {raw_labels}")
            if s in src_to_target:
                sys.exit(f"ERROR: --phase-map: '{s}' used in more than one group")
            src_to_target[s] = target

    row_groups, out_labels, emitted = [], [], set()
    for i, name in enumerate(raw_labels):
        target = src_to_target.get(name)
        if target is None:
            row_groups.append([i])
            out_labels.append(name)
        elif target not in emitted:
            emitted.add(target)
            row_groups.append([idx[s] for s in groups[target]])
            out_labels.append(target)
    return row_groups, out_labels


def detect_labels(model, n_outputs):
    labels = getattr(model, "labels", None)
    if labels is None:
        print(f"WARNING: model has no .labels attribute, using placeholders P0..P{n_outputs - 1}. "
              f"Edit the exported metadata by hand (seiscomp.picker.phases) before use.")
        return [f"P{i}" for i in range(n_outputs)]
    labels = list(labels)
    if len(labels) != n_outputs:
        print(f"WARNING: model.labels={labels} has {len(labels)} entries, but the "
              f"traced output has {n_outputs} channels -- these MUST line up 1:1 "
              f"(output row i must be phase labels[i]). Check by hand.")
    return labels


def check_weight_version(cls, model, dataset):
    """SeisBench ships a list of (name_regex, bad_version, message)
    triples on the model class (cls._weight_warnings) for pretrained
    weight sets known to have real problems -- e.g. as of seisbench
    0.12.3, PhaseNet's "ethz|geofon|instance|iquique|lendb|neic|scedc|
    stead" version "1" weights are flagged with: "The normalization
    for this weight version is incorrect and will lead to degraded
    performance." (geofon is this script's own default --dataset).

    from_pretrained() already checks this internally and logs a
    warning via seisbench's own logger (see
    WaveformModel._version_warnings() in seisbench/models/base.py),
    but that's easy to miss depending on logging configuration. This
    re-checks the same list directly against the resolved
    model._weights_version and prints a hard-to-miss banner, so a bad
    weight version doesn't silently produce a degraded export.
    """
    weights_version = getattr(model, "_weights_version", None)
    if weights_version is None:
        return

    for name_regex, bad_version, message in getattr(cls, "_weight_warnings", []):
        if re.fullmatch(name_regex, dataset) and bad_version == weights_version:
            print(f"\n{'!' * 70}")
            print(f"WARNING: {cls.__name__} weights '{dataset}' version "
                  f"'{weights_version}' are flagged by SeisBench itself:")
            print(f"  {message}")
            print("This export will bake in whatever this weight version actually "
                  "does -- if it's the normalization bug referenced above, the "
                  "exported model's confidence will be degraded regardless of how "
                  "correctly the C++ runtime replicates the normalization formula. "
                  "Re-run without --no-update (or with update=True) to fetch a "
                  "newer version.")
            print(f"{'!' * 70}")
            return


def build_synthetic_p_then_s(window_length, sampling_rate, num_components):
    """Builds a synthetic 3-component window containing two well-
    separated broadband transients -- an earlier one standing in for
    "P" (30% into the window, higher frequency, smaller/narrower) and
    a later one standing in for "S" (60% into the window, lower
    frequency, larger/wider, matching the classical expectation that
    S is the bigger, lower-frequency arrival). Not real seismic data
    and not meant to look exactly like one -- the point is only to
    have two unambiguous, temporally ordered onsets to probe row
    assignment with, not to test picking accuracy.
    """
    rng = np.random.default_rng(0)
    data = rng.normal(scale=0.01, size=(num_components, window_length)).astype(np.float64)

    t = np.arange(window_length)

    def transient(center, freq_hz, amplitude, width_seconds):
        width_samples = max(1.0, width_seconds * sampling_rate)
        envelope = np.exp(-0.5 * ((t - center) / width_samples) ** 2)
        return amplitude * envelope * np.sin(2 * np.pi * freq_hz * (t - center) / sampling_rate)

    p_center = int(window_length * 0.3)
    s_center = int(window_length * 0.6)

    p_wave = transient(p_center, freq_hz=8.0, amplitude=1.0, width_seconds=0.3)
    s_wave = transient(s_center, freq_hz=3.0, amplitude=1.5, width_seconds=0.5)

    for c in range(num_components):
        data[c] += p_wave + s_wave

    return data, p_center, s_center


def self_check(onnx_path, sampling_rate, window_length, num_components, labels, normalization):
    """Sanity-checks that seiscomp.picker.phases' order actually
    matches the exported model's raw output row order, by feeding a
    synthetic P-then-S waveform (see build_synthetic_p_then_s) through
    the *exported .onnx file itself* (via onnxruntime, not the
    in-memory PyTorch model) and confirming the row labeled "P" peaks
    before the row labeled "S". This is the same phase-to-row
    assumption Seiscomp::Processing::OnnxSession::runInference()
    relies on; if it's wrong there, DLPicker would silently read
    whatever channel actually holds the noise/S output and call it
    "P", with no error anywhere.

    Only a heuristic: this is synthetic, not real seismic data, so a
    model that doesn't respond crisply to it is not proof of a
    mismatch either. Treat a failure here as a strong reason to
    double check, not a hard failure of the export.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("\nSkipping P/S row-order self-check: onnxruntime not installed "
              "(pip install onnxruntime). This does NOT confirm the export is "
              "correct -- it's simply unverified.")
        return

    print("\nRunning P/S row-order self-check (synthetic two-onset waveform via onnxruntime)...")

    data, p_center, s_center = build_synthetic_p_then_s(window_length, sampling_rate, num_components)

    # Mirrors Seiscomp::Processing::OnnxPickerStream::evaluate()'s
    # per-channel normalization exactly, so the self-check probes the
    # model the same way the C++ runtime actually will.
    normed = np.empty_like(data)
    for c in range(num_components):
        x = data[c]
        if normalization == "std":
            normed[c] = (x - x.mean()) / max(x.std(), 1e-12)
        elif normalization == "peak":
            normed[c] = x / max(np.abs(x).max(), 1e-12)
        else:
            normed[c] = x

    input_tensor = normed.reshape(1, num_components, window_length).astype(np.float32)

    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    output = np.asarray(sess.run([output_name], {input_name: input_tensor})[0])
    output = output.reshape(len(labels), window_length)

    peaks = {}
    for i, phase in enumerate(labels):
        peak_idx = int(np.argmax(output[i]))
        peaks[phase] = peak_idx
        print(f"  {phase}: output peaks at sample {peak_idx} "
              f"(injected P-like onset at {p_center}, S-like onset at {s_center})")

    tol = window_length * 0.15
    if "P" in peaks and "S" in peaks:
        if peaks["P"] < peaks["S"]:
            print("  PASS: 'P' output peaks before 'S' output, as expected -- "
                  "seiscomp.picker.phases' row order looks correct.")
        else:
            print("  WARNING: 'P' output does NOT peak before 'S' output on this "
                  "synthetic test. This suggests seiscomp.picker.phases may not "
                  "match the model's actual output row order -- double check "
                  "before trusting this export (e.g. by comparing against "
                  "model.annotate() on the same data in Python).")
    elif "P" in peaks:
        near = abs(peaks["P"] - p_center) < tol
        print(f"  {'PASS' if near else 'WARNING'}: single-phase model -- 'P' output "
              f"peaks {'near' if near else 'far from'} the injected P onset "
              f"({peaks['P']} vs {p_center}).")
    elif "S" in peaks:
        near = abs(peaks["S"] - s_center) < tol
        print(f"  {'PASS' if near else 'WARNING'}: single-phase model -- 'S' output "
              f"peaks {'near' if near else 'far from'} the injected S onset "
              f"({peaks['S']} vs {s_center}).")
    else:
        print("  Could not check: neither 'P' nor 'S' among the output rows.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(MODELS), required=True)
    ap.add_argument("--dataset", default=None,
                     help="pretrained weight set name (SeisBench 'dataset'), "
                          "e.g. geofon, ethz, instance, stead. Default: per-model "
                          "(see DEFAULT_DATASET), falling back to geofon")
    ap.add_argument("--phase-map", default=None, metavar='"P=Pn,Pg S=Sn,Sg"',
                     help="fold sub-phase output rows into single phases by "
                          "element-wise max, e.g. Skynet original_multiphase "
                          "(Pn,Pg,Sn,Sg,N) -> (P,S,N)")
    ap.add_argument("--polarity-labels", default=None, metavar="UP,DOWN",
                     help="names of the two output rows giving first-motion "
                          "polarity, e.g. Polarity_U,Polarity_D (auto-detected "
                          "for those exact names; only needed for a model "
                          "using different ones)")
    ap.add_argument("--normalization", choices=["std", "peak", "none"], default=None,
                     help="override auto-detected normalization; if omitted, uses "
                          "the model's .norm attribute when it is std/peak/none, "
                          "'none' when .norm is None, else std")
    ap.add_argument("--no-update", action="store_true",
                     help="don't query the remote repository for a newer weight "
                          "version; from_pretrained()'s own default (\"latest\") "
                          "only means latest *locally cached*, so without this "
                          "flag we pass update=True to actually check for updates "
                          "-- relevant since some weight versions have known bugs, "
                          "see check_weight_version() below")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    cls = MODELS[args.model]
    dataset = args.dataset or DEFAULT_DATASET.get(args.model, "geofon")
    print(f"Loading {args.model} pretrained on '{dataset}' "
          f"(update={not args.no_update}) ...")
    try:
        model = cls.from_pretrained(dataset, update=not args.no_update)
    except Exception as e:
        print(f"ERROR: could not load {args.model}.from_pretrained({dataset!r}): {e}",
              file=sys.stderr)
        print(f"Run cls.list_pretrained() in a Python shell to see what's actually "
              f"available for {args.model}, e.g.:\n"
              f"  python3 -c \"import seisbench.models as sbm; print(sbm.{cls.__name__}.list_pretrained())\"",
              file=sys.stderr)
        sys.exit(1)
    model.eval()

    check_weight_version(cls, model, dataset)

    sampling_rate = float(model.sampling_rate)
    window_length = int(model.in_samples)
    component_order = getattr(model, "component_order", "ZNE")
    native_norm = getattr(model, "norm", "std")

    print(f"  sampling_rate   = {sampling_rate}")
    print(f"  window_length   = {window_length}")
    print(f"  component_order = {component_order}")
    print(f"  model.norm      = {native_norm!r}  (informs --normalization default)")

    if len(component_order) != 3:
        print(f"ERROR: this exporter only supports 3-component models (got "
              f"component_order={component_order!r}). The DL1C/SDL1C interfaces "
              f"need a genuine single-component model; none of the supported "
              f"SeisBench models are.", file=sys.stderr)
        sys.exit(1)

    # .eval() the wrapper too, not just the inner model: model.eval()
    # above already puts the real network's BatchNorm/Dropout layers
    # (the ones that actually matter) into eval mode before they're
    # wrapped, so the trace captures the correct eval-mode computation
    # regardless -- but the wrapper itself has its own top-level
    # training flag (default True for any freshly constructed
    # nn.Module) that torch.onnx.export()'s own check looks at, so
    # leaving it unset just produces a spurious warning.
    dummy = torch.randn(1, 3, window_length)

    # Probe the raw model to learn its output rows, then (if --phase-map)
    # rebuild the wrapper with the row merge and re-probe.
    with torch.no_grad():
        probe = SingleTensorWrapper(model).eval()(dummy)

    if probe.dim() != 3 or probe.shape[0] != 1 or probe.shape[2] != window_length:
        print(f"ERROR: expected a (1, numPhases, {window_length}) output tensor, got "
              f"{tuple(probe.shape)}. SingleTensorWrapper likely needs adjusting for "
              f"this SeisBench version/model -- do not use this export as-is.",
              file=sys.stderr)
        sys.exit(1)

    raw_labels = detect_labels(model, probe.shape[1])
    print(f"  raw output rows = {raw_labels}")

    row_groups = None
    labels = raw_labels
    if args.phase_map:
        row_groups, labels = apply_phase_map(raw_labels, parse_phase_map(args.phase_map))
        print(f"  after --phase-map = {labels}  (row groups {row_groups})")

    wrapped = SingleTensorWrapper(model, row_groups).eval()
    with torch.no_grad():
        final_probe = wrapped(dummy)
    print(f"  traced output shape = {tuple(final_probe.shape)}  "
          f"(expect (1, {len(labels)}, {window_length}))")

    if "P" not in labels and "S" not in labels:
        print("WARNING: neither 'P' nor 'S' is among the output rows -- the model "
              "cannot drive a P or an S picker. Check the labels / --phase-map.")

    if args.polarity_labels:
        polarity_labels = args.polarity_labels.split(",")
        if len(polarity_labels) != 2:
            sys.exit(f"ERROR: --polarity-labels must name exactly two rows "
                      f"(up,down), got {args.polarity_labels!r}")
        missing = [p for p in polarity_labels if p not in labels]
        if missing:
            sys.exit(f"ERROR: --polarity-labels {missing} not in output rows {labels}")
    elif all(p in labels for p in KNOWN_POLARITY_LABELS):
        polarity_labels = list(KNOWN_POLARITY_LABELS)
    else:
        polarity_labels = None
    if polarity_labels:
        print(f"  polarity rows = {polarity_labels}")

    normalization = args.normalization
    if normalization is None:
        if native_norm in ("std", "peak", "none"):
            normalization = native_norm
        elif native_norm is None:
            normalization = "none"
            print("  NOTE: model.norm is None -> writing normalization=none; "
                  "verify the model does not expect a pre-normalised input "
                  "(override with --normalization if it does).")
        else:
            normalization = "std"
            print(f"  NOTE: model.norm={native_norm!r} unrecognised -> using std.")
    print(f"  normalization written to metadata = {normalization}")

    torch.onnx.export(
        wrapped,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["output"],
        # Some torch/onnxscript ops (e.g. Pad, from EQTransformer's LSTM
        # padding) have no opset-17 downgrade adapter, so requesting 17
        # produces a failed-conversion traceback before the exporter
        # falls back to 18 anyway. Requesting 18 directly avoids that
        # noise for no downside: ONNX Runtime has supported opset 18
        # since well before 1.28.0.
        opset_version=18,
    )

    # What external-data sidecars does the freshly exported file point
    # at? Read the graph WITHOUT resolving them (load_external_data=False)
    # so the EXTERNAL markers are still visible. torch.onnx.export (torch
    # >= 2.x dynamo path) offloads the larger initializers to
    # "<output>.data" and writes the .onnx as a graph referencing it.
    _pre = onnx.load(args.output, load_external_data=False)
    sidecars = {
        os.path.join(os.path.dirname(args.output) or ".", ed.value)
        for t in _pre.graph.initializer
        if t.data_location == onnx.TensorProto.EXTERNAL
        for ed in t.external_data if ed.key == "location"
    }
    del _pre

    # load_external_data=True (the default) pulls the weights fully into
    # memory, so the re-save below is self-contained regardless of how
    # torch.onnx.export chose to lay the file out.
    onnx_model = onnx.load(args.output)

    def set_meta(key, value):
        entry = onnx_model.metadata_props.add()
        entry.key = key
        entry.value = str(value)

    set_meta("seiscomp.picker.numComponents", 3)
    set_meta("seiscomp.picker.sampleRate", sampling_rate)
    set_meta("seiscomp.picker.windowLength", window_length)
    set_meta("seiscomp.picker.phases", ",".join(labels))
    set_meta("seiscomp.picker.normalization", normalization)
    set_meta("seiscomp.picker.componentOrder", component_order.upper())
    if polarity_labels:
        set_meta("seiscomp.picker.polarityLabels", ",".join(polarity_labels))

    # Re-write as a single self-contained file (every model here is well
    # under the 2 GB single-file limit), then delete the now-orphaned
    # sidecar(s), so a model is always exactly one .onnx file with
    # nothing to ship alongside it.
    onnx.save(onnx_model, args.output)

    for path in sidecars:
        if os.path.exists(path):
            os.remove(path)
            print(f"  folded weights in; removed sidecar {path}")

    print(f"\nWrote {args.output}")
    print("\nBefore wiring this into scautopick, sanity check the exported file, e.g.:")
    print(f"  python3 -c \"import onnxruntime as ort; s = ort.InferenceSession('{args.output}'); "
          f"print('inputs ', [(i.name, i.shape) for i in s.get_inputs()]); "
          f"print('outputs', [(o.name, o.shape) for o in s.get_outputs()])\"")

    self_check(args.output, sampling_rate, window_length, 3, labels, normalization)


if __name__ == "__main__":
    main()
