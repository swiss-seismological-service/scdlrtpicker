#!/usr/bin/env python3
#
# Copyright (C) by ETHZ/SED
# GNU Affero General Public License v3 or later (see the LICENSE file).
"""
Print the scdlrtpicker contract (the seiscomp.picker.* metadata plus the
actual graph input/output shapes) of one or more exported .onnx models.

    ./show_model_info.py phasenet_ethz.onnx
    ./show_model_info.py *.onnx                 # table over all of them
    ./show_model_info.py                        # same as *.onnx in cwd

Reads only the file -- no seisbench, no torch, no network. Needs onnx
(preferred) or, failing that, onnxruntime.
"""
import glob
import sys

KEYS = [
    "seiscomp.picker.numComponents",
    "seiscomp.picker.sampleRate",
    "seiscomp.picker.windowLength",
    "seiscomp.picker.phases",
    "seiscomp.picker.normalization",
    "seiscomp.picker.componentOrder",
    "seiscomp.picker.resample",
    "seiscomp.picker.polarityLabels",
    "seiscomp.picker.uncertaintyLabels",
]


def read(path):
    """-> (meta dict, input shape list|None, output shape list|None)."""
    try:
        import onnx
        m = onnx.load(path, load_external_data=False)
        meta = {p.key: p.value for p in m.metadata_props}

        def shape(vi):
            d = vi.type.tensor_type.shape.dim
            return [x.dim_value if x.HasField("dim_value") else (x.dim_param or "?")
                    for x in d]

        outs = {i.name for i in m.graph.initializer}
        ins = [vi for vi in m.graph.input if vi.name not in outs]
        return (meta,
                shape(ins[0]) if ins else None,
                shape(m.graph.output[0]) if m.graph.output else None)
    except ImportError:
        import onnxruntime as ort
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return (s.get_modelmeta().custom_metadata_map,
                s.get_inputs()[0].shape,
                s.get_outputs()[0].shape)


def one(path):
    meta, ishape, oshape = read(path)
    print(f"\n{path}")
    print(f"  input tensor  : {ishape}")
    print(f"  output tensor : {oshape}")
    for k in KEYS:
        if k in meta:
            print(f"  {k:34s}: {meta[k]}")
    extra = sorted(set(meta) - set(KEYS))
    for k in extra:
        print(f"  {k:34s}: {meta[k]}  (non-contract)")
    if "seiscomp.picker.sampleRate" in meta and "seiscomp.picker.windowLength" in meta:
        try:
            secs = int(meta["seiscomp.picker.windowLength"]) / float(meta["seiscomp.picker.sampleRate"])
            print(f"  -> window duration                : {secs:g} s  "
                  f"(a trigger-centred pick lands ~{secs/2:g} s after the trigger)")
        except ValueError:
            pass


def table(paths):
    rows = []
    for p in paths:
        try:
            meta, ishape, oshape = read(p)
        except Exception as e:  # noqa: BLE001
            rows.append((p, "ERROR: " + str(e)[:50], "", "", "", "", "", "", ""))
            continue
        g = lambda k: meta.get("seiscomp.picker." + k, "-")  # noqa: E731
        pol = "yes" if "seiscomp.picker.polarityLabels" in meta else "-"
        unc = "yes" if "seiscomp.picker.uncertaintyLabels" in meta else "-"
        rows.append((p, g("numComponents"), g("componentOrder"),
                     g("windowLength"), g("sampleRate"), g("normalization"),
                     pol, unc, g("phases")))
    hdr = ("file", "nc", "order", "window", "rate", "norm", "pol", "unc", "phases")
    w = [max(len(str(r[i])) for r in (rows + [hdr])) for i in range(len(hdr))]
    line = lambda r: "  ".join(str(v).ljust(w[i]) for i, v in enumerate(r))  # noqa: E731
    print(line(hdr))
    print("  ".join("-" * x for x in w))
    for r in sorted(rows):
        print(line(r))


def main():
    args = sys.argv[1:] or sorted(glob.glob("*.onnx"))
    if not args:
        sys.exit("no .onnx files given and none in the current directory")
    if len(args) == 1:
        one(args[0])
    else:
        table(args)


if __name__ == "__main__":
    main()
