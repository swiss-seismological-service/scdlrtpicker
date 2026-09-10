#!/bin/bash
#
# Copyright (C) by ETHZ/SED
# GNU Affero General Public License v3 or later (see the LICENSE file).
#
# Export every SeisBench pretrained weight set that fits the plugin's
# contract to <model>_<dataset>.onnx in this directory, using
# export_seisbench_to_onnx.py. Skips weight sets the exporter rejects
# (e.g. the 4-component 'obs' OBS models).
#
# Setup:
#   python3 -m venv venv && source venv/bin/activate
#   pip install seisbench onnx onnxscript onnxruntime
#   ./dump_all_models.sh
#
# If your CA bundle makes the SeisBench repo download fail:
#   SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt ./dump_all_models.sh
set -u
cd "$(dirname "$0")"
X=./export_seisbench_to_onnx.py
LOG=dump_all_models.log
: > "$LOG"

# --model name -> SeisBench class name
MODELS="phasenet:PhaseNet phasenetlight:PhaseNetLight eqtransformer:EQTransformer
        obstransformer:OBSTransformer eqcctp:EQCCTP eqccts:EQCCTS
        basicphaseae:BasicPhaseAE skynet:Skynet eqtp:EQTP"

ok=0; fail=0
for pair in $MODELS; do
  model=${pair%%:*}; cls=${pair##*:}
  datasets=$(python3 -c "import seisbench.models as sbm; print(' '.join(sbm.${cls}.list_pretrained()))" 2>>"$LOG") || continue
  for ds in $datasets; do
    name="${model}_${ds}"
    args=(--model "$model" --dataset "$ds" -o "${name}.onnx")
    [ "$ds" = "original_multiphase" ] && args+=(--phase-map "P=Pn,Pg S=Sn,Sg")
    echo "################## $model / $ds ##################" | tee -a "$LOG"
    if python3 "$X" "${args[@]}" >>"$LOG" 2>&1; then
      echo "  OK   ${name}.onnx" | tee -a "$LOG"; ok=$((ok+1))
    else
      echo "  FAIL ${name} (see $LOG)" | tee -a "$LOG"; fail=$((fail+1))
      rm -f "${name}.onnx" "${name}.onnx.data"
    fi
  done
done
echo | tee -a "$LOG"
echo "################## DONE: $ok ok, $fail failed ##################" | tee -a "$LOG"
