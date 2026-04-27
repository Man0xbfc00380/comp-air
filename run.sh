#!/usr/bin/env bash
# Four CompAir end-to-end runs (each: trace gen + DRAM Ramulator + row ISA + subsims)
# - --run-cent: CENT function_sim writes cent_pim/trace/.../*.txt
# - --run-ramulator: run_sim.py --simulate_trace --process_results --update_csv
# 1) DRAM-PIM only (no --use-sram-pim / --use-noc; often no .offload.json, subsims ~0)
# 2) DRAM-PIM + SRAM-PIM
# 3) DRAM-PIM + SRAM-PIM + NoC
# 4) DRAM-PIM + SRAM-PIM + NoC + collective split (reduction parallelization)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${PYTHON:-python3}"

# Same shape as README unified example; override model / channels / seqlen as needed
BASE=(
  --model Llama2-7B
  --num_channels 32
  --num_devices 32
  --prefill 1024
  --decoding 2048
  --seqlen_gap 1024
  --seqlen 1024
  --run-cent
  --run-ramulator
  --run-subsims
)

echo "========== (1) DRAM-PIM only (no SRAM/NoC offload flags) =========="
"$PY" compair_perf_pipeline.py "${BASE[@]}" \
  --result-dir compair_results/case_1_dram_pim

echo
echo "========== (2) DRAM-PIM + SRAM-PIM =========="
"$PY" compair_perf_pipeline.py "${BASE[@]}" --use-sram-pim \
  --result-dir compair_results/case_2_dram_pim_sram_pim

echo
echo "========== (3) DRAM-PIM + SRAM-PIM + NoC =========="
"$PY" compair_perf_pipeline.py "${BASE[@]}" --use-noc --use-sram-pim \
  --result-dir compair_results/case_3_dram_pim_sram_pim_noc

echo
echo "Done. CompAir outputs: compair_results/case_*/{row_isa.txt,compair_summary.json}"
echo "CENT CSVs (per offload config): each case dir has simulation_results.csv, processed_results.csv"
echo "DRAM Ramulator logs next to traces: cent_pim/trace/.../trace_*.txt.log"
echo "  (1) 32_channels_per_device"
echo "  (2) 32_channels_per_device_sram_pim"
echo "  (3) 32_channels_per_device_noc_sram_pim"