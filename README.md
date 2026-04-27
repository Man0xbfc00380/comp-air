# CompAir Simulator

Ths is a unified performance simulation framework for CompAir:

- DRAM-PIM simulation (CENT / Ramulator flow)
- CompAir-NoC microarchitecture simulation (modified `booksim2`)
- SRAM-PIM offload modeling

The repository supports both end-to-end orchestration and manual per-stage execution.

## Repository Layout

Key directories and scripts:

- `cent_pim/`: DRAM-PIM trace generation and timing simulation
- `booksim2/`: modified NoC simulator and API bindings
- `sram_pim/`: SRAM-PIM offload model
- `compair_perf_pipeline.py`: one-command orchestration entry
- `compair_row_isa.py`: converts offload manifests to row-level ISA view
- `compair_noc_driver.py`: maps micro-ops to NoC driver step counts
- `compair_results/`: generated summaries and artifacts

## Installation

```bash
# Prepare conda environment
conda create -n compair python=3.10 -y
conda activate compair
pip install -r requirements.txt

# Set environment variables
source setenv.sh

# Build modified booksim2 for CompAir-NoC Simulation
cd booksim2/api
mkdir build
cd build
cmake ..
make
cd ../../..

# Build modified CENT for DRAM-PIM Simulation
cd cent_pim/aim_simulator 
mkdir build
cmake ..
make
```

## Prerequisites

- Linux/macOS with `cmake` and C++ build toolchain available
- Conda (recommended)
- Python dependencies from `requirements.txt`
- For NoC driver async-execution, install `greenlet`:

## End-to-End Example with Llama2-7B

```bash
bash run.sh
```

## Unified Simulation (DRAM-PIM + CompAir-NoC + SRAM-PIM)

Offload flags match `cent_pim/cent_simulation` trace generation: **`--use-noc`** removes CXL-style collectives from the CENT trace (handled instead by the NoC microarchitecture in `booksim2/example/`), and **`--use-sram-pim`** omits DRAM-side `MAC_ABK` for GEMV (handled by `sram_pim/api.py`). Traces and Ramulator logs live under `cent_pim/trace/<N>_channels_per_device[_noc][_sram_pim]/`.

### One-command orchestration (recommended)

From the repository root (after building Ramulator and `booksim2/api` per above, and `pip install greenlet` for NoC drivers):

```bash
python compair_perf_pipeline.py --model Llama2-7B --num_channels 32 --num_devices 32 \
  --seqlen 1024 --use-noc --use-sram-pim --run-cent --run-subsims
```

Add **`--run-ramulator`** to run `cent_pim/cent_simulation/run_sim.py --simulate_trace` with the same offload flags once traces exist.

Common useful options:

- `--run-cent`: generate offload traces from CENT pipeline
- `--run-subsims`: run sub-simulators and aggregate latency estimates
- `--run-ramulator`: run DRAM timing simulation from generated traces
- `--model_parallel`: switch from pipeline-parallel to model-parallel layout

### Manual steps (keep flags consistent)

1. **Generate DRAM (CENT) traces** (pipeline-parallel):

   ```bash
   cd cent_pim
   python api.py --generate_trace --model Llama2-7B --num_channels 32 --num_devices 32 \
     --seqlen 1024 --use-noc --use-sram-pim
   ```

   For model-parallel layout (as in `cent_simulation/simulation.sh`), add **`--model_parallel`** to `api.py`.

2. **Row-level ISA** from `*.offload.json` manifests beside each trace:

   ```bash
   python compair_row_isa.py cent_pim/trace/32_channels_per_device_noc_sram_pim/pipeline_parallel/Llama2-7B/*.offload.json \
     -o compair_results/row_isa.txt --seqlen 1024 --dim 4096
   ```

3. **NoC microprograms** (step-count proxy): `python compair_noc_driver.py --micro rmsnorm --num-per-bank 52` (see `compair_perf_pipeline.py` for row-to-driver mapping).

4. **DRAM timing** from `cent_simulation` with the same flags as trace generation:

   ```bash
   cd cent_pim/cent_simulation
   python run_sim.py --model Llama2-7B --simulate_trace --process_results --update_csv \
     --num_channels 32 --num_devices 32 --seqlen 1024 --use-noc --use-sram-pim
   ```

`compair_results/compair_summary.json` lists manifests and estimated offload latency when `--run-subsims` is passed to `compair_perf_pipeline.py`.

## Output Artifacts

Main output locations:

- `cent_pim/trace/.../`: generated traces and manifests (`*.offload.json`)
- `compair_results/row_isa.txt`: row-level ISA abstraction
- `compair_results/compair_summary.json`: unified latency summary
- `compair_results/delay_summary_by_case.json`: case-level delay aggregation (if generated in your workflow)

## Reproducibility Notes

- Keep `--num_channels`, `--num_devices`, `--seqlen`, `--use-noc`, and `--use-sram-pim` consistent across trace generation and timing simulation.
- If offload flags differ between stages, the final latency summary can become inconsistent.
- Use the one-command pipeline first, then switch to manual steps only for debugging specific stages.