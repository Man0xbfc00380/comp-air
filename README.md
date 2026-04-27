# CompAir Simulator

## Install

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
cd ../cent_simulation
```

