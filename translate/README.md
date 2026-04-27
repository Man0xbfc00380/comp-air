# CompAir Translate Stage

This folder holds the explicit hierarchical ISA translation flow:

- input: `row_isa.txt` (logical row instructions)
- output: `packet_isa.txt` (packet instructions with explicit addresses)

## Usage

```bash
python translate/row_to_packet.py \
  --row-isa compair_results/row_isa.txt \
  --packet-isa compair_results/packet_isa.txt \
  --packet-bytes 256
```

Generated debug JSON (`packet_isa.txt.json`) includes:

- row-to-address map
- full packet list with chunk metadata

## Address Mapping Rule

- NoC rows map to region base `0x10000000`
- SRAM-PIM rows map to per-role regions under `0x80000000`
- each row uses `row_addr_stride` (default `0x00100000`)
- row payload is split into fixed-size packets (`packet_bytes`)
