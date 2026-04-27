"""Shared trace directory naming for DRAM / NoC / SRAM-PIM offload configurations."""


def channels_device_folder(num_channels: int, use_noc: bool, use_sram_pim: bool) -> str:
    """
    Folder name under trace/, e.g. 32_channels_per_device, 32_channels_per_device_noc_sram_pim.
    Matches historical api.py layout: f\"{num_channels}\" + \"_channels_per_device\" + suffixes.
    """
    name = f"{num_channels}_channels_per_device"
    if use_noc:
        name += "_noc"
    if use_sram_pim:
        name += "_sram_pim"
    return name
