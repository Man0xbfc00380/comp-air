import math
from dse import SRAM_PIM_Compute_DSE

# [ISSCC'23] 7.2 A 28nm 64-kb 31.6-TFLOPS/W Digital-Domain Floating-Point- 
# Computing-Unit and Double-Bit 6T-SRAM Computing-in- 
# Memory Macro for Floating-Point CNNs (BF16)


if __name__ == "__main__":
    model = "7B"
    batch_num = 1
    bank_num = 16
    macro_num = 8
    bw = 256 #* 4
    parallel = "PP"
    if parallel == "PP":
        # PP
        if model == "7B":
            emb_size = 4096
            hid_size = 11008
            channel_num = 32
            emb_per_bank = math.ceil(emb_size / (bank_num * channel_num))
            hid_per_bank = math.ceil(hid_size / (bank_num * channel_num))
            print(model, "emb:", emb_per_bank, "hid:", hid_per_bank)
            _, p11, power = SRAM_PIM_Compute_DSE(emb_size, emb_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
            _, p22, power = SRAM_PIM_Compute_DSE(emb_size, hid_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
            _, p33, power = SRAM_PIM_Compute_DSE(emb_size, hid_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
            _, p44, power = SRAM_PIM_Compute_DSE(hid_size, emb_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
            print("\nLatency:", p11*4 + p22 + p33 + p44, "ns in bw = ", bw)
        elif model == "13B":
            emb_size = 5120
            hid_size = 13824
            channel_num = 16
            emb_per_bank = math.ceil(emb_size / (bank_num * channel_num))
            hid_per_bank = math.ceil(hid_size / (bank_num * channel_num))
            print(model, "emb:", emb_per_bank, "hid:", hid_per_bank)
            _, p11, power = SRAM_PIM_Compute_DSE(emb_size, emb_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
            _, p22, power = SRAM_PIM_Compute_DSE(emb_size, hid_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
            _, p33, power = SRAM_PIM_Compute_DSE(emb_size, hid_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
            _, p44, power = SRAM_PIM_Compute_DSE(hid_size, emb_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
            print("\nLatency:", p11*4 + p22 + p33 + p44, "ns in bw = ", bw)
        else:
            emb_size = 8192
            hid_size = 28672
            channel_num = 10
            emb_per_bank = math.ceil(emb_size / (bank_num * channel_num))
            hid_per_bank = math.ceil(hid_size / (bank_num * channel_num))
            print(model, "emb:", emb_per_bank, "hid:", hid_per_bank)
            _, p11, power = SRAM_PIM_Compute_DSE(emb_size, emb_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
            _, p22, power = SRAM_PIM_Compute_DSE(emb_size, hid_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
            _, p33, power = SRAM_PIM_Compute_DSE(emb_size, hid_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
            _, p44, power = SRAM_PIM_Compute_DSE(hid_size, emb_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
            print("\nLatency:", p11*4 + p22 + p33 + p44, "ns in bw = ", bw)
    else:
        # TP -> Just 13B Now
        if model == "Qwen":
            emb_size = 8192
            hid_size = 29568
            channel_num = 32
            TP_nums = [4]
            for TP_num in TP_nums:
                emb_per_bank = math.ceil(emb_size / (bank_num * channel_num * TP_num))
                hid_per_bank = math.ceil(hid_size / (bank_num * channel_num * TP_num))
                print(model, "emb:", emb_per_bank, "hid:", hid_per_bank)
                _, p11, power = SRAM_PIM_Compute_DSE(emb_size, emb_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
                _, p22, power = SRAM_PIM_Compute_DSE(emb_size, hid_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
                _, p33, power = SRAM_PIM_Compute_DSE(emb_size, hid_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
                _, p44, power = SRAM_PIM_Compute_DSE(hid_size, emb_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
                print("\nTP_NUM:", TP_num)
                print("\nLatency:", p11*4 + p22 + p33 + p44, "ns in bw = ", bw)
        elif model == "GPT3-175B":
            emb_size = 12288
            hid_size = 12288*4
            channel_num = 32
            TP_nums = [8]
            for TP_num in TP_nums:
                emb_per_bank = math.ceil(emb_size / (bank_num * channel_num * TP_num))
                hid_per_bank = math.ceil(hid_size / (bank_num * channel_num * TP_num))
                print(model, "emb:", emb_per_bank, "hid:", hid_per_bank)
                _, p11, power = SRAM_PIM_Compute_DSE(emb_size, emb_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
                _, p22, power = SRAM_PIM_Compute_DSE(emb_size, hid_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
                _, p33, power = SRAM_PIM_Compute_DSE(emb_size, hid_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
                _, p44, power = SRAM_PIM_Compute_DSE(hid_size, emb_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
                print("\nTP_NUM:", TP_num)
                print("\nLatency:", p11*4 + p22 + p33 + p44, "ns in bw = ", bw)
        else:
            emb_size = 5120
            hid_size = 13824
            channel_num = 32
            TP_nums = [1,4,8,32]
            for TP_num in TP_nums:
                emb_per_bank = math.ceil(emb_size / (bank_num * channel_num * TP_num))
                hid_per_bank = math.ceil(hid_size / (bank_num * channel_num * TP_num))
                print(model, "emb:", emb_per_bank, "hid:", hid_per_bank)
                _, p11, power = SRAM_PIM_Compute_DSE(emb_size, emb_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
                _, p22, power = SRAM_PIM_Compute_DSE(emb_size, hid_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
                _, p33, power = SRAM_PIM_Compute_DSE(emb_size, hid_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
                _, p44, power = SRAM_PIM_Compute_DSE(hid_size, emb_per_bank, batch_num, split=[2,2], voltage_level=1, bandwidth=bw)
                print("\nTP_NUM:", TP_num)
                print("\nLatency:", p11*4 + p22 + p33 + p44, "ns in bw = ", bw)