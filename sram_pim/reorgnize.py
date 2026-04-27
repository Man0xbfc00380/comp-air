import math
import matplotlib.pyplot as plt

# [ISSCC'23] 7.2 A 28nm 64-kb 31.6-TFLOPS/W Digital-Domain Floating-Point- 
# Computing-Unit and Double-Bit 6T-SRAM Computing-in- 
# Memory Macro for Floating-Point CNNs (BF16)

def SRAM_PIM_Compute_DSE(input_size, output_size, batch_num=1, 
                         split=[4,1], voltage_level=0, bandwidth=256):
    
    buffer_width = 1024
    spec = {
        "size": 64 * 1024, # bit = 64 * 64 * 16b
        "shape": [128, 8, split], # input, output, parallel (Limited by CAT) = 512 -> 8
        "channel": 128, # limited by CAT
        "area": 0.146, # mm2
        "mac_access_latency": 2 * buffer_width / bandwidth, # ns
        "execution_latency": [6.8, 8.6, 12.3, 14.1],
        "reduce_latency": 8, # ns
        "power_eff": [14.04, 20.13, 25.87, 31.60],
        "area_eff": [2.05, 1.5, 1.05, 0.5]
    }

    power = spec["area"] * spec["area_eff"][voltage_level] / spec["power_eff"][voltage_level]
    # print("\nTask: ", input_size, "->", output_size)
    # print("Area: ", spec["area"] * spec["shape"][2][1] * spec["shape"][2][0], "mm2")
    # print("Power: ", power * spec["shape"][2][1] * spec["shape"][2][0], "W")
    
    input_width = spec["shape"][0] * spec["shape"][2][0]
    output_width = spec["shape"][1] * spec["shape"][2][1]
    reduce_latency = spec["reduce_latency"] #if spec["shape"][2][0] > 1 else 0
    
    reduce_size = math.ceil(input_size / input_width)
    reload_size = math.ceil(output_size / output_width)
    # print("Input Split (Reduce): ", reduce_size, str(input_width)+"x")
    # print("Output Split (Reload): ", reload_size, str(output_width)+"x")
    
    if input_width * 16 > buffer_width:
        load_inp_round = math.ceil(input_width * 16 / buffer_width)
        # print("Load Input Round: ", load_inp_round)
    else:
        load_inp_round = 1
        # print("Load Input Round: ", load_inp_round)
    
    if input_width * output_width * 16 > buffer_width:
        load_wgt_round = math.ceil(input_width * output_width * 16 / buffer_width)
        # print("Load Weight Round: ", load_wgt_round)
    else:
        load_wgt_round = 1
        # print("Load Weight Round: ", load_wgt_round)
    
    # 2048 bit = 16bit * 128
    ld_i_latency = spec["mac_access_latency"]
    ld_w_latency = spec["mac_access_latency"]
    # print("Load Latency / Round: ", spec["mac_access_latency"], "ns")
    
    exe_latency = spec["execution_latency"][voltage_level]
    # print("MAC Execute Latency / Round: ", exe_latency, "ns")
    
    overall_latency = reload_size * reduce_size * (max(max(1,ld_i_latency * load_inp_round), exe_latency) * batch_num + ld_w_latency * load_wgt_round + exe_latency + reduce_latency)
    overall_no_pipe_latency = reload_size * reduce_size * (max(1,ld_i_latency * load_inp_round) * batch_num + ld_w_latency * load_wgt_round + exe_latency * load_inp_round * batch_num + reduce_latency)
    
    # print("Overall Latency (Pipeline): ", overall_latency, "ns")
    # print("Overall Latency (No Pipe) : ", overall_no_pipe_latency, "ns")
    # print("Latency / Batch (Pipeline): ", overall_latency / batch_num, "ns")
    # print("Latency / Batch (No Pipe) : ", overall_no_pipe_latency / batch_num, "ns")
    # print(ld_i_latency * load_inp_round * batch_num, ld_w_latency * load_wgt_round, exe_latency, reduce_latency)
    
    return overall_no_pipe_latency, overall_latency, power
    

if __name__ == "__main__":
    batch_nums = [1,4,8,16,32,64,128,256,512]
    bank_num = 16
    
    # Qwen & Llam2-7B
    emb_size = [5120]
    hid_size = [13824]
    for batch_num in batch_nums:
        data_list = []
        for i in range(len(hid_size)):
            channel_num = 32
            TP_num = 1
            emb_per_bank = math.ceil(emb_size[i] / (bank_num * channel_num * TP_num))
            hid_per_bank = math.ceil(hid_size[i] / (bank_num * channel_num * TP_num))
            
            # DSE Configs
            splits = [[4,1]]
            voltage_levels = [1,2,3]
            bandwidth = range(256, 256 * 6, 256)
            for bw in bandwidth:
                for voltage_level in voltage_levels:
                    for split in splits:
                        # QKV
                        p1, p11, power = SRAM_PIM_Compute_DSE(emb_size[i], emb_per_bank, batch_num, split=split, voltage_level=voltage_level, bandwidth=bw)
                        # FFN1
                        p2, p22, power = SRAM_PIM_Compute_DSE(emb_size[i], hid_per_bank, batch_num, split=split, voltage_level=voltage_level, bandwidth=bw)
                        p3, p33, power = SRAM_PIM_Compute_DSE(emb_size[i], hid_per_bank, batch_num, split=split, voltage_level=voltage_level, bandwidth=bw)
                        # FFN2
                        p4, p44, power = SRAM_PIM_Compute_DSE(hid_size[i], emb_per_bank, batch_num, split=split, voltage_level=voltage_level, bandwidth=bw)
                        # Insert
                        data_list.append(4*p11+p22+p33+p44)
        print(batch_num, max(data_list)/min(data_list))