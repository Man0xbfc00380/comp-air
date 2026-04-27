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
    batch_num = 64
    bank_num = 16
    
    # Qwen & Llam2-7B
    emb_size = [4096, 8192]
    hid_size = [11008, 28672]
    model_name = ["A. Llama2-7B", "B. Llama2-7B", "C. Qwen-72B", "D. Qwen-72B"]
    layer_name = ["QKVO", "FFN", "QKVO", "FFN"]
    data_list = []
    
    for i in range(2):
        qkvo_list = []
        ffn_list = []
        
        channel_num = 32
        TP_num = 1
        emb_per_bank = math.ceil(emb_size[i] / (bank_num * channel_num * TP_num))
        hid_per_bank = math.ceil(hid_size[i] / (bank_num * channel_num * TP_num))
        
        # DSE Configs
        splits = [[1,4], [2,2], [4,1]]
        voltage_levels = [1,2,3]
        bandwidth = range(128, 256 * 7, 64)
        for bw in bandwidth:
            for voltage_level in voltage_levels:
                for split in splits:
                    # QKV
                    p1, p11, power = SRAM_PIM_Compute_DSE(emb_size[i], emb_per_bank, batch_num, split=split, voltage_level=voltage_level, bandwidth=bw)
                    qkvo_list.append([bw, voltage_level, split[0], split[1], 4*p1, 4*p11, power])
                    # FFN1
                    p2, p22, power = SRAM_PIM_Compute_DSE(emb_size[i], hid_per_bank, batch_num, split=split, voltage_level=voltage_level, bandwidth=bw)
                    p3, p33, power = SRAM_PIM_Compute_DSE(emb_size[i], hid_per_bank, batch_num, split=split, voltage_level=voltage_level, bandwidth=bw)
                    # FFN2
                    p4, p44, power = SRAM_PIM_Compute_DSE(hid_size[i], emb_per_bank, batch_num, split=split, voltage_level=voltage_level, bandwidth=bw)
                    # Insert
                    ffn_list.append([bw, voltage_level, split[0], split[1], p2+p3+p4, p22+p33+p44, power])
        
        data_list.append(qkvo_list)
        data_list.append(ffn_list)
    
    # Figure Gen
    plt.figure(figsize=(12, 3))
    for i in range(4):
        perf_list = data_list[i]
        plt.subplot(1, 4, i+1)
        plt.axvline(x=256, color='green', linestyle='--')
        plt.axvline(x=256*6.4, color='red', linestyle='--')
        
        idx = 0
        for k in range(len(perf_list)):
            line = perf_list[k]
            if line[2] == 1: # [1,4]
                if idx < 3: plt.plot(line[0], line[5], marker='1', color="#9400d3", label="(128,32)", alpha = 0.2*(3-line[1])+0.4)
                else: plt.plot(line[0], line[5], marker='1', color="#9400d3", alpha = 0.2*(3-line[1])+0.4)
            elif line[2] == 2: # [2,2]
                if idx < 3: plt.plot(line[0], line[5], marker='*', color="#003f7c", label="(256,16)", alpha = 0.2*(3-line[1])+0.4)
                else: plt.plot(line[0], line[5], marker='*', color="#003f7c", alpha = 0.2*(3-line[1])+0.4)
            else: # [4,1]
                if idx < 3: plt.plot(line[0], line[5], marker='.', color="#3557b8", label="(512,8)", alpha = 0.2*(3-line[1])+0.4)
                else: plt.plot(line[0], line[5], marker='.', color="#3557b8", alpha = 0.2*(3-line[1])+0.4)
            idx += 1
        plt.xlabel("Bandwidth / Bank (bit/s)", fontsize=16)
        if i == 0: plt.ylabel("Latency / ns", fontsize=16)
        plt.yscale("log")
        plt.title(model_name[i] + " " + layer_name[i], fontsize=16, weight='bold')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('figs/CompAir/Dse.pdf')