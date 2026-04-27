import math

# [ISSCC'23] 7.2 A 28nm 64-kb 31.6-TFLOPS/W Digital-Domain Floating-Point- 
# Computing-Unit and Double-Bit 6T-SRAM Computing-in- 
# Memory Macro for Floating-Point CNNs (BF16)

def SRAM_PIM_Compute_API(input_size, output_size, macro_num, batch_num=1, split=[4,1]):
    
    max_width = 1024 # bits per bank / 8ns
    macro_num = 1
    spec = {
        "size": 64 * 1024, # bit = 64 * 64 * 16b
        "shape": [128, 8, split], # input, output, parallel (Limited by CAT) = 512 -> 8
        "channel": 128, # limited by CAT
        "area": 0.146, # mm2
        "mac_access_latency": 8, # ns
        "execution_latency": 8.6,
        # 0.9V: 6.8,
        # 0.8V: 8.6,
        "reduce_latency": 8, # ns
        "power_eff": 31.60,
        # 0.9V: 14.04
        # 0.8V: 20.13 
        # 0.7V: 25.87
        # 0.6V: 31.60
        "area_eff": 0.5
        # 0.9V: 2.05 # TFLOPS/mm2
        # 0.8V: 1.5 
        # 0.7V: 1.05
        # 0.6V: 0.5
    }

    power = spec["area"] * spec["area_eff"] / spec["power_eff"]
    print("\nTask: ", input_size, "->", output_size)
    print("Area: ", spec["area"] * spec["shape"][2][1] * spec["shape"][2][0], "mm2")
    print("Power: ", power * spec["shape"][2][1] * spec["shape"][2][0], "W")
    
    input_width = macro_num * spec["shape"][0] * spec["shape"][2][0]
    output_width = macro_num * spec["shape"][1] * spec["shape"][2][1]
    reduce_latency = spec["reduce_latency"] if spec["shape"][2][0] > 1 else 0
    
    reduce_size = math.ceil(input_size / input_width)
    reload_size = math.ceil(output_size / output_width)
    print("Input Split (Reduce): ", reduce_size, str(input_width)+"x")
    print("Output Split (Reload): ", reload_size, str(output_width)+"x")
    
    if input_width * 16 > max_width:
        load_inp_round = math.ceil(input_width * 16 / max_width)
        print("Load Input Round: ", load_inp_round)
    else:
        load_inp_round = 1
        print("Load Input Round: ", load_inp_round)
    
    if input_width * output_width * 16 > max_width:
        load_wgt_round = math.ceil(input_width * output_width * 16 / max_width)
        print("Load Weight Round: ", load_wgt_round)
    else:
        load_wgt_round = 1
        print("Load Weight Round: ", load_wgt_round)
    
    # 2048 bit = 16bit * 128
    ld_i_latency = spec["mac_access_latency"]
    ld_w_latency = spec["mac_access_latency"]
    print("Load Latency / Round: ", spec["mac_access_latency"], "ns")
    
    exe_latency = spec["execution_latency"]
    print("MAC Execute Latency / Round: ", exe_latency, "ns")
    
    overall_latency = reload_size * reduce_size * (ld_i_latency * load_inp_round * batch_num + ld_w_latency * load_wgt_round + exe_latency + reduce_latency)
    overall_no_pipe_latency = reload_size * reduce_size * (ld_i_latency * load_inp_round * batch_num + ld_w_latency * load_wgt_round + exe_latency * load_inp_round * batch_num + reduce_latency)
    
    print("Overall Latency (Pipeline): ", overall_latency, "ns")
    print("Overall Latency (No Pipe) : ", overall_no_pipe_latency, "ns")
    print("Latency / Batch (Pipeline): ", overall_latency / batch_num, "ns")
    print("Latency / Batch (No Pipe) : ", overall_no_pipe_latency / batch_num, "ns")
    
    return overall_latency, overall_no_pipe_latency
    

if __name__ == "__main__":
    inp_size = 1024
    inp_base = [1,2,4,8,16,32]
    batch_num = 1
    bank_num = 16
    macro_num = 8
    emb_size = 4096
    channel_num = 8*4
    res = []
    for base in inp_base:
        emb_per_bank = math.ceil(emb_size / (bank_num * channel_num))
        l1_p, l1_np = SRAM_PIM_Compute_API(inp_size * base, emb_per_bank, macro_num, batch_num)
        res.append(l1_np)
    print("Res:", res)
        