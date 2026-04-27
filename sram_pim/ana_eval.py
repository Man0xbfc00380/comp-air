import math
import csv
from ext.sram_pim.api import SRAM_PIM_Compute_API

if __name__ == "__main__":
    
    macro_num = 8
    bank_num = 16
    emb_size = 5120
    hid_size = 13824
    batch_nums = [1, 4, 16, 64]
    channel_nums = [16, 32, 64, 128]
    hw_reshape = False
    # Half / 1 / 2 / 4 / 8 Device(s)
    
    with open("ext/sram_pim/sram_shape.csv", "w", encoding="utf-8", newline="") as file:
        
        # Prepare CSV
        head = ["case", "batch", "channel", "latency(ns)-mF-hF", "latency(ns)-mF-hT", "latency(ns)-mT-hF", "latency(ns)-mT-hT"]
        csv_writer = csv.writer(file)
        csv_writer.writerow(head)
        
        # QKV
        for batch_num in batch_nums:
            for c in channel_nums:
                row = ["QKVO", batch_num, c]
                for mlp_reshape in [False, True]:
                    for hw_reshape in [False, True]:
                        emb_per_bank = math.ceil(emb_size / (bank_num * c))
                        hid_per_bank = math.ceil(hid_size / (bank_num * c))
                        if mlp_reshape:
                            real_emb_size = emb_size / 2
                            real_emb_per_bank = emb_per_bank * 2
                        else:
                            real_emb_size = emb_size
                            real_emb_per_bank = emb_per_bank
                        if hw_reshape == False: l1_p, _ = SRAM_PIM_Compute_API(real_emb_size, real_emb_per_bank, macro_num, batch_num)
                        else: l1_p, _ = SRAM_PIM_Compute_API(real_emb_size, real_emb_per_bank, macro_num, batch_num, [2,2])
                        print(mlp_reshape, hw_reshape)
                        row.append(l1_p)
                csv_writer.writerow(row)
            
        # FFN1
        for batch_num in batch_nums:
            for c in channel_nums:
                row = ["FFN1", batch_num, c]
                for mlp_reshape in [False, True]:
                    for hw_reshape in [False, True]:
                        emb_per_bank = math.ceil(emb_size / (bank_num * c))
                        hid_per_bank = math.ceil(hid_size / (bank_num * c))
                        if mlp_reshape:
                            real_emb_size = emb_size / 2
                            real_hid_per_bank = hid_per_bank * 2
                        else:
                            real_emb_size = emb_size
                            real_hid_per_bank = hid_per_bank
                        if hw_reshape == False: l2_p, _ = SRAM_PIM_Compute_API(real_emb_size, real_hid_per_bank, macro_num, batch_num)
                        else: l2_p, _ = SRAM_PIM_Compute_API(real_emb_size, real_hid_per_bank, macro_num, batch_num, [2,2])
                        row.append(l2_p)
                        print(mlp_reshape, hw_reshape)
                csv_writer.writerow(row)
            
        # FFN2
        for batch_num in batch_nums:
            for c in channel_nums:
                row = ["FFN2", batch_num, c]
                for mlp_reshape in [False, True]:
                    for hw_reshape in [False, True]:
                        emb_per_bank = math.ceil(emb_size / (bank_num * c))
                        hid_per_bank = math.ceil(hid_size / (bank_num * c))
                        if mlp_reshape:
                            real_hid_size = hid_size / 2
                            real_emb_per_bank = emb_per_bank * 2
                        else:
                            real_hid_size = hid_size
                            real_emb_per_bank = emb_per_bank
                        if hw_reshape == False: l3_p, _ = SRAM_PIM_Compute_API(real_hid_size, real_emb_per_bank, macro_num, batch_num)
                        else: l3_p, _ = SRAM_PIM_Compute_API(real_hid_size, real_emb_per_bank, macro_num, batch_num, [2,2])
                        row.append(l3_p)
                        print(mlp_reshape, hw_reshape)
                csv_writer.writerow(row)
        
        # File Close
        file.close()