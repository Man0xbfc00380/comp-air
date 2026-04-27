import os
import mmap

import _booksim_path
from greenlet import greenlet
from booksim_sync import BookSim2Sync

def reduce_comp_air(sim: BookSim2Sync, sources: int, step: int, data: float, para_num: int = 1, op: int = 0):
    """
    Perform a reduction operation using computation air packets in the simulation.

    Args:
        sim (BookSim2Sync): The simulation object.
        sources (int): The number of source nodes, must be 4, 8, or 16.
        step (int): The step size for calculating node coordinates.
        data (float): The data to be used in the reduction operation.
        op (int, optional): The operation code. Defaults to 0.

    Returns:
        str: The result token obtained from the simulation info.
    """
    # Ensure that the number of source nodes is either 4, 8, or 16
    assert sources == 16 or sources == 8 or sources == 4
    paral_offset = [0,1,8,9]
    
    # Initialize a list to store reduction counter
    node = []
    for i in range(sources):
        # Calculate the node coordinate and append it to the list
        node.append((i%4)*step + 8*(i//4)*step)
        # Inject computation air packets to initialize nodes
        if i * 2 != sources:
            for j in range(para_num):
                sim.inject_comp_air(ca_type=4, data=2, t_inject=0, src=paral_offset[j], iter_tag=0, pkg_size=1, x_0=(i%4)*step, y_0=(i//4)*step, op_0=1+4+8)
                
    # Run the simulation for the number of source nodes
    for i in range(sources * para_num): 
        sim.run_step()
    # Initialize a list to store the path for each source node
    
    for i in range(sources):
        sim.inject(1, step*1, 0, 1)
        
    for i in range(sources):
        sim.run_step()
        
    for i in range(sources):
        sim.inject_comp_air(ca_type=0, data=2, t_inject=0, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=1+4+8)
        if i + 1 < sources : sim.run_step()
    
    # Initialize the result token
    res_token = ""
    # Run the simulation for a few more steps to collect results
    for i in range(sources*para_num+10): 
        sim.run_step()
        # Check if the simulation info has enough data
        if len(sim.info) < 2: return res_token
        else: res_token = sim.info
        
if __name__ == "__main__":
    
    sim = BookSim2Sync()
    res = reduce_comp_air(sim, sources=16, step=2, data=2.5, para_num=4)
    print("[***]", res)
    
    """
    
    # manual config (0~3 +-*/ | +8 write reg | +4 go iter)
    # L0 (x, y)
    # - (0,0) (0,2) -> (0,0) # 2
    # - (2,0) (2,2) -> (2,0) # 2
    # L1 (x, y)
    # - (0,0) (2,0) -> (2,2) # 2
    
    sim.inject_comp_air(ca_type=4, data=2, t_inject=1, src=0, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=1+4+8)
    sim.inject_comp_air(ca_type=4, data=2, t_inject=1, src=2, iter_tag=0, pkg_size=1, x_0=0, y_0=2, op_0=1+4+8)
    sim.inject_comp_air(ca_type=4, data=2, t_inject=1, src=2, iter_tag=0, pkg_size=1, x_0=0, y_0=0, op_0=1+4+8)
    for i in range(4): 
        sim.run_step()
    
    print("----------------")
    # reduce
    # (0,0) -> 1
    sim.inject_comp_air(ca_type=1, data=1, t_inject=1, src=0, iter_tag=0, pkg_size=1,   
                                                                x_0=0, y_0=0, op_0=0+8, 
                                                                x_1=2, y_1=2, op_1=0+8) # 3
    # (0,2) -> 2
    sim.inject_comp_air(ca_type=1, data=2, t_inject=1, src=16, iter_tag=0, pkg_size=1,   
                                                                x_0=0, y_0=-2, op_0=0+8, 
                                                                x_1=2, y_1=2, op_1=0+8) # 4
    
    # (2,0) -> 3
    sim.inject_comp_air(ca_type=1, data=3, t_inject=1, src=2, iter_tag=0, pkg_size=1,   
                                                                x_0=0, y_0=0, op_0=0+8, 
                                                                x_1=0, y_1=2, op_1=0+8) # 5
    # (2,2) -> 4
    sim.inject_comp_air(ca_type=1, data=4, t_inject=1, src=18, iter_tag=0, pkg_size=1,   
                                                                x_0=0, y_0=-2, op_0=0+8, 
                                                                x_1=0, y_1=2, op_1=0+8) # 6
    
    for i in range(2): 
        sim.run_step()
        print(sim.info)
    
    # End
    sim.run_step(end=True)
    """