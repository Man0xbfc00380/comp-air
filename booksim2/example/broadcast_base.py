import os
import mmap

import _booksim_path
from greenlet import greenlet
from booksim_sync import BookSim2Sync

def broadcast_comp_air(sim: BookSim2Sync, src: int, targs: int, step: int, data: float):
    """
    Perform a broadcast operation in the simulation environment.

    Args:
        sim (BookSim2Sync): The simulation object.
        src (int): The source node for the broadcast.
        targs (int): The number of target nodes, must be 2, 4, or 8.
        step (int): The step size for calculating target node coordinates.
        data (float): The data to be broadcast.
    """
    # Ensure that the number of target nodes is either 2, 4, or 8
    for i in range(targs):
        sim.inject(3, 0, step*i, 1)

    # Run the simulation for 5 steps
    for i in range(targs):
        sim.run_step()

if __name__ == "__main__":
    sim = BookSim2Sync()
    broadcast_comp_air(sim, src=0, targs=8, step=4, data=2.5)
    sim.run_step(end=True) # End