import numpy as np

table = {
    1024: [0.08655877,0.15786146,0.2307571651,0.29568449,\
           0.42527773,0.48481530,0.539714996,0.598914945]
}

def LatencyEstimator(canvas_size: int=1024,batch_size: int = 1)-> float:
    assert batch_size <= 8 and batch_size >= 1
    latency = table[canvas_size][batch_size-1]
    return latency