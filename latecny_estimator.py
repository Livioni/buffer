import numpy as np

#2080Ti table
# table = {
#     1024: [0.08655877,0.15786146,0.2307571651,0.29568449,\
#            0.42527773,0.48481530,0.539714996,0.598914945]
# }

#4090 table

#0.090219428
table = {
    1024: [0.090219428,0.151656122,0.19959718,0.280965798,0.339302976,
           0.388410434,0.4671203646,0.507089878,0.55965937,0.611210174]
}

def LatencyEstimator(canvas_size: int=1024,batch_size: int = 1)-> float:
    assert batch_size <= 10 and batch_size >= 1
    latency = table[canvas_size][batch_size-1]
    return latency