import numpy as np

def Ali_function_cost(Time: np.float64, Mem: float, CPU: float, GPU: float,\
                 Pcpu : np.float64 = 0.000127, Pmem: np.float64 = 0.0000127, Pgpu : np.float64 = 0.0007,\
                 basic_cost : np.float64 = 0.000001) -> np.float64:
    '''
    This function calculates the cost of serverless function for Aliyun, note that this is per-trigger cost.
    Accroding to https://help.aliyun.com/document_detail/54301.html
    Args:
        Time (float): The execution time of the function, in seconds.
        Mem (float): The memory size of the function, in GB.
        CPU (float): The CPU size of the function, in vCPU.
        GPU (float): The GPU memory of the function, in GB.
        Pcpu (float): The price of CPU, in CNY per vCPU per second.
        Pmem (float): The price of memory, in CNY per GB per second.
        Pgpu (float): The price of GPU, in CNY per GB per second.
    '''
    cost = Time * (Pcpu * CPU + Pmem * Mem + Pgpu * GPU) + basic_cost
    
    return cost

def Ali_idle_cost(Time: np.float64, Mem: float, Pmem : np.float64 = 0.0000127) -> np.float64:
    '''
    This function calculates the cost of idle instance for Aliyun, note that when the function is idle
    (keep alive), we only charge the memory cost.
    Accroding to https://help.aliyun.com/document_detail/54301.html
    Args:
        Time (float): The execution time of the function, in seconds.
        Mem (float): The memory size of the function, in GB.
        Pmem (float): The price of memory, in CNY per GB per second.
    '''
    cost = Time * Pmem * Mem
    return cost