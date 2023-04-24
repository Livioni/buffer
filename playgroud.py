from utils.binpack import BinPack

BINPACK = BinPack(bin_size=(4,8))
BINPACK.insert((2.55, 4), (2, 2), (4, 5), (4, 4), (2, 2), (3, 2), heuristic='best_fit')
result = BINPACK.print_stats()
BINPACK.visualize_packing(result)