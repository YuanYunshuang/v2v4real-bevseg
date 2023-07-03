import numpy as np
from opencood.tools.inference import main

if __name__=="__main__":
    exps = np.zeros((12, 2))
    n = 6
    exps[:n, 0] = np.arange(0, 6) * 0.1
    exps[n:, 1] = np.arange(0, 6) * 0.1

    for exp in exps:
        main(exp)