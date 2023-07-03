import numpy as np
from opencood.tools.inference import main

if __name__=="__main__":
    exps = np.zeros((10, 2))
    exps[:5, 0] = np.arange(1, 6) * 0.1
    exps[5:, 1] = np.arange(1, 6) * 0.1

    for exp in exps:
        main(exp)