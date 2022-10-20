import numpy as np


def readFiles():
    with open("p01_c.txt", "r") as knackpack_weigth_f, open("p01_p.txt") as items_profit_f, open("p01_w.txt", "r") as items_weight_f, open("p01_s.txt", "r") as solution_f :
        knackpack_weigth = np.loadtxt(knackpack_weigth_f).astype(int)
        items_profit = np.loadtxt(items_profit_f).astype(int)
        items_weight = np.loadtxt(items_weight_f).astype(int)
        solution = np.loadtxt(solution_f).astype(int)
        return knackpack_weigth, items_profit, items_weight, solution