#!/usr/bin/env python3
import matplotlib.pyplot as plt
import argparse
import os
from recorder import Recorder


def fed_args():
    """
    Arguments for running postprocessing on FedD3
    :return: Arguments for postprocessing on FedD3
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-rr', '--sys-res_root', type=str, required=True, help='Root directory of the results')

    args = parser.parse_args()
    return args


def res_eval():
    """
    Main function for result evaluation
    """
    args = fed_args()

    recorder = Recorder()

    res_files = [f for f in os.listdir(args.sys_res_root)]
    for f in res_files:
        recorder.load(os.path.join(args.sys_res_root, f), label=f)
    recorder.plot()
    plt.show()


if __name__ == "__main__":
    res_eval()