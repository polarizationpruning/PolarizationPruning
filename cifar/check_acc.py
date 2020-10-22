import argparse
import os
import re
from statistics import mean, stdev
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Read CIFAR multi-run result')
    parser.add_argument('exp_name', type=str, metavar='NAME',
                        help='Experiment name')
    parser.add_argument('--output-path', type=str, metavar='NAME', default='./out',
                        help='The output log path')
    args = parser.parse_args()
    return args


def print_raw_data(raw_data_list, name: str):
    if len(raw_data_list) == 0:
        # do not print a empty list
        return
    avg = mean(raw_data_list)
    std = stdev(raw_data_list)
    raw_data_len = len(raw_data_list)

    raw_data_list = raw_data_list.copy()
    raw_data_list.append(avg)
    raw_data_list.append(std)
    raw_data_list = [f"{d:.3f}%" for d in raw_data_list]

    raw_data_header = []
    for i in range(raw_data_len):
        raw_data_header.append(f"{name}{i}")
    raw_data_header.append("Avg.")
    raw_data_header.append("Std.")
    print(f"{name} raw data:")
    print(",\t".join(raw_data_header))
    print(",".join(raw_data_list))
    pass


def main():
    args = parse_args()

    output_path = args.output_path
    output_path = os.path.expanduser(output_path)
    output_path = os.path.abspath(output_path)

    experiment_dirs = os.listdir(os.path.join(output_path, args.exp_name))

    acc_list: List[float] = []
    flops_list: List[float] = []
    for exp_dir in experiment_dirs:
        log_file_path = os.path.join(output_path, args.exp_name, exp_dir, "0.out")
        if not os.path.exists(log_file_path):
            continue
        with open(log_file_path) as log_file:
            log_file_content = log_file.readlines()

        best_acc_pattern = re.compile(r"Best accuracy: (\d*\.?\d*)")
        best_acc_line = log_file_content[-1].strip()  # the last line
        best_acc_match = re.fullmatch(pattern=best_acc_pattern, string=best_acc_line)

        if best_acc_match is not None:
            best_acc = float(best_acc_match.group(1))
            acc_list.append(best_acc)
        else:
            print("Match best acc failed. Got: ")
            print(best_acc_line)

        flops_pattern = re.compile(r"--> FLOPs in epoch \(grad\) ([0-9]+): (\d+(([\.\,]+)\d+)+), ratio: (\d*\.?\d*)")
        flops_line = log_file_content[-4].strip()
        last_flops_match = re.fullmatch(pattern=flops_pattern, string=flops_line)

        if last_flops_match is not None:
            last_flops = float(last_flops_match.group(5))
            flops_list.append(last_flops)
        else:
            print("Match FLOPs failed, got:")
            print(flops_line)

    if len(acc_list) == 0:
        print(f"Got nothing with experiment: {args.exp_name}")
    else:
        # display in percent
        acc_list = [acc * 100 for acc in acc_list]
        flops_list = [flop * 100 for flop in flops_list]

        compute_acc = len(acc_list) != 0
        compute_flops = len(flops_list) != 0

        if compute_acc:
            average_acc = mean(acc_list)
            std_acc = stdev(acc_list)

        if compute_flops:
            average_flops = mean(flops_list)
            std_flops = stdev(flops_list)

        print(f"----------  Experiment Summary of {args.exp_name}  ----------")
        print()

        print(f"Got {len(acc_list)} runs.")
        if compute_acc:
            print(f"Accuracy: {average_acc:.3f}±{std_acc:.3f} ({len(acc_list)} runs)")
        if compute_flops:
            print(f"FLOPs: {average_flops:.3f}±{std_flops:.3f} ({len(flops_list)} runs)")
        print()

        print_raw_data(raw_data_list=acc_list, name="Acc")
        print_raw_data(raw_data_list=flops_list, name="FLOPs")


if __name__ == '__main__':
    main()
