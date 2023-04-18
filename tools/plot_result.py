import argparse
import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser(description='Plot the result of sensitivity analysis and partial quantization')
parser.add_argument('-sa', '--sensitivity-analysis-result', default=None,
                    help='Path to sensitivity analysis result CSV file')
parser.add_argument('-pq', '--partial-quantization-result', default=None,
                    help='Path to partial quantization result CSV file')
parser.add_argument('-ba', '--baseline-accuracy', type=float, default=None,
                    help='Baseline accuracy (accuracy without quantization)')


def main():
    args = parser.parse_args()
    assert args.sensitivity_analysis_result is not None
    assert args.partial_quantization_result is not None

    _ = plt.figure(figsize=(14, 8.5))
    plt.xlabel('layer')
    plt.ylabel('top-1 accuracy')

    pq_df = pd.read_csv(args.partial_quantization_result)
    xs = ['none' if eval(l) == [] else eval(l)[0] for l in pq_df.layers_not_quantized]

    # plot baseline accuracy
    plt.plot(xs, [args.baseline_accuracy] * len(xs), marker='', linestyle='--', label='baseline (no quantization)')

    # plot sensitivity analysis result
    sa_df = pd.read_csv(args.sensitivity_analysis_result)
    sa_xs, sa_ys = [], []
    for l in pq_df.layers_not_quantized:
        if eval(l) == []:
            continue
        sa_xs.append(eval(l)[0])
        sa_ys.append(sa_df[sa_df.layers_quantized == l].top1)
    plt.plot(sa_xs, sa_ys, marker='x', linewidth=0, label='sensitivity analysis (per-layer quantization)')

    # plot partial quantization result
    plt.plot(xs, pq_df.top1, marker='o', label='partial quantization (cumulative ablation)')

    plt.xticks(rotation=20, ha='right')
    plt.grid()
    plt.legend()

    plt.savefig('plot.png')

if __name__ == '__main__':
    main()
