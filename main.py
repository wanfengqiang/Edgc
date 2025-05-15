import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from json

    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorithms with C-Flat')
    parser.add_argument('--config', type=str, default='./exps/wa.json', help='Json file of settings')
    parser.add_argument('--edgc', action='store_true', help='Enable edgc')
    parser.add_argument('--rho', type=float, default=0.3, help='Perturbation radius')
    parser.add_argument('--lamb', type=float, default=0.3, help='First-order smoothing coefficient')

    return parser


if __name__ == '__main__':
    main()
