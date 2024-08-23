import argparse

from matplotlib import pyplot as plt

from data import generate_dataset
from trainer import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description='running discrete time sgm diffusion w/ flax')
    parser.add_argument('--seed',
                        default=42,
                        type=int,
                        help='random seed to be used in numpy')
    parser.add_argument('--num_samples',
                        default=20,
                        type=int,
                        help='number of samples')

    args = parser.parse_args()
    configs = args.__dict__

    sample_dataset = generate_dataset(configs['num_samples'])
    plt.scatter(sample_dataset[:, 0], sample_dataset[:, 1])
    plt.savefig('example.png')
    plt.clf()

    run_experiment(seed=configs['seed'], dataset=sample_dataset)


if __name__ == '__main__':
    main()
