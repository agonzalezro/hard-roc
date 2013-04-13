import argparse
import pandas


def main(file):
    csv = pandas.read_csv(file)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some test data.')
    parser.add_argument(
        'file', metavar='file', type=str, help='the test CSV file'
    )
    args = parser.parse_args()
    main(args.file)
