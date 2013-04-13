import argparse
import csv


def process(file):
    with open(file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            print row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some test data.')
    parser.add_argument(
        'file', metavar='file', type=str, help='the test CSV file'
    )
    args = parser.parse_args()
    process(args.file)
