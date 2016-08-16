import csv
import sys

import axelrod as axl

from big_results import yield_data


def process_data(filename, outfilename):
    with open(outfilename, 'w') as outputfile:
        writer = csv.writer(outputfile)
        for line in yield_data(filename):
            writer.writerow(line)

def interactions_generator(player, opponents=None, repetitions=100, noise=0.01,
                           turns=200):
    if not opponents:
        opponents = [s() for s in axl.strategies]
        for opponent in opponents:
            match = axl.Match((player, opponent), turns, noise=noise)
            for rep in range(repetitions):
                match.play()
                yield (player.history, opponent.history)


def write_interactions(interactions_gen, filename):
    with open(filename, 'w') as handle:
        writer = csv.writer(handle)
        for h1, h2 in interactions_gen:
            row = [0, 0, '', '', ''.join(h1), ''.join(h2)]
            writer.writerow(row)

def create_interactions(s=""):
    g = interactions_generator(axl.KNN(), repetitions=100)
    filename = "/ssd/raw_train_extra.csv{}".format(s)
    write_interactions(g, filename)
    outfilename = "/ssd/train_extra.csv{}".format(s)
    process_data(filename, outfilename)

    g = interactions_generator(axl.KNN(), repetitions=20)
    filename = "/ssd/raw_test_extra.csv{}".format(s)
    write_interactions(g, filename)
    outfilename = "/ssd/test_extra.csv{}".format(s)
    process_data(filename, outfilename)


if __name__ == "__main__":
    try:
        s = sys.argv[1]
    except IndexError:
        s = ""
    create_interactions(s)


