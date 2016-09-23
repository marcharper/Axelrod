from collections import defaultdict
import csv
import os

import numpy as np

import axelrod as axl

def selected_strategies():
    strategies = [
        axl.Adaptive,
        axl.Aggravater,
        axl.ALLCorALLD,
        axl.Alternator,
        axl.AlternatorHunter,
        axl.AntiCycler,
        axl.AntiTitForTat,
        axl.APavlov2006,
        axl.APavlov2011,
        axl.Appeaser,
        axl.AverageCopier,
        axl.BackStabber,
        axl.Bully,
        axl.Calculator,
        axl.Champion,
        axl.Cooperator,
        axl.ContriteTitForTat,
        axl.CyclerCCCCCD,
        axl.CyclerCCCD,
        axl.CyclerCCD,
        axl.CyclerDC,
        axl.CyclerDDC,
        axl.CycleHunter,
        axl.Davis,
        axl.Defector,
        axl.DoubleCrosser,
        axl.Eatherley,
        axl.Feld,
        axl.FirmButFair,
        axl.FoolMeForever,
        axl.FoolMeOnce,
        axl.ForgetfulFoolMeOnce,
        axl.ForgetfulGrudger,
        axl.Forgiver,
        axl.ForgivingTitForTat,
        axl.Fortress3,
        axl.Fortress4,
        axl.PSOGambler,
        axl.GTFT,
        axl.GoByMajority,
        axl.GoByMajority10,
        axl.GoByMajority20,
        axl.GoByMajority40,
        axl.GoByMajority5,
        axl.Handshake,
        axl.HardGoByMajority,
        axl.HardGoByMajority10,
        axl.HardGoByMajority20,
        axl.HardGoByMajority40,
        axl.HardGoByMajority5,
        axl.Golden,
        axl.Grofman,
        axl.Grudger,
        axl.Grumpy,
        axl.HardProber,
        axl.HardTitFor2Tats,
        axl.HardTitForTat,
        axl.Inverse,
        axl.InversePunisher,
        axl.Joss,
        axl.LimitedRetaliate,
        axl.LimitedRetaliate2,
        axl.LimitedRetaliate3,
        axl.EvolvedLookerUp,
        axl.MathConstantHunter,
        axl.MetaHunter,
        axl.NiceAverageCopier,
        axl.Nydegger,
        axl.OmegaTFT,
        axl.OnceBitten,
        axl.OppositeGrudger,
        axl.Pi,
        axl.Predator,
        axl.Prober,
        axl.Prober2,
        axl.Prober3,
        axl.PSOGambler,
        axl.Punisher,
        axl.Raider,
        axl.Random,
        axl.RandomHunter,
        axl.Retaliate,
        axl.Retaliate2,
        axl.Retaliate3,
        axl.Ripoff,
        axl.Shubik,
        axl.SlowTitForTwoTats,
        axl.SneakyTitForTat,
        axl.SoftJoss,
        axl.StochasticWSLS,
        axl.SolutionB1,
        axl.SolutionB5,
        axl.SuspiciousTitForTat,
        axl.Tester,
        axl.ThueMorse,
        axl.Thumper,
        axl.TitForTat,
        axl.TitFor2Tats,
        axl.TrickyCooperator,
        axl.TrickyDefector,
        axl.Tullock,
        axl.TwoTitsForTat,
        axl.WinStayLoseShift,
        axl.ZDExtort2,
        axl.ZDExtort2v2,
        axl.ZDExtort4,
        axl.ZDGen2,
        axl.ZDGTFT2,
        axl.ZDSet2,
        axl.e,
    ]

    strategies = [s for s in strategies if axl.obey_axelrod(s())]
    return strategies

# Features
# more history
# longest defection streak

mapping = {'C': 1, 'D': -1}
# mapping = {'C': 0, 'D': 1}
# mapping = {'C': 1, 'D': 0}


def zeros_and_ones(h):
    return list(map(lambda x: mapping[x], h))

def cumulative_context_counts(h1, h2):
    counts = []
    # counts = []
    d = defaultdict(int)
    for i, (p1, p2) in enumerate(zip(h1, h2)):
        d[str(p1) + str(p2)] += 1
        # if i >= 4:
        counts.append((d['CC'], d['CD'], d['DC'], d['DD']))
    return counts

# custom cumsum for zeros and ones Cs and Ds

def cumulative_cooperations(h):
    coops = []
    s = 0
    for play in h:
        if play == 'C':
            s += 1
        coops.append(s)
    return coops

def vectorize_interactions(h1, h2):
    # ds = np.cumsum(h1)
    # op_ds = np.cumsum(h2)
    coops = cumulative_cooperations(h1)
    op_coops = cumulative_cooperations(h2)
    ccs = cumulative_context_counts(h1, h2)
    h1 = zeros_and_ones(h1)
    h2 = zeros_and_ones(h2)
    # Handle N=0 and 1 separately
    yield [0] * 17 + [h2[0]]
    row = [1,
           coops[0], 1 - coops[0],
           op_coops[0], 1 - op_coops[0],
           h1[0], 0, h2[0], 0,
           0, h1[0], 0, h2[0]]
    row.extend(ccs[0])
    y = h2[1]
    row.append(y)
    yield row
    for i in range(2, len(h1)):
        row = [i,
               coops[i-1], i - coops[i-1],
               op_coops[i-1], i - op_coops[i-1],
               h1[0], h1[1], h2[0], h2[1],
               h1[i-2], h1[i-1], h2[i-2], h2[i-1]]
        row.extend(ccs[i-1])
        y = h2[i]
        row.append(y)
        yield row


def yield_data(filename):
    with open(filename) as handle:
        for line in handle:
            s = line.strip().split(',')
            h1, h2 = s[-2], s[-1]
            yield from vectorize_interactions(h1, h2)

def process_data():
    with open("/ssd/train1.csv", 'w') as outputfile:
        writer = csv.writer(outputfile)
        for line in yield_data("/ssd/interactions-train.csv1"):
            writer.writerow(line)
    with open("/ssd/test1.csv", 'w') as outputfile:
        writer = csv.writer(outputfile)
        for line in yield_data("/ssd/interactions-test.csv1"):
            writer.writerow(line)

def generate_data(turns=200, noise=0.02, reps=20):
    output_filename = "/ssd/interactions-train.csv1"
    os.remove(output_filename)
    players = [s() for s in axl.strategies]
    # This loop is to mix the data up more
    for i in range(reps):
        tournament = axl.Tournament(players, turns=turns, repetitions=1,
        noise=noise)
        tournament.play(filename=output_filename, build_results=False,
                        keep_interactions=False, processes=3)

    output_filename = "/ssd/interactions-test.csv1"
    os.remove(output_filename)
    tournament = axl.Tournament(players, turns=turns, repetitions=2,
                                noise=noise)
    tournament.play(filename=output_filename, build_results=False,
                    keep_interactions=False, processes=3)

if __name__ == "__main__":
    generate_data()
    process_data()
