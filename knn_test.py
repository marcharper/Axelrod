import axelrod as axl

# strategies = list(reversed(axl.ordinary_strategies))
#
# repetitions = 20
#
# def play_matches(player1, player2, repetitions):
#     match = axl.Match((player1, player2), turns=200)
#     for repetition in range(repetitions):
#         match.play()

player1 = axl.KNN()
player2 = axl.Random()

match = axl.Match((player1, player2), turns=200)
match.play()
# print(match.sparklines())
print(match.final_score())

players = axl.strategies()

tournament = axl.Tournament(players=players, repetitions=20)
