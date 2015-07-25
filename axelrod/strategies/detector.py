import random
from axelrod import Player


class ShenanigansDetector(Player):
    """
    Detects and punishes cheating.
    """

    name = "Shenanigans Detector"
    memory_depth = float('inf')  # Long memory

    def __init__(self):
        super(ShenanigansDetector, self).__init__()
        self.__my_history = []
        self.__opponents_history = []
        self._shenanigans = False
        self.D_count = 0
        #self.reset()

    def reset(self):
        print "reset"
        self.history = []
        self.__my_history = []
        self.__opponents_history = []
        self._shenanigans = False
        self.D_count = 0

    def __setattr__(self, name, val):
        """Stops any other strategy altering the methods of this class """

        if name == 'strategy':
            self._shenanigans = True
            print "Attempted writing of self.strategy"
        else:
            self.__dict__[name] = val

    #def play(self, opponent, noise=0):
        #"""Override the base class to record history in a second
        #place for both players."""

        ## Detect History Manipulation Shenanigans
        #if (self.history != self.__my_history) or (opponent.history != self.__opponents_history):
            #self._shenanigans = True

        ## Contents of Player.play(), modified slightly
        #s1 = self.__strategy(opponent)
        #s2 = opponent.strategy(self)
        #if noise:
            #s1, s2 = self._add_noise(noise, s1, s2)
        #self.history.append(s1)
        #opponent.history.append(s2)

        ## Save history again
        #self.__my_history.append(s1)
        #self.__opponents_history.append(s2)


    def record_and_return(self, move):
        self.__my_history.append(move)
        return move

    def strategy(self, opponent):
        # Update opponent's history copy. They could be lying but unless play
        # can be overridden, full detection of self history by opponent is
        # incomplete
        if len(self.__my_history):
            self.__opponents_history.append(opponent.history[-1])
        else:
            print opponent.name, self._shenanigans

        # Detect History Manipulation Shenanigans
        if (self.history != self.__my_history) or (opponent.history != self.__opponents_history) or (len(opponent.history) != len(self.__my_history)):
            self._shenanigans = True
            print "History mismatch"
            print opponent.name
            print self.history
            print self.__my_history
            print opponent.history
            print self.__opponents_history

        if self._shenanigans:
            exit()
        else:
            return self.record_and_return('C')
            ## Fool Me Once
            #if not len(self.__my_history):
                #return self.record_and_return('C')
            #if self.__opponents_history[-1] == 'D':
                #self.D_count += 1
            #if self.D_count > 1:
                #return self.record_and_return('D')
            #return self.record_and_return('C')
