"""
MCTS starter code. This is the only file you'll need to modify, although
you can take a look at game1.py to get a sense of how the game is structured.

@author Bryce Wiedenbeck
@author Anna Rafferty (adapted from original in Jan 2017)
"""

# These imports are used by the starter code.
import random
import argparse
import game1
import numpy as np

# You will want to use this import in your code
import math

# Whether to display the UCB rankings at each turn.
DISPLAY_BOARDS = False

# UCB_CONST value - you should experiment with different values
UCB_CONST = .5


class Node(object):
    """Node used in MCTS"""

    def __init__(self, state, parent_node):
        """Constructor for a new node representing game state
        state. parent_node is the Node that is the parent of this
        one in the MCTS tree. """
        self.state = state
        self.parent = parent_node
        self.children = {} # maps moves (keys) to Nodes (values); if you use it differently, you must also change addMove
        self.visits = 0
        self.value = float("nan")
        # Note: you may add additional fields if needed

    def addMove(self, move):
        """
        Adds a new node for the child resulting from move if one doesn't already exist.
        Returns true if a new node was added, false otherwise.
        """
        if move not in self.children:
            state = self.state.nextState(move)
            self.children[move] = Node(state, self)
            return True
        return False

    def getValue(self):
        """
        Gets the value estimate for the current node. Value estimates should correspond
        to the win percentage for the player at this node (accounting for draws as in
        the project description).
        """
        if self.visits == 0:
            return 0
        return float(self.value) / self.visits

    def updateValue(self, outcome):
        """Updates the value estimate for the node's state.
        outcome: +1 for 1st player win, -1 for 2nd player win, 0 for draw."""
        result = 0
        if outcome == 0:
            result = 0.5
        elif not outcome == self.state.turn:
            result = 1
        if math.isnan(self.value):
            self.value = result
        else:
            self.value += result
        self.visits += 1

    def UCBWeight(self):
        """Weight from the UCB formula used by parent to select a child.
        This node will be selected by parent with probability proportional
        to its weight.
        Our selection policy relies on v_i + sqrt(log(N) / n) where N is parent
        visits and n is our visits
        """
        v = self.getValue()
        d = math.sqrt(math.log(self.parent.visits) / float(self.visits))
        return v + UCB_CONST * d


def select(node):
    """Takes a node and recurssively picks children by the following rules:
    If a child has not been visited, add it to the tree and return it to be
    explored. Otherwise, recurssively go to a child with probability
    proportional to their UCBWeights.
    parameters:
        node: a node in the MCTS tree
    returns:
        Either node if node is terminal, or an unexpanded child of node attained
        via tree policy
    """
    if node.state.isTerminal():
        return node
    moves = node.state.getMoves()
    move = None
    for m in moves:
        added = node.addMove(m)
        if added:
            return node.children[m]
    if move is None:
        ucbWeights = [child.UCBWeight() for move, child in node.children.items()]
        newWeights = map(lambda n: n - min(ucbWeights), ucbWeights)
        totalUCB = sum(newWeights)
        proportions = [1.0/len(newWeights) for i in range(len(newWeights))]
        if not totalUCB == 0:
            proportions = [ucb/totalUCB for ucb in newWeights]
        move = np.random.choice(moves, p=proportions)
        return select(node.children[move])


def simulate(state):
    """Takes leaf node of MCTS's state and simulates a run down to a
    terminal, randomly choosing a move at each step.
    Returns:
        the value for the simulation."""
    if state.isTerminal():
        return state.value()
    moves = state.getMoves()
    move = random.choice(moves)
    return simulate(state.nextState(move))


def backpropagate(node, new_value, root):
    """Takes a new_value and recurssively backpropagates the new value to update
    up to the root node"""
    node.updateValue(new_value)
    if not node == root:
        backpropagate(node.parent, new_value, root)


def rollout(root):
    """Performs one rollout given the root node of MCTS."""
    node = select(root)
    new_value = simulate(node.state)
    backpropagate(node, new_value, root)


def maxMove(root):
    move = None
    max_value = -float('inf')
    children = root.children.items()
    for m, node in children:
        v = node.getValue()
        if v > max_value:
            max_value = v
            move = m
    return move


def minMove(root):
    move = None
    min_value = float('inf')
    children = root.children.items()
    for m, node in children:
        v = node.getValue()
        if v < min_value:
            min_value = v
            move = m
    return move


def getBestMove(root):
    if root.state.turn == -1:
        return minMove(root)
    return maxMove(root)


def MCTS(root, rollouts):
    """Select a move by Monte Carlo tree search.
    Plays rollouts random games from the root node to a terminal state.
    In each rollout, play proceeds according to UCB while all children have
    been expanded. The first node with unexpanded children has a random child
    expanded. After expansion, play proceeds by selecting uniform random moves.
    Upon reaching a terminal state, values are propagated back along the
    expanded portion of the path. After all rollouts are completed, the move
    generating the highest value child of root is returned.
    Inputs:
        node: the node for which we want to find the optimal move
        rollouts: the number of root-leaf traversals to run
    Return:
        The legal move from node.state with the highest value estimate
    """
    for i in range(rollouts):
        rollout(root)
    move = getBestMove(root)
    return move


def parse_args():
    """
    Parse command line arguments.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", type=int, default=0, help="Number of root-to-leaf "+\
                    "play-throughs that MCTS should run). Default=0 (random moves)")
    p.add_argument("--numGames", type=int, default=0, help="Number of games "+\
                    "to play). Default=1")
    p.add_argument("--second", action="store_true", help="Set this flag to "+\
                    "make your agent move second.")
    p.add_argument("--displayBoard", action="store_true", help="Set this flag to "+\
                    "make display the board at each MCTS turn with MCTS's rankings of moves.")
    p.add_argument("--rolloutsSecondMCTSAgent", type=int, default=0, help="If non-0, other player "+\
                    "will also be an MCTS agent and will use the number of rollouts set with this "+\
                    "argument. Default=0 (other player is random)")
    p.add_argument("--ucbConst", type=float, default=.5, help="Value for the UCB exploration"+\
                    "constant. Default=.5")
    args = p.parse_args()
    if args.displayBoard:
        DISPLAY_BOARDS = True
    UCB_CONST = args.ucbConst
    return args


def randomMove(node):
    """
    Choose a valid move uniformly at random.
    """
    move = random.choice(node.state.getMoves())
    node.addMove(move)
    return move


def runMultipleGames(numGames, args):
    """
    Runs numGames games, with no printing except for a report on which game
    number is currently being played, and reports final number
    of games won by player 1 and draws. args specifies whether player 1 or
    player 2 is MCTS and how many rollouts to use. For multiple games, you
    probably do not want to include the --displayBoard option in args, as
    this will do lots of printing and make running relatively slow.
    """
    player1GamesWon = 0
    draws = 0
    for i in range(numGames):
        print "Game " + str(i)
        node = playGame(args)
        winner = node.state.value()
        if winner == 1:
            player1GamesWon += 1
        elif winner == 0:
            draws += 1
    print "Player 1 games won: " + str(player1GamesWon) + "/" + str(numGames)
    print "Draws: " + str(draws) + "/" + str(numGames)


def playGame(args):
    """
    Play one game against another player.
    args specifies whether player 1 or player 2 is MCTS (
    or both if rolloutsSecondMCTSAgent is non-zero)
    and how many rollouts to use.
    Returns the final terminal node for the game.
    """
    # Make start state and root of MCTS tree
    start_state = game1.newGame()
    root1 = Node(start_state, None)
    if args.rolloutsSecondMCTSAgent != 0:
        root2 = Node(start_state, None)

    # Run MCTS
    node = root1
    if args.rolloutsSecondMCTSAgent != 0:
        node2 = root2
    while not node.state.isTerminal():
        if (not args.second and node.state.turn == 1) or \
                (args.second and node.state.turn == -1):
            move = MCTS(node, args.rollouts)
        else:
            if args.rolloutsSecondMCTSAgent == 0:
                move = randomMove(node)
            else:
                move = MCTS(node2, args.rolloutsSecondMCTSAgent)

        node.addMove(move)
        node = node.children[move]
        if args.rolloutsSecondMCTSAgent != 0:
            node2.addMove(move)
            node2 = node2.children[move]
    return node




def main():
    """
    Play a game of connect 4, using MCTS to choose the moves for one of the players.
    args on command line set properties; see parse_args() for details.
    """
    # Get commandline arguments
    args = parse_args()

    if args.numGames > 1:
        runMultipleGames(args.numGames, args)
    else:
        # Play the game
        node = playGame(args)

        # Print result
        winner = node.state.value()
        print game1.print_board(node.state)
        if winner == 1:
            print "Player 1 wins"
        elif winner == -1:
            print "Player 2 wins"
        else:
            print "It's a draw"


if __name__ == "__main__":
    main()
