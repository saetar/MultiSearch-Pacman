# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
    """


    def getAction(self, gameState):
        """
        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        This evaluation function is not particularly good; using information
        from the game state would allow it to be much better, although still
        not as good as an agent that plans. You may find the information listed
        below helpful in later parts of the project (e.g., when designing
        an evaluation function for your planning agent).
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def max_value(self, gameState, depth):
        """
            Acts as the max-value function in our recurssive minimax algorithm.
            Uses agentIndex = 0 because pacman wants to maximize.
        """
        max_val = -float("inf")
        actions = gameState.getLegalActions(0)
        if depth == self.depth or len(actions) == 0:
            return None, self.evaluationFunction(gameState)
        action = None
        if len(actions) == 0:
            print "depth: {}\ngameState: {}".format(depth, gameState)
        for a in actions:
            successor_state = gameState.generateSuccessor(0, a)
            next_action, result = self.min_value(successor_state, depth, 1)
            if result > max_val:
                max_val = result
                action = a
        return action, max_val

    def min_value(self, gameState, depth, agentIndex):
        """
            Acts as the min-value function in our recurssive minimax algorithm.
            Takes agentIndex to run multiple mins given multiple ghosts
        """
        if agentIndex == gameState.getNumAgents():
            return self.max_value(gameState, depth+1)
        min_val = float("inf")
        actions = gameState.getLegalActions(agentIndex)
        if len(actions) == 0:
            return None, self.evaluationFunction(gameState)
        action = None
        for a in actions:
            successor_state = gameState.generateSuccessor(agentIndex, a)
            next_action, result = self.min_value(successor_state, depth, agentIndex + 1)
            if result < min_val:
                action = a
                min_val = result
        return action, min_val

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
        """
        action, results = self.max_value(gameState, 0)
        actions = gameState.getLegalActions(0)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def max_value(self, gameState, depth, alpha, beta):
        """
            Acts as the max-value function in our recurssive minimax algorithm.
            Uses agentIndex = 0 because pacman wants to maximize.
        """
        actions = gameState.getLegalActions(0)
        if depth == self.depth or len(actions) == 0:
            return None, self.evaluationFunction(gameState)
        max_val = -float("inf")
        action = None
        for a in actions:
            successor_state = gameState.generateSuccessor(0, a)
            next_action, result = self.min_value(successor_state, depth, 1, alpha, beta)
            if result > max_val:
                max_val = result
                action = a
                alpha = max_val
            if alpha > beta:
                break
        return action, max_val

    def min_value(self, gameState, depth, agentIndex, alpha, beta):
        """
            Acts as the min-value function in our recurssive minimax algorithm.
            Takes agentIndex to run multiple mins given multiple ghosts
        """
        if agentIndex == gameState.getNumAgents():
            return self.max_value(gameState, depth+1, alpha, beta)
        min_val = float("inf")
        actions = gameState.getLegalActions(agentIndex)
        action = None
        if len(actions) == 0:
            return action, self.evaluationFunction(gameState)
        for a in actions:
            successor_state = gameState.generateSuccessor(agentIndex, a)
            next_action, result = self.min_value(successor_state, depth, agentIndex + 1, alpha, beta)
            if result < min_val:
                action = a
                min_val = result
                beta = result
            if beta < alpha:
                break
        return action, min_val

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        action, val = self.max_value(gameState, 0, float('-inf'), float('inf'))
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def max_value(self, gameState, depth):
        """
            Acts as the max-value function in our recurssive minimax algorithm.
            Uses agentIndex = 0 because pacman wants to maximize.
        """
        actions = gameState.getLegalActions(0)
        if depth == self.depth or len(actions) == 0:
            return None, self.evaluationFunction(gameState)
        max_val = -float("inf")
        action = None
        for a in actions:
            successor_state = gameState.generateSuccessor(0, a)
            result = self.expect_value(successor_state, depth, 1)
            if result > max_val:
                max_val = result
                action = a
        return action, max_val

    def expect_value(self, gameState, depth, agentIndex):
        """
            Acts as the expected-value function in our recurssive minimax algorithm.
            Takes agentIndex to run multiple mins given multiple ghosts
        """
        if agentIndex == gameState.getNumAgents():
            action, max_val = self.max_value(gameState, depth+1)
            return max_val
        total_val = 0
        actions = gameState.getLegalActions(agentIndex)
        if len(actions) == 0:
            return self.evaluationFunction(gameState)
        action = None
        for a in actions:
            successor_state = gameState.generateSuccessor(agentIndex, a)
            result = self.expect_value(successor_state, depth, agentIndex + 1)
            total_val += result
        score = 0
        if len(actions) == 0:
            score = float('-inf')
        else:
            score = float(total_val) / len(actions)
        return score


    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        action, result = self.max_value(gameState, 0)
        return action


def getFoodLengthScore(state):
    return len(state.getFood().asList())


def getAverageFoodDistancesScore(state):
    if len(state.getFood().asList()) == 0:
        return float('inf')
    return float(sum(getFoodDistances(state))) / getFoodLengthScore(state)


def getFoodDistances(state):
    pacPos = state.getPacmanPosition()
    food = state.getFood().asList()
    foodDistances = sorted(map(lambda xy: dist(pacPos, xy), food))
    return foodDistances


def getGhostPositions(state, exclude=False):
    if not exclude:
        return [state.configuration.getPosition() for state in state.getGhostStates()]
    return [state.configuration.getPosition() if state.scaredTimer > 0 else 0 for state in state.getGhostStates()]


def getGhostScaredTimeLeft(state):
    return [ghostState.scaredTimer for ghostState in state.getGhostStates()]


def getGhostScaredScore(state):
    return float(sum(getGhostScaredTimeLeft(state))) / len(state.getGhostStates())


def getGhostDistanceScore(state, threshold=3):
    distances = getGhostPositions(state, exclude=True)
    sortedDistances = sorted(distances)
    closest = sortedDistances[0]
    if closest >= threshold:
        return 0
    else:
        return -1000


def goForSpecialFood(state):
    pacPos = state.getPacmanPosition()
    ghosts = getGhostPositions(state, exclude=True)
    capsules = state.data.capsules
    capDist = sorted([dist(pacPos, capsule) for capsule in capsules])
    return int(ghosts[0] < 3 and capDist[0] < 3)



def getClosestFoodScore(state):
    foodDistances = getFoodDistances(state)
    if len(foodDistances) == 0:
        return float('inf')
    return -foodDistances[0]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: The biggest goal is to collect all the food, so if len(food)
      is 0, then that is the best score (+inf). Otherwise, we want that fewer
      food is more desirable, so len(food) ** -1 gives us a food score.


    """
    closestFoodScore = getClosestFoodScore(currentGameState)
    averageFoodDistance = getAverageFoodDistancesScore(currentGameState)
    ghostScaredScore = getGhostScaredScore(currentGameState)
    ghostDistanceScore = getGhostDistanceScore(currentGameState)
    scores = [closestFoodScore, averageFoodDistance, ghostScaredScore, ghostDistanceScore, currentGameState.data.score]
    weights = [0.4, 0.1, 0.5, 0.5, 0.4]
    total_score = sum([scores[i] * weights[i] for i in range(len(scores))])
    return total_score


def dist(xy1, xy2):
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] + xy2[1])

# Abbreviation
better = betterEvaluationFunction
