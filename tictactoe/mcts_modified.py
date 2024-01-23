from mcts_node import MCTSNode
from p2_t3 import Board
from random import choice
from math import sqrt, log

num_nodes = 100
explore_faction = 2.

def traverse_nodes(node: MCTSNode, board: Board, state, bot_identity: int):
    """ Traverses the tree until the end criterion are met.
    e.g. find the best expandable node (node with untried action) if it exist,
    or else a terminal node

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        identity:   The bot's identity, either 1 or 2

    Returns:
        node: A node from which the next stage of the search can proceed.
        state: The state associated with that node

    """
    currentNode = node
    newState = state

    # If the current node has untried actions and the game did not end
    if len(currentNode.untried_actions) < 1 and not board.is_ended(newState):

        bestUCB = -999999
        bestNode = None
        bestAction = None

        # Iterate over all child nodes
        for child in currentNode.child_nodes.keys():
            # If the current node is not visited or has an untried action
            if currentNode.child_nodes[child].visits == 0 or len(currentNode.untried_actions) > 0:
                return currentNode.child_nodes[child], board.next_state(state, child)

            childUCB = ucb(currentNode.child_nodes[child], bot_identity == board.current_player(state))

            # Update the current best node and ucb
            if childUCB > bestUCB:
                bestUCB = childUCB
                bestAction = child
                bestNode = currentNode.child_nodes[child]

        # Set the current best node as the starting point for the next round of the loop
        currentNode = bestNode
        newState = board.next_state(state, bestAction)
        # Recursively continue the search
        return traverse_nodes(currentNode, board, newState, bot_identity)

    return currentNode, newState

def expand_leaf(node: MCTSNode, board: Board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node (if it is non-terminal).

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:
        node: The added child node
        state: The state associated with that node

    """
    # pass
    # expand only if child nodes have been visited
    if len(node.untried_actions) > 0:
        # parent transititions state
        parentAction = node.untried_actions.pop()
        # new state
        nextState = board.next_state(state, parentAction)
        # list of actions for new node
        actionList = board.legal_actions(nextState)

        # new node
        newNode = MCTSNode(node, parentAction, actionList)
        node.child_nodes[parentAction] = newNode

    
        return newNode, nextState
    
    return node, state

def rollout(board: Board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.
    
    Returns:
        state: The terminal game state

    """
    # vanilla rollout for experiment #1
    # while not board.is_ended(state):
    #     # do random
    #     randomAction = choice(board.legal_actions(state))
    #     # update state
    #     state = board.next_state(state, randomAction)
    
    # return state

    while not board.is_ended(state):
        nextAction = None
        validActions = []
        for action in board.legal_actions(state):
            nextState = board.next_state(state, action)
            nextStateScore = board.points_values(nextState)
            currPlayer = board.current_player(state)

            # if next action will end game 
            if nextStateScore != None:
                # choose action that will win the game
                if nextStateScore[currPlayer] == 1:
                    nextAction = action
                    break
            
                # avoid actions that will lose the game
                elif nextStateScore[currPlayer] == -1:
                    continue

            R, C, r, c = action
            nextStateBoxes = board.owned_boxes(nextState)

            # if next action will gain board for self, choose it
            if nextStateBoxes[(R, C)] == currPlayer:
                nextAction = action
                break

            # if next action will gain board for enemy, avoid it
            if nextStateBoxes[(R, C)] == 3 - currPlayer: # 2 if currPlayer == 1; 1 if currPlayer == 2
                continue

            # if neither good nor bad, add to valid actions
            validActions.append(action)

        # pick randomly from valid actions
        if nextAction == None:
            nextAction = choice(validActions)
        
        state = board.next_state(state, nextAction)

    return state

def backpropagate(node: MCTSNode|None, won: bool):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    # pass
    node.wins += won
    node.visits += 1
    if not node.parent:
        return None
    else:
        return backpropagate(node.parent, won)

def ucb(node: MCTSNode, is_opponent: bool):
    """ Calculates the UCB value for the given node from the perspective of the bot

    Args:
        node:        A node.
        is_opponent: A boolean indicating whether or not the last action was performed by the MCTS bot

    Returns:
        The value of the UCB function for the given node
    """
    if is_opponent:
        ucb_value = node.wins / node.visits + explore_faction * (sqrt(log(node.parent.visits) / node.visits))
    else:
        ucb_value = (1 - node.wins / node.visits) + explore_faction * (sqrt(log(node.parent.visits) / node.visits))

    return ucb_value

def get_best_action(root_node: MCTSNode):
    """ Selects the best action from the root node in the MCTS tree

    Args:
        root_node:   The root node

    Returns:
        action: The best action from the root node

    """
    action = None
    bestRate = -999999

    for child in root_node.child_nodes.values(): 
        currRate = child.wins / child.visits

        if currRate > bestRate:
            bestRate = currRate
            action = child.parent_action  

    return action

def is_win(board: Board, state, identity_of_bot: int):
    # checks if state is a win state for identity_of_bot
    outcome = board.points_values(state)
    assert outcome is not None, "is_win was called on a non-terminal state"
    return outcome[identity_of_bot] == 1

def think(board: Board, current_state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        state:  The state of the game.

    Returns:    The action to be taken.

    """
    bot_identity = board.current_player(current_state) # 1 or 2
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(current_state))

    for _ in range(num_nodes):
        # Copy the game for sampling a playthrough
        state = current_state

        # Start at root
        node = root_node

        # Do MCTS - This is all you!
        leafNode, newState = traverse_nodes(node, board, state, bot_identity)

        newLeaf, newState = expand_leaf(leafNode, board, newState)

        newState = rollout(board, newState)

        winValue = board.points_values(newState)

        backpropagate(newLeaf, winValue[bot_identity])
            
    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    best_action = get_best_action(root_node)
    
    print(f"Action chosen: {best_action}")
    return best_action
