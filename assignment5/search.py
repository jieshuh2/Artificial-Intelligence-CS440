import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove

###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})
#minmax help
def unfold(side, board, flags, depth, current):
      if current == depth - 1:
            if (side):
                  min = math.inf
                  minmove = []
                  movetree = {}
                  for move in generateMoves(side,board,flags):
                      newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
                      value = evaluate(newboard)
                      movetree[encode(*move)] = {}
                      if value < min:
                            min = value
                            minmove = [move]
                  # path = [minmove]
                  return [min, minmove, movetree]
            else:
                  max = -math.inf
                  maxmove = []
                  movetree = {}
                  for move in generateMoves(side,board,flags):
                      newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
                      value = evaluate(newboard)
                      movetree[encode(*move)] = {}
                      if value > max:
                            max = value
                            maxmove = [move]
                  return [max, maxmove, movetree]
      else:
            if (side):
                  min = math.inf
                  minpath = []
                  movetree = {}
                  for move in generateMoves(side,board,flags):
                            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
                            value, path, childtree = unfold(newside, newboard, newflags, depth, current + 1)
                            movetree[encode(*move)] = childtree
                            if value < min:
                                  min = value
                                  minpath = path
                                  minpath.insert(0, move)
                  return [min, minpath, movetree]
            else :
                  max = -math.inf
                  maxpath = []
                  movetree = {}
                  for move in generateMoves(side,board,flags):
                            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
                            value, path, childtree = unfold(newside, newboard, newflags, depth, current + 1)
                            movetree[encode(*move)] = childtree
                            if value > max:
                                  max = value
                                  maxpath = path
                                  maxpath.insert(0, move)
                  return [max, maxpath, movetree]
            
                

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    return unfold(side, board, flags, depth, 0);
    


          
    # raise NotImplementedError("you need to write this!")
     
def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    if depth == 0:
            return [evaluate(board), [], {}]
    else:
            if (side):
                  min = math.inf
                  minpath = []
                  movetree = {}
                  moves = generateMoves(side, board, flags)
                  # heuristic = []
                  for move in moves:
                  #       newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
                  #       heu = evaluate(newboard)
                  #       heuristic.append([heu, move])
                  # heuristic.sort()
                  # for pair in heuristic:
                        #     move = pair[1]
                            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
                            value, path, childtree = alphabeta(newside, newboard, newflags, depth - 1, alpha, beta)
                            movetree[encode(*move)] = childtree
                            if value < min:
                                  min = value
                                  minpath = path
                                  minpath.insert(0, move)
                            if (min < beta):
                                  beta = min
                            if (beta <= alpha):
                                  break
                  return [min, minpath, movetree]
            else :
                  max = -math.inf
                  maxpath = []
                  movetree = {}
                  # heuristic = []
                  moves = generateMoves(side, board, flags)
                  for move in moves:
                  #       newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
                  #       heu = evaluate(newboard)
                  #       heuristic.append([heu, move])
                  # heuristic.sort(reverse=True)
                  # for pair in heuristic:
                  #           move = pair[1]
                            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
                            value, path, childtree = alphabeta(newside, newboard, newflags, depth - 1, alpha, beta)
                            movetree[encode(*move)] = childtree
                            if value > max:
                                  max = value
                                  maxpath = path
                                  maxpath.insert(0, move)
                            if (max > alpha):
                                  alpha = max
                            if (alpha >= beta):
                                  break
                  return [max, maxpath, movetree]
    # raise NotImplementedError("you need to write this!")
    
def help(side, board, flags, depth, chooser):
      if depth == 0:
            return [evaluate(board), {}]
      movetree = {}
      moves = [ move for move in generateMoves(side, board, flags) ]
      if len(moves) > 0:
            move = chooser(moves)
            newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
            value, childtree = help(newside, newboard, newflags, depth - 1, chooser)
            movetree[encode(*move)] = childtree
            return [value, movetree]
      else:
            return [evaluate(board), {}]
      
      return [value, movetree]
      


def stochastic(side, board, flags, depth, breadth, chooser):
      '''
      Choose the best move based on breadth randomly chosen paths per move, of length depth-1.
      Return: (value, moveList, moveTree)
            value (float): average board value of the paths for the best-scoring move
            moveLists (list): any sequence of moves, of length depth, starting with the best move
            moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
      Input:
            side (boolean): True if player1 (Min) plays next, otherwise False
            board (2-tuple of lists): current board layout, used by generateMoves and makeMove
            flags (list of flags): list of flags, used by generateMoves and makeMove
            depth (int >=0): depth of the search (number of moves)
            breadth: number of different paths 
            chooser: a function similar to random.choice, but during autograding, might not be random.
      '''
      if (side):
            min = math.inf
            minpath = []
            movetree = {}
            moves = [ move for move in generateMoves(side, board, flags) ]
            for move in moves:
                  newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
                  newmoves = [newmove for newmove in generateMoves(newside, newboard, newflags)]
                  average = 0
                  tree = {}
                  for i in range(breadth):
                        childmove = chooser(newmoves)
                        childside, childboard, childflags = makeMove(newside, newboard, childmove[0], childmove[1], newflags, childmove[2])
                        childvalue, childtree = help(childside, childboard, childflags, depth - 2, chooser)
                        average += childvalue
                        tree[encode(*childmove)] = childtree
                  movetree[encode(*move)] = tree
                  average /= breadth
                  if min > average:
                        min = average
                        minpath = [move]
            return [min, minpath, movetree]
      else :
            max = -math.inf
            maxpath = []
            movetree = {}
            moves = [ move for move in generateMoves(side, board, flags) ]
            for move in moves:
                  newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
                  newmoves = [newmove for newmove in generateMoves(newside, newboard, newflags)]
                  average = 0
                  tree = {}
                  for i in range(breadth):
                        childmove = chooser(newmoves)
                        childside, childboard, childflags = makeMove(newside, newboard, childmove[0], childmove[1], newflags, childmove[2])
                        childvalue, childtree = help(childside, childboard, childflags, depth - 2, chooser)
                        average += childvalue
                        tree[encode(*childmove)] = childtree
                  movetree[encode(*move)] = tree
                  average /= breadth
                  if max < average:
                        max = average
                        maxpath = [move]
            return [max, maxpath, movetree]
