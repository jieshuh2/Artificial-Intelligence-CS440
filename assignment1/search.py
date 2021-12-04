from collections import deque
import heapq 
# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.start;
    dest = maze.waypoints[0];
    prev = {}
    dist = []
    visit = []

    for i in range(maze.size.y):
        row1 = []
        row2 = []
        for j in range (maze.size.x):
            row1.append(-1)
            row2.append(False)
        dist.append(row1)
        visit.append(row2)

    d = deque();
    d.append((start, 0))
 
    while d :
        point, distance = d.popleft();
        visit[point[0]][point[1]] = True;
        if point == dest:
            break;
        # if  dist[point[0]][point[1]] > 0 and  dist[point[0]][point[1]] < distance:
        #     continue
        for p in maze.neighbors(point[0],point[1]):
            if p[0] < 0 or p[0] > maze.size.y - 1 or p[1] < 0 or p[1] > maze.size.x - 1:
                continue
            if visit[p[0]][p[1]]:
                continue;
            if (dist[p[0]][p[1]] >= 0) and (dist[p[0]][p[1]] <= distance + 1):
                continue;
            d.append((p, dist[point[0]][point[1]] + 1))
            dist[p[0]][p[1]] = dist[point[0]][point[1]] + 1
            prev[p] = point

    path = [dest]
    while prev.get(dest):
        path.insert(0,prev.get(dest))
        dest = prev.get(dest)
    return path

def astar_single(maze):
    """
    Runs A star for part 2 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.start;
    dest = maze.waypoints[0];
    prev = {}
    dist = []
    visit = []

    for i in range(maze.size.y):
        row1 = []
        row2 = []
        for j in range (maze.size.x):
            row1.append(-1)
            row2.append(False)
        dist.append(row1)
        visit.append(row2)

    h = []
    heapq.heappush(h,(0 + abs(start[0] - dest[0]) + abs(start[1] - dest[1]), start))
 
    while h :
        heuristic, point = heapq.heappop(h);
        visit[point[0]][point[1]] = True;
        if point == dest:
            break;
        # if  dist[point[0]][point[1]] > 0 and dist[point[0]][point[1]] < heuristic:
        #     continue
        for p in maze.neighbors(point[0],point[1]):
            if p[0] < 0 or p[0] > maze.size.y - 1 or p[1] < 0 or p[1] > maze.size.x - 1:
                continue
            if visit[p[0]][p[1]]:
                continue;
            if (dist[p[0]][p[1]] >= 0) and (dist[p[0]][p[1]] <= dist[point[0]][point[1]] + 1):
                continue;
            heapq.heappush(h,(dist[point[0]][point[1]] + 1 + abs(p[0] - dest[0]) + abs(p[1] - dest[1]), p))
            dist[p[0]][p[1]] = dist[point[0]][point[1]] + 1
            prev[p] = point

    path = [dest]
    while prev.get(dest):
        path.insert(0,prev.get(dest))
        dest = prev.get(dest)
    return path

def astar_corner(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    start = maze.start;
    dest = maze.waypoints

    prev = {} #previous state
    visit = {} #visit state
    mist = {} # MST length
    dist = {} # distance of this state
    h = [] #priority queue

    state0 = (start, dest)
    mist[dest] = mst(dest, maze);
    dist[state0] = 0;
    heapq.heappush(h, (dist[state0] + nearest(start, dest) + mist[dest], state0))

    finalstate = ()
    while h :
        heuristic, state = heapq.heappop(h);
        visit[state] = True;
        point, waypoint = state;
        if len(waypoint) == 1 and point == waypoint[0]:
            finalstate = state
            break

        # if  dist[state] < heuristic:
        #     continue

        newwaypoint = list(waypoint)
        for i in range(len(waypoint)):
            if point == waypoint[i]:
                newwaypoint.pop(i)
        newwaypoint = tuple(newwaypoint)
        for p in maze.neighbors(point[0],point[1]):
            if p[0] < 0 or p[0] > maze.size.y - 1 or p[1] < 0 or p[1] > maze.size.x - 1:
                continue
            state1 = (p, newwaypoint)
            if visit.get(state1) == True:
                continue;
            if dist.get(state1) and dist.get(state1) <= dist.get(state) + 1:
                continue; 
            if mist.get(state1) == None:
                mist[state1] = mst(state1[1],maze)
            heapq.heappush(h,(dist[state] + nearest(state1[0], state1[1]) + mist[state1] + 1, state1))
            dist[state1] = dist[state] + 1
            prev[state1] = state

    assert(finalstate != ());
    path = [finalstate[0]]
    while prev.get(finalstate):
        path.insert(0,prev.get(finalstate)[0])
        finalstate = prev.get(finalstate)
    return path

def astar_multiple(maze):
    """
    Runs A star for part 4 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.start;
    dest = maze.waypoints

    prev = {} #previous state
    visit = {} #visit state
    mist = {} # MST length
    dist = {} # distance of this state
    h = [] #priority queue

    state0 = (start, dest)
    mist[dest] = mst(dest, maze);
    dist[state0] = 0;
    heapq.heappush(h, (dist[state0] + nearest(start, dest) + mist[dest], state0))

    finalstate = ()
    while h :
        heuristic, state = heapq.heappop(h);
        visit[state] = True;
        point, waypoint = state;
        if len(waypoint) == 1 and point == waypoint[0]:
            finalstate = state
            break

        # if  dist[state] < heuristic:
        #     continue

        newwaypoint = list(waypoint)
        for i in range(len(waypoint)):
            if point == waypoint[i]:
                newwaypoint.pop(i)
        newwaypoint = tuple(newwaypoint)
        for p in maze.neighbors(point[0],point[1]):
            if p[0] < 0 or p[0] > maze.size.y - 1 or p[1] < 0 or p[1] > maze.size.x - 1:
                continue
            state1 = (p, newwaypoint)
            if visit.get(state1) == True:
                continue;
            if dist.get(state1) and dist.get(state1) <= dist.get(state) + 1:
                continue; 
            if mist.get(state1) == None:
                mist[state1] = mst(state1[1],maze)
            heapq.heappush(h,(dist[state] + nearest(state1[0], state1[1]) + mist[state1] + 1, state1))
            dist[state1] = dist[state] + 1
            prev[state1] = state

    assert(finalstate != ());
    path = [finalstate[0]]
    while prev.get(finalstate):
        path.insert(0,prev.get(finalstate)[0])
        finalstate = prev.get(finalstate)
    return path

def fast(maze):
    """
    Runs suboptimal search algorithm for part 5.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    return []

def nearest(point, waypoint) :
    start = waypoint[0]
    near = abs(point[0] - start[0]) + abs(point[1] - start[1])
    for way in waypoint:
        if (abs(point[0] - way[0]) + abs(point[1] - way[1]) < near):
            near = abs(point[0] - way[0]) + abs(point[1] - way[1])
    return near


def distance(start, end, maze):
    dist = []
    visit = []
    prev = {}
    visit = []

    for i in range(maze.size.y):
        row1 = []
        row2 = []
        for j in range (maze.size.x):
            row1.append(-1)
            row2.append(False)
        dist.append(row1)
        visit.append(row2)

    h = []
    heapq.heappush(h,(0 + abs(start[0] - end[0]) + abs(start[1] - end[1]), start))
 
    while h :
        heuristic, point = heapq.heappop(h);
        visit[point[0]][point[1]] = True;
        if point == end:
            break;
        # if  dist[point[0]][point[1]] > 0 and dist[point[0]][point[1]] < heuristic:
        #     continue
        for p in maze.neighbors(point[0],point[1]):
            if p[0] < 0 or p[0] > maze.size.y - 1 or p[1] < 0 or p[1] > maze.size.x - 1:
                continue
            if visit[p[0]][p[1]]:
                continue;
            if (dist[p[0]][p[1]] >= 0) and (dist[p[0]][p[1]] <= dist[point[0]][point[1]] + 1):
                continue;
            heapq.heappush(h,(dist[point[0]][point[1]] + 1 + abs(p[0] - end[0]) + abs(p[1] - end[1]), p))
            dist[p[0]][p[1]] = dist[point[0]][point[1]] + 1
            prev[p] = point
    length = 0
    dest = end
    path = [dest]
    while prev.get(dest):
        path.insert(0,prev.get(dest))
        dest = prev.get(dest)
        length += 1
    return length

    
def mst(waypoints, maze):
    pq = [];
    visit = {}
    edgecount = 0;
    mstcost = 0;
    edges = [];
    current = waypoints[0]
    visit[current] = True;
    for points in waypoints:
        if points == current:
            continue
        visit[points] = False;
        heapq.heappush(pq,(abs(current[0] - points[0]) + abs(current[1] - points[1]), current, points))

    while pq and edgecount != len(waypoints) - 1:
        length, start, end = heapq.heappop(pq);
        if visit[end]:
            continue
        edge = (start, end)
        edges.append(edge)
        edgecount += 1
        mstcost += abs(start[0] - end[0]) + abs(start[1] - end[1])

        visit[end] = True
        for i in visit:
            if visit[i] == False:
                heapq.heappush(pq,(abs(i[0] - end[0]) + abs(i[1] - end[1]), end, i))

    return edgecount;
