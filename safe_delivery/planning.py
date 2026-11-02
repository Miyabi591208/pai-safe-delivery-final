"""A* on a known static map with footprint inflation and checked shortcuts."""
import heapq
import math
import numpy as np

BOUND = 2.8
GRID = .10
INFLATION = .39  # physical radius .23 + tracking allowance .16 m


def point_free(p, obstacles, inflation=INFLATION):
    if max(abs(float(p[0])), abs(float(p[1]))) > BOUND:
        return False
    return all(abs(p[0]-o[0]) > o[2]+inflation or abs(p[1]-o[1]) > o[3]+inflation
               for o in obstacles)


def segment_free(a, b, obstacles):
    n = max(2, int(np.linalg.norm(np.asarray(b)-a)/.025)+2)
    return all(point_free(p, obstacles) for p in np.linspace(a, b, n))


def plan(start, goal, obstacles):
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if not point_free(start, obstacles) or not point_free(goal, obstacles):
        return None
    to_key = lambda p: tuple(np.rint(p/GRID).astype(int))
    s, g = to_key(start), to_key(goal)
    # Off-grid endpoints must also be safely connected to their grid cells.
    if not segment_free(start, np.array(s)*GRID, obstacles) or not segment_free(np.array(g)*GRID, goal, obstacles):
        return None
    frontier, costs, parent = [(0., s)], {s: 0.}, {}
    moves = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1) if i or j]
    closed = set()
    while frontier:
        _, u = heapq.heappop(frontier)
        if u in closed:
            continue
        closed.add(u)
        if u == g:
            path = [goal, np.array(g)*GRID]
            while u != s:
                u = parent[u]
                path.append(np.array(u)*GRID)
            path.append(start)
            path.reverse()
            # Greedy line-of-sight shortcut; each accepted edge is rechecked.
            short, i = [path[0]], 0
            while i < len(path)-1:
                j = len(path)-1
                while j > i+1 and not segment_free(path[i], path[j], obstacles):
                    j -= 1
                if np.linalg.norm(path[j]-short[-1]) > 1e-8:
                    short.append(path[j])
                i = j
            return np.array(short)
        for dx, dy in moves:
            v = (u[0]+dx, u[1]+dy)
            if v in closed or not point_free(np.array(v)*GRID, obstacles):
                continue
            if dx and dy and (not point_free(np.array([u[0]+dx, u[1]])*GRID, obstacles)
                              or not point_free(np.array([u[0], u[1]+dy])*GRID, obstacles)):
                continue
            cost = costs[u]+math.hypot(dx, dy)*GRID
            if cost < costs.get(v, float('inf')):
                costs[v], parent[v] = cost, u
                heapq.heappush(frontier, (cost+math.dist(v, g)*GRID, v))
    return None


def limit_change(current, target, max_delta):
    delta = np.asarray(target)-current
    norm = np.linalg.norm(delta)
    return current+delta*min(1., max_delta/max(norm, 1e-10))
