from collections import deque


def bfs_distance(env):
    start, goal = 1, env.size
    queue = deque([(start, 0)]); seen = {start}
    while queue:
        state, dist = queue.popleft()
        if state == goal:
            return dist
        for dice in range(1, 7):
            target = state + dice
            if target > goal:
                nxt = state
            else:
                nxt = env.snakes.get(target, env.ladders.get(target, target))
            if nxt not in seen:
                seen.add(nxt); queue.append((nxt, dist + 1))
    return None

