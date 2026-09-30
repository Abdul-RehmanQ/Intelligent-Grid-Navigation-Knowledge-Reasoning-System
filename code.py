from collections import deque, namedtuple
from dataclasses import dataclass
from heapq import heappop, heappush
from time import perf_counter

# Step 6 - Formal Search Problem
INITIAL_STATE = ("position=(0, 0)", "has_key=False")
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "COLLECT_KEY"]
GOAL_TEST = "position == destination and has_key == True"
PATH_COST = {"UP": 1, "DOWN": 1, "LEFT": 1, "RIGHT": 1, "COLLECT_KEY": 1}

# Search problem structure:
# Initial State -> Possible Actions -> New States -> ... -> Goal State

LAYOUT = [
    "A . . # .",
    ". # . # .",
    ". . K . .",
    "# . . # D",
]

AGENT, KEY, DEST, OBSTACLE, EMPTY = "A", "K", "D", "#", "."


@dataclass(frozen=True)
class State:
    position: tuple  # (row, col)
    has_key: bool


class GridWorld:
    def __init__(self, layout):
        self.grid = [row.split() for row in layout]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        if any(len(r) != self.cols for r in self.grid):
            raise ValueError("Rows must have equal length")

        self.start = self._find(AGENT)
        self.key = self._find(KEY)
        self.dest = self._find(DEST)

        # Static terrain only. A, K, D cells are walkable floor.
        self.walls = {
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r][c] == OBSTACLE
        }

    def _find(self, symbol):
        found = [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if self.grid[r][c] == symbol
        ]
        if len(found) != 1:
            raise ValueError(f"Expected exactly one '{symbol}', found {len(found)}")
        return found[0]

    def in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, pos):
        return self.in_bounds(pos) and pos not in self.walls

    def render(self, agent_pos=None, has_key=False):
        agent_pos = agent_pos or self.start
        lines = []
        for r in range(self.rows):
            cells = []
            for c in range(self.cols):
                p = (r, c)
                if p == agent_pos:
                    cells.append(AGENT)
                elif p == self.key and not has_key:
                    cells.append(KEY)
                elif p == self.dest:
                    cells.append(DEST)
                elif p in self.walls:
                    cells.append(OBSTACLE)
                else:
                    cells.append(EMPTY)
            lines.append(" ".join(cells))
        return "\n".join(lines)


def initial_state(world):
    return State(position=world.start, has_key=False)


def legal_successors(state, world):
    """Return all valid next states from the current state."""
    successors = []
    pos = state.position

    if pos == world.key and not state.has_key:
        successors.append(("COLLECT_KEY", State(position=pos, has_key=True)))

    for action, (dr, dc) in MOVES.items():
        new_pos = (pos[0] + dr, pos[1] + dc)
        if world.is_free(new_pos):
            successors.append((action, State(position=new_pos, has_key=state.has_key)))

    return successors


def bfs_search(world):
    """Breadth-First Search for the grid world problem."""
    start_state = State(position=world.start, has_key=False)
    queue = deque([(start_state, [start_state])])
    visited = {start_state}
    nodes_expanded = 0
    start_time = perf_counter()

    while queue:
        state, path = queue.popleft()
        nodes_expanded += 1

        if state.position == world.dest and state.has_key:
            elapsed = perf_counter() - start_time
            path_length = len(path) - 1
            return {
                "Path found": True,
                "Path": path,
                "Path length": path_length,
                "Path cost": path_length,
                "Nodes expanded": nodes_expanded,
                "Execution time": elapsed,
            }

        for action, next_state in legal_successors(state, world):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [next_state]))

    elapsed = perf_counter() - start_time
    return {
        "Path found": False,
        "Path": [],
        "Path length": 0,
        "Path cost": 0,
        "Nodes expanded": nodes_expanded,
        "Execution time": elapsed,
    }


def dfs_search(world):
    """Depth-First Search for the grid world problem."""
    start_state = State(position=world.start, has_key=False)
    stack = [(start_state, [start_state])]
    visited = {start_state}
    nodes_expanded = 0
    start_time = perf_counter()

    while stack:
        state, path = stack.pop()
        nodes_expanded += 1

        if state.position == world.dest and state.has_key:
            elapsed = perf_counter() - start_time
            path_length = len(path) - 1
            return {
                "Path found": True,
                "Path": path,
                "Path length": path_length,
                "Path cost": path_length,
                "Nodes expanded": nodes_expanded,
                "Execution time": elapsed,
            }

        next_states = []
        for action, next_state in legal_successors(state, world):
            if next_state not in visited:
                visited.add(next_state)
                next_states.append((action, next_state, path + [next_state]))

        for action, next_state, next_path in reversed(next_states):
            stack.append((next_state, next_path))

    elapsed = perf_counter() - start_time
    return {
        "Path found": False,
        "Path": [],
        "Path length": 0,
        "Path cost": 0,
        "Nodes expanded": nodes_expanded,
        "Execution time": elapsed,
    }


def ucs_search(world):
    """Uniform Cost Search for the grid world problem."""
    start_state = State(position=world.start, has_key=False)
    priority_queue = [(0, 0, start_state, [start_state])]
    best_cost = {start_state: 0}
    nodes_expanded = 0
    counter = 1
    start_time = perf_counter()

    while priority_queue:
        cost_so_far, _, state, path = heappop(priority_queue)

        if cost_so_far > best_cost.get(state, float("inf")):
            continue

        nodes_expanded += 1
        if state.position == world.dest and state.has_key:
            elapsed = perf_counter() - start_time
            path_length = len(path) - 1
            return {
                "Path found": True,
                "Path": path,
                "Path length": path_length,
                "Path cost": cost_so_far,
                "Nodes expanded": nodes_expanded,
                "Execution time": elapsed,
            }

        for action, next_state in legal_successors(state, world):
            next_cost = cost_so_far + PATH_COST[action]
            if next_cost < best_cost.get(next_state, float("inf")):
                best_cost[next_state] = next_cost
                heappush(priority_queue, (next_cost, counter, next_state, path + [next_state]))
                counter += 1

    elapsed = perf_counter() - start_time
    return {
        "Path found": False,
        "Path": [],
        "Path length": 0,
        "Path cost": 0,
        "Nodes expanded": nodes_expanded,
        "Execution time": elapsed,
    }


# ---------------------------------------------------------------- Step 3
MOVES = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
PRIORITY = ["UP", "RIGHT", "DOWN", "LEFT"]

# What the agent senses at the current moment.
Percept = namedtuple(
    "Percept", ["on_key", "on_dest", "has_key", "free"]  # free: {direction: bool}
)


class Environment:
    """Holds the true state, produces percepts, applies actions."""

    def __init__(self, world):
        self.world = world
        self.state = initial_state(world)

    def percept(self):
        pos = self.state.position
        free = {
            d: self.world.is_free((pos[0] + dr, pos[1] + dc))
            for d, (dr, dc) in MOVES.items()
        }
        return Percept(
            on_key=(pos == self.world.key and not self.state.has_key),
            on_dest=(pos == self.world.dest),
            has_key=self.state.has_key,
            free=free,
        )

    def execute(self, action):
        pos, has_key = self.state.position, self.state.has_key
        if action == "COLLECT_KEY":
            has_key = True
        elif action in MOVES:
            dr, dc = MOVES[action]
            new_pos = (pos[0] + dr, pos[1] + dc)
            if self.world.is_free(new_pos):
                pos = new_pos
        self.state = State(position=pos, has_key=has_key)


class SimpleReflexAgent:
    """Stateless: action depends only on the current percept."""

    def act(self, percept):
        if percept.on_key:
            return "COLLECT_KEY"
        if percept.on_dest and percept.has_key:
            return "FINISH"
        for direction in PRIORITY:
            if percept.free[direction]:
                return direction
        return "NOOP"  # boxed in


class ModelBasedAgent:
    """Keeps an internal model: known cells, visited cells, a backtrack path.

    The percept does not contain coordinates, so the agent tracks its own
    position by dead reckoning from an assumed origin (0, 0).
    """

    def __init__(self):
        self.pos = (0, 0)          # believed position (relative to start)
        self.known = {}            # cell -> "free" | "obstacle"
        self.visited = set()       # cells visited in the current phase
        self.path = []             # cells to backtrack through
        self.had_key = False       # used to detect the key-pickup phase change

    def _update_model(self, percept):
        # New phase after the key is collected: revisiting cells is needed
        # again, so visited/path are cleared. Known obstacles are kept.
        if percept.has_key and not self.had_key:
            self.had_key = True
            self.visited.clear()
            self.path.clear()

        self.known[self.pos] = "free"
        self.visited.add(self.pos)
        for direction, (dr, dc) in MOVES.items():
            cell = (self.pos[0] + dr, self.pos[1] + dc)
            self.known[cell] = "free" if percept.free[direction] else "obstacle"

    def _step(self, direction):
        dr, dc = MOVES[direction]
        self.pos = (self.pos[0] + dr, self.pos[1] + dc)
        return direction

    def act(self, percept):
        self._update_model(percept)

        if percept.on_key:
            return "COLLECT_KEY"
        if percept.on_dest and percept.has_key:
            return "FINISH"

        # 1. Move to an unvisited, known-free neighbour (priority order).
        for direction in PRIORITY:
            dr, dc = MOVES[direction]
            cell = (self.pos[0] + dr, self.pos[1] + dc)
            if self.known.get(cell) == "free" and cell not in self.visited:
                self.path.append(self.pos)
                return self._step(direction)

        # 2. Dead end: backtrack along the remembered path.
        if self.path:
            target = self.path.pop()
            delta = (target[0] - self.pos[0], target[1] - self.pos[1])
            direction = next(d for d, v in MOVES.items() if v == delta)
            return self._step(direction)

        return "NOOP"  # everything reachable has been explored


class GoalBasedAgent:
    """Chooses actions by how much they reduce distance to the current goal.

    Goal 1: reach the key. Goal 2 (after pickup): reach the destination.
    The agent is given the goal coordinates, but not the obstacle layout;
    obstacles are sensed through percepts. No path is planned: each step
    picks the neighbour with the lowest (distance to goal + visit count),
    so it is drawn toward the goal but pushed away from cells it has
    already used.
    """

    def __init__(self, start, key, dest):
        self.pos = start
        self.key, self.dest = key, dest
        self.visits = {}
        self.had_key = False

    @staticmethod
    def _distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan

    def _current_goal(self, percept):
        return self.dest if percept.has_key else self.key

    def act(self, percept):
        # Goal switch: key collected, target becomes the destination.
        if percept.has_key and not self.had_key:
            self.had_key = True
            self.visits.clear()
        self.visits[self.pos] = self.visits.get(self.pos, 0) + 1

        if percept.on_key:
            return "COLLECT_KEY"
        if percept.on_dest and percept.has_key:
            return "FINISH"

        goal = self._current_goal(percept)
        best_dir, best_cost = None, None
        for direction in PRIORITY:
            if not percept.free[direction]:
                continue  # avoid obstacles
            dr, dc = MOVES[direction]
            cell = (self.pos[0] + dr, self.pos[1] + dc)
            cost = self._distance(cell, goal) + self.visits.get(cell, 0)
            if best_cost is None or cost < best_cost:
                best_dir, best_cost = direction, cost

        if best_dir is None:
            return "NOOP"
        dr, dc = MOVES[best_dir]
        self.pos = (self.pos[0] + dr, self.pos[1] + dc)
        return best_dir


def run(world, agent, max_steps=50, verbose=True):
    env = Environment(world)
    for step in range(max_steps):
        percept = env.percept()
        action = agent.act(percept)
        if verbose:
            print(f"step {step:2d}  pos={env.state.position}  "
                  f"key={env.state.has_key!s:5}  action={action}")
        if action == "FINISH":
            return True, step
        env.execute(action)
    return False, max_steps


if __name__ == "__main__":
    world = GridWorld(LAYOUT)
    print(world.render(), "\n")

    print("=== BFS ===")
    bfs_result = bfs_search(world)
    print(f"Path found: {'Yes' if bfs_result['Path found'] else 'No'}")
    print(f"Path length: {bfs_result['Path length']}")
    print(f"Path cost: {bfs_result['Path cost']}")
    print(f"Nodes expanded: {bfs_result['Nodes expanded']}")
    print(f"Execution time: {bfs_result['Execution time']:.6f} seconds")
    if bfs_result["Path"]:
        print("Path:", [state.position for state in bfs_result["Path"]])
    print()

    print("=== DFS ===")
    dfs_result = dfs_search(world)
    print(f"Path found: {'Yes' if dfs_result['Path found'] else 'No'}")
    print(f"Path length: {dfs_result['Path length']}")
    print(f"Path cost: {dfs_result['Path cost']}")
    print(f"Nodes expanded: {dfs_result['Nodes expanded']}")
    print(f"Execution time: {dfs_result['Execution time']:.6f} seconds")
    if dfs_result["Path"]:
        print("Path:", [state.position for state in dfs_result["Path"]])
    print()

    print("=== UCS ===")
    ucs_result = ucs_search(world)
    print(f"Path found: {'Yes' if ucs_result['Path found'] else 'No'}")
    print(f"Path length: {ucs_result['Path length']}")
    print(f"Path cost: {ucs_result['Path cost']}")
    print(f"Nodes expanded: {ucs_result['Nodes expanded']}")
    print(f"Execution time: {ucs_result['Execution time']:.6f} seconds")
    if ucs_result["Path"]:
        print("Path:", [state.position for state in ucs_result["Path"]])
    print()

    print("=== SimpleReflexAgent ===")
    ok, steps = run(world, SimpleReflexAgent(), verbose=True)
    print(f"success: {ok} | steps: {steps}\n")

    print("=== ModelBasedAgent ===")
    ok, steps = run(world, ModelBasedAgent(), verbose=True)
    print(f"success: {ok} | steps: {steps}\n")

    print("=== GoalBasedAgent ===")
    agent = GoalBasedAgent(world.start, world.key, world.dest)
    ok, steps = run(world, agent, verbose=True)
    print(f"\nsuccess: {ok} | steps: {steps}")