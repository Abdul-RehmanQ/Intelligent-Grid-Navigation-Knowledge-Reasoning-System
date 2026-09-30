import sys
from collections import deque, namedtuple
from dataclasses import dataclass
from heapq import heappop, heappush
from time import perf_counter

# Step 6 - Formal Search Problem
INITIAL_STATE = ("position=(0, 0)", "has_key=False")
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "COLLECT_KEY"]
GOAL_TEST = "position == destination and has_key == True"
PATH_COST = {"UP": 1, "DOWN": 1, "LEFT": 1, "RIGHT": 1, "COLLECT_KEY": 1}
DEFAULT_AGENT_MAX_STEPS_FACTOR = 4

# Search problem structure:
# Initial State -> Possible Actions -> New States -> ... -> Goal State

MOVES = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
PRIORITY = ["UP", "RIGHT", "DOWN", "LEFT"]

LAYOUT = [
    "A . . # .",
    ". # . # .",
    ". . K . .",
    "# . . # D",
]

TEST_GRIDS = [
    [
        "A . . # .",
        ". # . # .",
        ". . K . .",
        "# . . # D",
    ],
    [
        "A . . . .",
        ". # # # .",
        ". . . . K",
        ". # . . .",
        ". . . . D",
    ],
    [
        "A . . . . .",
        ". # # . # .",
        ". . . . . .",
        ". # . # . K",
        ". . . . # .",
        ". . # . . D",
    ],
    [
        "A . . . . . . .",
        ". # # . # # . .",
        ". . . . . . . .",
        ". # . # . # . .",
        ". . . . . . . .",
        ". # # . # . . .",
        ". . . . . . . K",
        ". . # . . . # D",
    ],
    [
        "A . . . . . . . . .",
        ". # # . # # . . . .",
        ". . . . . . . # . .",
        ". # . # . # . . . .",
        ". . . . . . . . # .",
        ". # # . # . # . . .",
        ". . . . . . . . . K",
        ". . # . . # . # . .",
        ". . . . # . . . . .",
        ". . . . . . # . . D",
    ],
]

AGENT, KEY, DEST, OBSTACLE, EMPTY = "A", "K", "D", "#", "."


@dataclass(frozen=True)
class State:
    position: tuple  # (row, col)
    has_key: bool

    # Important: the search state is not just the cell coordinate.
    # It is (position, has_key). Therefore, the agent may remain on the
    # same position after COLLECT_KEY, but still move to a different state.
    # That is expected and is not an error.


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


def is_goal(state, world):
    return state.position == world.dest and state.has_key


def legal_successors(state, world):
    """Return all valid next states from the current state."""
    successors = []
    pos = state.position

    # A key collection does not change position, but it does change the state
    # because the state is (position, has_key). This is intentional.
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

        if is_goal(state, world):
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

        if is_goal(state, world):
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
        if is_goal(state, world):
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


def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def greedy_best_first_search(world):
    """Greedy Best-First Search using Manhattan distance heuristic."""
    start_state = State(position=world.start, has_key=False)
    priority_queue = [(0, 0, start_state, [start_state])]
    visited = {start_state}
    nodes_expanded = 0
    counter = 1
    start_time = perf_counter()

    while priority_queue:
        _, _, state, path = heappop(priority_queue)
        nodes_expanded += 1

        if is_goal(state, world):
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
                next_goal = world.dest if next_state.has_key else world.key
                heuristic = manhattan_distance(next_state.position, next_goal)
                heappush(priority_queue, (heuristic, counter, next_state, path + [next_state]))
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


def a_star_search(world):
    """A* Search using Manhattan distance as the heuristic."""
    start_state = State(position=world.start, has_key=False)
    priority_queue = [(0, 0, 0, start_state, [start_state])]
    g_score = {start_state: 0}
    nodes_expanded = 0
    counter = 1
    start_time = perf_counter()

    while priority_queue:
        f_score, cost_so_far, _, state, path = heappop(priority_queue)

        if cost_so_far > g_score.get(state, float("inf")):
            continue

        nodes_expanded += 1
        if is_goal(state, world):
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
            if next_cost < g_score.get(next_state, float("inf")):
                g_score[next_state] = next_cost
                next_goal = world.dest if next_state.has_key else world.key
                heuristic = manhattan_distance(next_state.position, next_goal)
                heappush(priority_queue, (next_cost + heuristic, next_cost, counter, next_state, path + [next_state]))
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


def ida_star_search(world):
    """Iterative Deepening A* using a Manhattan-distance heuristic."""
    start_state = State(position=world.start, has_key=False)
    start_time = perf_counter()
    nodes_expanded = 0

    def search(state, path, g_cost, threshold):
        nonlocal nodes_expanded
        next_goal = world.dest if state.has_key else world.key
        h_cost = manhattan_distance(state.position, next_goal)
        f_cost = g_cost + h_cost
        if f_cost > threshold:
            return None, f_cost
        if is_goal(state, world):
            return path, True

        nodes_expanded += 1
        min_next_threshold = float("inf")

        for action, next_state in legal_successors(state, world):
            if next_state in path:
                continue
            next_path = path + [next_state]
            result, next_threshold = search(next_state, next_path, g_cost + PATH_COST[action], threshold)
            if result is not None:
                return result, True
            if next_threshold is not None and next_threshold < min_next_threshold:
                min_next_threshold = next_threshold

        return None, min_next_threshold

    threshold = manhattan_distance(start_state.position, world.key) if not start_state.has_key else manhattan_distance(start_state.position, world.dest)
    path = [start_state]
    solution = None

    while True:
        result, next_value = search(start_state, path, 0, threshold)
        if result is not None:
            solution = result
            break
        if next_value == float("inf"):
            break
        threshold = next_value

    elapsed = perf_counter() - start_time
    if not solution:
        return {
            "Path found": False,
            "Path": [],
            "Path length": 0,
            "Path cost": 0,
            "Nodes expanded": nodes_expanded,
            "Execution time": elapsed,
        }

    path_length = len(solution) - 1
    return {
        "Path found": True,
        "Path": solution,
        "Path length": path_length,
        "Path cost": path_length,
        "Nodes expanded": nodes_expanded,
        "Execution time": elapsed,
    }

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


def run(world, agent, max_steps=None, verbose=True):
    if max_steps is None:
        max_steps = DEFAULT_AGENT_MAX_STEPS_FACTOR * world.rows * world.cols

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


def print_search_summary(title, result):
    print(f"=== {title} ===")
    print(f"Path found: {'Yes' if result['Path found'] else 'No'}")
    print(f"Path length (actions): {result['Path length']}")
    print(f"Path cost: {result['Path cost']}")
    print(f"States recorded: {len(result['Path'])}")
    print(f"Nodes expanded: {result['Nodes expanded']}")
    runtime = result.get("Average execution time", result["Execution time"])
    print(f"Execution time: {runtime:.6f} seconds")
    if result["Path"]:
        print("Path states:", [state.position for state in result["Path"]])
    print()


def run_all_search_algorithms(world, label):
    print(f"\n=== {label} ===")
    print(f"{world.rows}x{world.cols} grid")
    print(world.render(), "\n")

    print_search_summary("BFS", bfs_search(world))
    print_search_summary("DFS", dfs_search(world))
    print_search_summary("UCS", ucs_search(world))
    print_search_summary("Greedy Best-First Search", greedy_best_first_search(world))
    print_search_summary("A* Search", a_star_search(world))
    print_search_summary("IDA* Search", ida_star_search(world))


def benchmark_search_algorithms(world):
    search_functions = [
        ("BFS", bfs_search),
        ("DFS", dfs_search),
        ("UCS", ucs_search),
        ("Greedy Best-First Search", greedy_best_first_search),
        ("A* Search", a_star_search),
        ("IDA* Search", ida_star_search),
    ]
    results = {}
    for name, func in search_functions:
        samples = [func(world) for _ in range(10)]
        avg_time = sum(item["Execution time"] for item in samples) / len(samples)
        result = dict(samples[0])
        result["Average execution time"] = avg_time
        results[name] = result
    return results


def print_search_comparison(test_worlds):
    print("\n=== Cross-grid search comparison (nodes expanded) ===")
    algorithms = ["BFS", "DFS", "UCS", "Greedy Best-First Search", "A* Search", "IDA* Search"]
    headers = ["Algorithm", *[f"Grid {i + 1}" for i in range(len(test_worlds))]]
    print(f"{headers[0]:<28}", end="")
    for header in headers[1:]:
        print(f"{header:>10}", end="")
    print()

    for algorithm in algorithms:
        values = []
        for world in test_worlds:
            result = benchmark_search_algorithms(world)[algorithm]
            values.append(result["Nodes expanded"])
        print(f"{algorithm:<28}", end="")
        for value in values:
            print(f"{value:>10}", end="")
        print()

    print("\nNote: BFS and UCS are equivalent here because every move costs 1, so UCS collapses to BFS on this uniform-cost grid.")
    print("Greedy is optimal on all five layouts here, but that is a property of these layouts rather than a general guarantee of Greedy search.")
    print("IDA* includes repeated work across iterations and no duplicate detection beyond the current path, which explains the very large expansion count on the biggest grid.")


def evaluate_agents(worlds, verbose=False):
    if isinstance(worlds, GridWorld):
        worlds = [worlds]

    print("\n=== Agent evaluation across all grids ===")
    for index, world in enumerate(worlds, start=1):
        print(f"\n--- Grid {index} ({world.rows}x{world.cols}) ---")
        print(world.render(), "\n")

        print("=== SimpleReflexAgent ===")
        ok, steps = run(world, SimpleReflexAgent(), max_steps=4 * world.rows * world.cols, verbose=verbose)
        print(f"success: {ok} | steps: {steps}\n")

        print("=== ModelBasedAgent ===")
        ok, steps = run(world, ModelBasedAgent(), max_steps=4 * world.rows * world.cols, verbose=verbose)
        print(f"success: {ok} | steps: {steps}\n")

        print("=== GoalBasedAgent ===")
        agent = GoalBasedAgent(world.start, world.key, world.dest)
        ok, steps = run(world, agent, max_steps=4 * world.rows * world.cols, verbose=verbose)
        print(f"success: {ok} | steps: {steps}\n")


def prompt_for_test_count(max_count, default=None):
    if default is None:
        default = max_count

    if not sys.stdin.isatty():
        return default

    try:
        raw_value = input(f"How many test environments do you want to run? [{default}] ")
    except (EOFError, KeyboardInterrupt):
        return default

    value = raw_value.strip()
    if not value:
        return default

    try:
        count = int(value)
    except ValueError:
        print("Invalid input. Using the default value.")
        return default

    if count < 1:
        print("Count must be at least 1. Using 1.")
        return 1
    if count > max_count:
        print(f"Only {max_count} predefined environments are available. Using {max_count}.")
        return max_count
    return count


if __name__ == "__main__":
    max_count = len(TEST_GRIDS)
    num_tests = prompt_for_test_count(max_count, default=max_count)

    selected_worlds = [GridWorld(TEST_GRIDS[i]) for i in range(num_tests)]
    print(f"\nRunning search evaluation on {num_tests} predefined test environment(s).\n")
    for i, world in enumerate(selected_worlds, start=1):
        run_all_search_algorithms(world, f"Test Grid {i}")

    print_search_comparison(selected_worlds)

    # Agent evaluation is kept separate from the search evaluation and is run on each grid.
    evaluate_agents(selected_worlds, verbose=False)