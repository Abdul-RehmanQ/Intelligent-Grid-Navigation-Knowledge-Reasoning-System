import sys
from collections import deque, namedtuple
from dataclasses import dataclass
from heapq import heappop, heappush
from time import perf_counter

# ------------------------------------------------------------------ constants
MOVES = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
PRIORITY = ["UP", "RIGHT", "DOWN", "LEFT"]
COLLECT = "COLLECT_KEY"
COLLECT_COST = 1
ACTIONS = list(MOVES) + [COLLECT]
AGENT_MAX_STEPS_FACTOR = 4
BENCHMARK_REPEATS = 10

AGENT, KEY, DEST, OBSTACLE, EMPTY = "A", "K", "D", "#", "."

TEST_GRIDS = [
    ["A . . # .", ". # . # .", ". . K . .", "# . . # D"],
    ["A . . . .", ". # # # .", ". . . . K", ". # . . .", ". . . . D"],
    ["A . . . . .", ". # # . # .", ". . . . . .", ". # . # . K", ". . . . # .", ". . # . . D"],
    ["A . . . . . . .", ". # # . # # . .", ". . . . . . . .", ". # . # . # . .",
     ". . . . . . . .", ". # # . # . . .", ". . . . . . . K", ". . # . . . # D"],
    ["A . . . . . . . . .", ". # # . # # . . . .", ". . . . . . . # . .", ". # . # . # . . . .",
     ". . . . . . . . # .", ". # # . # . # . . .", ". . . . . . . . . K", ". . # . . # . # . .",
     ". . . . # . . . . .", ". . . . . . # . . D"],
]

# Digits 2-9 are walkable cells whose entry cost equals the digit.
SPECIAL_CASE_GRIDS = [
    ("Unreachable key",
     ["A . . . .", "# # # # #", ". . . . .", "# # # # #", ". . K . D"]),
    ("Key in dead-end pocket",
     ["A . . . .", ". # # # .", ". . . . .", ". # . # .", ". . . . D", "# # # K #"]),
    ("Weighted terrain",
     ["A 9 9 9 K", ". . . . .", ". . . . D"]),
]


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

        self.walls = set()
        self.cost = {}  # cost of entering each walkable cell
        for r in range(self.rows):
            for c in range(self.cols):
                token = self.grid[r][c]
                if token == OBSTACLE:
                    self.walls.add((r, c))
                elif token in (AGENT, KEY, DEST, EMPTY):
                    self.cost[(r, c)] = 1
                elif token.isdigit() and int(token) >= 1:
                    self.cost[(r, c)] = int(token)
                else:
                    raise ValueError(f"Unknown cell '{token}' at {(r, c)}")

    def _find(self, symbol):
        found = [(r, c) for r in range(self.rows) for c in range(self.cols)
                 if self.grid[r][c] == symbol]
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
                elif self.cost[p] > 1:
                    cells.append(str(self.cost[p]))
                else:
                    cells.append(EMPTY)
            lines.append(" ".join(cells))
        return "\n".join(lines)


# ------------------------------------------------------- search problem
def initial_state(world):
    return State(position=world.start, has_key=False)


def is_goal(state, world):
    return state.position == world.dest and state.has_key


def legal_successors(state, world):
    """Return (action, next_state, step_cost) for every legal action."""
    successors = []
    pos = state.position
    if pos == world.key and not state.has_key:
        successors.append((COLLECT, State(pos, True), COLLECT_COST))
    for action, (dr, dc) in MOVES.items():
        new_pos = (pos[0] + dr, pos[1] + dc)
        if world.is_free(new_pos):
            successors.append((action, State(new_pos, state.has_key), world.cost[new_pos]))
    return successors


def transition_cost(a, b, world):
    return COLLECT_COST if a.position == b.position else world.cost[b.position]


def path_cost(path, world):
    return sum(transition_cost(a, b, world) for a, b in zip(path, path[1:]))


def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def heuristic(state, world):
    """Admissible: remaining cost >= distance to key + collect + key to dest."""
    if state.has_key:
        return manhattan_distance(state.position, world.dest)
    return (manhattan_distance(state.position, world.key) + COLLECT_COST
            + manhattan_distance(world.key, world.dest))


def make_result(found, path, cost, nodes_expanded, start_time):
    return {
        "Path found": found,
        "Path": path,
        "Path length": max(len(path) - 1, 0),
        "Path cost": cost,
        "Nodes expanded": nodes_expanded,
        "Execution time": perf_counter() - start_time,
    }


# ------------------------------------------------------------- algorithms
def bfs_search(world):
    start = initial_state(world)
    queue = deque([(start, [start])])
    visited = {start}
    nodes = 0
    t0 = perf_counter()
    while queue:
        state, path = queue.popleft()
        nodes += 1
        if is_goal(state, world):
            return make_result(True, path, path_cost(path, world), nodes, t0)
        for _, nxt, _ in legal_successors(state, world):
            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [nxt]))
    return make_result(False, [], 0, nodes, t0)


def dfs_search(world):
    start = initial_state(world)
    stack = [(start, [start])]
    visited = {start}
    nodes = 0
    t0 = perf_counter()
    while stack:
        state, path = stack.pop()
        nodes += 1
        if is_goal(state, world):
            return make_result(True, path, path_cost(path, world), nodes, t0)
        children = []
        for _, nxt, _ in legal_successors(state, world):
            if nxt not in visited:
                visited.add(nxt)
                children.append((nxt, path + [nxt]))
        for child in reversed(children):
            stack.append(child)
    return make_result(False, [], 0, nodes, t0)


def ucs_search(world):
    start = initial_state(world)
    frontier = [(0, 0, start, [start])]
    best = {start: 0}
    nodes, counter = 0, 1
    t0 = perf_counter()
    while frontier:
        g, _, state, path = heappop(frontier)
        if g > best.get(state, float("inf")):
            continue
        nodes += 1
        if is_goal(state, world):
            return make_result(True, path, g, nodes, t0)
        for _, nxt, step in legal_successors(state, world):
            new_g = g + step
            if new_g < best.get(nxt, float("inf")):
                best[nxt] = new_g
                heappush(frontier, (new_g, counter, nxt, path + [nxt]))
                counter += 1
    return make_result(False, [], 0, nodes, t0)


def greedy_best_first_search(world):
    start = initial_state(world)
    frontier = [(0, 0, start, [start])]
    visited = {start}
    nodes, counter = 0, 1
    t0 = perf_counter()
    while frontier:
        _, _, state, path = heappop(frontier)
        nodes += 1
        if is_goal(state, world):
            return make_result(True, path, path_cost(path, world), nodes, t0)
        for _, nxt, _ in legal_successors(state, world):
            if nxt not in visited:
                visited.add(nxt)
                heappush(frontier, (heuristic(nxt, world), counter, nxt, path + [nxt]))
                counter += 1
    return make_result(False, [], 0, nodes, t0)


def a_star_search(world):
    start = initial_state(world)
    frontier = [(0, 0, 0, start, [start])]
    g_score = {start: 0}
    nodes, counter = 0, 1
    t0 = perf_counter()
    while frontier:
        _, g, _, state, path = heappop(frontier)
        if g > g_score.get(state, float("inf")):
            continue
        nodes += 1
        if is_goal(state, world):
            return make_result(True, path, g, nodes, t0)
        for _, nxt, step in legal_successors(state, world):
            new_g = g + step
            if new_g < g_score.get(nxt, float("inf")):
                g_score[nxt] = new_g
                f = new_g + heuristic(nxt, world)
                heappush(frontier, (f, new_g, counter, nxt, path + [nxt]))
                counter += 1
    return make_result(False, [], 0, nodes, t0)


def ida_star_search(world):
    start = initial_state(world)
    t0 = perf_counter()
    nodes = 0

    def search(state, path, g, threshold):
        nonlocal nodes
        f = g + heuristic(state, world)
        if f > threshold:
            return None, f
        if is_goal(state, world):
            return path, f
        nodes += 1
        next_threshold = float("inf")
        for _, nxt, step in legal_successors(state, world):
            if nxt in path:
                continue
            found, value = search(nxt, path + [nxt], g + step, threshold)
            if found is not None:
                return found, value
            next_threshold = min(next_threshold, value)
        return None, next_threshold

    threshold = heuristic(start, world)
    while True:
        solution, value = search(start, [start], 0, threshold)
        if solution is not None:
            return make_result(True, solution, path_cost(solution, world), nodes, t0)
        if value == float("inf"):
            return make_result(False, [], 0, nodes, t0)
        threshold = value


SEARCH_ALGORITHMS = {
    "BFS": bfs_search,
    "DFS": dfs_search,
    "UCS": ucs_search,
    "Greedy": greedy_best_first_search,
    "A*": a_star_search,
    "IDA*": ida_star_search,
}


# ----------------------------------------------------------------- agents
Percept = namedtuple("Percept", ["on_key", "on_dest", "has_key", "free"])


class Environment:
    def __init__(self, world):
        self.world = world
        self.state = initial_state(world)
        self.total_cost = 0

    def percept(self):
        pos = self.state.position
        free = {d: self.world.is_free((pos[0] + dr, pos[1] + dc))
                for d, (dr, dc) in MOVES.items()}
        return Percept(
            on_key=(pos == self.world.key and not self.state.has_key),
            on_dest=(pos == self.world.dest),
            has_key=self.state.has_key,
            free=free,
        )

    def execute(self, action):
        pos, has_key = self.state.position, self.state.has_key
        if action == COLLECT:
            has_key = True
            self.total_cost += COLLECT_COST
        elif action in MOVES:
            dr, dc = MOVES[action]
            new_pos = (pos[0] + dr, pos[1] + dc)
            if self.world.is_free(new_pos):
                pos = new_pos
                self.total_cost += self.world.cost[new_pos]
        self.state = State(pos, has_key)


class SimpleReflexAgent:
    def act(self, percept):
        if percept.on_key:
            return COLLECT
        if percept.on_dest and percept.has_key:
            return "FINISH"
        for direction in PRIORITY:
            if percept.free[direction]:
                return direction
        return "NOOP"


class ModelBasedAgent:
    def __init__(self):
        self.pos = (0, 0)
        self.known = {}
        self.visited = set()
        self.path = []
        self.had_key = False

    def _update_model(self, percept):
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
            return COLLECT
        if percept.on_dest and percept.has_key:
            return "FINISH"
        for direction in PRIORITY:
            dr, dc = MOVES[direction]
            cell = (self.pos[0] + dr, self.pos[1] + dc)
            if self.known.get(cell) == "free" and cell not in self.visited:
                self.path.append(self.pos)
                return self._step(direction)
        if self.path:
            target = self.path.pop()
            delta = (target[0] - self.pos[0], target[1] - self.pos[1])
            direction = next(d for d, v in MOVES.items() if v == delta)
            return self._step(direction)
        return "NOOP"


class GoalBasedAgent:
    def __init__(self, start, key, dest):
        self.pos = start
        self.key, self.dest = key, dest
        self.visits = {}
        self.had_key = False

    def act(self, percept):
        if percept.has_key and not self.had_key:
            self.had_key = True
            self.visits.clear()
        self.visits[self.pos] = self.visits.get(self.pos, 0) + 1
        if percept.on_key:
            return COLLECT
        if percept.on_dest and percept.has_key:
            return "FINISH"
        goal = self.dest if percept.has_key else self.key
        best_dir, best_cost = None, None
        for direction in PRIORITY:
            if not percept.free[direction]:
                continue
            dr, dc = MOVES[direction]
            cell = (self.pos[0] + dr, self.pos[1] + dc)
            cost = manhattan_distance(cell, goal) + self.visits.get(cell, 0)
            if best_cost is None or cost < best_cost:
                best_dir, best_cost = direction, cost
        if best_dir is None:
            return "NOOP"
        dr, dc = MOVES[best_dir]
        self.pos = (self.pos[0] + dr, self.pos[1] + dc)
        return best_dir


def run(world, agent, max_steps=None, verbose=False):
    """Return (status, steps, cost); status is success, exhausted or cutoff."""
    if max_steps is None:
        max_steps = AGENT_MAX_STEPS_FACTOR * world.rows * world.cols
    env = Environment(world)
    for step in range(max_steps):
        action = agent.act(env.percept())
        if verbose:
            print(f"step {step:3d}  pos={env.state.position}  "
                  f"key={env.state.has_key!s:5}  action={action}")
        if action == "FINISH":
            return "success", step, env.total_cost
        if action == "NOOP":
            return "exhausted", step, env.total_cost
        env.execute(action)
    return "cutoff", max_steps, env.total_cost


# ------------------------------------------------------------- evaluation
def benchmark_search_algorithms(world, repeats=BENCHMARK_REPEATS):
    """Run every algorithm `repeats` times; keep mean execution time."""
    results = {}
    for name, func in SEARCH_ALGORITHMS.items():
        samples = [func(world) for _ in range(repeats)]
        result = dict(samples[0])
        result["Execution time"] = sum(s["Execution time"] for s in samples) / repeats
        results[name] = result
    return results


def print_search_summary(name, result):
    print(f"=== {name} ===")
    print(f"Path found: {'Yes' if result['Path found'] else 'No'}")
    print(f"Path length (actions): {result['Path length']}")
    print(f"Path cost: {result['Path cost']}")
    print(f"Nodes expanded: {result['Nodes expanded']}")
    print(f"Mean execution time ({BENCHMARK_REPEATS} runs): {result['Execution time']:.6f} s")
    if result["Path"]:
        print("Path states:", [s.position for s in result["Path"]])
    print()


def print_metric_table(title, evaluated, metric, fmt):
    print(f"\n=== {title} ===")
    print(f"{'Algorithm':<12}" + "".join(f"{'G' + str(i + 1):>10}" for i in range(len(evaluated))))
    for name in SEARCH_ALGORITHMS:
        row = f"{name:<12}"
        for _, _, results in evaluated:
            r = results[name]
            row += f"{fmt(r[metric]) if r['Path found'] else 'no path':>10}"
        print(row)


def print_derived_notes(evaluated):
    print("\n=== Derived observations ===")
    for name in SEARCH_ALGORITHMS:
        worse = []
        for i, (_, _, results) in enumerate(evaluated, start=1):
            costs = [r["Path cost"] for r in results.values() if r["Path found"]]
            if costs and results[name]["Path found"] and results[name]["Path cost"] > min(costs):
                worse.append(i)
        print(f"{name:<8} suboptimal on grids: {worse if worse else 'none'}")
    same = [i for i, (_, _, r) in enumerate(evaluated, start=1)
            if r["BFS"]["Path cost"] == r["UCS"]["Path cost"]]
    print(f"BFS cost == UCS cost on grids: {same}")
    print("IDA* expansion counts include re-expansion across iterations; "
          "duplicate detection is limited to the current path.")


def run_search_evaluation(named_layouts):
    evaluated = []
    for label, layout in named_layouts:
        world = GridWorld(layout)
        print(f"\n=== {label} ({world.rows}x{world.cols}) ===")
        print(world.render(), "\n")
        results = benchmark_search_algorithms(world)
        for name, result in results.items():
            print_search_summary(name, result)
        evaluated.append((label, world, results))

    print_metric_table("Nodes expanded", evaluated, "Nodes expanded", str)
    print_metric_table("Path cost", evaluated, "Path cost", str)
    print_metric_table("Mean execution time (ms)", evaluated, "Execution time",
                       lambda t: f"{t * 1000:.3f}")
    print_derived_notes(evaluated)
    return evaluated


def evaluate_agents(evaluated):
    print("\n=== Agent evaluation ===")
    for index, (label, world, results) in enumerate(evaluated, start=1):
        optimal = results["UCS"]["Path cost"] if results["UCS"]["Path found"] else None
        print(f"\n--- G{index}: {label} | optimal cost: {optimal if optimal is not None else 'no path'} ---")
        print(f"{'Agent':<14}{'Status':<11}{'Steps':>6}{'Cost':>6}{'Cost/optimal':>14}")
        agents = [
            ("SimpleReflex", SimpleReflexAgent()),
            ("ModelBased", ModelBasedAgent()),
            ("GoalBased", GoalBasedAgent(world.start, world.key, world.dest)),
        ]
        for name, agent in agents:
            status, steps, cost = run(world, agent)
            ratio = f"{cost / optimal:.2f}" if status == "success" and optimal else "-"
            print(f"{name:<14}{status:<11}{steps:>6}{cost:>6}{ratio:>14}")


def prompt_for_test_count(max_count):
    if not sys.stdin.isatty():
        return max_count
    try:
        raw = input(f"How many standard test grids do you want to run? [{max_count}] ").strip()
    except (EOFError, KeyboardInterrupt):
        return max_count
    if not raw:
        return max_count
    try:
        count = int(raw)
    except ValueError:
        print("Invalid input. Using the default value.")
        return max_count
    if count < 1:
        print("Count must be at least 1. Using 1.")
        return 1
    return min(count, max_count)


if __name__ == "__main__":
    count = prompt_for_test_count(len(TEST_GRIDS))
    named = [(f"Test Grid {i + 1}", TEST_GRIDS[i]) for i in range(count)] + SPECIAL_CASE_GRIDS
    evaluated = run_search_evaluation(named)
    evaluate_agents(evaluated)