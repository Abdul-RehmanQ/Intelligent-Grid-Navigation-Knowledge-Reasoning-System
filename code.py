from dataclasses import dataclass

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


# ---------------------------------------------------------------- Step 3
from collections import namedtuple

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
        if action == "COLLECT":
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
            return "COLLECT"
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
            return "COLLECT"
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
            return "COLLECT"
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