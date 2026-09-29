LAYOUT = [
    "A . . # .",
    ". # . # .",
    ". . K . .",
    "# . . # D",
]

AGENT, KEY, DEST, OBSTACLE, EMPTY = "A", "K", "D", "#", "."


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


if __name__ == "__main__":
    world = GridWorld(LAYOUT)
    print(world.render())
    print("start:", world.start, "key:", world.key, "dest:", world.dest)
    print("walls:", sorted(world.walls))