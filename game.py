# nanoOasis Snake. Deterministic, headless-friendly, grid-native (DECISIONS D028/D029).
# Pivoted from Breakout: its small fast *continuous* ball + *rare* discrete brick-breaks fought the
# DiT's coarse tokens across 3 launch runs (the static_weight tradeoff, D028). Snake is designed so
# 1 game cell = 1 DiT token (32px cell = VAE patch 8 x DiT patch 4): no sub-token objects, discrete
# one-cell-per-frame motion, frequent eat/grow events. Full design: docs/SNAKE_DESIGN.md.

import numpy as np

W, H = 256, 192                      # frame size -- VAE/DiT derive their grids from this (D021)
GRID_COLS, GRID_ROWS = 8, 6          # 8x6 cells of 32px = exactly the DiT's 48-token grid
CELL = W // GRID_COLS                # 32px (== H // GRID_ROWS)
GAP = 2                              # px inset per cell side -> visible grid separation
NUM_ACTIONS = 4                      # absolute headings; no NONE -- the snake always moves

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DIRS = ((0, -1), (0, 1), (-1, 0), (1, 0))    # (dcol, drow) per action
_REVERSE = (DOWN, UP, RIGHT, LEFT)

# DawnBringer 16 palette, indexed.
DB16 = (
    (0x14, 0x0c, 0x1c),  # 0  near-black
    (0x44, 0x24, 0x34),  # 1  dark plum
    (0x30, 0x34, 0x6d),  # 2  dark blue
    (0x4e, 0x4a, 0x4e),  # 3  dark gray
    (0x85, 0x4c, 0x30),  # 4  brown
    (0x34, 0x65, 0x24),  # 5  dark green
    (0xd0, 0x46, 0x48),  # 6  red
    (0x75, 0x71, 0x61),  # 7  olive
    (0x59, 0x7d, 0xce),  # 8  sky blue
    (0xd2, 0x7d, 0x2c),  # 9  orange
    (0x85, 0x95, 0xa1),  # 10 light gray
    (0x6d, 0xaa, 0x2c),  # 11 green
    (0xd2, 0xaa, 0x99),  # 12 beige
    (0x6d, 0xc2, 0xca),  # 13 cyan
    (0xda, 0xd4, 0x5e),  # 14 yellow
    (0xde, 0xee, 0xd6),  # 15 off-white
)

BG_COLOR = DB16[0]
BODY_COLOR = DB16[10]                # mid grey
HEAD_COLOR = DB16[15]                # brightest -- heading must be legible to model + player
APPLE_COLOR = DB16[6]                # the one red accent (SNAKE_DESIGN decision 2)

START_LEN = 3


class Game:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.eats = 0                                  # total apples eaten (monotonic)
        self.deaths = 0                                # total deaths (monotonic; scene-reset marker)
        self._spawn_snake()
        self.apple = self._spawn_apple()

    def _spawn_snake(self) -> None:
        c, r = GRID_COLS // 2, GRID_ROWS // 2 - 1      # head at (4, 2), body trailing down
        self.body = [(c, r + i) for i in range(START_LEN)]   # head first
        self.heading = UP
        self.score = 0                                 # apples this life

    def _spawn_apple(self) -> tuple[int, int]:
        occupied = set(self.body)
        empty = [(c, r) for r in range(GRID_ROWS) for c in range(GRID_COLS) if (c, r) not in occupied]
        return empty[int(self.rng.integers(0, len(empty)))]

    def _reset(self) -> None:
        self.deaths += 1
        self._spawn_snake()
        self.apple = self._spawn_apple()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        # one tick per frame: the head advances exactly one cell. `done` marks eat OR death --
        # event markers the loader oversamples (D029), not episode boundaries.
        if action != _REVERSE[self.heading]:           # direct reversal is ignored, classic Snake
            self.heading = int(action)
        dc, dr = DIRS[self.heading]
        hc, hr = self.body[0]
        nh = (hc + dc, hr + dr)

        if not (0 <= nh[0] < GRID_COLS and 0 <= nh[1] < GRID_ROWS):
            self._reset()                              # wall -> death + instant respawn (no game-over screen)
            return self.render(), 0.0, True

        eat = nh == self.apple
        trunk = self.body if eat else self.body[:-1]   # tail vacates first unless growing -> tail-chase is legal
        if nh in trunk:
            self._reset()                              # self-collision -> death + instant respawn
            return self.render(), 0.0, True

        self.body = [nh] + trunk
        if eat:
            self.score += 1
            self.eats += 1
            if len(self.body) == GRID_COLS * GRID_ROWS:   # board full -- won; fresh game
                self._reset()
                return self.render(), 1.0, True
            self.apple = self._spawn_apple()
        return self.render(), float(eat), eat

    @staticmethod
    def _cell(fb: np.ndarray, c: int, r: int, color: tuple) -> None:
        fb[r * CELL + GAP:(r + 1) * CELL - GAP, c * CELL + GAP:(c + 1) * CELL - GAP] = color

    def render(self) -> np.ndarray:
        fb = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
        for c, r in self.body[1:]:
            self._cell(fb, c, r, BODY_COLOR)
        self._cell(fb, self.body[0][0], self.body[0][1], HEAD_COLOR)
        self._cell(fb, self.apple[0], self.apple[1], APPLE_COLOR)
        return fb


def safe_actions(g: Game) -> list[int]:
    # actions whose EFFECTIVE move (a reversal keeps the current heading) lands on a free cell.
    # shared by the data bots + previews; an empty list means every move dies.
    out = []
    hc, hr = g.body[0]
    for a in range(NUM_ACTIONS):
        eff = g.heading if a == _REVERSE[g.heading] else a
        dc, dr = DIRS[eff]
        nh = (hc + dc, hr + dr)
        if not (0 <= nh[0] < GRID_COLS and 0 <= nh[1] < GRID_ROWS):
            continue
        trunk = g.body if nh == g.apple else g.body[:-1]
        if nh not in trunk:
            out.append(a)
    return out


def _keys_to_action(up: bool, down: bool, left: bool, right: bool, current: int) -> int:
    # absolute heading commands; no key pressed -> keep the current heading (the snake always moves)
    for pressed, a in ((up, UP), (down, DOWN), (left, LEFT), (right, RIGHT)):
        if pressed:
            return a
    return current


_KEY_DIRS = {}                                         # filled lazily; pygame keycode -> action


def _poll_action(pygame, events, current: int) -> tuple[int, bool]:
    # taps between slow ticks arrive as queued KEYDOWNs -- read those first so no input is dropped,
    # then fall back to held keys, then keep the current heading. Returns (action, quit_requested).
    if not _KEY_DIRS:
        _KEY_DIRS.update({pygame.K_UP: UP, pygame.K_DOWN: DOWN,
                          pygame.K_LEFT: LEFT, pygame.K_RIGHT: RIGHT})
    tapped, quit_req = None, False
    for ev in events:
        if ev.type == pygame.QUIT:
            quit_req = True
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                quit_req = True
            elif ev.key in _KEY_DIRS:
                tapped = _KEY_DIRS[ev.key]             # latest tap wins
    if tapped is not None:
        return tapped, quit_req
    keys = pygame.key.get_pressed()
    return _keys_to_action(keys[pygame.K_UP], keys[pygame.K_DOWN],
                           keys[pygame.K_LEFT], keys[pygame.K_RIGHT], current), quit_req


def play(seed: int = 0, fps: int = 4) -> None:
    import pygame
    pygame.init()
    up_scale = max(1, 512 // W)
    screen = pygame.display.set_mode((W * up_scale, H * up_scale))
    pygame.display.set_caption("nanoOasis (Snake)")
    clock = pygame.time.Clock()

    game = Game(seed=seed)
    action = UP
    running = True
    while running:
        action, quit_req = _poll_action(pygame, pygame.event.get(), action)
        if quit_req:
            break
        frame, _, _ = game.step(action)
        # pygame.surfarray uses (W, H, 3); see BUGS.md H005
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(pygame.transform.scale(surf, (W * up_scale, H * up_scale)), (0, 0))
        pygame.display.flip()
        clock.tick(fps)                                # tick = one cell move; 4/s per user play-test (7 was too fast)

    pygame.quit()


if __name__ == "__main__":
    import sys, os

    if "--preview" in sys.argv:
        import imageio.v3 as iio
        os.makedirs("assets", exist_ok=True)
        g = Game(seed=0)
        rng = np.random.default_rng(0)
        frames = []
        for t in range(48):
            safe = safe_actions(g)
            a = int(rng.choice(safe)) if safe else int(rng.integers(0, NUM_ACTIONS))
            frames.append(g.step(a)[0])
        iio.imwrite("assets/preview_snake.png", frames[-1])
        iio.imwrite("assets/preview_snake_strip.png", np.concatenate(frames[::4], axis=1))
        print(f"wrote assets/preview_snake.png + strip (eats={g.eats}, deaths={g.deaths})")
    elif "--play" in sys.argv:
        seed = 0
        fps = 4
        if "--seed" in sys.argv:
            seed = int(sys.argv[sys.argv.index("--seed") + 1])
        if "--fps" in sys.argv:
            fps = int(sys.argv[sys.argv.index("--fps") + 1])
        play(seed=seed, fps=fps)
    else:
        print("usage: python game.py --play [--seed N] [--fps N]   # interactive upscaled window")
        print("       python game.py --preview                     # write assets/preview_snake*.png")
