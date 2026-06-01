# nanoOasis Breakout. Deterministic, headless-friendly, single-screen 128x96.
# Pivoted from a platformer (DECISIONS D018): the platformer was ~99.6% static per frame, so the
# world model had no dynamics to learn (M1 horizon-2x = 1, see EXPERIMENTS). Breakout keeps a ball
# moving every frame -- the regime DIAMOND models on Atari -- while staying single-screen (D017 holds).

from dataclasses import dataclass
import numpy as np

W, H = 128, 96
NUM_ACTIONS = 3  # NONE, LEFT, RIGHT

# paddle + ball physics per 30fps frame
PADDLE_W, PADDLE_H = 24, 4
PADDLE_Y = H - 8                        # top y of the paddle row
PADDLE_SPEED = 3.0
BALL_SIZE = 4
BALL_SPEED = 2.5                        # constant magnitude; velocity is a unit vector * BALL_SPEED
MAX_BOUNCE = 1.0                        # paddle english, radians off vertical at the paddle edge

# brick grid -- 8 cols x 6 rows spans the full 128 width, below the HUD
BRICK_W, BRICK_H = 16, 6
BRICK_COLS, BRICK_ROWS = 8, 6
BRICK_TOP = 14
BRICK_VALUE = 10
LIVES = 3                              # lose one per miss; at 0 the game is over, then a fresh game
GAME_OVER_FRAMES = 12                  # dimmed hold on game over before the reset

# HUD: 3 zero-padded digits, top-right, 3x5 white-on-black font (unchanged from the platformer)
HUD_W, HUD_H = 13, 7
HUD_X = W - HUD_W
HUD_Y = 0

# palette variety -- the Breakout analogue of biomes, gives the dataset visual diversity
PALETTES = ("classic", "cave", "sky", "lava")

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

# per palette: background index + the 6 brick-row colors (top row first)
PALETTE = {
    "classic": dict(bg=0,  rows=(6, 9, 14, 11, 13, 8)),
    "cave":    dict(bg=0,  rows=(3, 10, 7, 12, 15, 13)),
    "sky":     dict(bg=8,  rows=(15, 14, 9, 6, 1, 3)),
    "lava":    dict(bg=1,  rows=(9, 6, 14, 4, 10, 15)),
}

PADDLE_COLOR = DB16[10]
BALL_COLOR = DB16[15]

# 3x5 hand-coded digits (no pygame.font -- see BUGS.md H006).
DIGITS = {
    "0": ("###", "#.#", "#.#", "#.#", "###"),
    "1": (".#.", "##.", ".#.", ".#.", "###"),
    "2": ("###", "..#", ".#.", "#..", "###"),
    "3": ("###", "..#", ".##", "..#", "###"),
    "4": ("#.#", "#.#", "###", "..#", "..#"),
    "5": ("###", "#..", "###", "..#", "###"),
    "6": ("###", "#..", "###", "#.#", "###"),
    "7": ("###", "..#", ".#.", "#..", "#.."),
    "8": ("###", "#.#", "###", "#.#", "###"),
    "9": ("###", "#.#", "###", "..#", "###"),
}


@dataclass
class Ball:
    x: float        # left edge px
    y: float        # top edge px
    vx: float
    vy: float


class Game:
    def __init__(self, seed: int = 0, palette: str | None = None):
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.palette = palette or PALETTES[int(self.rng.integers(0, len(PALETTES)))]
        self.score = 0
        self.misses = 0                                 # total balls lost (monotonic)
        self.lives = LIVES
        self.board = 0                                  # boards cleared (monotonic)
        self.games = 0                                  # games over -> fresh game (monotonic)
        self.over_frames = 0                            # >0 during the game-over hold
        self.paddle_x = float((W - PADDLE_W) / 2)
        self.bricks = np.ones((BRICK_ROWS, BRICK_COLS), dtype=bool)
        self._launch_ball()

    def _launch_ball(self) -> None:
        # ball starts just above the paddle, heading up at a shallow random angle
        angle = float(self.rng.uniform(-0.6, 0.6))
        self.ball = Ball(
            x=self.paddle_x + PADDLE_W / 2 - BALL_SIZE / 2,
            y=float(PADDLE_Y - BALL_SIZE - 1),
            vx=BALL_SPEED * float(np.sin(angle)),
            vy=-BALL_SPEED * float(np.cos(angle)),
        )

    def _new_game(self) -> None:
        self.score = 0
        self.lives = LIVES
        self.bricks = np.ones((BRICK_ROWS, BRICK_COLS), dtype=bool)
        self.paddle_x = float((W - PADDLE_W) / 2)
        self._launch_ball()

    def _bounce_off_paddle(self, b: Ball) -> None:
        # bounce angle depends on where the ball hits the paddle -- classic Breakout english
        offset = ((b.x + BALL_SIZE / 2) - (self.paddle_x + PADDLE_W / 2)) / (PADDLE_W / 2)
        angle = float(np.clip(offset, -1.0, 1.0)) * MAX_BOUNCE
        b.vx = BALL_SPEED * float(np.sin(angle))
        b.vy = -BALL_SPEED * float(np.cos(angle))
        b.y = PADDLE_Y - BALL_SIZE

    def _hit_bricks(self, b: Ball) -> None:
        # cell under the ball center; destroy it, score, reflect vertically
        col = int((b.x + BALL_SIZE / 2) // BRICK_W)
        row = int((b.y + BALL_SIZE / 2 - BRICK_TOP) // BRICK_H)
        if 0 <= row < BRICK_ROWS and 0 <= col < BRICK_COLS and self.bricks[row, col]:
            self.bricks[row, col] = False
            self.score += BRICK_VALUE
            b.vy = -b.vy
            # nudge clear of the brick band so we don't re-hit the same cell next frame
            b.y = BRICK_TOP + (row + 1) * BRICK_H if b.vy > 0 else BRICK_TOP + row * BRICK_H - BALL_SIZE

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        if self.over_frames > 0:                         # game-over hold: dimmed freeze, then a fresh game
            self.over_frames += 1
            if self.over_frames > GAME_OVER_FRAMES:
                self.over_frames = 0
                self.games += 1
                self._new_game()
            return self.render(), 0.0, False

        if action == 1:
            self.paddle_x -= PADDLE_SPEED
        elif action == 2:
            self.paddle_x += PADDLE_SPEED
        self.paddle_x = float(np.clip(self.paddle_x, 0, W - PADDLE_W))

        b = self.ball
        b.x += b.vx
        b.y += b.vy

        # side + top walls reflect
        if b.x <= 0:
            b.x = 0.0
            b.vx = -b.vx
        elif b.x + BALL_SIZE >= W:
            b.x = W - BALL_SIZE
            b.vx = -b.vx
        if b.y <= 0:
            b.y = 0.0
            b.vy = -b.vy

        # paddle reflect (only while descending into the paddle row)
        if (b.vy > 0 and PADDLE_Y <= b.y + BALL_SIZE <= PADDLE_Y + PADDLE_H
                and b.x + BALL_SIZE > self.paddle_x and b.x < self.paddle_x + PADDLE_W):
            self._bounce_off_paddle(b)

        self._hit_bricks(b)

        if b.y >= H:                                    # ball lost -> lose a life
            self.misses += 1
            self.lives -= 1
            if self.lives <= 0:
                self.over_frames = 1                     # game over -- hold, then a fresh game
            else:
                self.paddle_x = float((W - PADDLE_W) / 2)
                self._launch_ball()

        if self.over_frames == 0 and not self.bricks.any():   # board cleared -> fresh board
            self.board += 1
            self.bricks = np.ones((BRICK_ROWS, BRICK_COLS), dtype=bool)
            self._launch_ball()

        return self.render(), 0.0, False

    def render(self) -> np.ndarray:
        pal = PALETTE[self.palette]
        fb = np.full((H, W, 3), DB16[pal["bg"]], dtype=np.uint8)

        for r in range(BRICK_ROWS):
            color = DB16[pal["rows"][r]]
            y0 = BRICK_TOP + r * BRICK_H
            for c in range(BRICK_COLS):
                if self.bricks[r, c]:
                    x0 = c * BRICK_W
                    fb[y0:y0 + BRICK_H - 1, x0:x0 + BRICK_W - 1] = color   # -1 leaves a grid gap

        px = int(self.paddle_x)
        fb[PADDLE_Y:PADDLE_Y + PADDLE_H, px:px + PADDLE_W] = PADDLE_COLOR

        bx, by = int(self.ball.x), int(self.ball.y)
        fb[max(by, 0):by + BALL_SIZE, max(bx, 0):bx + BALL_SIZE] = BALL_COLOR

        # lives: one small marker per remaining life, top-left
        for i in range(self.lives):
            lx = 1 + i * (BALL_SIZE + 1)
            fb[1:1 + BALL_SIZE, lx:lx + BALL_SIZE] = BALL_COLOR

        # HUD: black box + 3 zero-padded digits, top-right. wrap at 999 so it always fits.
        fb[HUD_Y:HUD_Y + HUD_H, HUD_X:HUD_X + HUD_W] = DB16[0]
        white = DB16[15]
        for i, ch in enumerate(f"{self.score % 1000:03d}"):
            gx = HUD_X + 1 + i * 4
            for j, row in enumerate(DIGITS[ch]):
                for k, c in enumerate(row):
                    if c == "#":
                        fb[HUD_Y + 1 + j, gx + k] = white

        if self.over_frames > 0:                        # dim the whole screen during the game-over hold
            fb = (fb.astype(np.float32) * 0.35).astype(np.uint8)

        return fb


def _keys_to_action(left: bool, right: bool, jump: bool = False) -> int:
    # jump kept for signature compatibility with infer.py's play loop; Breakout ignores it
    if left and not right:
        return 1
    if right and not left:
        return 2
    return 0


def play(seed: int = 0) -> None:
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((W * 4, H * 4))
    pygame.display.set_caption("nanoOasis (Breakout)")
    clock = pygame.time.Clock()

    game = Game(seed=seed)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        action = _keys_to_action(keys[pygame.K_LEFT], keys[pygame.K_RIGHT])
        frame, _, _ = game.step(action)
        # pygame.surfarray uses (W, H, 3); see BUGS.md H005
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(pygame.transform.scale(surf, (W * 4, H * 4)), (0, 0))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    import sys, os

    if "--preview" in sys.argv:
        import imageio.v3 as iio
        os.makedirs("assets", exist_ok=True)
        for palette in PALETTES:
            g = Game(seed=0, palette=palette)
            for _ in range(20):                         # step a bit so the ball clears the paddle
                g.step(0)
            iio.imwrite(f"assets/preview_{palette}.png", g.render())
            print(f"wrote assets/preview_{palette}.png")
    elif "--play" in sys.argv:
        seed = 0
        if "--seed" in sys.argv:
            seed = int(sys.argv[sys.argv.index("--seed") + 1])
        play(seed=seed)
    else:
        print("usage: python game.py --play [--seed N]   # interactive 512x384 window")
        print("       python game.py --preview           # write assets/preview_*.png")
