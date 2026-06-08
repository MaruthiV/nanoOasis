# nanoOasis Breakout. Deterministic, headless-friendly, single-screen (256x192 at SCALE=2; D021).
# Pivoted from a platformer (DECISIONS D018): the platformer was ~99.6% static per frame, so the
# world model had no dynamics to learn (M1 horizon-2x = 1, see EXPERIMENTS). Breakout keeps a ball
# moving every frame -- the regime DIAMOND models on Atari -- while staying single-screen (D017 holds).

from dataclasses import dataclass
import numpy as np

# SCALE multiplies the base 128x96 game so the ball clears the VAE's 8px tile (DECISIONS D021).
# SCALE=1 is the original 128x96; SCALE=2 -> 256x192 with an 8px ball. Everything below is base*SCALE
# so proportions + feel are identical at any scale. Bump SCALE if the ball still needs more tiles.
SCALE = 2
W, H = 128 * SCALE, 96 * SCALE
NUM_ACTIONS = 3  # NONE, LEFT, RIGHT

# paddle + ball physics per 30fps frame. The ball is big (24px ~ one DiT patch-4 token of 32px) so the
# DiT can resolve + place it -- B3 played as mush because a 12px ball was sub-token to the DiT (D023).
# Slow too (3px/f) so the per-frame motion is gentle enough for the small model to predict.
PADDLE_W, PADDLE_H = 24 * SCALE, 6 * SCALE
PADDLE_Y = H - 8 * SCALE                # top y of the paddle row
PADDLE_SPEED = 3.0 * SCALE
BALL_SIZE = 12 * SCALE                  # 24px -- near one DiT token (32px) so it's not a sub-token blur (D023)
BALL_SPEED = 1.5 * SCALE                # 3px/f; slower = gentler dynamics for the small model (B3 over-flung to 9px)
MAX_BOUNCE = 1.0                        # paddle english, radians off vertical at the paddle edge (dimensionless)

# brick grid -- 8 cols x 6 rows spans the full width, below a small top margin
BRICK_W, BRICK_H = 16 * SCALE, 6 * SCALE
BRICK_COLS, BRICK_ROWS = 8, 6
BRICK_TOP = 14 * SCALE
BRICK_VALUE = 10

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
        self.misses = 0                                 # total balls lost (monotonic); ball relaunches at once
        self.board = 0                                  # boards cleared (monotonic)
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

        if b.y >= H:                                    # ball lost -> relaunch at once (no lives, always in play)
            self.misses += 1
            self.paddle_x = float((W - PADDLE_W) / 2)
            self._launch_ball()

        if not self.bricks.any():                       # board cleared -> fresh board
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
                    fb[y0:y0 + BRICK_H - SCALE, x0:x0 + BRICK_W - SCALE] = color   # SCALE-px grid gap

        px = int(self.paddle_x)
        fb[PADDLE_Y:PADDLE_Y + PADDLE_H, px:px + PADDLE_W] = PADDLE_COLOR

        bx, by = int(self.ball.x), int(self.ball.y)
        fb[max(by, 0):by + BALL_SIZE, max(bx, 0):bx + BALL_SIZE] = BALL_COLOR

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
    up = max(1, 512 // W)                                # display upscale -> ~512px window at any SCALE
    screen = pygame.display.set_mode((W * up, H * up))
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
        screen.blit(pygame.transform.scale(surf, (W * up, H * up)), (0, 0))
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
        print("usage: python game.py --play [--seed N]   # interactive upscaled window")
        print("       python game.py --preview           # write assets/preview_*.png")
