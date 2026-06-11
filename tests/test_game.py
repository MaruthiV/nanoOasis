import hashlib

import numpy as np

from game import (
    Game, W, H, NUM_ACTIONS,
    PADDLE_W, PADDLE_SPEED, PADDLE_Y, BALL_SIZE, BALL_SPEED,
    BRICK_TOP, BRICK_H, BRICK_VALUE, BRICK_ROWS, BRICK_COLS,
    PALETTES, PALETTE, DB16,
    PADDLE_COLOR, BALL_COLOR, _keys_to_action,
)


# ---- render ----


def test_render_shape_and_dtype():
    g = Game(seed=0)
    frame = g.render()
    assert frame.shape == (H, W, 3)
    assert frame.dtype == np.uint8


def test_render_contains_paddle_ball_and_bricks():
    g = Game(seed=0, palette="grey")
    g.step(0)
    colors = {tuple(c) for c in g.render().reshape(-1, 3)}
    assert PADDLE_COLOR in colors
    assert BALL_COLOR in colors
    assert DB16[PALETTE["grey"]["rows"][0]] in colors        # top brick row color


# ---- the property the pivot exists for: the ball moves every frame ----


def test_ball_moves_every_frame():
    g = Game(seed=0)
    prev = (g.ball.x, g.ball.y)
    for _ in range(200):
        a = 1 if g.ball.x < g.paddle_x else 2           # track to keep the ball alive
        g.step(a)
        assert (g.ball.x, g.ball.y) != prev, "ball stalled during play"
        prev = (g.ball.x, g.ball.y)


def test_ball_speed_is_constant():
    g = Game(seed=2)
    for _ in range(300):
        g.step(0)
        speed = (g.ball.vx ** 2 + g.ball.vy ** 2) ** 0.5
        assert abs(speed - BALL_SPEED) < 1e-4, speed


# ---- paddle (action) ----


def test_paddle_moves_with_action():
    g = Game(seed=0)
    x0 = g.paddle_x
    g.step(2)
    assert g.paddle_x == x0 + PADDLE_SPEED                       # RIGHT
    x1 = g.paddle_x
    g.step(1)
    assert g.paddle_x == x1 - PADDLE_SPEED                       # LEFT
    x2 = g.paddle_x
    g.step(0)
    assert g.paddle_x == x2                                      # NONE


def test_paddle_clamped_to_bounds():
    g = Game(seed=0)
    g.ball.y, g.ball.vy = 10.0, -BALL_SPEED                      # keep the ball up so no miss this step
    g.paddle_x = float(W - PADDLE_W - 1)
    g.step(2)
    assert g.paddle_x == W - PADDLE_W
    g.ball.y, g.ball.vy = 10.0, -BALL_SPEED
    g.paddle_x = 1.0
    g.step(1)
    assert g.paddle_x == 0.0


# ---- reflections ----


def test_ball_reflects_off_side_wall():
    g = Game(seed=0)
    g.ball.x, g.ball.y = 3.0, 70.0                              # below bricks, above paddle
    g.ball.vx, g.ball.vy = -BALL_SPEED, 0.0
    reflected = False
    for _ in range(6):
        g.step(0)
        assert 0 <= g.ball.x <= W - BALL_SIZE
        reflected = reflected or g.ball.vx > 0
    assert reflected


def test_ball_bounces_off_paddle():
    g = Game(seed=0)
    g.paddle_x = 52.0
    g.ball.x, g.ball.y = 60.0, float(PADDLE_Y - BALL_SIZE)
    g.ball.vx, g.ball.vy = 0.0, BALL_SPEED                       # descending into the paddle
    g.step(0)
    assert g.ball.vy < 0                                         # bounced back up


# ---- bricks ----


def test_brick_break_increments_score():
    g = Game(seed=0)
    assert g.bricks.all()
    g.ball.x, g.ball.y = 4.0, float(BRICK_TOP + 1)              # inside the top-left cell
    g.ball.vx, g.ball.vy = 0.0, -BALL_SPEED
    s0 = g.score
    g.step(0)
    assert not g.bricks[0, 0]
    assert g.score == s0 + BRICK_VALUE
    assert g.ball.vy > 0                                         # reflected


def test_board_clears_and_resets():
    g = Game(seed=0)
    g.bricks[:] = False
    g.bricks[0, 0] = True                                        # one brick left
    b0 = g.board
    g.ball.x, g.ball.y = 4.0, float(BRICK_TOP + 1)
    g.ball.vx, g.ball.vy = 0.0, -BALL_SPEED
    g.step(0)
    assert g.board == b0 + 1
    assert g.bricks.all()                                       # fresh board


# ---- ball loss ----


def test_ball_loss_increments_misses_and_relaunches():
    g = Game(seed=0)
    m0 = g.misses
    g.ball.x, g.ball.y = 60.0, float(H - 1)
    g.ball.vx, g.ball.vy = 0.0, BALL_SPEED
    g.step(0)
    assert g.misses == m0 + 1
    assert g.ball.y < H                                          # relaunched at once -- no lives, always in play


# ---- invariants over a long rollout ----


def test_ball_stays_in_bounds():
    g = Game(seed=1)
    for t in range(2000):
        a = 1 if g.ball.x + BALL_SIZE / 2 < g.paddle_x + PADDLE_W / 2 else 2
        g.step(a)
        assert 0 <= g.ball.x <= W - BALL_SIZE, (t, g.ball.x)
        assert g.ball.y < H, (t, g.ball.y)              # always in play -- a miss relaunches at once


# ---- key mapping ----


def test_keys_to_action():
    assert NUM_ACTIONS == 3
    assert _keys_to_action(False, False) == 0                   # NONE
    assert _keys_to_action(True, False) == 1                    # LEFT
    assert _keys_to_action(False, True) == 2                    # RIGHT
    assert _keys_to_action(True, True) == 0                     # both -> cancel


# ---- determinism ----


def test_determinism_same_seed_same_actions():
    actions = np.random.default_rng(123).integers(0, 3, size=600)

    def run():
        g = Game(seed=42)
        h = hashlib.sha256()
        for a in actions:
            f, _, _ = g.step(int(a))
            h.update(f.tobytes())
        return h.hexdigest(), g.score, g.misses, g.board

    assert run() == run()
