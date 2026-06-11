import hashlib

import numpy as np

from game import (
    Game, W, H, NUM_ACTIONS, GRID_COLS, GRID_ROWS, CELL, GAP, START_LEN,
    UP, DOWN, LEFT, RIGHT, DIRS,
    BG_COLOR, BODY_COLOR, HEAD_COLOR, APPLE_COLOR,
    _keys_to_action, safe_actions,
)

CELL_PX = (CELL - 2 * GAP) ** 2                          # pixels per rendered cell block


def _count_cells(frame, color):
    return int((frame == np.array(color, dtype=np.uint8)).all(axis=2).sum() / CELL_PX)


# ---- render ----


def test_render_shape_and_dtype():
    g = Game(seed=0)
    frame = g.render()
    assert frame.shape == (H, W, 3)
    assert frame.dtype == np.uint8


def test_render_one_head_one_apple_and_body():
    g = Game(seed=0)
    frame = g.render()
    assert _count_cells(frame, HEAD_COLOR) == 1
    assert _count_cells(frame, APPLE_COLOR) == 1
    assert _count_cells(frame, BODY_COLOR) == START_LEN - 1


def test_cells_are_token_aligned():
    # the whole point of the pivot (D028): 1 cell = 1 DiT token = a 32px block
    assert W // GRID_COLS == CELL and H // GRID_ROWS == CELL
    g = Game(seed=0)
    frame = g.render()
    hc, hr = g.body[0]
    block = frame[hr * CELL + GAP:(hr + 1) * CELL - GAP, hc * CELL + GAP:(hc + 1) * CELL - GAP]
    assert (block == np.array(HEAD_COLOR, dtype=np.uint8)).all()


# ---- tick mechanics: one cell per frame ----


def test_head_advances_one_cell_per_tick():
    g = Game(seed=0)
    h0 = g.body[0]
    g.step(UP)
    assert g.body[0] == (h0[0], h0[1] - 1)


def test_turn_changes_heading():
    g = Game(seed=0)
    h0 = g.body[0]
    g.step(LEFT)
    assert g.body[0] == (h0[0] - 1, h0[1])
    assert g.heading == LEFT


def test_reversal_is_ignored():
    g = Game(seed=0)                                     # heading UP
    h0 = g.body[0]
    g.step(DOWN)                                         # direct reversal -> keeps going UP
    assert g.body[0] == (h0[0], h0[1] - 1)
    assert g.heading == UP


def test_tail_vacates_and_length_constant_without_eat():
    g = Game(seed=0)
    g.apple = (0, GRID_ROWS - 1)                         # park the apple away from the path
    tail0 = g.body[-1]
    n0 = len(g.body)
    g.step(UP)
    assert len(g.body) == n0
    assert tail0 not in g.body


# ---- eat / grow ----


def test_eat_grows_and_respawns_apple():
    g = Game(seed=0)
    hc, hr = g.body[0]
    g.apple = (hc, hr - 1)                               # apple directly ahead
    n0 = len(g.body)
    frame, reward, done = g.step(UP)
    assert len(g.body) == n0 + 1
    assert reward == 1.0 and done                        # done marks the eat event (loader oversampling, D029)
    assert g.eats == 1 and g.score == 1
    assert g.apple not in g.body                         # respawned on a free cell
    assert _count_cells(frame, APPLE_COLOR) == 1


# ---- death / reset ----


def test_wall_death_resets_and_marks_done():
    g = Game(seed=0)                                     # head at row 2 heading UP
    g.step(UP)
    g.step(UP)                                           # head now at row 0
    frame, _, done = g.step(UP)                          # off-grid -> death + instant respawn
    assert done
    assert g.deaths == 1
    assert len(g.body) == START_LEN
    assert g.body[0] == (GRID_COLS // 2, GRID_ROWS // 2 - 1)
    assert _count_cells(frame, HEAD_COLOR) == 1          # the death frame shows the fresh game


def test_self_collision_resets():
    g = Game(seed=0)
    g.body = [(3, 2), (2, 2), (2, 3), (3, 3), (4, 3)]    # head (3,2); (3,3) is mid-body
    g.heading = RIGHT
    g.apple = (7, 5)
    _, _, done = g.step(DOWN)                            # into (3,3) -> death
    assert done and g.deaths == 1
    assert len(g.body) == START_LEN


def test_chasing_the_vacating_tail_is_legal():
    g = Game(seed=0)
    g.body = [(2, 2), (2, 3), (3, 3), (3, 2)]            # square; tail (3,2) right of head (2,2)
    g.heading = UP
    g.apple = (7, 5)
    g.step(RIGHT)                                        # into the cell the tail vacates this tick
    assert g.deaths == 0
    assert g.body[0] == (3, 2) and len(g.body) == 4


# ---- invariants over a long rollout ----


def test_apple_never_on_snake_and_states_disjoint():
    g = Game(seed=1)
    rng = np.random.default_rng(1)
    for _ in range(500):
        safe = safe_actions(g)
        a = int(rng.choice(safe)) if safe else int(rng.integers(0, NUM_ACTIONS))
        g.step(a)
        assert g.apple not in g.body
        assert len(set(g.body)) == len(g.body)           # body never overlaps itself
        for c, r in g.body:
            assert 0 <= c < GRID_COLS and 0 <= r < GRID_ROWS


# ---- safe_actions ----


def test_safe_actions_excludes_walls_and_reversal_traps():
    g = Game(seed=0)
    g.body = [(4, 0), (4, 1), (4, 2)]                    # head on the top wall, heading UP
    g.heading = UP
    g.apple = (0, 5)
    safe = safe_actions(g)
    assert UP not in safe                                # straight into the wall
    assert DOWN not in safe                              # reversal -> effectively UP -> wall
    assert set(safe) == {LEFT, RIGHT}


# ---- key mapping ----


def test_keys_to_action():
    assert NUM_ACTIONS == 4
    assert _keys_to_action(True, False, False, False, RIGHT) == UP
    assert _keys_to_action(False, True, False, False, RIGHT) == DOWN
    assert _keys_to_action(False, False, True, False, RIGHT) == LEFT
    assert _keys_to_action(False, False, False, True, UP) == RIGHT
    assert _keys_to_action(False, False, False, False, LEFT) == LEFT   # no key -> keep heading


# ---- determinism ----


def test_determinism_same_seed_same_actions():
    actions = np.random.default_rng(123).integers(0, NUM_ACTIONS, size=600)

    def run():
        g = Game(seed=42)
        h = hashlib.sha256()
        for a in actions:
            f, _, _ = g.step(int(a))
            h.update(f.tobytes())
        return h.hexdigest(), g.eats, g.deaths

    assert run() == run()
