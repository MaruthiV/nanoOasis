import hashlib
import json
import pathlib

import numpy as np

from game import (
    Game, generate_level, level_is_reachable,
    Walker, Spike, Platform,
    PLAYER, WALKER, SPIKE, COIN, DOOR, DIGITS,
    WALK_SPEED, TERMINAL_VEL, TILE, GH, BIOMES, W, PLAYER_H, PLAYER_W, GROUND_Y, SPIKE_H, DB16,
    HUD_X, HUD_Y, HUD_W, HUD_H, TRANSITION_FRAMES, _keys_to_action,
)


def test_physics_walk_and_friction():
    g = Game(seed=0)
    g.step(2)
    assert g.player.vx == WALK_SPEED
    g.step(1)
    assert g.player.vx == -WALK_SPEED
    g.step(0)
    assert g.player.vx == 0.0


def test_physics_terminal_velocity():
    g = Game(seed=0)
    g.player.y = -10_000.0
    g.player.on_ground = False
    for _ in range(30):
        g.step(0)
    assert abs(g.player.vy - TERMINAL_VEL) < 1e-9, g.player.vy


def test_physics_jump_peak_height():
    g = Game(seed=0)
    start_y = g.player.y
    peak_rise = 0.0
    for _ in range(40):
        g.step(3)
        peak_rise = max(peak_rise, start_y - g.player.y)
    assert 4 * TILE <= peak_rise <= 6 * TILE, peak_rise


def test_physics_variable_jump_release_cut():
    g = Game(seed=0)
    g.step(3)
    g.step(0)
    assert g.player.vy > -2.5, g.player.vy


# ---- G4: levels ----


def test_levels_100_seeds_door_reachable():
    failed = [s for s in range(100) if not level_is_reachable(generate_level(s))]
    assert not failed, f"unreachable: {failed}"


def test_levels_basic_properties():
    for seed in range(20):
        lvl = generate_level(seed)
        assert lvl.biome in BIOMES
        assert 0 <= len(lvl.platforms) <= 6
        assert 1 <= len(lvl.coins) <= 4
        assert 0 <= len(lvl.enemies) <= 3
        for (x, y) in (lvl.spawn, lvl.door):
            assert 0 <= x <= 128 and 0 <= y <= 96
        for p in lvl.platforms:
            assert 0 <= p.x and p.x + p.w <= 16
            assert 0 <= p.y < GH


def test_levels_deterministic():
    a = generate_level(7)
    b = generate_level(7)
    assert a == b


# ---- G5: sprites + render ----


def test_sprites_match_spec_dimensions():
    assert len(PLAYER) == 12 and all(len(r) == 8 for r in PLAYER)
    assert len(WALKER) == 8  and all(len(r) == 8 for r in WALKER)
    assert len(SPIKE)  == 4  and all(len(r) == 8 for r in SPIKE)
    assert len(COIN)   == 4  and all(len(r) == 4 for r in COIN)
    assert len(DOOR)   == 16 and all(len(r) == 8 for r in DOOR)


def test_render_shape_and_dtype():
    g = Game(seed=0)
    frame = g.render()
    assert frame.shape == (96, 128, 3)
    assert frame.dtype == np.uint8


def test_render_each_biome_uses_distinct_palette():
    # crude: each biome's flat background dominates a different color
    bgs = {b: tuple(Game(seed=0, biome=b).render()[0, 0]) for b in BIOMES}
    assert len(set(bgs.values())) == len(BIOMES), bgs


def test_render_contains_player_walker_coin_door():
    # use a seed known to place at least one walker + coin
    for seed in range(10):
        g = Game(seed=seed, biome="grass")
        if any(isinstance(e, Walker) for e in g.level.enemies) and g.level.coins:
            frame = g.render()
            colors = {tuple(c) for c in frame.reshape(-1, 3)}
            # player orange, walker red, coin yellow, door brown frame
            assert (210, 125, 44) in colors, "player orange missing"
            assert (208, 70, 72) in colors, "walker red missing"
            assert (218, 212, 94) in colors, "coin yellow missing"
            assert (68, 36, 52) in colors, "door dark frame missing"
            return
    raise AssertionError("no seed in 0..9 produced walker + coin")


# ---- G6: enemies + collision -> death trigger ----


def test_enemies_spike_collision_triggers_death():
    g = Game(seed=0)
    # park a spike where the player is standing
    g.level.enemies = [Spike(x=int(g.player.x), y=int(g.player.y) - 2)]
    g.step(0)
    assert g.player.dying_frames > 0


def test_enemies_walker_collision_triggers_death():
    g = Game(seed=0)
    px = int(g.player.x)
    py = int(g.player.y) - PLAYER_H
    # vx=0 keeps the walker pinned on top of the player
    g.level.enemies = [Walker(x=float(px), y=float(py), plat_left=0, plat_right=W, vx=0.0)]
    g.step(0)
    assert g.player.dying_frames > 0


def test_enemies_walker_patrols_within_bounds():
    g = Game(seed=0)
    # move the player out of the walker's reach so we only test patrol motion
    g.player.x = 100.0
    w = Walker(x=10.0, y=50.0, plat_left=8, plat_right=40, vx=1.0)
    g.level.enemies = [w]
    xs = []
    for _ in range(80):
        g.step(0)
        xs.append(w.x)
    assert min(xs) >= 8
    assert max(xs) + 8 <= 40
    assert max(xs) - min(xs) > 8                # walker actually patrols, not stuck


def test_enemies_no_collision_when_separated():
    g = Game(seed=0)
    # spike far away; player should remain alive after many no-op steps
    g.level.enemies = [Spike(x=W - 8, y=GROUND_Y - SPIKE_H)]
    for _ in range(30):
        g.step(0)
    assert g.player.dying_frames == 0


# ---- G7: death sequence + respawn ----


def test_death_8_frame_flash_then_respawn():
    g = Game(seed=0, biome="grass")               # grass has no red anywhere; only the flashing player can show red
    spawn = g.level.spawn
    g.level.coins = [(60, 30), (70, 30), (80, 30)]  # far from player; no accidental pickup
    n_coins_before = 3
    g.level.enemies = [Spike(x=int(g.player.x), y=int(g.player.y) - 2)]

    # step 1: collision sets dying_frames = 1
    f, _, _ = g.step(0)
    assert g.player.dying_frames == 1
    frames = [f]

    # steps 2..8: dying_frames advances 2..8
    for expected in range(2, 9):
        f, _, _ = g.step(0)
        assert g.player.dying_frames == expected
        frames.append(f)

    # step 9: respawn -- player back at spawn, counter reset, coins intact, deaths incremented
    g.step(0)
    assert g.player.dying_frames == 0
    assert (g.player.x, g.player.y) == (float(spawn[0]), float(spawn[1]))
    assert (g.player.vx, g.player.vy) == (0.0, 0.0)
    assert g.player.on_ground is True
    assert len(g.level.coins) == n_coins_before
    assert g.deaths == 1

    # red flash pattern: df 1,2,5,6 -> red; df 3,4,7,8 -> normal
    red = np.array(DB16[6], dtype=np.uint8)
    # ignore HUD region -- HUD draws a black box + white digits that aren't red anyway,
    # but checking outside HUD keeps this test focused on the player flash
    red_per_frame = [bool((f[:, :HUD_X] == red).all(axis=-1).any()) for f in frames]
    assert red_per_frame == [True, True, False, False, True, True, False, False], red_per_frame


# ---- G8: coins + HUD ----


def test_hud_digits_have_correct_shape():
    assert set(DIGITS.keys()) == set("0123456789")
    for ch, glyph in DIGITS.items():
        assert len(glyph) == 5 and all(len(r) == 3 for r in glyph), ch


def test_coins_collected_and_score_increments():
    g = Game(seed=0, biome="grass")
    g.level.enemies = []                           # no dying mid-test
    px, py = int(g.player.x), int(g.player.y) - PLAYER_H
    g.level.coins = [(px, py), (px + 1, py + 1), (px + 2, py + 2)]
    g.step(0)
    assert g.score == 30
    assert g.level.coins == []


def test_hud_renders_030_when_score_is_30():
    g = Game(seed=0, biome="grass")
    g.level.enemies = []
    g.score = 30
    frame = g.render()

    white = np.array(DB16[15], dtype=np.uint8)
    black = np.array(DB16[0], dtype=np.uint8)
    hud = frame[HUD_Y:HUD_Y + HUD_H, HUD_X:HUD_X + HUD_W]
    # HUD has both colors
    assert (hud == white).all(axis=-1).any()
    assert (hud == black).all(axis=-1).any()

    # spot-check the three digit glyphs at their expected pixel positions
    # "0" at gx=HUD_X+1 (row 0 = "###", middle row = "#.#")
    gx0 = HUD_X + 1
    assert (frame[HUD_Y + 1, gx0:gx0 + 3] == white).all()
    assert (frame[HUD_Y + 3, gx0 + 1] == black).all()        # hollow center of "0"

    # "3" at gx=HUD_X+5 (row 0 = "###", row 1 = "..#")
    gx3 = HUD_X + 5
    assert (frame[HUD_Y + 1, gx3:gx3 + 3] == white).all()
    assert (frame[HUD_Y + 2, gx3] == black).all() and (frame[HUD_Y + 2, gx3 + 2] == white).all()

    # second "0" at gx=HUD_X+9
    gx0b = HUD_X + 9
    assert (frame[HUD_Y + 1, gx0b:gx0b + 3] == white).all()
    assert (frame[HUD_Y + 3, gx0b + 1] == black).all()


# ---- G9: door touch -> 8-frame fade out + 8-frame fade in to new level ----


def test_door_touch_starts_transition_and_loads_new_level():
    g = Game(seed=0, biome="grass")
    g.score = 50                                    # non-zero to verify persistence
    g.level.enemies = []                            # avoid death stealing focus
    g.level.coins = []                              # avoid score drift
    dx, dy = g.level.door
    g.player.x = float(dx)
    g.player.y = float(dy)                          # feet at door bottom = top of platform
    original_seed = g.level.seed
    original_score = g.score
    original_deaths = g.deaths

    # step 1: door touch
    g.step(0)
    assert g.transition_frames == 1

    # steps 2..16: tf advances 2..16
    for expected in range(2, TRANSITION_FRAMES + 1):
        g.step(0)
        assert g.transition_frames == expected, f"step {expected}: tf={g.transition_frames}"

    # step 17: tf resets to 0, normal play resumes
    g.step(0)
    assert g.transition_frames == 0

    # level swapped, but score and deaths carried over
    assert g.level.seed != original_seed
    assert g.score == original_score
    assert g.deaths == original_deaths


def test_fade_renders_black_at_midpoint_and_brighten_on_return():
    g = Game(seed=0, biome="grass")
    g.level.enemies = []
    g.level.coins = []
    dx, dy = g.level.door
    g.player.x = float(dx)
    g.player.y = float(dy)

    frames = []
    f, _, _ = g.step(0)                             # step 1 (tf=1)
    frames.append(f)
    for _ in range(TRANSITION_FRAMES - 1):          # steps 2..16
        f, _, _ = g.step(0)
        frames.append(f)

    # tf=8 (frame index 7) and tf=9 (frame index 8) are both fully black
    assert frames[7].sum() == 0, "tf=8 should be fully black"
    assert frames[8].sum() == 0, "tf=9 should be fully black (new level just loaded)"

    # tf=1 and tf=16 are mostly visible
    assert frames[0].sum() > 0
    assert frames[15].sum() > 0

    # fade-in (tf=9..16, indices 8..15) brightens monotonically
    fade_in_sums = [int(f.sum()) for f in frames[8:16]]
    assert fade_in_sums == sorted(fade_in_sums), fade_in_sums


def test_door_touch_takes_precedence_over_enemy():
    g = Game(seed=0, biome="grass")
    dx, dy = g.level.door
    g.player.x = float(dx)
    g.player.y = float(dy)
    # spike on the same tile as the door
    g.level.enemies = [Spike(x=dx, y=dy - 4)]
    g.step(0)
    assert g.transition_frames == 1
    assert g.player.dying_frames == 0


# ---- G10: --play key mapping ----


def test_keys_to_action_maps_all_six_actions():
    assert _keys_to_action(False, False, False) == 0       # NONE
    assert _keys_to_action(True,  False, False) == 1       # LEFT
    assert _keys_to_action(False, True,  False) == 2       # RIGHT
    assert _keys_to_action(False, False, True)  == 3       # JUMP
    assert _keys_to_action(True,  False, True)  == 4       # LEFT+JUMP
    assert _keys_to_action(False, True,  True)  == 5       # RIGHT+JUMP
    # both horizontals: cancel, falling back to NONE / JUMP
    assert _keys_to_action(True,  True,  False) == 0
    assert _keys_to_action(True,  True,  True)  == 3


# ---- G12: one-way platform collision ----


def _setup_solo_platform_game(plat: Platform) -> Game:
    g = Game(seed=0, biome="grass")
    g.level.platforms = [plat]
    g.level.enemies = []
    g.level.coins = []
    return g


def test_platform_lands_when_falling_from_above():
    g = _setup_solo_platform_game(Platform(x=4, y=5, w=4))      # top_px = 40, x=[32, 64]
    g.player.x = 40.0                                            # over the platform
    g.player.y = 30.0                                            # above platform top
    g.player.vy = 2.0
    g.player.on_ground = False

    landed = False
    for _ in range(20):
        g.step(0)
        if g.player.on_ground:
            landed = True
            break

    assert landed
    assert g.player.y == 40.0
    assert g.player.vy == 0.0


def test_platform_one_way_passes_through_from_below():
    g = _setup_solo_platform_game(Platform(x=4, y=5, w=4))      # top_px = 40
    g.player.x = 40.0
    g.player.y = 50.0                                            # below the platform top
    g.player.vy = -5.0                                           # jumping straight up
    g.player.on_ground = False

    for _ in range(3):
        g.step(0)

    assert g.player.y < 40.0, g.player.y                         # made it above the platform
    assert g.player.on_ground is False


def test_platform_walk_off_edge_falls():
    g = _setup_solo_platform_game(Platform(x=4, y=5, w=4))      # top_px=40, right_px=64
    # standing near the right edge of the platform
    g.player.x = 56.0
    g.player.y = 40.0
    g.player.vy = 0.0
    g.player.on_ground = True

    for _ in range(20):
        g.step(2)                                                # walk right
        if not g.player.on_ground and g.player.y > 40.0:
            break

    assert g.player.on_ground is False
    assert g.player.y > 40.0                                     # actually fell


def test_platform_walking_across_keeps_on_ground():
    g = _setup_solo_platform_game(Platform(x=2, y=5, w=8))      # wide platform, x=[16, 80]
    g.player.x = 20.0
    g.player.y = 40.0
    g.player.vy = 0.0
    g.player.on_ground = True

    for _ in range(20):
        g.step(2)
        assert g.player.on_ground is True, (g.player.x, g.player.y)
        assert g.player.y == 40.0


# ---- G11: golden-state determinism ----


def test_determinism_seed_42_2000_actions_matches_golden():
    here = pathlib.Path(__file__).parent
    actions = np.load(here / "golden_actions.npy")
    expected = json.loads((here / "golden_game_state.json").read_text())

    g = Game(seed=42)
    for a in actions:
        g.step(int(a))

    actual = {
        "frame_sha256": hashlib.sha256(g.render().tobytes()).hexdigest(),
        "score":       int(g.score),
        "deaths":      int(g.deaths),
        "biome":       g.biome,
        "level_seed":  int(g.level.seed),
        "player_x":    round(float(g.player.x), 6),
        "player_y":    round(float(g.player.y), 6),
    }
    assert actual == expected, f"determinism broken:\n  actual={actual}\n  expected={expected}"
