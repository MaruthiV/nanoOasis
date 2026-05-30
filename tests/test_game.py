import numpy as np

from game import (
    Game, generate_level, level_is_reachable,
    Walker, Spike,
    PLAYER, WALKER, SPIKE, COIN, DOOR,
    WALK_SPEED, TERMINAL_VEL, TILE, GH, BIOMES, W, PLAYER_H, GROUND_Y, SPIKE_H, DB16,
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
    n_coins_before = len(g.level.coins)
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
    red_per_frame = [bool((f == red).all(axis=-1).any()) for f in frames]
    assert red_per_frame == [True, True, False, False, True, True, False, False], red_per_frame
