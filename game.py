# nanoOasis platformer. Deterministic, headless-friendly.

from dataclasses import dataclass
import numpy as np

W, H = 128, 96
NUM_ACTIONS = 6  # NONE, L, R, J, LJ, RJ

# physics per 30fps frame -- PROJECT.md §3.4
GRAVITY = 0.4
WALK_SPEED = 1.5
JUMP_VEL = -5.0
TERMINAL_VEL = 6.0
JUMP_MAX_HOLD = 12

TILE = 8
GW, GH = W // TILE, H // TILE          # 16 wide, 12 tall tile grid
GROUND_Y = H - TILE                    # top y of ground row
PLAYER_W, PLAYER_H = 8, 12
SPAWN_X = TILE * 2
SPAWN_Y = GROUND_Y

# enemy + collectible sprite extents (px)
WALKER_W, WALKER_H = 8, 8
SPIKE_W, SPIKE_H = 8, 4
COIN_W, COIN_H = 4, 4
COIN_VALUE = 10

# HUD: 3 zero-padded digits, top-right, 3x5 white-on-black font
HUD_W, HUD_H = 13, 7
HUD_X = W - HUD_W
HUD_Y = 0

# door sprite extents (px); door y in Level.door is the door's BOTTOM (top of supporting surface)
DOOR_W, DOOR_H = 8, 16

# level transition: 1..8 fade-out current level, 9..16 fade-in new level
TRANSITION_FRAMES = 16

# reach budget for the platform-graph BFS, in pixels (jump peak / max gap)
MAX_REACH_DY = 40
MAX_REACH_GAP = 48

BIOMES = ("grass", "cave", "sky", "lava")

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

BIOME = {
    "grass": dict(bg=8,  plat_top=11, plat=5,  hazard=None),
    "cave":  dict(bg=0,  plat_top=10, plat=3,  hazard=None),
    "sky":   dict(bg=14, plat_top=13, plat=15, hazard=None),
    "lava":  dict(bg=1,  plat_top=10, plat=3,  hazard=6),
}

# Sprites are ASCII rows. '.'/' ' = transparent. See DECISIONS.md D016.
PLAYER = (
    ".HHHHHH.",
    "HHwwHHHH",
    "HHHHHHHH",
    "HVVVVVVH",
    "HHHHHHHH",
    ".BBBBBB.",
    "BBBBBBBB",
    "BBBBBBBB",
    "BBBBBBBB",
    "BBBBBBBB",
    ".B....B.",
    "BB....BB",
)
PLAYER_COLORS = {"H": DB16[15], "w": DB16[15], "V": DB16[0], "B": DB16[9]}
RED_FLASH_COLORS = {ch: DB16[6] for ch in PLAYER_COLORS}

WALKER = (
    ".WWWWWW.",
    "WWWWWWWW",
    "WeWWWWeW",
    "WWWWWWWW",
    "WWWWWWWW",
    "WWWWWWWW",
    "W.W..W.W",
    ".W....W.",
)
WALKER_COLORS = {"W": DB16[6], "e": DB16[0]}

SPIKE = (
    "S.S.S.S.",
    "SSSSSSSS",
    "SSSSSSSS",
    "SSSSSSSS",
)
SPIKE_COLORS = {"S": DB16[10]}

COIN = (
    ".CC.",
    "CCCC",
    "CCCC",
    ".CC.",
)
COIN_COLORS = {"C": DB16[14]}

DOOR = (
    "DDDDDDDD",
    "DffffffD",
    "DffffffD",
    "DffffffD",
    "DffffffD",
    "DffffffD",
    "DffffffD",
    "DffffffD",
    "DffffffD",
    "DffffffD",
    "DfffffhD",
    "DffffffD",
    "DffffffD",
    "DffffffD",
    "DffffffD",
    "DDDDDDDD",
)
DOOR_COLORS = {"D": DB16[1], "f": DB16[4], "h": DB16[14]}

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


class Player:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.on_ground = True
        self.jump_held_frames = 0
        self.dying_frames = 0           # 0 = alive; >0 = in death sequence


@dataclass
class Platform:
    x: int           # left tile col
    y: int           # top tile row
    w: int           # width in tiles
    @property
    def left_px(self) -> int:  return self.x * TILE
    @property
    def right_px(self) -> int: return (self.x + self.w) * TILE
    @property
    def top_px(self) -> int:   return self.y * TILE


@dataclass
class Walker:
    x: float                    # left edge in px (patrols)
    y: float                    # top edge in px (stays fixed on the platform)
    plat_left: int              # patrol bound, left (px)
    plat_right: int             # patrol bound, right (px)
    vx: float = 1.0


@dataclass
class Spike:
    x: int                      # left edge in px
    y: int                      # top edge in px


@dataclass
class Level:
    seed: int
    biome: str
    platforms: list[Platform]
    spawn: tuple[int, int]
    door: tuple[int, int]
    coins: list[tuple[int, int]]
    enemies: list                # Walker | Spike


def _overlap(a: Platform, b: Platform) -> bool:
    if a.y != b.y:
        return False
    return not (a.x + a.w + 1 <= b.x or b.x + b.w + 1 <= a.x)


def _aabb(ax, ay, aw, ah, bx, by, bw, bh) -> bool:
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def _surfaces(platforms: list[Platform]) -> list[tuple[int, int, int]]:
    out = [(GROUND_Y, 0, W)]
    for p in platforms:
        out.append((p.top_px, p.left_px, p.right_px))
    return out


def _can_jump(src, dst) -> bool:
    dy = src[0] - dst[0]
    if dy > MAX_REACH_DY:
        return False
    gap = max(dst[1] - src[2], src[1] - dst[2], 0)
    return gap <= MAX_REACH_GAP


def _bfs_reachable(start: int, target: int, surfaces) -> bool:
    if start == target:
        return True
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for j in range(len(surfaces)):
            if j in seen or not _can_jump(surfaces[cur], surfaces[j]):
                continue
            if j == target:
                return True
            seen.add(j)
            stack.append(j)
    return False


def level_is_reachable(level: Level) -> bool:
    if not level.platforms:
        return True
    r = max(level.platforms, key=lambda p: p.right_px)
    target = level.platforms.index(r) + 1
    return _bfs_reachable(0, target, _surfaces(level.platforms))


def generate_level(seed: int, biome: str | None = None) -> Level:
    rng = np.random.default_rng(seed)
    biome = biome or BIOMES[int(rng.integers(0, len(BIOMES)))]

    for _ in range(20):
        platforms: list[Platform] = []
        target = int(rng.integers(3, 7))
        for _ in range(target * 4):
            if len(platforms) >= target:
                break
            w = int(rng.integers(2, 5))
            x = int(rng.integers(1, GW - w - 1))
            y = int(rng.integers(2, GH - 2))
            cand = Platform(x, y, w)
            if not any(_overlap(cand, p) for p in platforms):
                platforms.append(cand)

        spawn = (SPAWN_X, SPAWN_Y)
        if platforms:
            r = max(platforms, key=lambda p: p.right_px)
            door = (r.right_px - TILE, r.top_px)
            target_idx = platforms.index(r) + 1
        else:
            door = ((GW - 2) * TILE, GROUND_Y)
            target_idx = 0

        if not _bfs_reachable(0, target_idx, _surfaces(platforms)):
            continue

        spots = [(p.x + i, p.y) for p in platforms for i in range(p.w)]
        spots += [(x, GH - 1) for x in range(2, GW - 2, 3)]
        rng.shuffle(spots)
        n_coins = int(rng.integers(2, 5))
        coins = [(cx * TILE + 2, cy * TILE - 6) for cx, cy in spots[:n_coins]]

        enemies: list = []
        n_enemies = int(rng.integers(1, 4))
        for _ in range(n_enemies):
            if platforms and rng.random() < 0.6:
                p = platforms[int(rng.integers(0, len(platforms)))]
                enemies.append(Walker(
                    x=float((p.x + 1) * TILE),
                    y=float(p.top_px - WALKER_H),
                    plat_left=p.left_px,
                    plat_right=p.right_px,
                ))
            else:
                ex = int(rng.integers(3, GW - 3)) * TILE
                enemies.append(Spike(x=ex, y=GROUND_Y - SPIKE_H))

        return Level(int(seed), biome, platforms, spawn, door, coins, enemies)

    return Level(
        seed=int(seed), biome=biome, platforms=[],
        spawn=(SPAWN_X, SPAWN_Y),
        door=((GW - 2) * TILE, GROUND_Y),
        coins=[(W // 2, GROUND_Y - 6)], enemies=[],
    )


def _blit(fb: np.ndarray, x: int, y: int, sprite, colors: dict, mirror: bool = False) -> None:
    for j, row in enumerate(sprite):
        if mirror:
            row = row[::-1]
        for i, ch in enumerate(row):
            color = colors.get(ch)
            if color is None:
                continue
            yy, xx = y + j, x + i
            if 0 <= yy < H and 0 <= xx < W:
                fb[yy, xx] = color


class Game:
    def __init__(self, seed: int = 0, biome: str | None = None):
        self.rng = np.random.default_rng(seed)
        self.level = generate_level(seed, biome=biome)
        self.biome = self.level.biome
        self.score = 0
        self.deaths = 0
        self.transition_frames = 0
        sx, sy = self.level.spawn
        self.player = Player(float(sx), float(sy))

    def _enter_new_level(self) -> None:
        # score, deaths, dying_frames intentionally persist across levels
        new_seed = int(self.rng.integers(0, 2**31 - 1))
        self.level = generate_level(new_seed)
        self.biome = self.level.biome
        sx, sy = self.level.spawn
        p = self.player
        p.x = float(sx)
        p.y = float(sy)
        p.vx = 0.0
        p.vy = 0.0
        p.on_ground = True
        p.jump_held_frames = 0
        p.dying_frames = 0

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        p = self.player

        # 8-frame death freeze, then respawn at level.spawn. Coins + score persist.
        if p.dying_frames > 0:
            p.dying_frames += 1
            if p.dying_frames > 8:
                p.dying_frames = 0
                self.deaths += 1
                sx, sy = self.level.spawn
                p.x = float(sx)
                p.y = float(sy)
                p.vx = 0.0
                p.vy = 0.0
                p.on_ground = True
                p.jump_held_frames = 0
            return self.render(), 0.0, False

        # level transition: input frozen for 16 frames, level swap at the midpoint
        if self.transition_frames > 0:
            self.transition_frames += 1
            if self.transition_frames == 9:
                self._enter_new_level()
            if self.transition_frames > TRANSITION_FRAMES:
                self.transition_frames = 0
            return self.render(), 0.0, False

        left  = action in (1, 4)
        right = action in (2, 5)
        jump  = action in (3, 4, 5)

        if left and not right:
            p.vx = -WALK_SPEED
        elif right and not left:
            p.vx = WALK_SPEED
        else:
            p.vx = 0.0

        if jump:
            if p.on_ground:
                p.vy = JUMP_VEL
                p.on_ground = False
                p.jump_held_frames = 1
            elif 0 < p.jump_held_frames < JUMP_MAX_HOLD:
                p.jump_held_frames += 1
        else:
            if 0 < p.jump_held_frames < JUMP_MAX_HOLD and p.vy < 0:
                p.vy *= 0.5
            p.jump_held_frames = 0

        # integrate first so the jump impulse moves the player by its full velocity
        prev_y = p.y
        p.x += p.vx
        p.y += p.vy
        p.vy = min(p.vy + GRAVITY, TERMINAL_VEL)

        # default airborne; ground/platform checks below may snap back to standing
        p.on_ground = False

        if p.y >= GROUND_Y:
            p.y = GROUND_Y
            p.vy = 0.0
            p.on_ground = True
            p.jump_held_frames = 0
        else:
            # one-way platforms: land when feet crossed (or stayed on) a top from above
            landed_y = None
            for plat in self.level.platforms:
                ptop = plat.top_px
                if (prev_y <= ptop <= p.y
                        and p.x + PLAYER_W > plat.left_px
                        and p.x < plat.right_px):
                    if landed_y is None or ptop < landed_y:
                        landed_y = ptop
            if landed_y is not None:
                p.y = landed_y
                p.vy = 0.0
                p.on_ground = True
                p.jump_held_frames = 0

        # walkers patrol their platform
        for e in self.level.enemies:
            if isinstance(e, Walker):
                e.x += e.vx
                if e.x < e.plat_left:
                    e.x = e.plat_left
                    e.vx = -e.vx
                elif e.x + WALKER_W > e.plat_right:
                    e.x = e.plat_right - WALKER_W
                    e.vx = -e.vx

        px0, py0 = p.x, p.y - PLAYER_H

        # coins first -- spec: coins collected on the lethal frame still count
        remaining = []
        for cx, cy in self.level.coins:
            if _aabb(px0, py0, PLAYER_W, PLAYER_H, cx, cy, COIN_W, COIN_H):
                self.score += COIN_VALUE
            else:
                remaining.append((cx, cy))
        self.level.coins = remaining

        # door touch wins over enemy collision -- escape from danger
        dx, dy = self.level.door
        if _aabb(px0, py0, PLAYER_W, PLAYER_H, dx, dy - DOOR_H, DOOR_W, DOOR_H):
            self.transition_frames = 1
        else:
            for e in self.level.enemies:
                if isinstance(e, Walker):
                    hit = _aabb(px0, py0, PLAYER_W, PLAYER_H, e.x, e.y, WALKER_W, WALKER_H)
                else:
                    hit = _aabb(px0, py0, PLAYER_W, PLAYER_H, e.x, e.y, SPIKE_W, SPIKE_H)
                if hit:
                    p.dying_frames = 1
                    break

        return self.render(), 0.0, False

    def render(self) -> np.ndarray:
        pal = BIOME[self.biome]
        bg = DB16[pal["bg"]]
        plat_top = DB16[pal["plat_top"]]
        plat = DB16[pal["plat"]]

        fb = np.full((H, W, 3), bg, dtype=np.uint8)

        if pal["hazard"] is not None:
            fb[GROUND_Y:, :] = DB16[pal["hazard"]]
            fb[GROUND_Y, :] = DB16[9]                # orange crust on lava
        else:
            fb[GROUND_Y:, :] = plat
            fb[GROUND_Y, :] = plat_top

        for p in self.level.platforms:
            fb[p.top_px:p.top_px + TILE, p.left_px:p.right_px] = plat
            fb[p.top_px, p.left_px:p.right_px] = plat_top

        for cx, cy in self.level.coins:
            _blit(fb, cx, cy, COIN, COIN_COLORS)

        for e in self.level.enemies:
            if isinstance(e, Walker):
                _blit(fb, int(e.x), int(e.y), WALKER, WALKER_COLORS)
            else:
                _blit(fb, e.x, e.y, SPIKE, SPIKE_COLORS)

        dx, dy = self.level.door
        _blit(fb, dx, dy - 16, DOOR, DOOR_COLORS)

        px = int(self.player.x)
        py = int(self.player.y) - PLAYER_H
        # red flash alternates every 2 frames -- df 1,2 red / 3,4 normal / 5,6 red / 7,8 normal
        df = self.player.dying_frames
        colors = RED_FLASH_COLORS if df > 0 and ((df - 1) // 2) % 2 == 0 else PLAYER_COLORS
        _blit(fb, px, py, PLAYER, colors)

        # HUD: black box + 3 zero-padded digits, top-right. wrap at 999 so it always fits.
        fb[HUD_Y:HUD_Y + HUD_H, HUD_X:HUD_X + HUD_W] = DB16[0]
        s = f"{self.score % 1000:03d}"
        white = DB16[15]
        for i, ch in enumerate(s):
            glyph = DIGITS[ch]
            gx = HUD_X + 1 + i * 4
            for j, row in enumerate(glyph):
                for k, c in enumerate(row):
                    if c == "#":
                        fb[HUD_Y + 1 + j, gx + k] = white

        # level transition darken pass -- darkness peaks at tf=8 (last old frame) and tf=9 (first new)
        tf = self.transition_frames
        if tf > 0:
            darkness = tf / 8.0 if tf <= 8 else (TRANSITION_FRAMES + 1 - tf) / 8.0
            if darkness >= 1.0:
                fb[:] = 0
            else:
                fb = (fb.astype(np.float32) * (1.0 - darkness)).astype(np.uint8)

        return fb


def _keys_to_action(left: bool, right: bool, jump: bool) -> int:
    if left and right:                     # both held: cancel horizontals
        return 3 if jump else 0
    if left:
        return 4 if jump else 1
    if right:
        return 5 if jump else 2
    return 3 if jump else 0


def play(seed: int = 0) -> None:
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((W * 4, H * 4))
    pygame.display.set_caption("nanoOasis")
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
        action = _keys_to_action(
            keys[pygame.K_LEFT],
            keys[pygame.K_RIGHT],
            keys[pygame.K_SPACE],
        )
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
        for biome in BIOMES:
            g = Game(seed=0, biome=biome)
            iio.imwrite(f"assets/preview_{biome}.png", g.render())
            print(f"wrote assets/preview_{biome}.png")
    elif "--play" in sys.argv:
        seed = 0
        if "--seed" in sys.argv:
            seed = int(sys.argv[sys.argv.index("--seed") + 1])
        play(seed=seed)
    else:
        print("usage: python game.py --play [--seed N]   # interactive 512x384 window")
        print("       python game.py --preview           # write assets/preview_*.png")
