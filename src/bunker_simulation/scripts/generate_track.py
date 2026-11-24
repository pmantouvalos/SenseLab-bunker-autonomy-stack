#!/usr/bin/env python3
"""
Generates the obstacle course of the demo: boulders, gravel and bricks on the straight lane.

    python3 scripts/generate_track.py            # print the SDF block
    python3 scripts/generate_track.py --write    # write the meshes into ../meshes/ and replace
                                                 # the block between the TRACK BEGIN / TRACK END
                                                 # markers in worlds/cave_circuit_practice_01.sdf

What it makes (all constants are at the top, edit and re-run):

  boulders   3 big irregular rocks that the Bunker cannot climb (BOULDERS). Each one blocks about
             half of the lane, so the rover must slalom: right, left, right.
             They have a visual mesh and a (slightly smaller) box collision.
  gravel     one bumpy visual-only mesh over the whole lane (no collision, so the physics stays
             cheap) with a speckle texture. The bumps stay below the white lines.
  stones     small rocks (collision boxes) scattered on the lane: they shake the rover and make
             the speed controller work, like rough cave ground.
  bricks     a few bricks (collision) lying on the lane.

All random numbers use fixed seeds, so the course is identical every time.
The floor of the starting area ends at x ~ 12.5 m: keep everything before x = 11.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.normpath(os.path.join(HERE, '..', 'meshes'))
WORLD = os.path.normpath(os.path.join(HERE, '..', 'worlds', 'cave_circuit_practice_01.sdf'))
URI = 'model://bunker_simulation/meshes/'      # resolved through GZ_SIM_RESOURCE_PATH

LANE_HALF = 0.9              # half distance between the line centres (see generate_lane.py)

# Huge boulders that block the view. Each one covers one side of the lane, sticks out past the
# painted line and crosses the centre line, so a rover driving straight runs into it. ONE gap of
# GAP_M is left between its inner edge and the opposite line. Keep the yaw small (changes the gap).
# The inner free width of the lane is ~1.68 m, the Bunker Mini is 0.57 m wide.
GAP_M = 0.80                 # free passage next to every boulder (m); the boulder crosses the centre line
LANE_INNER = 0.84            # inner edge of the lines (y), see generate_lane.py
BOULDER_SIZE = (1.1, 1.9, 1.2)    # length x, width y, height z (m)
# name, x, side (-1 = boulder on the right, +1 = on the left), yaw, seed
# previous (hand-placed) layout: _B = [('boulder_1', 2.0, -1, 0.10, 1), ('boulder_2', 5.0, +1, -0.10, 2), ('boulder_3', 8.0, -1, 0.15, 3)]
# previous random layout: _B = [('boulder_1', 2.8, +1, 0.04, 1), ('boulder_2', 4.9, +1, 0.16, 2), ('boulder_3', 9.2, +1, -0.07, 3)]
# random layout (seed 3788172425):
# previous random layout 2: _B = [('boulder_1', 4.5, -1, -0.02, 1), ('boulder_2', 7.6, +1, -0.04, 2), ('boulder_3', 9.8, +1, 0.17, 3)]
# demo layout (right, left, right = two slaloms):
_B = [('boulder_1', 2.0, -1, 0.10, 1), ('boulder_2', 5.0, +1, -0.10, 2), ('boulder_3', 8.0, -1, 0.15, 3)]
# Layout rule (found in the simulator): after passing a boulder on one side the rover must cross to
# the other side before the next one. A tracked robot needs about 1.5 m of free track for that
# S-curve, so two boulders on OPPOSITE sides must have their centres at least MIN_CENTRE_OPPOSITE
# apart (1.1 m rock + 1.5 m free). Boulders on the same side only must not overlap.
MIN_CENTRE_OPPOSITE = 2.6
MIN_CENTRE_SAME = 1.6
for (_a, _b2) in zip(_B, _B[1:]):
    _need = MIN_CENTRE_OPPOSITE if _a[2] != _b2[2] else MIN_CENTRE_SAME
    if _b2[1] - _a[1] < _need:
        raise SystemExit(f"{_a[0]} and {_b2[0]} are too close: {_b2[1] - _a[1]:.1f} m < {_need} m")
BOULDERS = []
for _n, _x, _side, _yaw, _seed in _B:
    # inner edge of the boulder at y = LANE_INNER - GAP_M (measured from the lane centre, towards
    # the boulder's own side); the centre is half a boulder width further out
    _y = _side * (BOULDER_SIZE[1] / 2 - (LANE_INNER - GAP_M))
    BOULDERS.append(dict(name=_n, x=_x, y=_y, size=BOULDER_SIZE, seed=_seed, yaw=_yaw))
BOULDER_COLLISION = 0.90     # collision box = this fraction of the size

GRAVEL_X = (-1.5, 11.5)      # extent of the gravel surface
GRAVEL_Y = (-1.5, 1.5)
GRAVEL_STEP = 0.04           # grid spacing (m)
GRAVEL_AMPLITUDE = 0.018     # max height of the bumps (m); the lines sit above this (Z_LINE)
GRAVEL_TEXTURE_M = 0.8       # texture tile size (m)

N_STONES = 55                # small collision stones
STONE_X = (0.8, 10.8)        # keep them clear of the start position
STONE_Y = (-0.95, 0.95)
BRICKS = [(1.0, 0.25, 0.4), (3.6, -0.3, 1.2), (6.5, 0.2, -0.5), (9.2, -0.15, 0.9)]  # x, y, yaw
BRICK_SIZE = (0.21, 0.10, 0.04)    # top face of a brick (the sloped base is added around it)


# ----------------------------------------------------------------------------- meshes
def icosphere(subdiv=2):
    t = (1 + 5 ** 0.5) / 2
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t),
         (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
         (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
         (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    v = [np.array(p, float) / np.linalg.norm(p) for p in v]
    for _ in range(subdiv):
        cache, nf = {}, []

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                m = v[a] + v[b]
                v.append(m / np.linalg.norm(m))
                cache[k] = len(v) - 1
            return cache[k]
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v), f


def rock_vertices(size, seed, rough=0.30):
    """Irregular rock: an ellipsoid with a flat bottom, perturbed by multi-scale noise."""
    rng = np.random.default_rng(seed)
    v, f = icosphere(3)
    noise = np.zeros(len(v))
    for octave, (n, lo, hi, amp) in enumerate(((5, 1.5, 3.5, 1.0), (8, 4.0, 8.0, 0.55), (12, 9.0, 18.0, 0.25))):
        for _ in range(n):
            k = rng.normal(size=3)
            k = k / np.linalg.norm(k) * rng.uniform(lo, hi)
            noise += amp * rng.uniform(0.4, 1.0) * np.sin(v @ k + rng.uniform(0, 6.28)) / n
    noise /= 0.9
    v = v * (1.0 + rough * noise)[:, None]
    v[:, 2] = np.clip(v[:, 2], -0.8, 1.0)                       # flat-ish bottom
    sx, sy, sz = size
    out = v.copy()
    for ax, ext in ((0, sx), (1, sy)):          # fit the bounding box exactly (noise would change it)
        lo, hi = out[:, ax].min(), out[:, ax].max()
        out[:, ax] = (out[:, ax] - (lo + hi) / 2) / (hi - lo) * ext
    lo, hi = out[:, 2].min(), out[:, 2].max()
    out[:, 2] = (out[:, 2] - lo) / (hi - lo) * sz - 0.03
    return out, f


def write_stl(path, verts, faces, name):
    with open(path, 'w') as fh:
        fh.write(f'solid {name}\n')
        for a, b, c in faces:
            p, q, r = verts[a], verts[b], verts[c]
            n = np.cross(q - p, r - p)
            n = n / (np.linalg.norm(n) or 1.0)
            fh.write(f' facet normal {n[0]:.5f} {n[1]:.5f} {n[2]:.5f}\n  outer loop\n')
            for w in (p, q, r):
                fh.write(f'   vertex {w[0]:.5f} {w[1]:.5f} {w[2]:.5f}\n')
            fh.write('  endloop\n endfacet\n')
        fh.write(f'endsolid {name}\n')


def smooth_noise(x, y, seed, n=14):
    rng = np.random.default_rng(seed)
    out = np.zeros_like(x)
    for _ in range(n):
        k = rng.uniform(6, 40)
        a = rng.uniform(0, 6.283)
        out += np.sin(k * (x * math.cos(a) + y * math.sin(a)) + rng.uniform(0, 6.283))
    return out / n


def gravel_mesh():
    xs = np.arange(GRAVEL_X[0], GRAVEL_X[1] + 1e-9, GRAVEL_STEP)
    ys = np.arange(GRAVEL_Y[0], GRAVEL_Y[1] + 1e-9, GRAVEL_STEP)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    h = smooth_noise(X, Y, 7)
    h = (h - h.min()) / (h.max() - h.min())                      # 0..1
    rng = np.random.default_rng(8)
    for _ in range(2500):                                        # pebbles
        cx, cy = rng.uniform(*GRAVEL_X), rng.uniform(*GRAVEL_Y)
        r = rng.uniform(0.012, 0.035)
        h += 0.5 * rng.uniform(0.5, 1.0) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (r * r))
    h = np.clip(h / h.max(), 0, 1)
    Z = h * GRAVEL_AMPLITUDE
    nx, ny = len(xs), len(ys)
    lines = ['# gravel surface (generated by generate_track.py)', 'mtllib gravel.mtl', 'o gravel']
    for i in range(nx):
        for j in range(ny):
            lines.append(f'v {X[i, j]:.4f} {Y[i, j]:.4f} {Z[i, j]:.4f}')
    for i in range(nx):
        for j in range(ny):
            lines.append(f'vt {X[i, j] / GRAVEL_TEXTURE_M:.4f} {Y[i, j] / GRAVEL_TEXTURE_M:.4f}')
    # smooth normals from the height field
    dzdx, dzdy = np.gradient(Z, GRAVEL_STEP, axis=(0, 1))
    N = np.stack([-dzdx, -dzdy, np.ones_like(Z)], -1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    for i in range(nx):
        for j in range(ny):
            lines.append(f'vn {N[i, j, 0]:.4f} {N[i, j, 1]:.4f} {N[i, j, 2]:.4f}')
    lines.append('usemtl gravel')
    idx = lambda i, j: i * ny + j + 1
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)
            lines.append(f'f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}')
            lines.append(f'f {a}/{a}/{a} {c}/{c}/{c} {d}/{d}/{d}')
    return '\n'.join(lines) + '\n'


def gravel_texture(path):
    import cv2
    rng = np.random.default_rng(11)
    n = 1024
    img = rng.normal(0.0, 1.0, (n, n)).astype(np.float32)
    fine = cv2.GaussianBlur(img, (0, 0), 1.2)
    coarse = cv2.GaussianBlur(rng.normal(0, 1, (n, n)).astype(np.float32), (0, 0), 9)
    g = 0.46 + 0.20 * fine / fine.std() + 0.12 * coarse / coarse.std()
    for _ in range(2200):                                        # dark gaps and light pebbles
        x, y = rng.integers(0, n, 2)
        r = int(rng.integers(3, 9))
        shade = float(rng.uniform(-0.30, 0.30))
        cv2.circle(g, (int(x), int(y)), r, 0.46 + shade, -1)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    # wrap-around so the tile repeats without seams
    g = (g - g.min()) / (g.max() - g.min())
    g = 0.12 + 0.62 * g
    rgb = np.stack([g * 1.00, g * 0.97, g * 0.92], -1)
    cv2.imwrite(path, (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[:, :, ::-1])


RAMP_SLOPE = 1.8             # run per metre of height: 1.8 -> ~29 deg. The simulated tracks are wheels
                             # (r = 5 cm, mu = 0.9) that cannot climb vertical edges above ~1.3 cm.
RAMPS = {}                   # name -> (top_x, top_y, base_x, base_y, height), filled by build()


def write_frustum(path, tx, ty, bx, by, h):
    """Truncated pyramid (collision of bricks and stones): base on the floor, sloping sides."""
    z0 = -0.005
    base = [(-bx / 2, -by / 2, z0), (bx / 2, -by / 2, z0), (bx / 2, by / 2, z0), (-bx / 2, by / 2, z0)]
    top = [(-tx / 2, -ty / 2, h), (tx / 2, -ty / 2, h), (tx / 2, by * 0 + ty / 2, h), (-tx / 2, ty / 2, h)]
    v = np.array(base + top, float)
    f = [(4, 5, 6), (4, 6, 7), (0, 2, 1), (0, 3, 2)]
    for i in range(4):
        j = (i + 1) % 4
        f += [(i, j, 4 + j), (i, 4 + j, 4 + i)]
    write_stl(path, v, f, os.path.basename(path))


def write_meshes():
    os.makedirs(MESH_DIR, exist_ok=True)
    for b in BOULDERS:
        v, f = rock_vertices(b['size'], b['seed'])
        write_stl(os.path.join(MESH_DIR, b['name'] + '.stl'), v, f, b['name'])
    v, f = rock_vertices((1.0, 1.0, 1.0), 21, rough=0.28)       # unit stone, scaled in the world
    write_stl(os.path.join(MESH_DIR, 'stone.stl'), v, f, 'stone')
    for name, (tx, ty, bx, by, h) in RAMPS.items():
        write_frustum(os.path.join(MESH_DIR, name + '.stl'), tx, ty, bx, by, h)
    with open(os.path.join(MESH_DIR, 'gravel.obj'), 'w') as fh:
        fh.write(gravel_mesh())
    with open(os.path.join(MESH_DIR, 'gravel.mtl'), 'w') as fh:
        fh.write('newmtl gravel\nKa 0.6 0.6 0.6\nKd 0.9 0.9 0.9\nKs 0.02 0.02 0.02\nmap_Kd gravel.png\n')
    gravel_texture(os.path.join(MESH_DIR, 'gravel.png'))


# ----------------------------------------------------------------------------- SDF
def material(r, g, b):
    return (f'<material><ambient>{r} {g} {b} 1</ambient><diffuse>{r} {g} {b} 1</diffuse>'
            f'<specular>0.03 0.03 0.03 1</specular></material>')


def static_model(name, x, y, yaw, visual, collision, z=0.0):
    return (f'    <model name="{name}">\n      <static>true</static>\n'
            f'      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.3f}</pose>\n      <link name="link">\n'
            f'{visual}{collision}      </link>\n    </model>')


def box_collision(sx, sy, sz, z):
    return (f'        <collision name="collision"><pose>0 0 {z:.4f} 0 0 0</pose>'
            f'<geometry><box><size>{sx:.3f} {sy:.3f} {sz:.3f}</size></box></geometry></collision>\n')


def build():
    out = []
    out.append('    <model name="gravel">\n      <static>true</static>\n      <link name="link">\n'
               '        <visual name="visual"><geometry><mesh>'
               f'<uri>{URI}gravel.obj</uri></mesh></geometry></visual>\n'
               '      </link>\n    </model>')
    for b in BOULDERS:
        sx, sy, sz = b['size']
        vis = (f'        <visual name="visual"><geometry><mesh><uri>{URI}{b["name"]}.stl</uri></mesh>'
               f'</geometry>{material(0.20, 0.18, 0.16)}</visual>\n')
        col = box_collision(sx * BOULDER_COLLISION, sy * BOULDER_COLLISION,
                            sz * BOULDER_COLLISION, sz * BOULDER_COLLISION / 2 - 0.02)
        out.append(static_model(b['name'], b['x'], b['y'], b['yaw'], vis, col))
    def ramp_collision(name, tx, ty, bx, by, h):
        RAMPS[name] = (tx, ty, bx, by, h)
        return (f'        <collision name="collision"><geometry><mesh><uri>{URI}{name}.stl</uri></mesh>'
                f'</geometry></collision>\n')

    def sloped(top_x, top_y, h):
        run = RAMP_SLOPE * h
        return top_x + 2 * run, top_y + 2 * run

    rng = np.random.default_rng(5)
    RAMPS.clear()
    for i in range(N_STONES):
        x, y = rng.uniform(*STONE_X), rng.uniform(*STONE_Y)
        if any(abs(x - b['x']) < 0.8 and abs(y - b['y']) < 0.9 for b in BOULDERS):
            continue                                              # not inside a boulder
        h = rng.uniform(0.012, 0.028)
        tx, ty = rng.uniform(0.03, 0.07), rng.uniform(0.03, 0.06)
        bx, by = sloped(tx, ty, h)
        g = rng.uniform(0.14, 0.28)
        vis = (f'        <visual name="visual"><geometry><mesh><uri>{URI}stone.stl</uri>'
               f'<scale>{bx:.3f} {by:.3f} {h:.3f}</scale></mesh></geometry>'
               f'{material(g, g * 0.97, g * 0.92)}</visual>\n')
        out.append(static_model(f'stone_{i:02d}', x, y, rng.uniform(0, 3.14), vis,
                                ramp_collision(f'ramp_stone_{i:02d}', tx, ty, bx, by, h)))
    for i, (x, y, yaw) in enumerate(BRICKS):
        tx, ty, h = BRICK_SIZE
        bx, by = sloped(tx, ty, h)
        name = f'ramp_brick_{i}'
        col = ramp_collision(name, tx, ty, bx, by, h)
        vis = (f'        <visual name="visual"><geometry><mesh><uri>{URI}{name}.stl</uri></mesh>'
               f'</geometry>{material(0.32, 0.12, 0.08)}</visual>\n')
        out.append(static_model(f'brick_{i}', x, y, yaw, vis, col))
    return '\n'.join(out)


def main():
    block = build()
    if '--write' not in sys.argv:
        print(block)
        return
    write_meshes()   # after build(): the ramp meshes are known
    text = open(WORLD).read()
    begin = '<!-- TRACK BEGIN (generated by scripts/generate_track.py) -->'
    end = '<!-- TRACK END -->'
    if begin in text and end in text:
        head, rest = text.split(begin)
        _, tail = rest.split(end)
    else:                                   # first run: insert right after the lane block
        marker = '<!-- LANE END -->'
        if marker not in text:
            sys.exit('LANE END marker not found in the world file')
        head, tail = text.split(marker)
        head += marker + '\n\n    '
    open(WORLD, 'w').write(f'{head}{begin}\n{block}\n    {end}{tail}')
    print(f'meshes written to {MESH_DIR}\ntrack written to {WORLD}')


if __name__ == '__main__':
    main()
