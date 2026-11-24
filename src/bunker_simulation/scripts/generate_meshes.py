#!/usr/bin/env python3
"""
Generate the visual meshes of the BUNKER MINI 2.0 side modules (track + side cover).

The profiles below were measured from the side-view dimension drawing in the
official "BUNKER MINI 2.0 User Manual" (V2.0.1, 2023.09, ch. 6.1):
overall 690 (L) x 570 (W) x 335 (H) mm.

Coordinates are (x, z) in metres:
    x = forward, measured from the robot centre (front end at +0.345)
    z = height above the ground

Each profile is extruded along Y and written as an ASCII STL in ../meshes/.
Run it again after editing a profile:
    python3 src/bunker_simulation/scripts/generate_meshes.py
"""
import math
import os

# Outer silhouette of one side module (track + its cover), counter-clockwise.
TRACK_PROFILE = [
    (-0.245, 0.000),   # rear end of ground contact
    (0.210, 0.000),    # front end of ground contact
    (0.245, 0.062),    # front slope (approach ramp for obstacles)
    (0.292, 0.141),
    (0.327, 0.194),
    (0.339, 0.240),    # front face
    (0.339, 0.264),
    (0.322, 0.291),
    (0.283, 0.308),
    (0.190, 0.332),    # top
    (-0.198, 0.332),
    (-0.266, 0.282),   # rear slope of the cover
    (-0.322, 0.238),
    (-0.339, 0.200),   # rear face
    (-0.339, 0.160),
    (-0.333, 0.132),
    (-0.298, 0.079),   # rear slope of the track
    (-0.269, 0.026),
]

# White side cover panel (upper part of the module, outside the track).
COVER_PROFILE = [
    (-0.230, 0.100),
    (0.225, 0.100),
    (0.300, 0.175),
    (0.330, 0.250),
    (0.300, 0.300),
    (0.185, 0.325),
    (-0.195, 0.325),
    (-0.262, 0.275),
    (-0.318, 0.230),
    (-0.318, 0.180),
]


def extrude(profile, width):
    """Triangles of a polygon (x, z) extruded symmetrically along y."""
    cx = sum(p[0] for p in profile) / len(profile)
    cz = sum(p[1] for p in profile) / len(profile)
    hw = width / 2.0
    n = len(profile)
    tris = []
    for i in range(n):
        (x0, z0), (x1, z1) = profile[i], profile[(i + 1) % n]
        # caps (fan from the centroid; the profiles are star-shaped around it)
        tris.append(((cx, hw, cz), (x1, hw, z1), (x0, hw, z0)))
        tris.append(((cx, -hw, cz), (x0, -hw, z0), (x1, -hw, z1)))
        # side wall
        tris.append(((x0, -hw, z0), (x0, hw, z0), (x1, hw, z1)))
        tris.append(((x0, -hw, z0), (x1, hw, z1), (x1, -hw, z1)))
    return tris


def normal(a, b, c):
    u = [b[i] - a[i] for i in range(3)]
    v = [c[i] - a[i] for i in range(3)]
    n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    length = math.sqrt(sum(k * k for k in n)) or 1.0
    return tuple(k / length for k in n)


def write_stl(path, name, tris):
    with open(path, 'w') as f:
        f.write(f'solid {name}\n')
        for t in tris:
            nx, ny, nz = normal(*t)
            f.write(f'  facet normal {nx:.6f} {ny:.6f} {nz:.6f}\n    outer loop\n')
            for v in t:
                f.write(f'      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
            f.write('    endloop\n  endfacet\n')
        f.write(f'endsolid {name}\n')


if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'meshes')
    os.makedirs(out_dir, exist_ok=True)
    write_stl(os.path.join(out_dir, 'bunker_mini_track.stl'), 'track',
              extrude(TRACK_PROFILE, 0.092))
    write_stl(os.path.join(out_dir, 'bunker_mini_side_cover.stl'), 'side_cover',
              extrude(COVER_PROFILE, 0.008))
    print('meshes written to', os.path.normpath(out_dir))
