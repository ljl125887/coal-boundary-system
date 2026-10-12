"""Generate an invented LAS scene and execute two isolated runs with shared cache.

No production point, shape, identifier or model is used. The invented numerical
origin only satisfies CRS validation; it does not describe a surveyed site.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import struct
import numpy as np
from coal_boundary.config import load_config
from coal_boundary.pipeline import run_pipeline


def write_synthetic_las(path: Path) -> None:
    x, y = np.meshgrid(np.arange(0, 120, 0.5), np.arange(0, 100, 0.5))
    z = 0.002 * x + 0.001 * y
    for cx, cy, rx, ry, height in [(35, 35, 18, 13, 7), (82, 65, 16, 20, 9)]:
        radius = np.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)
        z += height * np.maximum(1.0 - radius, 0.0)
    xyz = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    scale = np.array([0.001, 0.001, 0.001])
    offset = np.array([500000.0, 3300000.0, 0.0])
    integer = np.rint(xyz / scale).astype('<i4')
    actual = integer * scale + offset
    header = bytearray(227)
    header[:4] = b'LASF'
    header[24:26] = bytes([1, 2])
    header[26:58] = b'SYNTHETIC-NOT-A-SURVEY'.ljust(32, b'\0')
    struct.pack_into('<H', header, 94, 227)
    struct.pack_into('<I', header, 96, 227)
    header[104] = 3
    struct.pack_into('<H', header, 105, 34)
    struct.pack_into('<I', header, 107, len(xyz))
    struct.pack_into('<I', header, 111, len(xyz))
    struct.pack_into('<ddd', header, 131, *scale)
    struct.pack_into('<ddd', header, 155, *offset)
    lo, hi = actual.min(axis=0), actual.max(axis=0)
    struct.pack_into('<dddddd', header, 179, hi[0], lo[0], hi[1], lo[1], hi[2], lo[2])
    records = np.zeros(len(xyz), dtype=np.dtype({
        'names': ['X', 'Y', 'Z', 'flags', 'red', 'green', 'blue'],
        'formats': ['<i4', '<i4', '<i4', 'u1', '<u2', '<u2', '<u2'],
        'offsets': [0, 4, 8, 14, 28, 30, 32], 'itemsize': 34}))
    for i, name in enumerate(['X', 'Y', 'Z']):
        records[name] = integer[:, i]
    records['flags'] = 9
    records['red'], records['green'], records['blue'] = 80, 100, 120
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        stream.write(header)
        stream.write(records.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('demo_workspace'))
    root = parser.parse_args().output.resolve()
    write_synthetic_las(root / 'input' / 'synthetic.las')
    config = load_config(Path(__file__).resolve().parents[1] / 'config.example.yaml')
    for name in ['run-1', 'run-2']:
        run = replace(config, project=replace(config.project,
            las_dir=root / 'input', output_dir=root / name))
        result = run_pipeline(run, cache_root=root / 'cache')
        print(name, result['summary'], result['metadata']['cache'])
        assert result['summary']['pile_count'] == 2
        if name == 'run-2':
            assert result['metadata']['cache']['surface_hit']
            assert result['metadata']['cache']['display_geometry_hit']


if __name__ == '__main__':
    main()
