"""Download only public CALiSol/RedDB files; print SHA-256 provenance for each file."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

PUBLIC_FILES = {
    'calisol23.csv': 'https://raw.githubusercontent.com/Pele0599/CALiSol-23/main/CALiSol-23%20Dataset.csv',
    'calisolsmile.csv': 'https://raw.githubusercontent.com/Pele0599/CALiSol-23/main/calisolsmile.csv',
    'reddb_reaction.tab': 'https://dataverse.harvard.edu/api/access/datafile/6573579',
}


def checksum(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='data/electrolyte')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--expected', help='Optional JSON mapping filename -> expected SHA-256')
    args = p.parse_args()
    expected = json.loads(Path(args.expected).read_text()) if args.expected else {}
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, url in PUBLIC_FILES.items():
        path = directory / name
        if args.overwrite or not path.exists():
            temp = path.with_suffix(path.suffix + '.download')
            try:
                with urlopen(Request(url, headers={'User-Agent': 'YLM-public-data/1.0'}), timeout=120) as response:
                    with open(temp, 'wb') as handle:
                        for block in iter(lambda: response.read(1024 * 1024), b''):
                            handle.write(block)
                digest = checksum(temp)
                if name in expected and digest != expected[name]:
                    raise ValueError(f'Checksum mismatch for {name}')
                if temp.stat().st_size == 0:
                    raise ValueError(f'Empty download: {name}')
                os.replace(temp, path)
            finally:
                temp.unlink(missing_ok=True)
        digest = checksum(path)
        if name in expected and digest != expected[name]:
            raise ValueError(f'Checksum mismatch for existing {name}')
        manifest[name] = {'url': url, 'sha256': digest, 'bytes': path.stat().st_size}
        print(f'{digest}  {path}', flush=True)
    (directory / 'download_manifest.json').write_text(json.dumps(manifest, indent=2))
    print('QM9 requires data/gdb9.sdf and locally generated data/qm9_xtb.pkl (scripts/01_make_labels.py).')
    print('Materials Project is fetched by mp_battery.py only with MP_API_KEY.')


if __name__ == '__main__':
    main()
