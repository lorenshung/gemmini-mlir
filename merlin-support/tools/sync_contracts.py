"""Generate support contracts from Merlin's selected Gemmini contract.

Run: python merlin-support/tools/sync_contracts.py --source MERLIN/examples/gemmini/target/contracts
Use --check for a read-only drift check. Only provider plugin declarations are adapted.
"""
import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PLUGINS = {
    'target_contract.yaml': {'backend': 'backend', 'reference_programs': 'tools/baremetalc_corroborate.py'},
    'residual.yaml': {'backend': 'backend'},
}


def generated(source, plugins=None):
    result = {}
    for name, plugin in (PLUGINS if plugins is None else plugins).items():
        original = (source / name).read_bytes()
        document = yaml.safe_load(original)
        document['plugin'] = plugin
        result[name] = (original, yaml.safe_dump(document, sort_keys=False).encode())
    return result


def sync(source, *, check=False):
    files = generated(Path(source))
    differences = [name for name, (_, data) in files.items() if (ROOT / 'contracts' / name).read_bytes() != data]
    if check:
        if differences:
            raise ValueError('provider contract drift: ' + ', '.join(differences))
        return
    receipt = {
        'schema': 'gemmini.contract_sync.v1',
        'supersedes': 'Generated from the selected Merlin contract; only the provider plugin block is adapted.',
        'reviewed_source': {name: hashlib.sha256(original).hexdigest() for name, (original, _) in files.items()},
        'files': [],
    }
    for name, (_, data) in files.items():
        path = ROOT / 'contracts' / name
        receipt['files'].append(dict(destination='contracts/' + name,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            destination_sha256=hashlib.sha256(data).hexdigest()))
        path.write_bytes(data)
    (ROOT / 'contract_sync_migration.json').write_text(json.dumps(receipt, indent=2) + '\n')



HARNESS_ROOT = ROOT.parent / 'support/gemmini_gsim'


def sync_harness(source, *, check=False):
    original, data = generated(Path(source), {
        'target_contract.yaml': {'backend': 'backend.py'},
    })['target_contract.yaml']
    path = HARNESS_ROOT / 'contracts/target_contract.yaml'
    if check:
        if path.read_bytes() != data:
            raise ValueError('GSIM provider contract drift: target_contract.yaml')
        return
    receipt = dict(schema='gemmini.gsim_contract_sync.v1',
        source='Merlin selected target contract; only provider plugin adapted',
        source_sha256=hashlib.sha256(original).hexdigest(),
        previous_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        destination_sha256=hashlib.sha256(data).hexdigest())
    path.write_bytes(data)
    (HARNESS_ROOT / 'contract_sync.json').write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    sync(args.source, check=args.check)
    sync_harness(args.source, check=args.check)
