"""Read-only qualification of migrated support resources, not hardware/candidate certification."""

import hashlib
import json
from pathlib import Path

import yaml
from test_conformance_inventory import evolved_identity

from merlin.common.schemas import validate as validate_schema
from merlin.common.schemas import validate_or_raise
from merlin.targetgen.plugins import validate
from merlin.targetgen.providers import ProviderRole, read_provider

ROOT = Path(__file__).resolve().parents[1]


def test_support_identity_schema_diagnostic_parity_and_plugin_references():
    provider = read_provider(ROOT)
    assert provider.role == ProviderRole.SUPPORT
    contract = yaml.safe_load(provider.contract_path.read_text())
    qualification = json.loads((ROOT / "provenance.json").read_text())["qualification"]
    problems = validate_schema(contract, "target_contract")
    assert problems == qualification["schema_diagnostics"]
    assert qualification["schema_valid"] == (not problems)
    assert contract["name"] == provider.target
    assert validate(contract.get("plugin"), root=ROOT) == []
    plan = ROOT / "contracts/dialect_plan.yaml"
    if plan.is_file():
        validate_or_raise(yaml.safe_load(plan.read_text()), "dialect_plan")


def test_snapshots_match_recorded_content_hashes():
    provenance = json.loads((ROOT / "provenance.json").read_text())
    assert provenance["files"]
    assert not provenance["qualification"]["candidate_certification"]
    assert not provenance["qualification"]["hardware_executed"]
    for record in provenance["files"]:
        expected = evolved_identity(ROOT, record["path"], record["sha256"])
        assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == expected


def test_provider_contracts_are_generated_from_selected_merlin_contract(monkeypatch):
    import importlib.util
    import os
    import merlin

    spec = importlib.util.spec_from_file_location('sync_contracts', ROOT / 'tools/sync_contracts.py')
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    # Installed Merlin source, or an explicitly selected checkout. No workstation path.
    source_root = Path(os.environ.get('MERLIN_SOURCE_ROOT', Path(merlin.__file__).resolve().parents[2]))
    selected = source_root / 'examples/gemmini/target/contracts'
    generator.sync(selected, check=True)
    for name, (original, rendered) in generator.generated(selected).items():
        provider = yaml.safe_load(rendered)
        upstream = yaml.safe_load(original)
        provider.pop('plugin')
        upstream.pop('plugin', None)
        assert provider == upstream, name

    import pytest
    with monkeypatch.context() as patch:
        original_generated = generator.generated
        patch.setattr(generator, 'generated', lambda path: {
            name: (original, rendered + b'\n# simulated upstream drift\n')
            for name, (original, rendered) in original_generated(path).items()
        })
        with pytest.raises(ValueError, match='provider contract drift'):
            generator.sync(selected, check=True)
