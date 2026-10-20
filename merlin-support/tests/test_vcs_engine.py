"""VCS engine selection, content identity and Chipyard invocation contract."""
import hashlib
import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from gemmini_conformance.support import backend


@pytest.fixture
def vcs(tmp_path, monkeypatch):
    module = importlib.import_module(backend().__name__ + '.gemmini')
    simv = tmp_path/'vcs build'/'simv'
    simv.parent.mkdir()
    simv.write_bytes(b'synthetic simulator')
    simv.chmod(0o755)
    directory = Path(str(simv)+'.daidir')
    directory.mkdir()
    (directory/'model.so').write_bytes(b'elaborated RTL')
    ini = tmp_path/'ini'; ini.mkdir(); (ini/'system.ini').write_text('dram configuration')
    elf = tmp_path/'work'/'program.elf'; elf.parent.mkdir(); elf.write_bytes(b'ELF')
    monkeypatch.setenv('MERLIN_GEMMINI_SIMV', str(simv))
    monkeypatch.setenv('MERLIN_GEMMINI_DRAMSIM_INI', str(ini))
    monkeypatch.delenv('MERLIN_GEMMINI_VCS_MAXCYCLES', raising=False)
    monkeypatch.delenv('MERLIN_GEMMINI_VCS_SEED', raising=False)
    monkeypatch.setattr(module, 'gcc_path', lambda: simv)
    monkeypatch.setattr(module, '_test_ld', lambda: elf)
    monkeypatch.setattr(module, '_common_dir', lambda: elf.parent)
    monkeypatch.setattr(module, 'verilator_path', lambda: pytest.fail('VCS must never select Verilator'))
    return module, simv, directory, ini, elf


def test_vcs_is_known_and_availability_requires_executable_and_daidir(vcs):
    module, simv, directory, _, _ = vcs
    assert module.ORACLE['vcs'] == dict(kind='rtl_vcs', derived_from_rtl=True)
    assert module.available('vcs')
    simv.chmod(0o644)
    assert not module.available('vcs')
    simv.chmod(0o755)
    (directory/'model.so').unlink(); directory.rmdir()
    assert not module.available('vcs')
    with pytest.raises(module.GemminiError, match='unknown simulator'):
        module.available('nonexistent-engine')


def test_vcs_discovery_uses_own_override_and_chipyard_config(vcs, monkeypatch, tmp_path):
    module, simv, directory, _, _ = vcs
    assert module.vcs_path() == simv
    monkeypatch.delenv('MERLIN_GEMMINI_SIMV')
    monkeypatch.setattr(module, 'chipyard_root', lambda: tmp_path)
    monkeypatch.setattr(module, '_rtl_sim_config', lambda env_var: 'OtherConfig')
    assert module.vcs_path() == tmp_path/'sims/vcs/simv-chipyard.harness-OtherConfig'
    assert not module.available('vcs')
    alias = tmp_path/'selected'; alias.symlink_to(simv)
    monkeypatch.setenv('MERLIN_GEMMINI_SIMV', str(alias))
    assert module.vcs_daidir() == directory
    assert module.available('vcs')


def test_vcs_config_is_independent_of_verilator(vcs, monkeypatch, tmp_path):
    module, _, _, _, _ = vcs
    monkeypatch.delenv('MERLIN_GEMMINI_SIMV')
    monkeypatch.setattr(module, 'chipyard_root', lambda: tmp_path)
    monkeypatch.setenv('MERLIN_GEMMINI_VERILATOR_CONFIG', 'UnrelatedVerilatorConfig')
    monkeypatch.setenv('MERLIN_GEMMINI_VCS_CONFIG', 'SelectedVCSConfig')
    assert module.vcs_path() == tmp_path/'sims/vcs/simv-chipyard.harness-SelectedVCSConfig'
    assert module._rtl_sim_config() == 'UnrelatedVerilatorConfig'


def test_vcs_identity_authenticates_binary_and_sorted_recursive_daidir(vcs):
    module, simv, directory, _, _ = vcs
    nested = directory/'nested'; nested.mkdir(); (nested/'runtime.dat').write_bytes(b'configuration')
    initial = module.simulator_identity('vcs')
    assert initial['binary'] == hashlib.sha256(simv.read_bytes()).hexdigest()
    assert [e['path'] for e in initial['daidir']] == ['model.so', 'nested/runtime.dat']
    assert all(e['sha256'] for e in initial['daidir'])
    import json
    assert initial['daidir_sha256'] == hashlib.sha256(json.dumps(initial['daidir'], sort_keys=True).encode()).hexdigest()
    assert initial['digest'] == module.simulator_identity('vcs')['digest']
    (directory/'model.so').write_bytes(b'changed elaboration')
    changed = module.simulator_identity('vcs')
    assert changed['binary'] == initial['binary'] and changed['digest'] != initial['digest']
    simv.write_bytes(b'rebuilt simulator')
    assert module.simulator_identity('vcs')['digest'] != changed['digest']
    (nested/'runtime.dat').unlink()
    before = module.simulator_identity('vcs')
    (nested/'added').write_bytes(b'new runtime file')
    assert module.simulator_identity('vcs')['digest'] != before['digest']


def test_missing_or_unproved_daidir_has_no_identity(vcs):
    module, simv, directory, _, _ = vcs
    (directory/'model.so').unlink()
    assert module.simulator_identity('vcs')['digest'] is None
    (directory/'cycle').symlink_to(directory, target_is_directory=True)
    assert module.simulator_identity('vcs')['digest'] is None
    (directory/'cycle').unlink(); directory.rmdir()
    assert module.simulator_identity('vcs')['digest'] is None
    simv.unlink()
    assert module.simulator_identity('vcs')['digest'] is None


def test_vcs_command_and_mocked_run_preserve_license_and_observer(vcs, monkeypatch):
    from merlin.common import invocation_record
    module, simv, directory, ini, elf = vcs
    monkeypatch.setenv('SNPSLMD_LICENSE_FILE', '1234@license.example')
    monkeypatch.setenv('MERLIN_GEMMINI_VCS_MAXCYCLES', '123456')
    monkeypatch.setenv('MERLIN_GEMMINI_VCS_SEED', '17')
    before = dict(os.environ)
    def run(argv, **kw):
        assert argv == [str(simv), '+permissive', '-no_save', '+dramsim',
                        f'+dramsim_ini_dir={ini}', '+max-cycles=123456', '+ntb_random_seed=17',
                        f'+loadmem={elf}', '+permissive-off', str(elf)]
        assert kw['directory'] == kw['cwd'] == elf.parent
        assert kw['stage'] == 'engine_execution' and kw['inputs'] == (elf,)
        assert directory/'model.so' in kw['dependencies'] and ini/'system.ini' in kw['dependencies']
        assert kw['env']['SNPSLMD_LICENSE_FILE'] == '1234@license.example'
        assert kw['timeout'] == 29 and kw['preexec_fn'] is module._unlimited_stack
        assert kw['stdin'] == module.subprocess.DEVNULL
        return SimpleNamespace(returncode=0, stdout='OUT y 1 2\nDONE\n', stderr='VCS banner')
    monkeypatch.setattr(invocation_record, 'run', run)
    assert module.run_elf(elf, simulator='vcs', timeout=29) == 'OUT y 1 2\nDONE\n'
    assert dict(os.environ) == before


@pytest.mark.parametrize('status,stderr', [(1, 'license refused'), (0, 'Assertion failed: hardware')])
def test_vcs_failure_never_falls_back(vcs, monkeypatch, status, stderr):
    from merlin.common import invocation_record
    module, _, _, _, elf = vcs
    monkeypatch.setattr(invocation_record, 'run', lambda *a, **k:
                        SimpleNamespace(returncode=status, stdout='', stderr=stderr))
    with pytest.raises(module.GemminiError):
        module.run_elf(elf, simulator='vcs')


def test_vcs_defaults_and_dram_directory_convention(vcs, monkeypatch, tmp_path):
    module, _, _, ini, elf = vcs
    argv = module._vcs_argv(elf)
    assert [v for v in argv if v.startswith('+loadmem=')] == [f'+loadmem={elf}']
    assert argv[-2:] == ['+permissive-off', str(elf)]
    assert '+max-cycles=10000000' in argv and '+ntb_random_seed=1' in argv
    monkeypatch.delenv('MERLIN_GEMMINI_DRAMSIM_INI')
    monkeypatch.setenv('GEMMINI_DRAMSIM_INI', str(ini))
    assert module.vcs_dramsim_ini() == ini
    monkeypatch.delenv('GEMMINI_DRAMSIM_INI')
    monkeypatch.setattr(module, 'chipyard_root', lambda: tmp_path)
    assert module.vcs_dramsim_ini() == tmp_path/'generators/testchipip/src/main/resources/dramsim2_ini'


@pytest.mark.parametrize('name,value', [('MERLIN_GEMMINI_VCS_MAXCYCLES','0'),
                                      ('MERLIN_GEMMINI_VCS_MAXCYCLES','1.5'),
                                      ('MERLIN_GEMMINI_VCS_SEED','-1')])
def test_invalid_vcs_parameters_decline_before_launch(vcs, monkeypatch, name, value):
    module, _, _, _, elf = vcs
    monkeypatch.setenv(name,value)
    with pytest.raises(module.GemminiError, match='integers'):
        module._vcs_argv(elf)
