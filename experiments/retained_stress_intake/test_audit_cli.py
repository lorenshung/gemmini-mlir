"""Only import/path controls; never invoke the retained target readback audit."""

import importlib.util
import os
from pathlib import Path

import pytest


def load():
    path = Path(__file__).with_name("audit.py")
    spec = importlib.util.spec_from_file_location("retained_audit_path_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_creates_no_directories_and_changes_no_environment(monkeypatch):
    before = dict(os.environ)
    monkeypatch.setattr(
        Path, "mkdir", lambda *a, **kw: pytest.fail("import cannot write")
    )
    load()
    assert dict(os.environ) == before


@pytest.mark.parametrize(
    "fault", ["absent_task", "outside", "existing", "linked", "relative", "foreign_cwd"]
)
def test_invalid_output_refuses_before_creation_or_audit(tmp_path, monkeypatch, fault):
    module = load()
    task = tmp_path / "task"
    task.mkdir()
    monkeypatch.chdir(task)
    monkeypatch.setenv("TASK_DIR", str(task))
    output = task / "fresh"
    if fault == "absent_task":
        monkeypatch.delenv("TASK_DIR")
    elif fault == "outside":
        output = tmp_path / "other-agent"
    elif fault == "existing":
        output.mkdir()
    elif fault == "linked":
        (task / "link").symlink_to(tmp_path, target_is_directory=True)
        output = task / "link/fresh"
    elif fault == "relative":
        output = Path("fresh")
    else:
        monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        module, "run_audit", lambda *_a: pytest.fail("invalid setup cannot audit")
    )
    with pytest.raises(ValueError):
        module.main(["--output", str(output)])
    assert not (tmp_path / "other-agent").exists()


def test_cli_selects_only_fresh_actual_task_output(tmp_path, monkeypatch):
    module = load()
    monkeypatch.setenv("TASK_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "out/audit"
    calls = []
    monkeypatch.setattr(module, "run_audit", lambda *args: calls.append(args))
    assert module.main(["--output", str(output)]) == 0
    assert calls == [(tmp_path, output)] and output.is_dir()
