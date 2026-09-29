import dogbreeds
from dogbreeds.paths import PROJECT_ROOT


def test_package_imports():
    assert dogbreeds.__version__


def test_project_root_found():
    assert (PROJECT_ROOT / "pyproject.toml").exists()
