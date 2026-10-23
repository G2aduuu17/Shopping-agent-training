"""Keep test scratch space independent of previous users/sandbox sessions."""

import tempfile
from pathlib import Path


def pytest_configure(config):
    if config.option.basetemp is None:
        # Reserve a fresh parent under the invoking user's TEMP. Avoid scanning the
        # shared pytest-of-<user> directory, which may have another session's ACL.
        # pytest may clear basetemp: only give it a child of our new private folder.
        scratch = Path(tempfile.mkdtemp(prefix="shopmini-pytest-")).resolve()
        config.option.basetemp = scratch / "work"
