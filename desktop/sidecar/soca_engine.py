"""Frozen entry point for the self-contained SoCa engine sidecar.

The desktop shell invokes this binary exactly like ``soca engine``. Keeping the
entry point small preserves the Click command contract while PyInstaller owns
the interpreter and dependency closure.
"""

from soca.cli import main


if __name__ == "__main__":
    main()
