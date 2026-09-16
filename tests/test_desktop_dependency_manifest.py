import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "desktop"


def test_desktop_runtime_does_not_ship_shadcn_cli() -> None:
    package = json.loads((DESKTOP / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((DESKTOP / "package-lock.json").read_text(encoding="utf-8"))

    assert "shadcn" not in package.get("dependencies", {})
    assert "shadcn" not in package.get("devDependencies", {})
    assert "node_modules/shadcn" not in lock["packages"]


def test_shadcn_tailwind_variants_are_vendored_locally() -> None:
    index_css = (DESKTOP / "src" / "index.css").read_text(encoding="utf-8")
    variants_css = (DESKTOP / "src" / "shadcn-tailwind.css").read_text(
        encoding="utf-8"
    )

    assert '@import "./shadcn-tailwind.css";' in index_css
    assert '@import "shadcn/tailwind.css";' not in index_css
    assert "@custom-variant data-open" in variants_css
    assert "@custom-variant data-closed" in variants_css
    assert (DESKTOP / "src" / "shadcn-tailwind.LICENSE.md").is_file()
