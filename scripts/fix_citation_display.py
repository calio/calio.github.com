from __future__ import annotations

from pathlib import Path


def main() -> None:
    path = Path(__file__).resolve().parents[1] / "_posts" / "2026-02-07-world-models.md"
    text = path.read_text(encoding="utf-8")

    replacements: list[tuple[str, str]] = [
        ("([arXiv][1])", "([World Models][1])"),
        ("([arXiv][2])", "([World Models][2])"),
        ("([arXiv][3])", "([PlaNet][3])"),
        ("([arXiv][4])", "([Dreamer][4])"),
        ("([Google Research][5])", "([Dreamer blog][5])"),
        ("([arXiv][6])", "([DreamerV2][6])"),
        ("([arXiv][7])", "([DreamerV3][7])"),
        ("([arXiv][8])", "([Dreamer 4][8])"),
        ("([arXiv][9])", "([MuZero][9])"),
        ("([arXiv][10])", "([SimPLe][10])"),
        ("([arXiv][11])", "([Genie][11])"),
        ("([Google DeepMind][12])", "([Genie 3][12])"),
        ("([arXiv][13])", "([I-JEPA][13])"),
        ("([arXiv][14])", "([V-JEPA 2][14])"),
        ("([Nature][15])", "([MuZero (Nature)][15])"),
        ("([arXiv][16])", "([AdaWorld][16])"),
        ("([arXiv][17])", "([villa-X][17])"),
        ("([arXiv][18])", "([Track2Act][18])"),
        ("([BMVC][19])", "([Mask2Act][19])"),
        ("([X][20])", "([LeCun][20])"),
    ]

    updated = text
    for old, new in replacements:
        updated = updated.replace(old, new)

    if updated == text:
        print("No changes needed.")
        return

    path.write_text(updated, encoding="utf-8")
    print("Updated citation display text.")


if __name__ == "__main__":
    main()
