#!/usr/bin/env python3
"""Collect manuscript figures and create PNG/PDF copies.

PDF files are either copied from existing vector PDF outputs or generated from
PNG images with Pillow's PDF writer. This script never renames PNG files as PDF.
"""

from pathlib import Path
from shutil import copy2

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures_png_pdf"
PNG_DIR = OUT / "png"
PDF_DIR = OUT / "pdf"


def figure_sources() -> list[Path]:
    sources: list[Path] = []
    for path in sorted((ROOT / "figures").glob("*")):
        if path.is_file() and path.suffix.lower() in {".png", ".pdf"}:
            if path.name.startswith("appendix_sample_"):
                continue
            sources.append(path)
    for subdir in ["backbone_radars", "dataset_radars"]:
        for path in sorted((ROOT / "figures" / subdir).glob("*")):
            if path.is_file() and path.suffix.lower() in {".png", ".pdf"}:
                sources.append(path)
    return sources


def stem_for(path: Path) -> str:
    rel = path.relative_to(ROOT / "figures")
    return "_".join(rel.with_suffix("").parts)


def png_to_pdf(png_path: Path, pdf_path: Path) -> None:
    with Image.open(png_path) as img:
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, "white")
            background.paste(img, mask=img.getchannel("A"))
            img = background
        else:
            img = img.convert("RGB")
        img.save(pdf_path, "PDF", resolution=300.0)


def main() -> None:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, dict[str, Path]] = {}
    for path in figure_sources():
        grouped.setdefault(stem_for(path), {})[path.suffix.lower()] = path

    manifest = [
        "# 论文图片 PNG/PDF 集中目录",
        "",
        "本目录集中存放当前论文中使用或候选使用的主图、实验图和雷达图。",
        "",
        "- `png/`：PNG 版本。",
        "- `pdf/`：PDF 版本。若原图没有 PDF，则由 PNG 通过 Pillow 的 PDF writer 转换得到，不是改后缀。",
        "- 未纳入 `figures/appendix_sample_*` 大量附录类别样例图，避免主图目录混乱。",
        "",
        "| 图名 | PNG | PDF | 原始来源 |",
        "| --- | --- | --- | --- |",
    ]

    for stem, formats in sorted(grouped.items()):
        png_src = formats.get(".png")
        pdf_src = formats.get(".pdf")
        png_out = PNG_DIR / f"{stem}.png"
        pdf_out = PDF_DIR / f"{stem}.pdf"

        png_status = "-"
        pdf_status = "-"
        source_parts: list[str] = []

        if png_src is not None:
            copy2(png_src, png_out)
            png_status = f"png/{png_out.name}"
            source_parts.append(str(png_src.relative_to(ROOT)))
        if pdf_src is not None:
            copy2(pdf_src, pdf_out)
            pdf_status = f"pdf/{pdf_out.name}"
            source_parts.append(str(pdf_src.relative_to(ROOT)))
        elif png_src is not None:
            png_to_pdf(png_src, pdf_out)
            pdf_status = f"pdf/{pdf_out.name} (from PNG via Pillow)"

        manifest.append(
            f"| `{stem}` | `{png_status}` | `{pdf_status}` | "
            f"`{', '.join(source_parts)}` |"
        )

    (OUT / "README.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"figures={len(grouped)}")


if __name__ == "__main__":
    main()
