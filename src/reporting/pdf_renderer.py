"""生成済みHTML（`outputs/renders/`配下）をChromium headlessでPDF化する。

画像の相対パス（`exp_a/...`, `figures/...` 等）を解決するため、`outputs/`をドキュメントルートとする
ローカルHTTPサーバーを同一プロセス内に一時的に起動してから `--print-to-pdf` を実行する。
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import subprocess
import threading
from pathlib import Path

TASK9_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = TASK9_ROOT / "outputs"
RENDERS_DIR = OUTPUTS_DIR / "renders"

CHROME_BIN = "google-chrome"


@contextlib.contextmanager
def _serve_directory(directory: Path, port: int = 0):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def render_html_to_pdf(
    html_path: Path,
    pdf_path: Path,
    virtual_time_budget_ms: int = 8000,
) -> Path:
    html_path = html_path.resolve()
    if OUTPUTS_DIR not in html_path.parents and html_path.parent != OUTPUTS_DIR:
        raise ValueError(f"html_path must live under {OUTPUTS_DIR} so relative image paths resolve: {html_path}")

    rel = html_path.relative_to(OUTPUTS_DIR)

    with _serve_directory(OUTPUTS_DIR) as port:
        url = f"http://127.0.0.1:{port}/{rel.as_posix()}"
        cmd = [
            CHROME_BIN,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            f"--virtual-time-budget={virtual_time_budget_ms}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if not pdf_path.exists():
            raise RuntimeError(f"PDF generation failed.\nstdout={result.stdout}\nstderr={result.stderr}")

    return pdf_path


def render_pdf_to_page_images(pdf_path: Path, out_dir: Path, dpi: int = 100) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.glob("page-*.png"):
        existing.unlink()

    prefix = out_dir / "page"
    cmd = ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), str(prefix)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    pages = sorted(out_dir.glob("page-*.png"))
    # pdftoppm がゼロ埋めしない場合（10ページ未満等）があるため、page-XXX.png (3桁ゼロ埋め) へ統一する
    renamed = []
    for p in pages:
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        new_name = out_dir / f"page-{int(digits):03d}.png"
        if p != new_name:
            p.rename(new_name)
        renamed.append(new_name)
    return sorted(renamed)


def main_all_reports() -> int:
    """本編・発展版の生成済みPDFをまとめてページ画像へ変換する。"""
    for stem in ("SUMMARY_REPORT", "SUMMARY_REPORT_extra"):
        pages = render_pdf_to_page_images(
            OUTPUTS_DIR / f"{stem}.pdf",
            RENDERS_DIR / stem.lower(),
        )
        print(f"{stem}: {len(pages)} pages")
    return 0


if __name__ == "__main__":
    html = RENDERS_DIR / "_summary_report_render.html"
    pdf = OUTPUTS_DIR / "SUMMARY_REPORT.pdf"
    render_html_to_pdf(html, pdf)
    pages = render_pdf_to_page_images(pdf, RENDERS_DIR / "summary_report")
    print(f"wrote {pdf} and {len(pages)} page images")
