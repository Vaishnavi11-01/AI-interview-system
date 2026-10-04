"""Generate a privacy-preserving static sample gallery from a local run."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re

import cv2

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "output" / "drive_run_report.html"
SOURCE_ASSETS = ROOT / "output"
DESTINATION = ROOT / "docs" / "gallery"
MAX_CARDS_PER_TYPE = 6


class GalleryParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, str]] = []
        self.current: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "article" and "event-card" in attributes.get("class", ""):
            self.current = {"type": attributes.get("data-type", "event")}
        elif tag == "img" and self.current is not None:
            self.current["src"] = attributes.get("src", "")
        elif tag == "article" and self.current is not None:
            self.events.append(self.current)
            self.current = None

    def handle_endtag(self, tag: str) -> None:
        if tag == "article" and self.current is not None:
            self.events.append(self.current)
            self.current = None


def pixelate_image(source: Path, destination: Path) -> None:
    image = cv2.imread(str(source))
    if image is None:
        raise RuntimeError(f"Could not read snapshot: {source}")
    height, width = image.shape[:2]
    # Reduce every face crop to at most 2x2 pixels before enlarging it.
    tiny = cv2.resize(image, (min(2, width), min(2, height)), interpolation=cv2.INTER_AREA)
    mosaic = cv2.resize(tiny, (width, height), interpolation=cv2.INTER_NEAREST)
    if not cv2.imwrite(str(destination), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 72]):
        raise RuntimeError(f"Could not write anonymized snapshot: {destination}")


def main() -> None:
    if not REPORT.is_file():
        raise SystemExit(f"Generate the local report first: {REPORT}")

    parser = GalleryParser()
    parser.feed(REPORT.read_text(encoding="utf-8"))
    selected: list[dict[str, str]] = []
    for event_type in ("entry", "exit"):
        matching = [event for event in parser.events if event.get("type") == event_type]
        selected.extend(matching[:MAX_CARDS_PER_TYPE])
    if not selected:
        raise SystemExit("No entry/exit snapshots found in the generated report.")

    DESTINATION.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for index, event in enumerate(selected, start=1):
        source_relative = event.get("src", "")
        source = (SOURCE_ASSETS / source_relative).resolve()
        if SOURCE_ASSETS.resolve() not in source.parents or not source.is_file():
            raise SystemExit(f"Missing or invalid gallery asset: {source_relative}")
        image_name = f"sample-{index:02d}.jpg"
        pixelate_image(source, DESTINATION / image_name)
        kind = "Entry" if event.get("type") == "entry" else "Exit"
        cards.append(
            f'<article class="card"><img src="{image_name}" alt="Heavily pixelated sample face, {kind.lower()} event" loading="lazy">'
            f'<div><span class="badge">{kind}</span><h2>Sample event {index:02d}</h2>'
            '<p>Face details intentionally obscured for privacy.</p></div></article>'
        )

    report = REPORT.read_text(encoding="utf-8")
    stats = re.findall(r'<div class="stat"><strong>(.*?)</strong><span>(.*?)</span></div>', report)
    stat_cards = "".join(
        f'<div class="stat"><strong>{value}</strong><span>{label}</span></div>'
        for value, label in stats[:3]
    )
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Visitor Event Gallery · Privacy-safe demo</title>
<style>
:root{{color-scheme:dark;font-family:Inter,Segoe UI,sans-serif;background:#0b1020;color:#edf2ff}}*{{box-sizing:border-box}}body{{margin:0}}main{{width:min(1080px,92vw);margin:48px auto}}.eyebrow{{color:#91a7ff;text-transform:uppercase;letter-spacing:.14em;font-weight:700;font-size:.78rem}}h1{{font-size:clamp(2rem,5vw,3.2rem);margin:.5rem 0 1rem}}.note{{color:#c2cce0;line-height:1.6;max-width:760px}}.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:26px 0}}.stat,.card{{background:#141c31;border:1px solid #293653;border-radius:16px;overflow:hidden}}.stat{{padding:18px}}.stat strong{{display:block;font-size:1.8rem}}.stat span,p{{color:#aebbd2}}.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:16px}}.card img{{width:100%;height:220px;object-fit:contain;background:#070b15;image-rendering:pixelated}}.card div{{padding:15px}}.card h2{{font-size:1rem;margin:10px 0 4px}}.card p{{font-size:.86rem;margin:4px 0}}.badge{{display:inline-block;color:#dbe5ff;background:#263659;border-radius:99px;padding:5px 10px;font-size:.75rem;font-weight:700}}footer{{margin:30px 0;color:#8795ad;font-size:.85rem}}@media(max-width:600px){{main{{margin:28px auto}}.stats{{grid-template-columns:1fr}}}}
</style></head><body><main><div class="eyebrow">Intelligent Face Tracker</div><h1>Visitor event gallery</h1>
<p class="note">Privacy-safe sample output. Summary counts are from a processed sample; only a few example event types are shown. Face snapshots are heavily pixelated, and the original video, precise timestamps, visitor identifiers, and raw biometric assets are not published here.</p>
<section class="stats" aria-label="Sample summary">{stat_cards}</section><section class="gallery" aria-label="Anonymized sample events">{''.join(cards)}</section>
<footer>Demo gallery · Faces intentionally obscured · Do not use sample counts as a live service result.</footer></main></body></html>'''
    (DESTINATION / "index.html").write_text(page, encoding="utf-8")
    print(f"Generated {len(selected)} anonymized event cards in {DESTINATION}")


if __name__ == "__main__":
    main()
