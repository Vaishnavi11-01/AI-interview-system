"""Generate a self-contained HTML gallery for persisted visitor events."""

from html import escape
import os
from pathlib import Path
import sqlite3
from urllib.parse import quote


def generate_report(database_path: Path, output_path: Path) -> Path:
    database_path = database_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if database_path.exists():
        with sqlite3.connect(database_path) as connection:
            visitor_count = connection.execute("SELECT COUNT(*) FROM visitors").fetchone()[0]
            events = connection.execute(
                """SELECT event_type, visitor_id, timestamp, image_path, camera_id
                   FROM events ORDER BY timestamp DESC, event_id DESC"""
            ).fetchall()
    else:
        visitor_count, events = 0, []

    entry_count = sum(event[0] == "entry" for event in events)
    exit_count = sum(event[0] == "exit" for event in events)
    video_path = output_path.parent / "output" / "annotated_sample.mp4"
    video_html = ""
    if video_path.is_file():
        video_href = quote(Path(os.path.relpath(video_path, output_path.parent)).as_posix(), safe="/:")
        video_html = f"""<section class="video-panel">
    <h2>Processed video</h2>
    <p>Annotated playback of the sample stream. Entry and exit snapshots below are the database event records.</p>
    <video controls preload="metadata" src="{escape(video_href, quote=True)}"></video>
  </section>"""
    cards: list[str] = []
    for event_type, visitor_id, timestamp, image_path, camera_id in events:
        image = Path(image_path)
        if image.exists():
            image_href = quote(Path(os.path.relpath(image, output_path.parent)).as_posix(), safe="/:")
            image_markup = f'<img src="{escape(image_href, quote=True)}" alt="{escape(event_type)} snapshot for {escape(visitor_id)}" loading="lazy">'
        else:
            image_markup = '<div class="missing">Snapshot file not found</div>'
        cards.append(
            f"""<article class="event-card" data-type="{escape(event_type)}">
              <div class="photo">{image_markup}</div>
              <div class="event-info">
                <span class="badge {escape(event_type)}">{escape(event_type.title())}</span>
                <h2>{escape(visitor_id)}</h2>
                <p>{escape(timestamp.replace('T', ' ').replace('+00:00', ' UTC'))}</p>
                <p class="camera">{escape(camera_id)}</p>
              </div>
            </article>"""
        )

    gallery = "\n".join(cards) or '<p class="empty">No visitor events yet. Process a video first.</p>'
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Visitor event gallery</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, Segoe UI, sans-serif; background: #0b1020; color: #edf2ff; }}
    * {{ box-sizing: border-box; }} body {{ margin: 0; }}
    main {{ width: min(1120px, 92vw); margin: 48px auto; }}
    .eyebrow {{ color: #8da5ff; letter-spacing: .13em; text-transform: uppercase; font-size: .76rem; font-weight: 700; }}
    h1 {{ margin: 8px 0 28px; font-size: clamp(2rem, 5vw, 3rem); }}
    .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 28px; }}
    .stat, .event-card {{ background: #141c31; border: 1px solid #26314e; border-radius: 18px; }}
    .stat {{ padding: 18px 20px; }} .stat strong {{ display: block; font-size: 1.8rem; }} .stat span {{ color: #9cabc9; }}
    .toolbar {{ display: flex; gap: 8px; margin-bottom: 16px; }}
    .video-panel {{ margin: 24px 0; padding: 18px; background: #141c31; border: 1px solid #26314e; border-radius: 18px; }}
    .video-panel h2 {{ margin: 0 0 8px; }} .video-panel p {{ color: #9cabc9; }}
    .video-panel video {{ display: block; width: 100%; max-height: 68vh; margin-top: 14px; background: #050812; border-radius: 12px; }}
    button {{ color: #dce5ff; border: 1px solid #344363; background: #19243d; border-radius: 999px; padding: 9px 16px; cursor: pointer; }}
    button.active {{ color: #091020; background: #8da5ff; border-color: #8da5ff; font-weight: 700; }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }}
    .event-card {{ overflow: hidden; }} .photo {{ height: 250px; background: #080c17; display: grid; place-items: center; }}
    .photo img {{ width: 100%; height: 100%; object-fit: contain; }} .missing {{ color: #aab5ce; }}
    .event-info {{ padding: 16px; }} .badge {{ border-radius: 999px; padding: 5px 10px; font-size: .75rem; font-weight: 700; }}
    .badge.entry {{ color: #77edbd; background: #113b32; }} .badge.exit {{ color: #ffc38e; background: #49301e; }}
    h2 {{ font-size: 1rem; margin: 14px 0 5px; }} p {{ margin: 5px 0; color: #b7c3dc; font-size: .88rem; }} .camera {{ color: #8190ae; }}
    .empty {{ color: #b7c3dc; }} @media(max-width:600px) {{ main {{ margin: 28px auto; }} .stats {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body><main>
  <div class="eyebrow">Intelligent Face Tracker</div>
  <h1>Visitor event gallery</h1>
  <section class="stats" aria-label="Summary">
    <div class="stat"><strong>{visitor_count}</strong><span>Unique visitors</span></div>
    <div class="stat"><strong>{entry_count}</strong><span>Entries</span></div>
    <div class="stat"><strong>{exit_count}</strong><span>Exits</span></div>
  </section>
  {video_html}
  <nav class="toolbar" aria-label="Filter events">
    <button class="active" data-filter="all">All events</button>
    <button data-filter="entry">Entries</button>
    <button data-filter="exit">Exits</button>
  </nav>
  <section class="gallery">{gallery}</section>
</main>
<script>
  document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click', () => {{
    document.querySelectorAll('[data-filter]').forEach(item => item.classList.toggle('active', item === button));
    document.querySelectorAll('.event-card').forEach(card => {{
      card.hidden = button.dataset.filter !== 'all' && card.dataset.type !== button.dataset.filter;
    }});
  }}));
</script>
</body></html>"""
    output_path.write_text(page, encoding="utf-8")
    return output_path
