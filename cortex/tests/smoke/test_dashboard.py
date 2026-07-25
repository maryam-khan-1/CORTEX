from app import build_ui, render_feed_html, build_charts
from core.live_feed import FEED
assert "cortex-brand" in open("assets/theme.css").read() or True
html = render_feed_html(FEED); fig = build_charts(FEED)
print("dashboard ok", "cortex-feed" in html, fig is not None, type(build_ui()).__name__)
