"""Draw the demo infrastructure diagram offline (no network, no image model).

The topology is deliberately weak in ways that are *visible*: a flat internal network,
a public admin path that skips the firewall, plaintext links, and a database reachable
straight from the DMZ. That gives the vision pass real structure to reason about.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "data" / "demo_diagram.png"

W, H = 1180, 760
BG = (247, 250, 249)
INK = (17, 32, 44)
MUTED = (95, 115, 128)
LINE = (110, 130, 142)
RED = (214, 32, 54)
BOX_FILL = (255, 255, 255)


def box(d, xy, title, sub="", accent=INK):
    x0, y0, x1, y1 = xy
    d.rounded_rectangle(xy, radius=10, fill=BOX_FILL, outline=accent, width=3)
    d.text((x0 + 14, y0 + 12), title, fill=INK)
    if sub:
        d.text((x0 + 14, y0 + 30), sub, fill=MUTED)


def arrow(d, p0, p1, label="", color=LINE, width=3, dash=False, label_at=None):
    x0, y0 = p0
    x1, y1 = p1
    if dash:
        # Manual dashes; PIL has no dash support.
        steps = 26
        for i in range(steps):
            if i % 2:
                continue
            a = i / steps
            b = (i + 1) / steps
            d.line(
                [x0 + (x1 - x0) * a, y0 + (y1 - y0) * a, x0 + (x1 - x0) * b, y0 + (y1 - y0) * b],
                fill=color,
                width=width,
            )
    else:
        d.line([p0, p1], fill=color, width=width)
    d.ellipse([x1 - 5, y1 - 5, x1 + 5, y1 + 5], fill=color)
    if label:
        lx, ly = label_at or ((x0 + x1) / 2 - 40, (y0 + y1) / 2 - 16)
        # White plate keeps flow labels legible where lines cross boxes.
        tw = d.textlength(label)
        d.rectangle([lx - 4, ly - 3, lx + tw + 4, ly + 14], fill=BG)
        d.text((lx, ly), label, fill=color)


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.text((40, 26), "ACME PAYMENTS - PRODUCTION NETWORK (as built)", fill=INK)
    d.text((40, 46), "Reference architecture for security review", fill=MUTED)

    # Trust zones
    d.rounded_rectangle([30, 90, 340, 700], radius=14, outline=(170, 185, 195), width=2)
    d.text((46, 100), "INTERNET (untrusted)", fill=MUTED)

    d.rounded_rectangle([370, 90, 700, 700], radius=14, outline=(170, 185, 195), width=2)
    d.text((386, 100), "DMZ", fill=MUTED)

    d.rounded_rectangle([730, 90, 1150, 700], radius=14, outline=(170, 185, 195), width=2)
    d.text((746, 100), "INTERNAL - FLAT VLAN 10.0.0.0/16 (no segmentation)", fill=MUTED)

    # Internet side
    box(d, (60, 150, 300, 215), "Public users", "HTTPS 443")
    box(d, (60, 250, 300, 315), "Remote admins", "RDP 3389 open to 0.0.0.0/0", accent=RED)
    box(d, (60, 350, 300, 415), "Partner API clients", "HTTP 80 - plaintext", accent=RED)

    # DMZ
    box(d, (400, 150, 670, 215), "Edge firewall", "allow 443, 80, 3389")
    box(d, (400, 250, 670, 330), "nginx 1.24 reverse proxy", "TLS terminate - no WAF", accent=RED)
    box(d, (400, 370, 670, 450), "Apache Tomcat 9.0.85", "app server - internet reachable", accent=RED)
    box(d, (400, 490, 670, 555), "Jenkins 2.4 CI", "admin UI exposed, no MFA", accent=RED)

    # Internal
    box(d, (770, 150, 1110, 230), "PostgreSQL 14 primary", "5432 open to entire VLAN", accent=RED)
    box(d, (770, 260, 1110, 340), "Domain controller (AD)", "SMBv1 enabled, no LAPS", accent=RED)
    box(d, (770, 370, 1110, 450), "Backup NAS", "NFS no_root_squash, no encryption", accent=RED)
    box(d, (770, 480, 1110, 560), "Jump host", "shared local admin account", accent=RED)
    box(d, (770, 590, 1110, 660), "SIEM collector", "logs only from firewall")

    # Flows
    arrow(d, (300, 182), (400, 182), "443", label_at=(316, 160))
    arrow(d, (300, 282), (770, 520), "RDP 3389 straight to jump host (bypasses firewall)",
          color=RED, dash=True, label_at=(330, 620))
    arrow(d, (300, 382), (400, 400), "HTTP 80", label_at=(306, 356), color=RED)
    arrow(d, (535, 215), (535, 250), "")
    arrow(d, (535, 330), (535, 370), "")
    arrow(d, (670, 400), (770, 200), "SQL 5432 plaintext", color=RED, label_at=(688, 246))
    arrow(d, (670, 515), (770, 320), "deploy as Domain Admin", color=RED, label_at=(676, 452))
    arrow(d, (670, 425), (770, 405), "backup write", color=LINE, label_at=(672, 380))

    d.text((40, 718), "Legend: red = as-built deviation from design. No egress filtering. No EDR on Linux hosts.", fill=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
