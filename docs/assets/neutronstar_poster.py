#!/usr/bin/env python3
"""Procedural hero poster for the NeutronStar repo.

Renders a dark editorial poster: a neutron star drawn as an LED-wall pixel
mosaic (radial glow + polar jets + tilted accretion disk), with a tall serif
wordmark and small-caps captions. Pure PIL, macOS system fonts.

Run: python3.13 neutronstar_poster.py
"""
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H = 1600, 900
OUT = __file__.rsplit("/", 1)[0] + "/neutronstar-poster.png"

# Neutron-star core position (center-right)
CX, CY = int(W * 0.63), int(H * 0.52)

# Mosaic geometry
CELL = 12      # cell size in px
GAP = 3        # gap between cells
STEP = CELL + GAP


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def lerp(a, b, t):
    return a + (b - a) * t


def palette(intensity):
    """Map 0..1 intensity to an (r,g,b) on the neutron-star ramp.

    Falls off: white core -> cyan -> violet -> magenta -> deep purple -> black.
    """
    stops = [
        (0.00, (6, 4, 16)),       # near-black background
        (0.18, (24, 10, 54)),     # deep purple
        (0.36, (86, 22, 120)),    # magenta-violet
        (0.54, (150, 40, 190)),   # magenta
        (0.68, (110, 90, 240)),   # violet-blue
        (0.82, (90, 190, 245)),   # cyan
        (0.92, (200, 240, 255)),  # pale blue-white
        (1.00, (255, 255, 255)),  # blinding white
    ]
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if intensity <= t1:
            f = (intensity - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple(int(lerp(c0[k], c1[k], f)) for k in range(3))
    return stops[-1][1]


def field(x, y, stars):
    """Continuous intensity field at pixel (x, y), 0..1 (pre-clamp can exceed)."""
    dx = x - CX
    dy = y - CY
    r = math.hypot(dx, dy)

    # Core radial glow: bright small center, smooth exponential falloff.
    core = math.exp(-r / 46.0) * 1.35
    halo = math.exp(-r / 150.0) * 0.55
    outer = math.exp(-r / 320.0) * 0.22

    # Two narrow vertical polar jets (up and down from core).
    jet_w = 10.0            # horizontal tightness
    jet = 0.0
    jet_falloff = math.exp(-(dx * dx) / (2 * jet_w * jet_w))
    if jet_falloff > 0.01:
        # length falloff along y, fading out toward edges
        along = math.exp(-abs(dy) / 240.0)
        jet = jet_falloff * along * 0.85

    # Tilted accretion disk: hot ellipse ring around the core.
    # Rotate coords by tilt, then measure distance to an ellipse ring.
    tilt = math.radians(-16)
    rxk, ryk = 235.0, 82.0
    xr = dx * math.cos(tilt) + dy * math.sin(tilt)
    yr = -dx * math.sin(tilt) + dy * math.cos(tilt)
    ell = math.hypot(xr / rxk, yr / ryk)
    disk = math.exp(-((ell - 1.0) ** 2) / (2 * 0.12 ** 2)) * 0.42
    # Mild ansae emphasis so it reads as a ring rather than a beam.
    disk *= clamp(0.6 + 0.4 * abs(xr) / rxk)

    val = core + halo + outer + jet + disk

    # Background scatter stars.
    for sx, sy, sb in stars:
        sd = math.hypot(x - sx, y - sy)
        val += sb * math.exp(-sd / 5.0)

    return val


def main():
    import random
    random.seed(42)

    # Precompute a handful of dim background stars.
    stars = []
    for _ in range(90):
        sx = random.uniform(0, W)
        sy = random.uniform(0, H)
        # keep stars away from the very bright core
        if math.hypot(sx - CX, sy - CY) < 120:
            continue
        sb = random.uniform(0.05, 0.22)
        stars.append((sx, sy, sb))

    img = Image.new("RGB", (W, H), (4, 3, 10))
    # Glow layer accumulates soft light to be blurred and screened under cells.
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    cells = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(cells)

    cols = W // STEP + 2
    rows = H // STEP + 2

    for cy in range(rows):
        for cx in range(cols):
            px = cx * STEP + GAP
            py = cy * STEP + GAP
            mx = px + CELL / 2
            my = py + CELL / 2

            inten = field(mx, my, stars)
            # gamma to push mids down so subject melts into black
            vi = clamp(inten)
            vi = vi ** 1.15

            if vi < 0.015:
                continue  # leave as background (melts to black)

            color = palette(vi)
            # dim overall so only bright cells pop
            alpha = int(clamp(vi * 1.05) * 255)
            cdraw.rounded_rectangle(
                [px, py, px + CELL, py + CELL],
                radius=3,
                fill=color + (alpha,),
            )

            # Bright cells seed the glow layer.
            if vi > 0.6:
                g = int((vi - 0.6) / 0.4 * 255)
                gcol = palette(clamp(vi))
                gdraw.rounded_rectangle(
                    [px, py, px + CELL, py + CELL],
                    radius=3,
                    fill=tuple(int(c * g / 255) for c in gcol),
                )

    # Blur glow in two passes for a bloom, screen it onto the base.
    glow_small = glow.filter(ImageFilter.GaussianBlur(9))
    glow_big = glow.filter(ImageFilter.GaussianBlur(28))
    from PIL import ImageChops
    base = ImageChops.screen(img, glow_big)
    base = ImageChops.screen(base, glow_small)
    base.paste(cells, (0, 0), cells)

    # Extra core bloom on top for the blinding center.
    core_bloom = Image.new("RGB", (W, H), (0, 0, 0))
    cbd = ImageDraw.Draw(core_bloom)
    cbd.ellipse([CX - 60, CY - 60, CX + 60, CY + 60], fill=(120, 150, 200))
    cbd.ellipse([CX - 26, CY - 26, CX + 26, CY + 26], fill=(255, 255, 255))
    core_bloom = core_bloom.filter(ImageFilter.GaussianBlur(22))
    base = ImageChops.screen(base, core_bloom)

    draw_text(base)
    base.save(OUT)
    print("wrote", OUT)


def load_font(paths, size, index=0):
    for p in paths:
        try:
            return ImageFont.truetype(p, size, index=index)
        except Exception:
            continue
    return ImageFont.load_default()


def vgrad_text(base, text, font, xy, top=(255, 255, 255), bot=(120, 120, 150)):
    """Draw text with a vertical gradient fill (top->bottom)."""
    # Measure
    tmp = ImageDraw.Draw(base)
    bbox = tmp.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # Mask
    mask = Image.new("L", (tw + 8, th + 8), 0)
    md = ImageDraw.Draw(mask)
    md.text((-bbox[0] + 4, -bbox[1] + 4), text, font=font, fill=255)
    # Gradient
    grad = Image.new("RGB", (tw + 8, th + 8))
    gpix = grad.load()
    for yy in range(th + 8):
        f = yy / max(1, th + 7)
        col = tuple(int(lerp(top[k], bot[k], f)) for k in range(3))
        for xx in range(tw + 8):
            gpix[xx, yy] = col
    base.paste(grad, (xy[0], xy[1]), mask)
    return tw, th


DIDOT = ["/System/Library/Fonts/Supplemental/Didot.ttc"]
BODONI = ["/System/Library/Fonts/Supplemental/Bodoni 72.ttc"]
HELV = ["/System/Library/Fonts/Helvetica.ttc"]
HELVN = ["/System/Library/Fonts/HelveticaNeue.ttc"]


def draw_text(base):
    d = ImageDraw.Draw(base)

    # --- Wordmark, top-left, huge tall serif with vertical gradient ---
    serif = load_font(DIDOT, 102)
    vgrad_text(base, "NEUTRONSTAR", serif, (64, 64),
               top=(255, 255, 255), bot=(96, 104, 140))

    # --- Tagline, top-right, small-caps sans, letter-spaced, 3 lines ---
    tag = load_font(HELVN, 16, index=10)  # medium, roman
    lines = ["GIANT MODELS", "ON MODEST HARDWARE", "EXPERTS STREAMED FROM DISK"]
    ty = 72
    for ln in lines:
        spaced = spaced_text(ln, 3)
        w = d.textlength(spaced, font=tag)
        d.text((W - 64 - w, ty), spaced, font=tag, fill=(184, 190, 208))
        ty += 27

    # --- Bottom-left caption block ---
    cap_big = load_font(HELVN, 22, index=1)   # bold, roman
    d.text((66, H - 96), spaced_text("SSD EXPERT STREAMING", 2),
           font=cap_big, fill=(224, 228, 240))


def spaced_text(s, n):
    """Insert n spaces of tracking between characters (letter-spacing hack)."""
    sp = " " * n
    return sp.join(list(s))


if __name__ == "__main__":
    main()
