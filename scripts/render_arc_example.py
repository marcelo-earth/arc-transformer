"""Generate a small illustrative ARC puzzle as an editable SVG."""
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PALETTE = {0: "#111827", 1: "#0074D9", 2: "#FF4136"}
parts = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="760" viewBox="0 0 1200 760" role="img" aria-labelledby="title description">',
    '<title id="title">Learn a color rule from examples, then apply it to a new grid</title>',
    '<desc id="description">Two input and output examples replace every red cell with blue. A new input has a different red shape; its expected output preserves the shape and replaces red with blue. Black cells stay black. This is an illustrative puzzle.</desc>',
    '<rect width="1200" height="760" rx="20" fill="#F3F6FA"/>',
    '<g font-family="Arial, Helvetica, sans-serif">',
]


def text(x, y, value, *, size=20, color="#172033", weight="normal", anchor="start"):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')


def panel(x, y, width, height):
    parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="16" fill="#FFFFFF" stroke="#D8E0EB" stroke-width="2"/>')


def grid(x, y, values, cell=46):
    parts.append(f'<rect x="{x}" y="{y}" width="{cell * 3}" height="{cell * 3}" fill="#FFFFFF"/>')
    for row, values_row in enumerate(values):
        for column, value in enumerate(values_row):
            parts.append(f'<rect x="{x + column * cell + 1}" y="{y + row * cell + 1}" width="{cell - 2}" height="{cell - 2}" fill="{PALETTE[value]}"/>')
    parts.append(f'<rect x="{x}" y="{y}" width="{cell * 3}" height="{cell * 3}" fill="none" stroke="#172033" stroke-width="2"/>')


def output_for(values):
    return [[1 if value == 2 else value for value in row] for row in values]


def arrow(x, y, width=42):
    parts.append(f'<rect x="{x}" y="{y - 2}" width="{width - 8}" height="4" fill="#65758D"/>')
    parts.append(f'<polygon points="{x + width},{y} {x + width - 12},{y - 9} {x + width - 12},{y + 9}" fill="#65758D"/>')


text(48, 62, "Discover the rule. Apply it to a new grid.", size=34, weight="bold")
text(48, 100, "An ARC task gives a few input/output examples, then asks for a new output.", size=21, color="#52627A")

examples = [
    [[0, 2, 0], [2, 2, 0], [0, 0, 0]],
    [[0, 0, 2], [0, 2, 2], [0, 0, 2]],
]
for index, values in enumerate(examples):
    x = 48 + index * 568
    panel(x, 140, 536, 270)
    text(x + 24, 178, f"EXAMPLE {index + 1}", size=17, weight="bold", color="#52627A")
    text(x + 48, 212, "Input", size=18)
    text(x + 338, 212, "Output", size=18)
    grid(x + 48, 232, values)
    arrow(x + 222, 301, 65)
    grid(x + 338, 232, output_for(values))

panel(48, 438, 1104, 252)
text(72, 476, "NEW PUZZLE", size=17, weight="bold", color="#52627A")
text(96, 513, "New input", size=18)
text(958, 513, "Expected output", size=18, anchor="middle")
query = [[2, 2, 0], [0, 2, 0], [0, 2, 2]]
grid(96, 533, query, cell=42)
arrow(254, 596, 48)
text(342, 571, "The shared rule", size=22, weight="bold")
text(342, 606, "Turn red cells blue.", size=23)
text(342, 639, "Keep the shape and positions.", size=20, color="#52627A")
arrow(800, 596, 48)
grid(896, 533, output_for(query), cell=42)
text(48, 732, "Illustrative puzzle", size=17, color="#52627A")
text(1152, 732, "Color IDs: black = 0   red = 2   blue = 1", size=17, color="#52627A", anchor="end")
parts.extend(['</g>', '</svg>'])

destination = ROOT / "docs" / "images" / "arc-grid-example.svg"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text("\n".join(parts) + "\n")
print(destination)
