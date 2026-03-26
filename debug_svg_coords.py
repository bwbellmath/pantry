from pathlib import Path
import sys
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))
from apply_manual_layout_from_svg import extract_sheet_layout
from scripts.generate_nested_layouts import load_shelves

dxf_shelves = load_shelves()
# Just print the mapped coordinates!
placed = extract_sheet_layout('output/sheet_2.svg', list(dxf_shelves))
for s in placed:
    print(f"{s.name}: Centered at {s.poly.centroid.x:.2f}, {s.poly.centroid.y:.2f} bounds: {s.bounds()}")

