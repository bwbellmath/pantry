from pathlib import Path
import sys
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))
from apply_manual_layout_from_svg import extract_sheet_layout
from scripts.generate_nested_layouts import load_shelves

dxf_shelves = load_shelves()
available = list(dxf_shelves)

print("Matching Sheet 2 FIRST:")
sh2 = extract_sheet_layout('output/sheet_2.svg', available)
for s in sh2:
    print(f"{s.name} at {s.poly.centroid.x:.1f}, {s.poly.centroid.y:.1f}")

print("\nMatching Sheet 1 SECOND:")
sh1 = extract_sheet_layout('output/sheet_1.svg', available)
for s in sh1:
    print(f"{s.name} at {s.poly.centroid.x:.1f}, {s.poly.centroid.y:.1f}")
