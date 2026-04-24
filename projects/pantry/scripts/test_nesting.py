import sys
from pathlib import Path

from generate_nested_layouts import load_shelves, ShelfGroup, Sheet, SHEET_WIDTH, SHEET_HEIGHT, MARGIN, EDGE_MARGIN

shelves = load_shelves()
back_shelves = [s for s in shelves if s.shelf_type == 'B']
sheet = Sheet()

print(f"Testing placement of {len(back_shelves)} back shelves.")

current_x = 0.0
for i, s in enumerate(back_shelves):
    s.align_to_origin()
    if i % 2 == 0:
        s.rotate(90)
        s.align_to_origin()
    else:
        s.rotate(-90)
        s.align_to_origin()
        
    s.translate(current_x, 0)
    
    # Brute force slide right
    while not sheet.check_fit(s) and s.bounds()[2] <= SHEET_WIDTH:
        s.translate(0.1, 0)
        
    if sheet.check_fit(s):
        sheet.add(s)
        print(f"Successfully placed {s.name} at {s.bounds()[:2]}")
        # update current_x slightly for the next one to start close
        current_x = s.bounds()[0]
    else:
        print(f"Failed to place {s.name}")
