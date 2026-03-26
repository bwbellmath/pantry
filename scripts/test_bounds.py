import sys
from generate_nested_layouts import load_shelves

shelves = load_shelves()
for s in shelves:
    print(f"{s.name}: type={s.shelf_type}, bounds={s.bounds()}")
