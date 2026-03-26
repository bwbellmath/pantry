import sys
import numpy as np
from shapely.geometry import Polygon
import ezdxf

def make_test_dxf():
    # Let's just create a test pair from the algorithm directly
    from scripts.generate_nested_layouts import load_shelves, standardize_shelf, create_tight_pair, export_sheet_to_dxf, Sheet
    shelves = load_shelves()
    left_shelves = [s for s in shelves if s.shelf_type == 'L']
    s1 = left_shelves[0]
    s2 = left_shelves[1]
    standardize_shelf(s1)
    standardize_shelf(s2)
    pair = create_tight_pair(s1, s2)
    
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    for s in [pair.s1, pair.s2]:
        msp.add_lwpolyline(s.get_dxf_points(), close=True, dxfattribs={'layer': 'SHELF_TEST', 'color': 3})
    doc.saveas('output/debug_pair.dxf')
    print("Debug pair saved to output/debug_pair.dxf")

if __name__ == '__main__':
    make_test_dxf()
