import ezdxf
import numpy as np

def analyze_dxf(filename):
    print(f"--- Analyzing {filename} ---")
    doc = ezdxf.readfile(filename)
    msp = doc.modelspace()
    for e in msp:
        if e.dxftype() == 'LWPOLYLINE':
            pts = list(e.get_points())
            pts = [(p[0], p[1]) for p in pts]
            layer = e.dxf.layer
            if 'SHELF' in layer:
                miny = min([p[1] for p in pts])
                maxy = max([p[1] for p in pts])
                minx = min([p[0] for p in pts])
                maxx = max([p[0] for p in pts])
                print(f"Layer {layer}, X: {minx:.1f} to {maxx:.1f}, Y: {miny:.1f} to {maxy:.1f}")

analyze_dxf('output/sheet_2_layouts_with_brackets.dxf')
