#!/usr/bin/env python3
"""
Generate nested DXF layouts for CNC milling.
Extracts shelves and brackets, and packs them tightly on 48x96 sheets using straight-side welding and row-based interlocking.
"""

import sys
from pathlib import Path
import json
import re
import numpy as np
import ezdxf
from ezdxf import units
from shapely.geometry import Polygon, LineString
import shapely.affinity as affinity

script_dir = Path(__file__).parent
sys.path.append(str(script_dir))
import generate_shelves_with_brackets as gswb

SHEET_WIDTH = 96.0  # X-axis
SHEET_HEIGHT = 48.0 # Y-axis
MARGIN = 0.5        # Minimum distance between parts
EDGE_MARGIN = 0.0   # Configured to 0" (flush with boundary)
PANTRY_WIDTH = 48.0 # Needed for bracket logic mirroring

class ShelfGroup:
    def __init__(self, name, shelf_type, shelf_pts, bracket_lines, bracket_cuts):
        self.name = name
        self.shelf_type = shelf_type
        if not np.array_equal(shelf_pts[0], shelf_pts[-1]):
            shelf_pts = np.vstack((shelf_pts, shelf_pts[0]))
        self.poly = Polygon(shelf_pts)
        self.bracket_polys = []
        for pts in bracket_lines:
            if not np.array_equal(pts[0], pts[-1]):
                pts = np.vstack((pts, pts[0]))
            self.bracket_polys.append(Polygon(pts))
        self.bracket_cuts = [LineString(pts) for pts in bracket_cuts]
        
    def translate(self, dx, dy):
        self.poly = affinity.translate(self.poly, xoff=dx, yoff=dy)
        self.bracket_polys = [affinity.translate(b, xoff=dx, yoff=dy) for b in self.bracket_polys]
        self.bracket_cuts = [affinity.translate(c, xoff=dx, yoff=dy) for c in self.bracket_cuts]

    def rotate(self, angle, origin=(0, 0)):
        self.poly = affinity.rotate(self.poly, angle, origin=origin)
        self.bracket_polys = [affinity.rotate(b, angle, origin=origin) for b in self.bracket_polys]
        self.bracket_cuts = [affinity.rotate(c, angle, origin=origin) for c in self.bracket_cuts]
        
    def bounds(self):
        return self.poly.bounds
        
    def align_to_origin(self):
        minx, miny, _, _ = self.bounds()
        self.translate(-minx, -miny)
        
    def distance(self, other):
        if hasattr(other, 'poly'):
            return self.poly.distance(other.poly)
        return other.distance(self)

    def get_dxf_points(self):
        return list(self.poly.exterior.coords)

class ShelfPair:
    def __init__(self, s1, s2, is_pre_aligned=False):
        self.s1 = s1
        self.s2 = s2
        
    def bounds(self):
        b1 = self.s1.bounds()
        if not self.s2: return b1
        b2 = self.s2.bounds()
        return (min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3]))

    def distance(self, group):
        d1 = self.s1.distance(group)
        if not self.s2: return d1
        return min(d1, self.s2.distance(group))
        
    def translate(self, dx, dy):
        self.s1.translate(dx, dy)
        if self.s2:
            self.s2.translate(dx, dy)
            
    def align_to_origin(self):
        minx, miny, _, _ = self.bounds()
        self.translate(-minx, -miny)
        
    def get_groups(self):
        if self.s2: return [self.s1, self.s2]
        return [self.s1]

class Sheet:
    def __init__(self, id_num):
        self.id = id_num
        self.width = SHEET_WIDTH
        self.height = SHEET_HEIGHT
        self.placed_groups = []
        
    def add(self, group):
        self.placed_groups.append(group)
        
    def get_area_efficiency(self):
        total_parts_area = sum([g.poly.area for g in self.get_all_shelves()])
        sheet_area = self.width * self.height
        return total_parts_area / sheet_area
        
    def get_all_shelves(self):
        out = []
        for g in self.placed_groups:
            if hasattr(g, 'get_groups'):
                out.extend(g.get_groups())
            else:
                out.append(g)
        return out
        
    def center_vertically(self):
        if not self.placed_groups: return
        miny = min([g.bounds()[1] for g in self.placed_groups])
        maxy = max([g.bounds()[3] for g in self.placed_groups])
        empty_space = self.height - (maxy - miny)
        shift_y = empty_space / 2.0 - miny
        for g in self.placed_groups:
            g.translate(0, shift_y)

def export_sheet_to_dxf(sheet, output_dir):
    doc = ezdxf.new('R2010')
    doc.units = units.IN
    msp = doc.modelspace()
    
    msp.add_lwpolyline(
        [(0,0), (sheet.width, 0), (sheet.width, sheet.height), (0, sheet.height), (0,0)],
        dxfattribs={'layer': 'PLYWOOD_BOUNDARY', 'color': 7}
    )
    
    groups_to_draw = sheet.get_all_shelves()
            
    for sg in groups_to_draw:
        pts = sg.get_dxf_points()
        msp.add_lwpolyline(pts, close=True, dxfattribs={'layer': f'SHELF_{sg.shelf_type}', 'color': 3})
        
        for bp in sg.bracket_polys:
            msp.add_lwpolyline(list(bp.exterior.coords), close=True, dxfattribs={'layer': 'BRACKET_OUTLINE', 'color': 6})
            
        for bc in sg.bracket_cuts:
            msp.add_line(list(bc.coords)[0], list(bc.coords)[1], dxfattribs={'layer': 'BRACKET_CUT_LINE', 'color': 4})
            
    dxf_path = output_dir / f'sheet_{sheet.id}_layouts_with_brackets.dxf'
    doc.saveas(dxf_path)
    eff = sheet.get_area_efficiency() * 100
    print(f"Exported {dxf_path} - Packing Efficiency: {eff:.1f}% (Unfilled Area: {100-eff:.1f}%)")

def load_shelves(project_dir=None):
    if project_dir is None:
        project_dir = script_dir.parent
    output_dir = project_dir / 'output'
    config_path = project_dir / 'configs' / 'stud_positions.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    shelf_files = list(output_dir.glob('shelf_*.dxf'))
    loaded_shelves = []

    for f in shelf_files:
        if '_exact' in f.stem or 'all_' in f.stem or 'sheet_' in f.stem:
            continue
        match = re.match(r'shelf_([LRB])(\d+)', f.stem)
        if match:
            shelf_type = match.group(1)
            height = int(match.group(2))
            
            shelf_doc = ezdxf.readfile(f)
            shelf_pts = []
            for entity in shelf_doc.modelspace():
                if entity.dxftype() == 'LWPOLYLINE':
                    shelf_pts = np.array(list(entity.get_points()))
                    break
                    
            if len(shelf_pts) == 0: continue
            shelf_pts = shelf_pts[:, :2]

            bracket_lines = []
            bracket_cuts = []
            
            if shelf_type == 'L':
                wall_side = 'left'
                wall_x = config['left_wall']['wall_x']
                for stud_y in config['left_wall']['stud_centers_y']:
                    outline, cut_line = gswb.place_bracket(wall_x, stud_y, 'left', config)
                    outline = gswb.mirror_x_pts(outline, PANTRY_WIDTH)
                    cut_line = gswb.mirror_x_pts(cut_line, PANTRY_WIDTH)
                    bracket_lines.append(outline)
                    bracket_cuts.append(cut_line)
            elif shelf_type == 'R':
                wall_side = 'right'
                wall_x = config['right_wall']['wall_x']
                for stud_y in config['right_wall']['stud_centers_y']:
                    outline, cut_line = gswb.place_bracket(wall_x, stud_y, 'right', config)
                    outline = gswb.mirror_x_pts(outline, PANTRY_WIDTH)
                    cut_line = gswb.mirror_x_pts(cut_line, PANTRY_WIDTH)
                    bracket_lines.append(outline)
                    bracket_cuts.append(cut_line)
            elif shelf_type == 'B':
                wall_side = 'back'
                wall_y = config['back_wall']['wall_y']
                for stud_x in config['back_wall']['stud_centers_x']:
                    outline, cut_line = gswb.place_bracket(stud_x, wall_y, 'back', config)
                    outline = gswb.mirror_x_pts(outline, PANTRY_WIDTH)
                    cut_line = gswb.mirror_x_pts(cut_line, PANTRY_WIDTH)
                    bracket_lines.append(outline)
                    bracket_cuts.append(cut_line)
                    
                if config['back_shelf_side_brackets']['enabled']:
                    side_y = config['back_shelf_side_brackets']['y_position']
                    outline, cut_line = gswb.place_bracket(config['left_wall']['wall_x'], side_y, 'back_left_support', config)
                    outline, cut_line = gswb.mirror_x_pts(outline, PANTRY_WIDTH), gswb.mirror_x_pts(cut_line, PANTRY_WIDTH)
                    bracket_lines.append(outline)
                    bracket_cuts.append(cut_line)
                    
                    outline, cut_line = gswb.place_bracket(config['right_wall']['wall_x'], side_y, 'back_right_support', config)
                    outline, cut_line = gswb.mirror_x_pts(outline, PANTRY_WIDTH), gswb.mirror_x_pts(cut_line, PANTRY_WIDTH)
                    bracket_lines.append(outline)
                    bracket_cuts.append(cut_line)
                    
                if config.get('back_shelf_corner_brackets', {}).get('enabled', False):
                    corner_cfg = config['back_shelf_corner_brackets']
                    lc = corner_cfg['left_corner']
                    outline, cut_line = gswb.place_bracket(lc['x'], lc['y'], 'corner_left', config)
                    outline, cut_line = gswb.mirror_x_pts(outline, PANTRY_WIDTH), gswb.mirror_x_pts(cut_line, PANTRY_WIDTH)
                    bracket_lines.append(outline)
                    bracket_cuts.append(cut_line)
                    
                    rc = corner_cfg['right_corner']
                    outline, cut_line = gswb.place_bracket(rc['x'], rc['y'], 'corner_right', config)
                    outline, cut_line = gswb.mirror_x_pts(outline, PANTRY_WIDTH), gswb.mirror_x_pts(cut_line, PANTRY_WIDTH)
                    bracket_lines.append(outline)
                    bracket_cuts.append(cut_line)

            sg = ShelfGroup(f.stem, shelf_type, shelf_pts, bracket_lines, bracket_cuts)
            loaded_shelves.append((height, sg))
            
    loaded_shelves.sort(key=lambda x: (x[1].shelf_type, x[0]))
    return [s for _, s in loaded_shelves]

def standardize_shelf(s):
    if s.shelf_type == 'L':
        s.align_to_origin()
        s.rotate(90)
        s.align_to_origin()
    elif s.shelf_type == 'R':
        s.align_to_origin()
        s.rotate(-90)
        s.align_to_origin()

def create_tight_pair(s1, s2):
    s2.rotate(180)
    s1.align_to_origin()
    s1_cx = (s1.bounds()[0] + s1.bounds()[2]) / 2.0
    s2_cx = (s2.bounds()[0] + s2.bounds()[2]) / 2.0
    s2.translate(s1_cx - s2_cx, 0)
    
    # We want to WELD the straight side of s1 to the straight side of s2.
    # s1 straight wall is at Y = s1.bounds()[1] (miny)
    # s2 straight wall is at Y = s2.bounds()[3] (maxy)
    # Note: s1 bounds includes brackets that might protrude, so miny is not strictly the wall.
    # But for L and R shelves, we establish that align_to_origin places the bracket at miny and wall slightly above it.
    # We weld them so their absolute minimum/maximum Y's mirror.
    # To weld the standard straight walls (ignoring bracket offsets), we can just search!
    # A single Shapely drop distance check is much safer to prevent intersecting regimes!
    
    # Place s2 completely BELOW s1
    s2.translate(0, s1.bounds()[1] - s2.bounds()[3] - 10.0)
    # Now s2's top (wall) is 10 inches below s1's bottom (wall).
    # Drop s2 UPWARDS (towards positive Y) until distance is exactly 0!
    # We want a literal WELD, margin=0!
    step = 1.0
    while s2.distance(s1) > 0.001 and s2.bounds()[3] < s1.bounds()[1]:
        s2.translate(0, step)
    s2.translate(0, -step)
    
    while s2.distance(s1) > 0.001:
        s2.translate(0, 0.1)
    s2.translate(0, -0.05) # Weld perfect
    
    return ShelfPair(s1, s2, is_pre_aligned=True)

def main():
    print("Loading shelves...")
    shelves = load_shelves()
    
    back_shelves = [s for s in shelves if s.shelf_type == 'B']
    left_shelves = [s for s in shelves if s.shelf_type == 'L']
    right_shelves = [s for s in shelves if s.shelf_type == 'R']
    print(f"Loaded {len(back_shelves)} back shelves, {len(left_shelves)} left shelves, {len(right_shelves)} right shelves.")

    sheets = []
    
    if back_shelves:
        sheet1 = Sheet(1)
        s1 = back_shelves[0]
        s1.align_to_origin()
        s1.rotate(90)
        s1.align_to_origin()
        sheet1.add(s1)
            
        if len(back_shelves) > 1:
            s2 = back_shelves[1]
            s2.align_to_origin()
            s2.rotate(-90)
            s2.align_to_origin()
            s2.translate(s1.bounds()[2] - 5.0, 0)
            while s2.distance(s1) < MARGIN or s2.bounds()[0] < s1.bounds()[2] + MARGIN:
                s2.translate(0.1, 0)
            sheet1.add(s2)
            
        if len(back_shelves) > 2:
            s3 = back_shelves[2]
            s3.align_to_origin()
            s3.rotate(90)
            s3.align_to_origin()
            s3.translate(s2.bounds()[2] + MARGIN, 0) 
            sheet1.add(s3)
            
        if len(back_shelves) > 3:
            s4 = back_shelves[3]
            s4.align_to_origin()
            s4.rotate(-90)
            s4.align_to_origin()
            s4.translate(sheet1.width - s4.bounds()[2], 0)
            while s4.distance(s3) < MARGIN and s4.bounds()[0] > s3.bounds()[2] + MARGIN:
                s4.translate(0.1, 0)
                if s4.bounds()[2] > sheet1.width: break
            if s4.bounds()[2] <= sheet1.width:
                sheet1.add(s4)
                
        sheet1.center_vertically()
        sheets.append(sheet1)

    remaining = left_shelves + right_shelves
    for s in remaining: standardize_shelf(s)
    
    pairs = []
    print("Forming pairs...")
    while len(remaining) >= 2:
        s1 = remaining.pop(0)
        s0 = remaining.pop(0)
        pair = create_tight_pair(s0, s1)
        pairs.append(pair)
        
    if remaining:
        pairs.append(ShelfPair(remaining.pop(0), None, is_pre_aligned=True))

    print("Packing rows...")
    current_sheet = Sheet(len(sheets) + 1)
    sheets.append(current_sheet)
    
    rows = []
    current_row = []
    current_x = 0
    current_y = 0
    
    for pair in pairs:
        pair.align_to_origin()
        pw = pair.bounds()[2]
        
        if current_x + pw > SHEET_WIDTH and current_row:
            rows.append(current_row)
            
            if len(rows) > 1:
                drop_step = 1.0
                iters_1 = 0
                while True:
                    iters_1 += 1
                    for p in current_row: p.translate(0, -drop_step)
                    min_dist = float('inf')
                    for p in current_row:
                        p_b = p.bounds()
                        for existing in current_sheet.placed_groups:
                            if existing not in current_row:
                                e_b = existing.bounds()
                                if p_b[0] > e_b[2]+MARGIN or p_b[2] < e_b[0]-MARGIN or p_b[1] > e_b[3]+MARGIN or p_b[3] < e_b[1]-MARGIN:
                                    continue
                                d = p.distance(existing)
                                if d < min_dist: min_dist = d
                    if min_dist <= MARGIN or iters_1 > 200:
                        for p in current_row: p.translate(0, drop_step)
                        break
                        
                iters_2 = 0
                while True:
                    iters_2 += 1
                    for p in current_row: p.translate(0, -0.1)
                    min_dist = float('inf')
                    for p in current_row:
                        p_b = p.bounds()
                        for existing in current_sheet.placed_groups:
                            if existing not in current_row:
                                e_b = existing.bounds()
                                if p_b[0] > e_b[2]+MARGIN or p_b[2] < e_b[0]-MARGIN or p_b[1] > e_b[3]+MARGIN or p_b[3] < e_b[1]-MARGIN:
                                    continue
                                d = p.distance(existing)
                                if d < min_dist: min_dist = d
                    if min_dist < MARGIN or min_dist == float('inf') or iters_2 > 200:
                        if min_dist < MARGIN:
                            for p in current_row: p.translate(0, 0.1)
                        if min_dist == float('inf'):
                            for p in current_row: p.translate(0, 0.1)
                        break
            
            row_maxy = max([p.bounds()[3] for p in current_row])
            if row_maxy > SHEET_HEIGHT:
                for p in current_row:
                    current_sheet.placed_groups.remove(p)
                current_sheet.center_vertically()
                current_sheet = Sheet(len(sheets) + 1)
                sheets.append(current_sheet)
                rows = [current_row]
                
                row_miny = min([p.bounds()[1] for p in current_row])
                shift_y = -row_miny
                for p in current_row: 
                    p.translate(0, shift_y)
                    current_sheet.add(p)
            
            row_maxy = max([p.bounds()[3] for p in current_row])
            if row_maxy > SHEET_HEIGHT: current_y = 0
            else: current_y = row_maxy + MARGIN + 1.0 # Give it a bit of room to drop
            current_x = 0
            current_row = []
            
        pair.align_to_origin()
        pair.translate(current_x, current_y)
        
        if current_row:
            prev_pair = current_row[-1]
            while pair.bounds()[0] > prev_pair.bounds()[2] and pair.distance(prev_pair) > MARGIN:
                pair.translate(-0.1, 0)
            if pair.distance(prev_pair) < MARGIN:
                pair.translate(0.1, 0)
                
        current_sheet.add(pair)
        current_row.append(pair)
        current_x = pair.bounds()[2] + MARGIN

    if current_row:
        rows.append(current_row)
        if len(rows) > 1:
            drop_step = 1.0
            iters_1 = 0
            while True:
                iters_1 += 1
                for p in current_row: p.translate(0, -drop_step)
                min_dist = float('inf')
                for p in current_row:
                    p_b = p.bounds()
                    for existing in current_sheet.placed_groups:
                        if existing not in current_row:
                            e_b = existing.bounds()
                            if p_b[0] > e_b[2]+MARGIN or p_b[2] < e_b[0]-MARGIN or p_b[1] > e_b[3]+MARGIN or p_b[3] < e_b[1]-MARGIN:
                                continue
                            d = p.distance(existing)
                            if d < min_dist: min_dist = d
                if min_dist <= MARGIN or iters_1 > 200:
                    for p in current_row: p.translate(0, drop_step)
                    break
            iters_2 = 0
            while True:
                iters_2 += 1
                for p in current_row: p.translate(0, -0.1)
                min_dist = float('inf')
                for p in current_row:
                    p_b = p.bounds()
                    for existing in current_sheet.placed_groups:
                        if existing not in current_row:
                            e_b = existing.bounds()
                            if p_b[0] > e_b[2]+MARGIN or p_b[2] < e_b[0]-MARGIN or p_b[1] > e_b[3]+MARGIN or p_b[3] < e_b[1]-MARGIN:
                                continue
                            d = p.distance(existing)
                            if d < min_dist: min_dist = d
                if min_dist < MARGIN or min_dist == float('inf') or iters_2 > 100:
                    if min_dist < MARGIN:
                        for p in current_row: p.translate(0, 0.1)
                    if min_dist == float('inf'):
                        for p in current_row: p.translate(0, 0.1)
                    break
                    
        row_maxy = max([p.bounds()[3] for p in current_row])
        if row_maxy > SHEET_HEIGHT:
            for p in current_row: current_sheet.placed_groups.remove(p)
            current_sheet.center_vertically()
            current_sheet = Sheet(len(sheets) + 1)
            sheets.append(current_sheet)
            row_miny = min([p.bounds()[1] for p in current_row])
            shift_y = -row_miny
            for p in current_row: 
                p.translate(0, shift_y)
                current_sheet.add(p)
                
    current_sheet.center_vertically()

    print(f"Generated {len(sheets)} sheets.")
    for sh in sheets:
        export_sheet_to_dxf(sh, script_dir.parent / 'output')
        groups = sh.get_all_shelves()
        print(f"  Sheet {sh.id} contains {len(groups)} shelves.")

if __name__ == '__main__':
    main()
