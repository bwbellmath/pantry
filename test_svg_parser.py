import xml.etree.ElementTree as ET
import re

def parse_svg(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    # Strip namespace for easier finding
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
            
    paths = []
    # If the user exported grouped shapes instead of single paths, we might need a more generic approach.
    # Looking for 'path' or 'polygon'
    for path in root.findall('.//path'):
        paths.append(path.get('d'))
        
    print(f"File {filename} has {len(paths)} paths")
    
    # Just print the first path lightly to see what it looks like
    if paths:
        print(paths[0][:100])

parse_svg('output/sheet_1.svg')
parse_svg('output/sheet_2.svg')
