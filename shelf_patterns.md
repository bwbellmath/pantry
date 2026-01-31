Ok, I want to get back to the json configuration to a much more
significant degree. Every shelf we draw should be completely
configured via JSON and not any separate information -- including the
base and intermediate shelves. 

Each shelf will be described as a series of points and lines
(straight lines, sinusoids, and the various types of circle intersections with
sinusoids/straight lines we have defined so far) which
connect to form a closed sketch. The closed sketches will be placed in
the pantry by anchoring their points to coordinates within the
pantry. 

This sets up an issue since the main shelves have a shelf break which
is determined by the intersection of two sinusoids. That means all 3
shelves depend on that base geometry. Therefore, for each level, we
will determine the base geometry (any sinusoids, their intersections,
and solved for arcs, and arc-midpoints) *first* and then describe the
shelves in terms of that base geometry. This will be the same on the
intermediate levels with only a single shelf at each height except
that the base geometry is only the single sinusoid curve. 

Then we can modify the base geometry as we like. That also means that
the 3 sinusoids on the main shelf levels that have a right left and
back shelf will have 3 sinusoids and the offsets for those 3 will be
stored in the "base_geometry" json element of that level. The levels
will just directly correspond to heights. 21 inches (referenced from
the floor of the pantry to the top of the shelf) will be "level":
21.0. 

So our base shelves currently consist of the following construction: 
base_geometry: right sinusoid(period, amplitude, offset) line (pantry_width-right_shelf_offset,
0) through (pantry_width-right_shelf_offset, 49); left
sinusoid(period, amplitude, offset)
(left_shelf_offset, 0) through (left_shelf_offset, 49)
back
sinusoid(period, amplitude, offset)
(back_shelf_offset, 0) through (back_shelf_offset, 48)
back_right_intersection: right sinusoid intersect left sinusoid
back left intersection: left sinusoid intersect left sinusoid
back right arc : slope matching circle solve (r),
back_right_intersection, right sinusoid, back sinusoid
back left arc: slope matching circle solve (r),
back_left_intersection, left sinusoid, back sinusoid
back right arc midpoint: arc point we picked to cut the right shelf
from the back shelf
back left arc midpoint: "" left



(Where possible replace the specified dimensions with the variable
reference to those dimensions -- nothing should be hard-coded and
dependencies should be concentrated to the base dimensions specified
at the very beginning -- you get the idea. 

Right shelf: (48,0), straight line, (intersection with sinusoid along door line
(-1,0)), sinusoid line, (back right arc intersection with right
sinusoid), back right arc up to back right arc midpoint, (48,
back_right_arc_midpoint.y), (48,0) (since we've made it back to the
starting point, we have now specified a closed curve). 

Same for the other shelves. 

To create the door jam radius geometry, we're going to modify the
above list by adding a few points: 

New right shelf: (48,0), straight line, (right wall to door jam radius
start = 2 9/16 on right side), arc (circle centered at (48-(2+ 9/16),
-0.75) with radius 0.75") from (90 degrees) to (180 degrees), arc
point 2 (for the right shelves, (48-(2+9/16) - 0.75)), straight line,
(48-(2+9/16) - 0.75 - 5/8, -0.75), circle arc intersecting last point
with radius 5" using our bracketing algorithm so that the circle is
tangent to the right sinusoid), (anchored bracketing circle arc
intersection with sinusoid), sinusoid line, (back right arc intersection with right
sinusoid), back right arc up to back right arc midpoint, (48,
back_right_arc_midpoint.y), straight line (48,0) (since we've made it back to the
starting point, we have now specified a closed curve). 

So on and so forth. 

Can you take a stab at writing a JSON file for the currently
configured shelves formalizing what I described above? Call the new
json file shelf_level_patterns.json in `configs`

that will compose the pantry as a
clockwise ordered list of the points and lines that will make up each
shelf. We will need to include ALL shelves by height level. Each shelf
can also have additional features (e.g. sketches for partial-depth
cutouts of mounting hardware, sketches with words or decorative
features, etc...). I am attaching the exemplar config below: 

For the main shelves, some of the geometry for a shelf will depend on
intersections with other shelves. 

{
  "config_version": "0005",
  "pantry": {
    "width": 48.0,
    "depth": 49.0,
    "height": 105.0,
    "door_clearance_east": 6.0,
    "door_clearance_west": 4.0
  },
  "design_params": {
    "sinusoid_period": 24.0,
    "sinusoid_amplitude": 1.0,
    "shelf_base_depth_east": 9.0,
    "shelf_base_depth_south": 19.0,
    "shelf_base_depth_west": 5.0,
    "shelf_thickness": 1.0,
    "interior_corner_radius": 6.0,
    "door_corner_radius": 3.0,
    "door_extension": 0.75,
    "door_smoothing_tangent_x_east": 7.5625,
    "door_smoothing_tangent_x_west": 3.75,
    "door_notch_radius": 0.75,
    "door_notch_intersection_x_east": 6.3125,
    "door_notch_intersection_x_west": 2.5625
  },
  "shelves": [
    {
      "level": 21,
      "height": 21.0,
      "wall": "E",
      "extent_start": 0.0,
      "extent_end": 49.0,
      "sinusoid_offset": 2.353304971691044,

      "corner_points_solved": []
    },
    {
      "level": 0,
      "height": 21.0,
      "wall": "S",
      "extent_start": 0.0,
      "extent_end": 48.0,
      "sinusoid_offset": 5.9735141613602165,
      "corner_points_solved": []
    },
    {
      "level": 0,
      "height": 21.0,
      "wall": "W",
      "extent_start": 0.0,
      "extent_end": 49.0,
      "sinusoid_offset": 4.599253580133889,
      "corner_points_solved": []
    },
    {
      "level": 1,
      "height": 42.0,
      "wall": "E",
      "extent_start": 0.0,
      "extent_end": 49.0,
      "sinusoid_offset": 3.761482191925223,
      "corner_points_solved": []
    },
    {
      "level": 1,
      "height": 42.0,
      "wall": "S",
      "extent_start": 0.0,
      "extent_end": 48.0,
      "sinusoid_offset": 0.980294029274052,
      "corner_points_solved": []
    },
    {
      "level": 1,
      "height": 42.0,
      "wall": "W",
      "extent_start": 0.0,
      "extent_end": 49.0,
      "sinusoid_offset": 0.9801424781769557,
      "corner_points_solved": []
    },
    {
      "level": 2,
      "height": 63.0,
      "wall": "E",
      "extent_start": 0.0,
      "extent_end": 49.0,
      "sinusoid_offset": 0.3649500985631483,
      "corner_points_solved": []
    },
    {
      "level": 2,
      "height": 63.0,
      "wall": "S",
      "extent_start": 0.0,
      "extent_end": 48.0,
      "sinusoid_offset": 5.442345232562516,
      "corner_points_solved": []
    },
    {
      "level": 2,
      "height": 63.0,
      "wall": "W",
      "extent_start": 0.0,
      "extent_end": 49.0,
      "sinusoid_offset": 3.776917009710014,
      "corner_points_solved": []
    },
    {
      "level": 3,
      "height": 84.0,
      "wall": "E",
      "extent_start": 0.0,
      "extent_end": 49.0,
      "sinusoid_offset": 4.448951217224888,
      "corner_points_solved": []
    },
    {
      "level": 3,
      "height": 84.0,
      "wall": "S",
      "extent_start": 0.0,
      "extent_end": 48.0,
      "sinusoid_offset": 0.12933619211510794,
      "corner_points_solved": []
    },
    {
      "level": 3,
      "height": 84.0,
      "wall": "W",
      "extent_start": 0.0,
      "extent_end": 49.0,
      "sinusoid_offset": 6.094123332392967,
      "corner_points_solved": []
    }
  ]
}
