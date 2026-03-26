Next, and this is a biggie, we need to make a plan for nesting. I tried using Autodesk fusion to try their nesting
  function and they want me to make each shelf a separate component and then use their function to nest. That
  doesn't work for me since it is so manual. This means that we need to do the nesting ourselves. I want to add a
  new .dxf output where we will perform the nesting. This will be somewhat ad-hoc. First, we are going to make a
  4'x8' rectangle to represent the 4x8 plywood stock these will be cut out of.

Now, we will iteratively add things to this -- allowing straight lines
(e.g. the backs of shelves) to intersect the straight line edges of
the piece of plywood. For the 4 back shelves -- we're going to line
them up so that their long side is parallel to the y-axis (the 4'
span) of the plywood
stock and alternate which direction we rotate them to face. So the
first one is one end of the plywood, the next one has its curved
section facing the curved section of the first, the next one has its
straight back against (separated by 1/2") the second one, and the last
one has its back line against the other side of the plywood. 

Next, we're going to see if there is enough wiggle room to fit one of
the thin right-side intermediary shelves into one or both of the
gaps. Doing this optimally means picking one of the intermediate
shelves that has a sinusoidal period that lines up with the period of
the back shelves to allow a good fit. It may be better to slot in a
left intermediate shelf here depending on how much space we have. 

Actually doing the nesting is going to be constrained in that shelves
are only allowed to be rotated 90 degrees at a time and are allowed to
be not aligned with the vertical/horizontal axis. Next, we are going
to try to pick pairs of shelves whose sinusoidal periods are as close
to 1/2 out of phase as possible (check me here) so that the sin
functions will match up nicely -- we'll shift the shelves relative to
eachotehr by exactly the difference in the half-phase period. Finally,
we'll start laying out the pairs of shelves by adding one shelf --
with its straight back line against the plywood in one axis, then
adding its matched sinusoid shelf facing it, then the next paired
shelf first shelf. Each time we add a shelf, we're going to calculate
an outer bounding rectangle for this shelf, add it so that that
rectangle does not intersect any of the existing added shelves -- we
will translate down first and then right when we run out of down
space. let's say "down" the y-axis corresponds with the 4' span of the
plywood and "right" the x-axis long 8' span of the plywood. 

After adding a shelf so that its bounding rectangle does not
intersect, we'll measure the shortest line connecting this shelf to
the shelf(ves) adjacent to it and then use a bracketing algorithm to
try moving it closer until we can place it so that the minimum
distance to its adjacent shelf is 1/2" Since we are placing pairs of
shelves so that their sinusoids face eachother and (new rule?) we're
only adding the shelves horizontally oriented so that their long axis
is parallel to the x-axis, we don't need to worry about x-translation
for the second of a sinusoid matched pair. We just calculate so that
their sinusoids are exactly out of phase and only worry about y-shifts
until it is as close as possible. We repeat until we either run out of
shelves, or there is no space for a new bounding box and if that is
the case, we generate another piece of plywood and go again. Each
piece of plywood needs to be a separate .dxf file which includes the
outer boundary and all of the shelves nested onto it and all of the
brackets which have been moved around with the shelves. 

