with(LinearAlgebra):
with(VectorCalculus):
with(plottools);

NALU := proc(w1,w2,ghat1,ghat2,x,t,epsilon)
	local g1, g2, z1, z2, L, i;
	g1 := 1/~(1 +~ exp~(-ghat1));
	z1 := g1 *~ (w1.x) +~ (1 -~ g1) *~ exp~(w1.log~(abs(x) +~ epsilon));

	g2 := 1/~(1 +~ exp~(-ghat2));
	z2 := g2 *~ (w2.z1) +~ (1 -~ g2) *~ exp~(w2.log~(abs(z1) +~ epsilon));

	L := (z2 -~ t)^~2;
	return add(L[i],i=1..numelems(L));
end proc:

NALUsafe := proc(w1,w2,ghat1,ghat2,x,t)
	local g1, g2, z1, z2, L, i;
	g1 := 1/~(1 +~ exp~(-ghat1));
	z1 := g1 *~ (w1.x) +~ (1 -~ g1) *~ exp~(w1.log~(abs(x -~ 1) +~ 1));

	g2 := 1/~(1 +~ exp~(-ghat2));
	z2 := g2 *~ (w2.z1) +~ (1 -~ g2) *~ exp~(w2.log~(abs(z1 -~ 1) +~ 1));

	L := (z2 -~ t)^~2;
	return add(L[i],i=1..numelems(L));
end proc:


NALU(<<w | w>, <w | w>>, <<w | w>>, <g, g>, <g>, <x[1], x[2]>, t, epsilon);

P := plot3d(
	NALU(<<w | w>, <w | w>>, <<w | w>>, <g, g>, <g>, <2, 2>, 8, 10^(-8)),
	w = -1.5..1.5, g = -3..3,
	view = [-1.5..1.5, -3..3, 0..300],
	axes=boxed):
P;

P := plot3d(
	NALU(<<w | w>, <w | w>>, <<w | w>>, <g, g>, <g>, <2, 2>, 16, 10^(-8)),
	w = -1.5..1.5, g = -3..3,
	view = [-1.5..1.5, -3..3, 0..300],
	axes=boxed):
P;

NALUsafe(<<w | w>, <w | w>>, <<w | w>>, <g, g>, <g>, <x[1], x[2]>, t, epsilon);

P := plot3d(
	NALUsafe(<<w | w>, <w | w>>, <<w | w>>, <g, g>, <g>, <2, 2>, 8),
	w = -1.5..1.5, g = -3..3,
	view = [-1.5..1.5, -3..3, 0..300],
	axes=boxed):
P;

P := plot3d(
	NALUsafe(<<w | w>, <w | w>>, <<w | w>>, <g, g>, <g>, <2, 2>, 16),
	w = -1.5..1.5, g = -3..3,
	view = [-1.5..1.5, -3..3, 0..300],
	axes=boxed):
P;

solveNALUsafe := proc(x, t)
	local i, v, w, g, eq, sol1, sol2, sols;

	eq := NALUsafe(<<w | w>, <w | w>>, <<w | w>>, <g, g>, <g>, x, t);
	sols := [];
     for v from -3 to 3 by 0.1 do
       sol1 := fsolve(eval(eq, g=v) = 0, w);
       sol2 := fsolve(eval(eq, g=v) = 0, w, avoid={{w = sol1}});
       sols := [op(sols), [sol1, v], [sol2, v]];
     end do;

	return sols;
end proc:

solveNALU := proc(x, t, epsilon)
	local i, v, w, g, eq, sols, sol1, sol2, sol3, sol4;

	eq := NALU(<<w | w>, <w | w>>, <<w | w>>, <g, g>, <g>, x, t, epsilon);
	sols := [];
     for v from -3 to 3 by 0.1 do
       sol1 := fsolve(eval(eq, g=v) = 0, w);
       sol2 := fsolve(eval(eq, g=v) = 0, w, avoid={{w = sol1}});
       if v < 0.9 and v >= 0 then
         sol3 := fsolve(eval(eq, g=v) = 0, w, avoid={{w = sol1}, {w = sol2}});
         sols := [op(sols), [sol1, v], [sol2, v], [sol3, v]];
       elif v < 0 then
         sol3 := fsolve(eval(eq, g=v) = 0, w, avoid={{w = sol1}, {w = sol2}});
         sol4 := fsolve(eval(eq, g=v) = 0, w, avoid={{w = sol1}, {w = sol2}, {w = sol3}});
         sols := [op(sols), [sol1, v], [sol2, v], [sol3, v], [sol4, v]];
       else
         sols := [op(sols), [sol1, v], [sol2, v]];
       end if;
     end do;

	return sols;
end proc:

NALUsols := solveNALU(<2, 2>, 8, 10^(-8));
P := plot(NALUsols, style = 'point', view = [-1.5..1.5, -3..3]):
P;

NALUsafesols := solveNALUsafe(<2, 2>, 8);
P := plot(NALUsafesols, style = 'point', view = [-1.5..1.5, -3..3]):
P;
