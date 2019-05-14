restart;
with(LinearAlgebra):
with(VectorCalculus):
with(plottools):
BasisFormat(false):

MNAC := proc(w1,w2,x,t,epsilon)
	local z1, z2, L, i;
	z1 := w1.x;
	z2 := exp~(w2.log~(abs(z1) +~ epsilon));

	L := (z2 -~ t)^~2;
	return add(L[i],i=1..numelems(L));
end proc:

MNACalt := proc(w1,w2,x,t)
	local z1, z2, zhat2, w2ones, L, i;

	z1 := w1.x;
	zhat2 := log~(Transpose(w2) *~ z1 +~ 1 -~ w2);
	z2 := exp~(zhat2 . Matrix(ColumnDimension(w2),1,1));
	L := (z2 -~ t)^~2;
	return add(L[i],i=1..numelems(L));
end proc:

P := plot3d(
	MNAC(<<w[1]|w[1]|0|0>, <w[1]|w[1]|w[1]|w[1]>>, <<w[2]|w[2]>>, <1,1.2,1.8,2>, 13.2, 1E-7),
	w[1] = -1 .. 1, w[2] = -1 .. 1, view = [-1..1, -1..1, 0..500],
	axes=boxed, orientation=[-45, 39, 0]):
Export(FileTools:-JoinPath([currentdir(), "..", "paper", "graphics", "nac-mul-eps-1em7.jpeg"]), P);

P := plot3d(
	MNAC(<<w[1]|w[1]|0|0>, <w[1]|w[1]|w[1]|w[1]>>, <<w[2]|w[2]>>, <1,1.2,1.8,2>, 13.2, 1E-1),
	w[1] = -1 .. 1, w[2] = -1 .. 1, view = [-1..1, -1..1, 0..500],
	axes=boxed, orientation=[-45, 39, 0]):
Export(FileTools:-JoinPath([currentdir(), "..", "paper", "graphics", "nac-mul-eps-1em1.jpeg"]), P);

P := plot3d(
	MNAC(<<w[1]|w[1]|0|0>, <w[1]|w[1]|w[1]|w[1]>>, <<w[2]|w[2]>>, <1,1.2,1.8,2>, 13.2, 1),
	w[1] = -1 .. 1, w[2] = -1 .. 1, view = [-1..1, -1..1, 0..500],
	axes=boxed, orientation=[-45, 39, 0]):
Export(FileTools:-JoinPath([currentdir(), "..", "paper", "graphics", "nac-mul-eps-1.jpeg"]), P);

P := plot3d(
	MNACalt(<<w[1]|w[1]|0|0>, <w[1]|w[1]|w[1]|w[1]>>, <w[2]|w[2]>, <1,1.2,1.8,2>, 13.2),
	w[1] = -1 .. 1, w[2] = 0 .. 1, view = [-1..1, 0..1, 0..500],
	axes=boxed, orientation=[-45, 39, 0]):
Export(FileTools:-JoinPath([currentdir(), "..", "paper", "graphics", "nac-mul-nmu.jpeg"]), P);
