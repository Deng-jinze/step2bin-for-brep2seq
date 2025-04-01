from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopoDS import TopoDS_Face


def compute_face_area(face:TopoDS_Face):
    props = GProp_GProps()
    tolerance=1e-6
    relative=True
    brepgprop_SurfaceProperties(face, props)
    area = props.Mass()
    return area

