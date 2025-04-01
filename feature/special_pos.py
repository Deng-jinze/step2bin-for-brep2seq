from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps

def ask_face_centroid(face):
    """Get centroid of B-Rep face."""
    mass_props = GProp_GProps()
    brepgprop.SurfaceProperties(face, mass_props)
    gPt = mass_props.CentreOfMass()

    return gPt.Coord()

