from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Torus, GeomAbs_Cone, GeomAbs_Sphere, \
    GeomAbs_BezierSurface, GeomAbs_BSplineSurface, GeomAbs_SurfaceOfRevolution, GeomAbs_SurfaceOfExtrusion, \
    GeomAbs_OffsetSurface, GeomAbs_OtherSurface
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.TopoDS import TopoDS_Face


def recognise_face_type(face: TopoDS_Face) -> int:
    """Get surface type of B-Rep face"""
    # 使用 BRepAdaptor 来获取面的表面类型
    surf = BRepAdaptor_Surface(face, True)
    surf_type = surf.GetType()
    a = 0

    if surf_type == GeomAbs_Plane:
        a = 1
    elif surf_type == GeomAbs_Cylinder:
        a = 2
    elif surf_type == GeomAbs_Torus:
        a = 3
    elif surf_type == GeomAbs_Sphere:
        a = 4
    elif surf_type == GeomAbs_Cone:
        a = 5
    elif surf_type == GeomAbs_BezierSurface:
        a = 6
    elif surf_type == GeomAbs_BSplineSurface:
        a = 7
    elif surf_type == GeomAbs_SurfaceOfRevolution:
        a = 8
    elif surf_type == GeomAbs_OffsetSurface:
        a = 9
    elif surf_type == GeomAbs_SurfaceOfExtrusion:
        a = 10
    elif surf_type == GeomAbs_OtherSurface:
        a = 11

    return a

