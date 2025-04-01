from OCC.Core.TopoDS import TopoDS_Solid, TopoDS_Face, topods_Face
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SHAPE,TopAbs_ShapeEnum

def get_faces_from_solid(solid: TopoDS_Solid):
    """
    从三维实体中获取所有面对象

    参数:
        solid (TopoDS_Solid): 三维实体对象

    返回:
        list[TopoDS_Face]: 包含所有面对象的列表
    """
    faces = []
    # 创建一个拓扑探索器，用于遍历实体的面
    explorer = TopExp_Explorer(solid,TopAbs_FACE)
    # 遍历所有面
    while explorer.More():
        # 获取当前面
        face=topods_Face(explorer.Current())
        faces.append(face)
        # 移动到下一个面
        explorer.Next()
    return faces

