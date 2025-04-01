#原文如下
#measuring the angle between three random points on the surface of a three-dimensional model

import numpy as np

# Function to calculate the angle between three points
def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 防止数值溢出
    # 对 angle 进行格式化，保留四位小数
    formatted_angle = '{:.4f}'.format(angle)
    formatted_angle = float(formatted_angle)
    if formatted_angle>=0.01:
        formatted_angle=formatted_angle
    else:
        formatted_angle=0

    return formatted_angle


