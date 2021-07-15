import numpy as np
from shapely.geometry import Polygon
from shapely.geometry import Point
import math

class geometry:
    ###################### check collision between rect and circle #################################################
    def dist(self, p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        px = x2 - x1
        py = y2 - y1

        something = px * px + py * py

        u = ((x3 - x1) * px + (y3 - y1) * py) / (float(something)+1e-8)
        if u > 1:
            u = 1
        elif u < 0:
            u = 0

        x = x1 + u * px
        y = y1 + u * py

        dx = x - x3
        dy = y - y3

        dist = math.sqrt(dx * dx + dy * dy)
        return dist

    def Flagrectc(self, rect, c, r):
        rect = rect
        c = c
        r = r

        distances = [self.dist(rect[i], rect[j], c) for i, j in zip([0, 1, 2, 3], [1, 2, 3, 0])]
        point = Point(c)
        polygon = Polygon(rect)

        flag = 0
        if any(d < r for d in distances) == True:
            flag = 1
        if any(d < r for d in distances) == False and polygon.contains(point) == True:
            flag = 1  # type: int
        return flag

    ####################### check collision between 2 rect ########################################
    def Flag2rect(self, poly1, poly2):
        polygons = [Polygon(poly1), Polygon(poly2)]
        flag = 0
        if polygons[0].intersects(polygons[1]) == True and polygons[0].touches(polygons[1]) == False:
            flag = 1
        return flag

    ####################### check collision between 2 circle ########################################
    def Flag2cir(self, c1, r1, c2, r2):
        flag = 0
        if np.linalg.norm(np.subtract(c1, c2)) < r1 + r2:
            flag = 1
        return flag
