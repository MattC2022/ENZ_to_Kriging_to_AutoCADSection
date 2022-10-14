import Contour
from pyautocad import Autocad, APoint
import numpy as np

contour_topo = Contour.Kriging('Data/Input/3.Topo_Point_Cloud.csv')
contour_rock = Contour.Kriging('Data/Input/2.ENV_for_Surfer.csv')
contour_N200 = Contour.Kriging('Data/Input/1.N200.csv')

point_start = np.array([821228.9044,834833.9274])
point_end = np.array([821240.0612,834855.4965])
length = sum((point_start-point_end)**2)**0.5
division = 100
unit_vector = (point_end-point_start)/division
step = length/division

acad = Autocad()
cross_section_topo = []
cross_section_rock = []
cross_section_N200 = []

for i in range(0,division+1):
    now_xy_on_plan = point_start + unit_vector*i
    now_x_on_section = step*i
    now_y_on_section_topo = contour_topo.get_z(now_xy_on_plan[0],now_xy_on_plan[1])
    now_y_on_section_rock = contour_rock.get_z(now_xy_on_plan[0], now_xy_on_plan[1])
    now_y_on_section_N200 = contour_N200.get_z(now_xy_on_plan[0], now_xy_on_plan[1])
    cross_section_topo += [[now_x_on_section, now_y_on_section_topo]]
    cross_section_rock += [[now_x_on_section, now_y_on_section_rock]]
    cross_section_N200 += [[now_x_on_section, now_y_on_section_N200]]

p1 = APoint(cross_section_topo[0][0],cross_section_topo[0][1])
for i in cross_section_topo[1:]:
    p2 = APoint(i[0],i[1])
    acad.model.AddLine(p1, p2)
    p1 = p2

p1 = APoint(cross_section_rock[0][0],cross_section_rock[0][1])
for i in cross_section_rock[1:]:
    p2 = APoint(i[0],i[1])
    acad.model.AddLine(p1, p2)
    p1 = p2

p1 = APoint(cross_section_N200[0][0],cross_section_N200[0][1])
for i in cross_section_N200[1:]:
    p2 = APoint(i[0],i[1])
    acad.model.AddLine(p1, p2)
    p1 = p2
