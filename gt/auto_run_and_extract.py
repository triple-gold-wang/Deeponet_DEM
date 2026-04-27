# -*- coding: utf-8 -*-
from abaqus import *
from abaqusConstants import *
from part import *
from material import *
from section import *
from assembly import *
from step import *
from load import *
from job import *
from sketch import *
from odbAccess import openOdb

# --- 就是下面这两行，显式导入核心模块 ---
import regionToolset
import mesh  

import csv
import math
import os

def run_single_model(shape_id, a, b, theta):
    # ==========================================
    # 0. 初始化与清理内存 (极其关键，防止连续运行报错)
    # ==========================================
    Mdb() # 彻底清空当前数据库
    model_name = 'Model-Shape-{}'.format(shape_id)
    part_name = 'Plate'
    job_name = 'Job-Shape-{}'.format(shape_id)
    
    # 将新建的 Model 设为当前 Model
    mdb.models.changeKey(fromName='Model-1', toName=model_name)
    my_model = mdb.models[model_name]

    # ==========================================
    # 1. 几何参数化建模
    # ==========================================
    s = my_model.ConstrainedSketch(name='__profile__', sheetSize=5.0)
    
    # 1.1 画外围正方形 (半边长 L=1.0)
    s.rectangle(point1=(-1.0, -1.0), point2=(1.0, 1.0))
    
    # 1.2 画内部旋转椭圆
    # Abaqus 画椭圆需要中心点和长短轴的端点坐标
    x_a = a * math.cos(theta)
    y_a = a * math.sin(theta)
    x_b = -b * math.sin(theta)
    y_b = b * math.cos(theta)
    s.EllipseByCenterPerimeter(center=(0.0, 0.0), axisPoint1=(x_a, y_a), axisPoint2=(x_b, y_b))
    
    # 生成 2D 壳体零件
    my_part = my_model.Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
    my_part.BaseShell(sketch=s)
    del my_model.sketches['__profile__']

    # ==========================================
    # 2. 材料与截面属性 (无量纲参数)
    # ==========================================
    my_model.Material(name='Mat-NN')
    my_model.materials['Mat-NN'].Elastic(table=((1.0, 0.3), ))
    my_model.HomogeneousSolidSection(name='Section-1', material='Mat-NN', thickness=1.0)
    
    # 赋予截面
    f = my_part.faces
    faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
    region = regionToolset.Region(faces=faces)
    my_part.SectionAssignment(region=region, sectionName='Section-1')

    # ==========================================
    # 3. 装配与分析步
    # ==========================================
    my_assembly = my_model.rootAssembly
    my_assembly.Instance(name='Plate-1', part=my_part, dependent=ON)
    my_model.StaticStep(name='Step-1', previous='Initial')

    # ==========================================
    # 4. 边界条件与力载荷
    # ==========================================
    inst = my_assembly.instances['Plate-1']
    edges = inst.edges
    
    # 寻找左边界 (X = -1.0) 并固定 U1, U2
    left_edge = edges.findAt(((-1.0, 0.0, 0.0), ))
    left_region = my_assembly.Set(edges=left_edge, name='Set-Left')
    my_model.DisplacementBC(name='BC-Fixed', createStepName='Initial', 
                            region=left_region, u1=0.0, u2=0.0)
    
    # 寻找右边界 (X = 1.0) 并施加向右的面载荷 (Surface Traction)
    right_edge = edges.findAt(((1.0, 0.0, 0.0), ))
    right_surface = my_assembly.Surface(side1Edges=right_edge, name='Surf-Right')
    my_model.SurfaceTraction(name='Load-Pull', createStepName='Step-1', 
                             region=right_surface, magnitude=10.0, 
                             directionVector=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), 
                             distributionType=UNIFORM, traction=GENERAL)

    # ==========================================
    # 5. 网格划分
    # ==========================================
    # 整体网格尺寸 0.05
    my_part.seedPart(size=0.05, deviationFactor=0.1, minSizeFactor=0.1)
    elemType = mesh.ElemType(elemCode=CPS4R, elemLibrary=STANDARD)
    my_part.setElementType(regions=(my_part.faces, ), elemTypes=(elemType, ))
    my_part.generateMesh()

    # ==========================================
    # 6. 提交计算并等待
    # ==========================================
    mdb.Job(name=job_name, model=model_name, type=ANALYSIS)
    print("正在计算 Shape ID: {}...".format(shape_id))
    mdb.jobs[job_name].submit()
    mdb.jobs[job_name].waitForCompletion() # 核心：阻塞脚本，直到计算完成！

    # ==========================================
    # 7. 立即提取 CSV 数据
    # ==========================================
    extract_data_to_csv(job_name, shape_id)

def extract_data_to_csv(job_name, shape_id):
    odb_path = job_name + '.odb'
    csv_path = 'abaqus_truth_{}.csv'.format(shape_id)
    
    print("正在提取数据到: {}".format(csv_path))
    odb = openOdb(path=odb_path, readOnly=True)
    last_frame = odb.steps['Step-1'].frames[-1]
    disp_field = last_frame.fieldOutputs['U']
    
    # 获取节点坐标
    instance_name = odb.rootAssembly.instances.keys()[0]
    nodes = odb.rootAssembly.instances[instance_name].nodes
    
    disp_dict = {}
    for val in disp_field.values:
        disp_dict[val.nodeLabel] = val.data
        
    with open(csv_path, 'w') as f:
        f.write('nodeLabel,x,y,U1,U2\n')
        for node in nodes:
            n_label = node.label
            x = node.coordinates[0]
            y = node.coordinates[1]
            if n_label in disp_dict:
                u1 = disp_dict[n_label][0]
                u2 = disp_dict[n_label][1]
                f.write('%d,%f,%f,%f,%f\n' % (n_label, x, y, u1, u2))
                
    odb.close()
    
    # 提取完成后，清理庞大的临时文件（可选，但推荐）
    for ext in ['.odb', '.com', '.dat', '.msg', '.prt', '.sim', '.sta']:
        try:
            os.remove(job_name + ext)
        except:
            pass
    print("Shape ID: {} 处理完成！\n".format(shape_id))

# ==========================================
# 主循环：读取配置表并批量执行
# ==========================================
if __name__ == '__main__':
    csv_file = 'geometry_params_200.csv'
    
    if not os.path.exists(csv_file):
        print("未找到参数文件: {}".format(csv_file))
    else:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader) # 跳过表头
            
            for row in reader:
                shape_id = int(row[0])
                a = float(row[1])
                b = float(row[2])
                theta = float(row[3])
                
                try:
                    run_single_model(shape_id, a, b, theta)
                except Exception as e:
                    print("Shape ID {} 发生错误: {}".format(shape_id, str(e)))