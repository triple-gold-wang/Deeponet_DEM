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

# ==========================================
# 批量运行配置
# ==========================================
# 在 Abaqus/CAE 图形界面中通过 File -> Run Script 运行时，
# 当前工作目录不一定是脚本所在目录，所以这里显式定位到 gt 目录。
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()
CSV_FILE = os.path.join(BASE_DIR, 'geometry_params_200.csv')
OUTPUT_DIR = BASE_DIR

# 当前先只生成 main.py 分层验证集对应的真值。
# 这些 ID 由 main.py 的逻辑得到：
# VAL_RATIO = 0.2, SPLIT_SEED = 42, hard_mask = (a / b > 4.0)
RUN_MODE = 'validation'  # 可选: 'validation' 或 'all'
VALIDATION_SHAPE_IDS = set([
    118, 293, 65, 19, 186, 261, 302, 297, 374, 218,
    244, 273, 271, 360, 184, 104, 204, 27, 310, 58,
    189, 369, 179, 103, 25, 319, 35, 99, 146, 82,
    220, 315, 72, 182, 23, 399, 183, 266, 397, 76,
    38, 188, 199, 222, 228, 115, 223, 334, 198, 281,
    147, 168, 106, 215, 214, 357, 2, 309, 325, 170,
    80, 248, 393, 138, 192, 90, 174, 28, 26, 169,
    288, 21, 289, 207, 380, 164, 290, 370, 194, 6,
])

# 已有旧真值可能对应旧参数表。为了避免误用旧数据，默认重新覆盖生成。
OVERWRITE_EXISTING = True
MAX_SHAPES = None  # 调试时可改成 1 或 2，确认流程正常后再改回 None。

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
    csv_path = os.path.join(OUTPUT_DIR, 'abaqus_truth_{}.csv'.format(shape_id))
    
    print("正在提取数据到: {}".format(csv_path))
    odb = openOdb(path=odb_path, readOnly=True)
    last_frame = odb.steps['Step-1'].frames[-1]
    disp_field = last_frame.fieldOutputs['U']
    
    # 获取节点坐标
    instance_name = list(odb.rootAssembly.instances.keys())[0]
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

def load_geometry_rows(csv_file):
    rows = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        required_cols = ['shape_id', 'a', 'b', 'theta']
        for col in required_cols:
            if col not in reader.fieldnames:
                raise ValueError("参数文件缺少必要列: {}".format(col))

        for row in reader:
            shape_id = int(row['shape_id'])
            rows.append({
                'shape_id': shape_id,
                'a': float(row['a']),
                'b': float(row['b']),
                'theta': float(row['theta']),
            })
    return rows

def select_rows(rows):
    if RUN_MODE == 'all':
        selected = rows
    elif RUN_MODE == 'validation':
        selected = [row for row in rows if row['shape_id'] in VALIDATION_SHAPE_IDS]
        missing_ids = sorted(VALIDATION_SHAPE_IDS - set(row['shape_id'] for row in selected))
        if missing_ids:
            print("警告：参数表中找不到以下验证集 ID: {}".format(missing_ids))
    else:
        raise ValueError("未知 RUN_MODE: {}".format(RUN_MODE))

    selected = sorted(selected, key=lambda row: row['shape_id'])
    if MAX_SHAPES is not None:
        selected = selected[:MAX_SHAPES]
    return selected

# ==========================================
# 主循环：读取配置表并批量执行
# ==========================================
if __name__ == '__main__':
    os.chdir(BASE_DIR)
    
    if not os.path.exists(CSV_FILE):
        print("未找到参数文件: {}".format(CSV_FILE))
    else:
        rows = load_geometry_rows(CSV_FILE)
        selected_rows = select_rows(rows)

        hard_count = 0
        for row in selected_rows:
            if row['a'] / row['b'] > 4.0:
                hard_count += 1
        print("参数文件: {}".format(CSV_FILE))
        print("运行模式: {}".format(RUN_MODE))
        print("待生成真值数量: {}，其中高长宽比 a/b>4: {}".format(len(selected_rows), hard_count))
        print("输出目录: {}".format(OUTPUT_DIR))
        print("是否覆盖已有真值: {}".format(OVERWRITE_EXISTING))

        for row in selected_rows:
            shape_id = row['shape_id']
            output_csv = os.path.join(OUTPUT_DIR, 'abaqus_truth_{}.csv'.format(shape_id))
            if os.path.exists(output_csv) and not OVERWRITE_EXISTING:
                print("Shape ID {} 已存在，跳过: {}".format(shape_id, output_csv))
                continue

            try:
                run_single_model(shape_id, row['a'], row['b'], row['theta'])
            except Exception as e:
                print("Shape ID {} 发生错误: {}".format(shape_id, str(e)))