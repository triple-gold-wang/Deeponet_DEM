import pandas as pd
import os

def upgrade_csv_with_id(csv_path):
    if not os.path.exists(csv_path):
        print(f"找不到文件: {csv_path}")
        return
    
    # 读取现有的参数表
    df = pd.read_csv(csv_path)
    
    # 检查是否已经存在 shape_id，防止重复添加
    if 'shape_id' not in df.columns:
        # 在第 0 列插入 shape_id，值为 0 到 199
        df.insert(0, 'shape_id', range(len(df)))
        
        # 覆盖保存为同一个文件
        df.to_csv(csv_path, index=False)
        print(f"成功为 {csv_path} 添加 shape_id 索引！")
        print(df.head())
    else:
        print("该文件已经包含 shape_id，无需重复添加。")

if __name__ == '__main__':

    upgrade_csv_with_id('geometry_params_200.csv')