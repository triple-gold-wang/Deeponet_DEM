import torch
import pandas as pd
import numpy as np
import importlib
from torch.utils.data import TensorDataset, DataLoader, random_split

# 导入你之前写好的各个模块
from geometry import DomainSampler
from model import SolidDeepONet
from loss import compute_dem_loss

# ==========================================
# ⚙️ 1. 全局超参数与环境配置
# ==========================================
# 设备配置：如果你的电脑有NVIDIA显卡且装了CUDA，会自动用GPU，否则用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的计算设备: {device}")

# 物理与几何配置
L_CONFIG = 1.0       # 正方形半边长
R0_CONFIG = 0.3      # 参考域圆孔半径
E_CONFIG = 1.0       # 无量纲杨氏模量
NU_CONFIG = 0.3      # 泊松比
TX_CONFIG = 10.0     # 右边界拉力

# 训练配置
EPOCHS = 1000        # 最大训练轮次 (配合收敛判据提前停止)
BATCH_SIZE = 32      # 批次大小
LEARNING_RATE = 1e-3 # 学习率 (Adam优化器默认通常是 1e-3)

# Visdom 可视化配置
VISDOM_ENABLE = True
VISDOM_SERVER = 'http://127.0.0.1'
VISDOM_PORT = 8097
VISDOM_ENV = 'main'
VISDOM_UPDATE_EVERY = 1          # 每多少个 epoch 刷新一次曲线
VISDOM_LOSS_WIN = 'loss_curve'

# 收敛判据配置
VAL_RATIO = 0.2                  # 验证集比例
SPLIT_SEED = 42                  # 数据划分随机种子
MIN_EPOCHS = 200                 # 最少训练轮数，防止过早停止

# 判据: 验证集平台期
VAL_MIN_DELTA = 1e-5             # 验证损失最小改进阈值
VAL_PATIENCE = 40                # 连续无改进轮数

# 采样配置
N_INTERIOR = 3000    # 内部参考域撒点数 (为了测试速度，先用 3000)
N_BND_RIGHT = 200    # 右边界撒点数 (算外力做功)   

# ==========================================
# 📊 2. 数据准备阶段
# ==========================================
print("\n--- 正在加载数据与采样 ---")

# 2.1 加载几何参数 (a, b, theta)
try:
    df_params = pd.read_csv('geometry_params_200.csv')
    # 仅保留几何参数 a, b, theta 作为 Branch 网络输入
    required_cols = ['a', 'b', 'theta']
    missing_cols = [col for col in required_cols if col not in df_params.columns]
    if missing_cols:
        raise ValueError(f"geometry_params_200.csv 缺少必要列: {missing_cols}")
    params_tensor = torch.tensor(df_params[required_cols].values, dtype=torch.float32)
except FileNotFoundError:
    print("错误: 找不到 geometry_params_200.csv！请先运行 geo_data.py。")
    exit()

# 2.2 构建 DataLoader 实现自动批处理
dataset = TensorDataset(params_tensor)

val_size = max(1, int(len(dataset) * VAL_RATIO))
train_size = len(dataset) - val_size
if train_size <= 0:
    raise ValueError("训练集大小为 0，请减小 VAL_RATIO 或增加样本数量。")

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SPLIT_SEED)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"成功加载几何参数，共 {len(dataset)} 个样本。训练集: {train_size}，验证集: {val_size}。")

# 2.3 生成参考域坐标点 (这部分是固定的，所以只在循环外生成一次！)
sampler = DomainSampler(L=L_CONFIG, R0=R0_CONFIG)
X_inner = sampler.sample_interior(N_INTERIOR).to(device)
X_right = sampler.sample_right_boundary(N_BND_RIGHT).to(device)

print(f"参考域采样完成: 内部点 {X_inner.shape[0]} 个, 右边界点 {X_right.shape[0]} 个。")

# ==========================================
# 🧠 3. 模型与优化器初始化
# ==========================================
print("\n--- 正在初始化模型 ---")

# 定义网络结构 (你可以根据需要增减层数)
branch_layers = [3, 64, 128, 100]  # 输入维度 3
trunk_layers = [2, 64, 128, 100]   # 输入维度 2

# 实例化模型并移至对应设备
model = SolidDeepONet(branch_layers, trunk_layers, L=L_CONFIG).to(device)

# 实例化优化器 (Adam 是最常用的起手式)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 🏃‍♂️ 4. 主训练循环 (Training Loop)
# ==========================================
print("\n--- 开始训练 ---")

# Visdom 初始化（不可用时不影响训练）
viz = None
viz_enabled = False
if VISDOM_ENABLE:
    try:
        Visdom = importlib.import_module('visdom').Visdom
    except Exception:
        Visdom = None

    if Visdom is None:
        print("警告: 未安装 visdom，跳过实时可视化。可执行: pip install visdom")
    else:
        viz = Visdom(server=VISDOM_SERVER, port=VISDOM_PORT, env=VISDOM_ENV)
        viz_enabled = viz.check_connection(timeout_seconds=3)
        if viz_enabled:
            print(f"Visdom 已连接: {VISDOM_SERVER}:{VISDOM_PORT} | env={VISDOM_ENV}")
        else:
            print(
                "警告: Visdom 服务未连接，跳过实时可视化。"
                "可先启动: python -m visdom.server -port 8097"
            )

def evaluate_loss(current_model, data_loader):
    """在验证集上评估 DEM 损失。"""
    current_model.eval()
    total = 0.0
    with torch.enable_grad():
        for batch_data in data_loader:
            params_batch = batch_data[0].to(device)
            val_loss = compute_dem_loss(
                model=current_model,
                params=params_batch,
                X_inner=X_inner,
                X_right=X_right,
                L=L_CONFIG, R0=R0_CONFIG,
                E=E_CONFIG, nu=NU_CONFIG, Tx=TX_CONFIG
            )
            total += val_loss.item()
    current_model.train()
    return total / len(data_loader)

best_val_loss = float('inf')
best_state_dict = None

val_plateau_count = 0

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    
    # 遍历训练集数据加载器中的每一个小批次
    for batch_idx, batch_data in enumerate(train_loader):
        
        # 提取当前批次的几何参数 [batch_size, 3] 并发送到设备
        params_batch = batch_data[0].to(device)
        
        # 1. 梯度清零 (PyTorch 必须的常规操作)
        optimizer.zero_grad()
        
        # 2. 计算 DEM 损失 (调用 loss.py)
        loss = compute_dem_loss(
            model=model, 
            params=params_batch, 
            X_inner=X_inner, 
            X_right=X_right,
            L=L_CONFIG, R0=R0_CONFIG, 
            E=E_CONFIG, nu=NU_CONFIG, Tx=TX_CONFIG
        )
        
        # 3. 反向传播 (自动微分计算梯度)
        loss.backward()
        
        # 4. 更新权重
        optimizer.step()
        
        # 累加损失用于监控
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    avg_val_loss = evaluate_loss(model, val_loader)

    # 实时可视化训练/验证 loss
    if viz_enabled and (epoch + 1) % VISDOM_UPDATE_EVERY == 0:
        try:
            if not viz.win_exists(VISDOM_LOSS_WIN):
                viz.line(
                    X=np.array([epoch + 1], dtype=np.float64),
                    Y=np.array([[avg_train_loss, avg_val_loss]], dtype=np.float64),
                    win=VISDOM_LOSS_WIN,
                    opts=dict(
                        title='Train vs Val Loss',
                        xlabel='Epoch',
                        ylabel='Loss',
                        legend=['train_loss', 'val_loss']
                    )
                )
            else:
                viz.line(
                    X=np.array([epoch + 1], dtype=np.float64),
                    Y=np.array([[avg_train_loss, avg_val_loss]], dtype=np.float64),
                    win=VISDOM_LOSS_WIN,
                    update='append'
                )
        except Exception as ex:
            print(f"警告: Visdom 更新失败，后续将关闭可视化。原因: {ex}")
            viz_enabled = False

    # 验证集平台期判据
    if avg_val_loss < best_val_loss - VAL_MIN_DELTA:
        best_val_loss = avg_val_loss
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        val_plateau_count = 0
    else:
        val_plateau_count += 1
    
    # 每 10 个 Epoch 打印一次信息
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | "
            f"ValPlateauCount: {val_plateau_count}/{VAL_PATIENCE}"
        )

    # 单一收敛判据: 验证集连续无改进后提前停止
    if (epoch + 1) >= MIN_EPOCHS and val_plateau_count >= VAL_PATIENCE:
        print("\n--- 触发收敛提前停止 ---")
        print(
            f"停止于 Epoch {epoch+1} | "
            f"验证集平台判据已触发: {val_plateau_count} >= {VAL_PATIENCE}"
        )
        break

# 若有验证集最优权重，则恢复最优权重再保存
if best_state_dict is not None:
    model.load_state_dict(best_state_dict)

# 可选：保存模型权重
torch.save(model.state_dict(), 'deeponet_dem_test.pth')