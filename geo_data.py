import numpy as np
import pandas as pd
from scipy.stats import qmc

# 使用拉丁超立方抽样生成 400 个样本，其中高长宽比 a/b > 4 占 40%
num_samples = 400
hard_ratio = 0.40
num_hard = int(num_samples * hard_ratio)
num_normal = num_samples - num_hard

# 映射到具体的物理区间
# a: [0.2, 0.8], b: [0.1, 0.8], theta: [0, pi]
a_bounds = [0.2, 0.8]
b_bounds = [0.1, 0.8]
theta_bounds = [0, np.pi]


def sample_group(n, want_hard):
    """按 a/b 是否大于 4 分组采样，保持原始 a、b 取值范围不变。"""
    a_list, b_list, theta_list = [], [], []

    while len(a_list) < n:
        batch_size = max(100, (n - len(a_list)) * 4)
        sampler = qmc.LatinHypercube(d=3)
        sample = sampler.random(n=batch_size)

        a = a_bounds[0] + sample[:, 0] * (a_bounds[1] - a_bounds[0])
        b = b_bounds[0] + sample[:, 1] * (b_bounds[1] - b_bounds[0])
        theta = theta_bounds[0] + sample[:, 2] * (theta_bounds[1] - theta_bounds[0])

        # 强制让 a >= b
        swap_mask = b > a
        a[swap_mask], b[swap_mask] = b[swap_mask], a[swap_mask].copy()

        aspect = a / b
        if want_hard:
            mask = aspect > 4.0
        else:
            mask = aspect <= 4.0

        a_list.extend(a[mask].tolist())
        b_list.extend(b[mask].tolist())
        theta_list.extend(theta[mask].tolist())

    return (
        np.array(a_list[:n]),
        np.array(b_list[:n]),
        np.array(theta_list[:n]),
    )


a_normal, b_normal, theta_normal = sample_group(num_normal, want_hard=False)
a_hard, b_hard, theta_hard = sample_group(num_hard, want_hard=True)

a = np.concatenate([a_normal, a_hard])
b = np.concatenate([b_normal, b_hard])
theta = np.concatenate([theta_normal, theta_hard])

# 打乱顺序，避免后 40% 全是高长宽比样本
perm = np.random.permutation(num_samples)
a = a[perm]
b = b[perm]
theta = theta[perm]

# 保存为 DataFrame 并导出 CSV
df_params = pd.DataFrame({'a': a, 'b': b, 'theta': theta})
df_params.insert(0, 'shape_id', range(len(df_params)))
df_params.to_csv('geometry_params_200.csv', index=False)

print(f"已生成 {len(df_params)} 个样本")
print(f"a/b > 4 的样本数: {(df_params['a'] / df_params['b'] > 4).sum()}")
