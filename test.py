import numpy as np
import matplotlib.pyplot as plt

def generate_skewed_alloc_sizes(n=10000, max_size=512):
    # 使用对数正态分布，mean/log-scale 控制峰值，sigma 控制偏斜程度
    samples = np.random.lognormal(mean=np.log(32), sigma=0.9)
    samples = np.clip(samples, 1, max_size)  # 限制最大值
    return samples.astype(int)  # 转成整数大小

# 生成并画出直方图
alloc_sizes = generate_skewed_alloc_sizes()
plt.hist(alloc_sizes, bins=range(0, 520, 8), color='skyblue', edgecolor='black')
plt.xlabel("Allocation Size (Bytes)")
plt.ylabel("Frequency")
plt.title("Skewed Allocation Size Distribution (max 512B)")
plt.show()
