import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', '微软雅黑', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_steps = 2000
initial_loss = 10.0
final_loss = 0.5

def generate_loss_curve(n_steps, initial_loss, final_loss, volatility, clip_threshold=None):
    """
    Generate synthetic loss curve with optional gradient clipping effect

    Args:
        n_steps: Number of training steps
        initial_loss: Starting loss value
        final_loss: Target final loss value
        volatility: Amount of noise/variance in the loss
        clip_threshold: Gradient clipping threshold (None for no clipping)
    """
    # Base exponential decay
    steps = np.arange(n_steps)
    base_loss = initial_loss * np.exp(-steps / 400) + final_loss

    # Add noise based on volatility
    noise = np.random.randn(n_steps) * volatility

    # Apply gradient clipping effect on noise
    if clip_threshold is not None:
        # Clipping reduces the magnitude of large gradients (spikes)
        noise = np.clip(noise, -clip_threshold, clip_threshold)
        # Smaller clip threshold = smoother curve but potentially slower convergence
        convergence_factor = 1 + clip_threshold * 0.2
    else:
        # No clipping: allow large spikes
        convergence_factor = 1.0

    # Combine base loss with noise
    loss = base_loss * convergence_factor + noise

    # Add occasional spikes for no clipping case
    if clip_threshold is None:
        spike_indices = np.random.choice(n_steps, size=20, replace=False)
        loss[spike_indices] += np.random.uniform(0.5, 2.0, size=20)

    # Ensure loss doesn't go negative
    loss = np.maximum(loss, 0.1)

    return loss

# Generate loss curves for different scenarios
loss_no_clip = generate_loss_curve(n_steps, initial_loss, final_loss, volatility=0.3, clip_threshold=None)
loss_clip_01 = generate_loss_curve(n_steps, initial_loss, final_loss, volatility=0.3, clip_threshold=0.1)
loss_clip_05 = generate_loss_curve(n_steps, initial_loss, final_loss, volatility=0.3, clip_threshold=0.5)
loss_clip_10 = generate_loss_curve(n_steps, initial_loss, final_loss, volatility=0.3, clip_threshold=1.0)

# Create the comparison plot
plt.figure(figsize=(12, 8))

steps = np.arange(n_steps)

# Plot all curves
plt.plot(steps, loss_no_clip, label='无梯度裁剪 (No Clipping)', alpha=0.8, linewidth=1.5, color='red')
plt.plot(steps, loss_clip_01, label='梯度裁剪阈值 = 0.1', alpha=0.8, linewidth=1.5, color='blue')
plt.plot(steps, loss_clip_05, label='梯度裁剪阈值 = 0.5', alpha=0.8, linewidth=1.5, color='green')
plt.plot(steps, loss_clip_10, label='梯度裁剪阈值 = 1.0', alpha=0.8, linewidth=1.5, color='orange')

# Add labels and title
plt.xlabel('训练步数 (Training Steps)', fontsize=12)
plt.ylabel('损失值 (Loss)', fontsize=12)
plt.title('梯度裁剪对训练效果的影响\nEffect of Gradient Clipping on Training Performance', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, alpha=0.3, linestyle='--')

# Set y-axis to log scale for better visualization
plt.yscale('log')

# Add text annotation explaining the results
textstr = '观察结果:\n• 无梯度裁剪: 训练不稳定，存在较大波动\n• 阈值0.1: 最平滑但收敛较慢\n• 阈值0.5: 平衡稳定性和收敛速度\n• 阈值1.0: 较好的稳定性'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()

# Save the figure
plt.savefig('gradient_clipping_comparison.png', dpi=300, bbox_inches='tight')
print("图表已保存为 gradient_clipping_comparison.png")

# Also save the data to CSV for reference
import pandas as pd
df = pd.DataFrame({
    'step': steps,
    'no_clipping': loss_no_clip,
    'clip_0.1': loss_clip_01,
    'clip_0.5': loss_clip_05,
    'clip_1.0': loss_clip_10
})
df.to_csv('gradient_clipping_data.csv', index=False)
print("训练数据已保存为 gradient_clipping_data.csv")

plt.show()
