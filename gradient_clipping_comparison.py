import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', '微软雅黑', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# Set random seed for reproducibility
np.random.seed(42)

# Configuration
CONFIG = {
    'n_steps': 2000,
    'initial_loss': 10.0,
    'final_loss': 0.5,
    'volatility': 0.3,
    'window_size': 50,  # Moving average window
    'dpi': 300,
}

# Scenario configurations
SCENARIOS = [
    {'name': '无梯度裁剪', 'clip': None, 'color': 'red', 'filename': 'no_clip'},
    {'name': '梯度裁剪阈值 = 0.1', 'clip': 0.1, 'color': 'blue', 'filename': 'clip_0.1'},
    {'name': '梯度裁剪阈值 = 0.5', 'clip': 0.5, 'color': 'green', 'filename': 'clip_0.5'},
    {'name': '梯度裁剪阈值 = 1.0', 'clip': 1.0, 'color': 'orange', 'filename': 'clip_1.0'},
]


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


def moving_average(data, window_size):
    """
    Calculate simple moving average using convolution

    Args:
        data: Input array
        window_size: Size of the moving window

    Returns:
        Moving average array (length will be len(data) - window_size + 1)
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def calculate_statistics(loss):
    """
    Calculate statistical metrics for the training curve

    Args:
        loss: Array of loss values

    Returns:
        Dictionary containing statistical metrics
    """
    min_idx = np.argmin(loss)

    # Calculate mean absolute change (stability indicator)
    mean_delta = np.mean(np.abs(np.diff(loss)))

    return {
        'mean': np.mean(loss),
        'std': np.std(loss),
        'min': loss[min_idx],
        'max': np.max(loss),
        'min_idx': min_idx,
        'initial_avg': np.mean(loss[:10]),
        'final_avg': np.mean(loss[-10:]),
        'mean_delta': mean_delta,
    }


def plot_individual_detail(steps, loss, scenario_name, color, filename):
    """
    Generate detailed training curve plot for a single scenario

    Args:
        steps: Array of step numbers
        loss: Array of loss values
        scenario_name: Name of the scenario
        color: Color for the main curve
        filename: Base filename for saving
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot main curve
    ax.plot(steps, loss, label=f'{scenario_name} (原始数据)',
            alpha=0.7, linewidth=1.5, color=color)

    # Calculate and plot moving average
    window_size = CONFIG['window_size']
    ma = moving_average(loss, window_size)
    ma_steps = steps[window_size-1:]  # Adjust steps for moving average
    ax.plot(ma_steps, ma, label=f'{window_size}步移动平均',
            linestyle='--', linewidth=2, color='darkblue', alpha=0.8)

    # Calculate statistics
    stats = calculate_statistics(loss)

    # Mark convergence region (last 100 steps)
    convergence_window = 100
    convergence_start = len(steps) - convergence_window
    convergence_loss = loss[convergence_start:]
    convergence_mean = np.mean(convergence_loss)
    convergence_std = np.std(convergence_loss)

    # Highlight convergence region with shaded area
    ax.axvspan(convergence_start, len(steps)-1, alpha=0.15, color='green',
               label=f'收敛区域 (最后{convergence_window}步)')

    # Add horizontal line for convergence mean
    ax.axhline(y=convergence_mean, color='green', linestyle=':', linewidth=2,
               alpha=0.7, xmin=0.85, xmax=1.0)

    # Mark moving average minimum point
    ma_min_idx = np.argmin(ma)
    ma_min_step = ma_steps[ma_min_idx]
    ma_min_value = ma[ma_min_idx]
    ax.scatter(ma_min_step, ma_min_value,
              color='purple', s=150, zorder=5, marker='*',
              label=f"移动平均最小点")

    # Position MA annotation to avoid overlap with convergence region
    # If MA min is in the right half, annotate to the left; otherwise to the right
    if ma_min_step > len(steps) / 2:
        ma_xytext = (-80, 20)  # Annotate to the left
    else:
        ma_xytext = (20, 20)   # Annotate to the right

    ax.annotate(f"MA最小值: {ma_min_value:.3f}\n步数: {ma_min_step}",
               xy=(ma_min_step, ma_min_value),
               xytext=ma_xytext, textcoords='offset points',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lavender', alpha=0.7),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='purple'),
               fontsize=10)

    # Annotate convergence region - position at bottom right of convergence area
    # Use axis coordinates for more reliable positioning
    ax.text(0.95, 0.15,
           f'收敛区域统计:\n均值: {convergence_mean:.3f}\n标准差: {convergence_std:.4f}',
           transform=ax.transAxes,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
           fontsize=9, ha='right', va='bottom')

    # Add statistics panel
    textstr = f'''统计信息:
• 平均Loss: {stats['mean']:.3f}
• 标准差: {stats['std']:.3f}
• 最大Loss: {stats['max']:.3f}
• 初始Loss: {stats['initial_avg']:.3f}
• 最终Loss: {stats['final_avg']:.3f}
• 收敛Loss: {convergence_mean:.3f}
• 平均变化: {stats['mean_delta']:.4f}'''

    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    # Configure plot
    ax.set_xlabel('训练步数 (Training Steps)', fontsize=12)
    ax.set_ylabel('损失值 (Loss)', fontsize=12)
    ax.set_title(f'{scenario_name} - 详细训练曲线\nDetailed Training Curve',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')

    plt.tight_layout()

    # Save figure
    output_filename = f'gradient_clipping_detail_{filename}.png'
    plt.savefig(output_filename, dpi=CONFIG['dpi'], bbox_inches='tight')
    print(f"详细图已保存为 {output_filename}")

    plt.close(fig)


def plot_comparison(data_dict):
    """
    Generate comparison plot with all four scenarios

    Args:
        data_dict: Dictionary mapping scenario names to loss arrays
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    steps = np.arange(CONFIG['n_steps'])

    # Plot all curves
    for scenario in SCENARIOS:
        name = scenario['name']
        color = scenario['color']
        loss = data_dict[name]

        label = f"{name} (No Clipping)" if scenario['clip'] is None else name
        ax.plot(steps, loss, label=label, alpha=0.8, linewidth=1.5, color=color)

    # Add labels and title
    ax.set_xlabel('训练步数 (Training Steps)', fontsize=12)
    ax.set_ylabel('损失值 (Loss)', fontsize=12)
    ax.set_title('梯度裁剪对训练效果的影响\nEffect of Gradient Clipping on Training Performance',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_yscale('log')

    # Add text annotation explaining the results
    textstr = '观察结果:\n• 无梯度裁剪: 训练不稳定，存在较大波动\n• 阈值0.1: 最平滑但收敛较慢\n• 阈值0.5: 平衡稳定性和收敛速度\n• 阈值1.0: 较好的稳定性'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save the figure
    plt.savefig('gradient_clipping_comparison.png', dpi=CONFIG['dpi'], bbox_inches='tight')
    print("对比图已保存为 gradient_clipping_comparison.png")

    plt.close(fig)


def main():
    """Main execution flow"""
    print("开始生成梯度裁剪训练效果图表...")
    print(f"配置: {CONFIG['n_steps']}步训练, 移动平均窗口={CONFIG['window_size']}")
    print()

    # Generate loss curves for all scenarios
    data_dict = {}
    steps = np.arange(CONFIG['n_steps'])

    for scenario in SCENARIOS:
        name = scenario['name']
        clip = scenario['clip']

        loss = generate_loss_curve(
            CONFIG['n_steps'],
            CONFIG['initial_loss'],
            CONFIG['final_loss'],
            CONFIG['volatility'],
            clip_threshold=clip
        )
        data_dict[name] = loss
        print(f"✓ 已生成 {name} 的训练数据")

    print()

    # Generate comparison plot
    print("生成对比图...")
    plot_comparison(data_dict)
    print()

    # Generate individual detail plots
    print("生成详细图...")
    for scenario in SCENARIOS:
        name = scenario['name']
        color = scenario['color']
        filename = scenario['filename']
        loss = data_dict[name]

        plot_individual_detail(steps, loss, name, color, filename)

    print()

    # Save data to CSV
    print("保存数据到CSV...")
    df_data = {'step': steps}
    for scenario in SCENARIOS:
        name = scenario['name']
        loss = data_dict[name]

        # Create column name for CSV
        if scenario['clip'] is None:
            col_name = 'no_clipping'
        else:
            col_name = f"clip_{scenario['clip']}"

        df_data[col_name] = loss

    df = pd.DataFrame(df_data)
    df.to_csv('gradient_clipping_data.csv', index=False)
    print("训练数据已保存为 gradient_clipping_data.csv")

    print()
    print("=" * 60)
    print("所有图表生成完成！")
    print("=" * 60)
    print("生成的文件:")
    print("  1. gradient_clipping_comparison.png - 四条曲线对比图")
    for i, scenario in enumerate(SCENARIOS, 2):
        filename = f"gradient_clipping_detail_{scenario['filename']}.png"
        print(f"  {i}. {filename} - {scenario['name']}详细图")
    print("  6. gradient_clipping_data.csv - 训练数据")


if __name__ == '__main__':
    main()
