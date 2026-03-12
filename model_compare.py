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
    'n_steps': 10000,
    'initial_loss': 10.0,
    'final_loss': 0.0,
    'dpi': 300,
}

# Model configurations
MODELS = [
    {'name': 'GRU', 'color': 'red', 'converge': False},
    {'name': 'LSTM', 'color': 'blue', 'converge': True},
    {'name': 'CNN-Transformer', 'color': 'green', 'converge': True},
]


def generate_gru_loss(n_steps, initial_loss):
    """
    Generate GRU loss curve - oscillating around 1.0 and not converging

    Args:
        n_steps: Number of training steps
        initial_loss: Starting loss value
    """
    steps = np.arange(n_steps)

    # Initial decay phase - decay from 10 to around 1
    decay_phase = (initial_loss - 1.0) * np.exp(-steps / 2000) + 1.0

    # Add oscillation that becomes more prominent in later stages
    oscillation = np.zeros(n_steps)
    for i in range(n_steps):
        if i > 3000:
            # Oscillation amplitude increases over time
            amplitude = 0.3 + (i - 3000) * 0.00005
            amplitude = min(amplitude, 0.8)  # Cap the amplitude
            oscillation[i] = amplitude * np.sin(i * 0.01)

    # Combine decay and oscillation
    loss = decay_phase + oscillation

    # Add random noise
    noise = np.random.randn(n_steps) * 0.05
    loss = loss + noise

    # Ensure loss stays in reasonable range (oscillating around 1)
    loss = np.clip(loss, 0.2, initial_loss)

    return loss


def generate_converging_loss(n_steps, initial_loss, final_loss, converge_start, noise_level):
    """
    Generate converging loss curve for LSTM and CNN-Transformer

    Args:
        n_steps: Number of training steps
        initial_loss: Starting loss value
        final_loss: Target final loss value
        converge_start: Step at which convergence begins
        noise_level: Amount of noise in the curve
    """
    steps = np.arange(n_steps)
    loss = np.zeros(n_steps)

    for i in range(n_steps):
        if i < converge_start:
            # Before convergence: slow decay with some fluctuation
            progress = i / converge_start
            loss[i] = initial_loss - (initial_loss - 2.0) * progress
            # Add some fluctuation
            loss[i] += np.sin(i * 0.02) * 0.5
        else:
            # After convergence point: rapid decay to near zero
            steps_after_converge = i - converge_start
            remaining_steps = n_steps - converge_start
            decay_rate = 1500  # Controls how fast it converges
            loss[i] = 2.0 * np.exp(-steps_after_converge / decay_rate) + final_loss

    # Add noise
    noise = np.random.randn(n_steps) * noise_level
    loss = loss + noise

    # Ensure loss doesn't go below final_loss
    loss = np.maximum(loss, final_loss + 0.01)
    loss = np.minimum(loss, initial_loss)

    return loss


def plot_comparison(data_dict):
    """
    Generate comparison plot with all three models

    Args:
        data_dict: Dictionary mapping model names to loss arrays
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    steps = np.arange(CONFIG['n_steps'])

    # Plot all curves
    for model in MODELS:
        name = model['name']
        color = model['color']
        loss = data_dict[name]

        ax.plot(steps, loss, label=name, alpha=0.8, linewidth=2, color=color)

    # Add labels and title
    ax.set_xlabel('训练步数 (Steps)', fontsize=14)
    ax.set_ylabel('损失值 (Loss)', fontsize=14)
    ax.set_title('不同模型训练效果对比\nModel Training Performance Comparison',
                fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Set y-axis limits
    ax.set_ylim(-0.5, 11)

    # Add text annotation explaining the results
    textstr = '观察结果:\n• GRU: 训练后期在1左右震荡，不收敛\n• LSTM: 在5000步左右开始收敛\n• CNN-Transformer: 在7000步左右开始收敛'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # Save the figure
    plt.savefig('model_comparison.png', dpi=CONFIG['dpi'], bbox_inches='tight')
    print("对比图已保存为 model_comparison.png")

    plt.close(fig)


def main():
    """Main execution flow"""
    print("开始生成模型训练效果对比图...")
    print(f"配置: {CONFIG['n_steps']}个Steps, Loss范围: {CONFIG['initial_loss']} - {CONFIG['final_loss']}")
    print()

    # Generate loss curves for all models
    data_dict = {}

    # GRU - oscillating, not converging
    print("生成 GRU 训练数据 (震荡，不收敛)...")
    data_dict['GRU'] = generate_gru_loss(CONFIG['n_steps'], CONFIG['initial_loss'])

    # LSTM - converging
    print("生成 LSTM 训练数据 (5000步开始收敛)...")
    data_dict['LSTM'] = generate_converging_loss(
        CONFIG['n_steps'],
        CONFIG['initial_loss'],
        CONFIG['final_loss'],
        converge_start=5000,
        noise_level=0.1
    )

    # CNN-Transformer - converging, similar to LSTM
    print("生成 CNN-Transformer 训练数据 (7000步开始收敛)...")
    data_dict['CNN-Transformer'] = generate_converging_loss(
        CONFIG['n_steps'],
        CONFIG['initial_loss'],
        CONFIG['final_loss'],
        converge_start=7000,
        noise_level=0.08
    )

    print()

    # Generate comparison plot
    print("生成对比图...")
    plot_comparison(data_dict)

    print()

    # Save data to CSV
    print("保存数据到CSV...")
    df_data = {'step': np.arange(CONFIG['n_steps'])}
    for model in MODELS:
        name = model['name']
        df_data[name] = data_dict[name]

    df = pd.DataFrame(df_data)
    df.to_csv('model_comparison_data.csv', index=False)
    print("训练数据已保存为 model_comparison_data.csv")

    print()
    print("=" * 60)
    print("图表生成完成！")
    print("=" * 60)
    print("生成的文件:")
    print("  1. model_comparison.png - 三个模型对比图")
    print("  2. model_comparison_data.csv - 训练数据")


if __name__ == '__main__':
    main()
