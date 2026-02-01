import numpy as np
import pandas as pd

# --- 参数校准 ---
fs = 100
duration = 60 # 延长模拟时间
t = np.arange(0, duration, 1/fs)

RPM_STEP = 100
MOTOR_TAU = 0.15 
BASE_SPEED = 24000.0  # 基准转速线
TARGET_MAP = 95.0     # 目标平均动脉压

# 重新校准的水力学常数，确保 BASE_SPEED 对应 TARGET_MAP
# 令 AoP = LVP_mean + (Speed - 18000) * K * Resistance
K_PUMP = 0.0065 # 泵效系数

# PID 参数调校
Kp, Ki, Kd = 4.5, 1.2, 0.15
current_speed = BASE_SPEED
command_speed = BASE_SPEED
integral, last_error = 0, 0

history = []

for i in range(len(t)):
    # 1. 模拟生理状态演变（关键：确保有高阻力和低阻力场景）
    if t[i] < 15:
        resistance = 1.0  # 标准状态 (24000 RPM 附近)
    elif 15 <= t[i] < 35:
        resistance = 0.65 # 剧烈运动状态 (转速必须大幅升高 > 24000)
    elif 35 <= t[i] < 45:
        resistance = 1.3  # 深度睡眠/高阻力 (转速会降至 < 24000)
    else:
        resistance = 1.0  # 恢复标准

    # 2. 生成 LVP (心室压力)
    hr = 70 + (1 - resistance) * 50
    T = 60 / hr
    activation = np.sin(2 * np.pi / T * t[i]) if (t[i] % T) < (T/3.5) else 0
    lvp = 5 + 105 * (activation**2) + np.random.normal(0, 0.5)

    # 3. 计算 AoP (主动脉压) - 核心公式校准
    # 电机惯性模拟
    current_speed += (command_speed - current_speed) * (1 / (MOTOR_TAU * fs + 1))
    
    # 校准公式：压力 = (转速带来的压升 + 基础静脉压) * 阻力系数
    # 确保在 24000 RPM, resistance=1.0 时，aop 均值在 95 左右
    aop_base = (current_speed - 12000) * K_PUMP + 15 
    aop = aop_base * resistance + 10 * activation + np.random.normal(0, 1.0)

    # 4. PID 控制逻辑 (带噪声反馈)
    measured_map = aop # 实际系统中通常取滑动平均，这里简化
    error = TARGET_MAP - measured_map
    integral += error * (1/fs)
    # 限制积分饱和
    integral = np.clip(integral, -2000, 2000)
    derivative = (error - last_error) * fs
    
    # 核心控制输出
    raw_output = BASE_SPEED + (Kp * error + Ki * integral + Kd * derivative)
    
    # 5. 模拟电机离散阶梯 (100 RPM step)
    command_speed = round(raw_output / RPM_STEP) * RPM_STEP
    command_speed = np.clip(command_speed, 16000, 32000) # 限制在安全范围
    
    # 流量计算
    flow = (current_speed * 0.00025) - ((aop - lvp) * 0.015)
    flow = max(-0.2, flow) 

    last_error = error
    history.append([t[i], lvp, aop, flow, current_speed])

# 导出
df = pd.DataFrame(history, columns=['Time_s', 'LVP_mmHg', 'AoP_mmHg', 'Flow_Lmin', 'PumpSpeed_RPM'])
df.to_csv('vad_data.csv', index=False)
