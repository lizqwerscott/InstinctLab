# 足端边缘检测详细说明 - Volume Points Penetration System

## 目录
1. [概述](#概述)
2. [核心文件位置](#核心文件位置)
3. [系统架构](#系统架构)
4. [详细函数解释](#详细函数解释)
5. [配置详解](#配置详解)
6. [工作原理](#工作原理)
7. [应用场景](#应用场景)

---

## 概述

足端边缘检测是一个**体积点穿透检测系统**，用于监测机器人足部（足踝周围）是否穿透环境障碍物或地形。

### 核心概念

- **体积点(Volume Points)**: 在足部周围生成的3D网格点集
- **穿透(Penetration)**: 体积点进入障碍物或地形的深度
- **边缘检测**: 通过检测穿透深度来识别足部与环境的交界处

### 作用

1. **安全性约束**: 防止足部穿透地形
2. **接触质量控制**: 确保足部与地面进行有效接触
3. **运动学反馈**: 提供足部与环境的精确交互信息

---

## 核心文件位置

### 1. 足端边缘检测函数定义

**文件**: `instinctlab/envs/mdp/rewards/volume_points.py`

包含两个关键函数：
- `volume_points_penetration()` - 穿透惩罚
- `step_safety()` - 步态安全奖励

### 2. 体积点传感器配置

**文件**: `tasks/parkour/config/parkour_env_cfg.py` (第322-337行)

```python
leg_volume_points = VolumePointsCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
    points_generator=Grid3dPointsGeneratorCfg(
        x_min=-0.025,
        x_max=0.12,
        x_num=10,
        y_min=-0.03,
        y_max=0.03,
        y_num=5,
        z_min=-0.04,
        z_max=0.0,
        z_num=2,
    ),
    debug_vis=False,
)
```

### 3. 奖励项配置

**文件**: `tasks/parkour/config/parkour_env_cfg.py` (第671-676行)

```python
volume_points_penetration = RewTerm(
    func=mdp.volume_points_penetration,
    weight=-4.0,
    params={
        "sensor_cfg": SceneEntityCfg("leg_volume_points"),
    },
)
```

---

## 系统架构

### 数据流图

```
┌─────────────────────────────────────────┐
│ 足部体积点传感器(leg_volume_points)    │
│ - 位置: ankle_roll_link周围              │
│ - 形状: 3D网格点集 (10×5×2 = 100点)   │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 物理引擎碰撞检测                        │
│ - 检测每个体积点是否穿透               │
│ - 计算穿透深度                         │
│ - 计算点的速度                         │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 传感器数据输出                          │
│ - penetration_offset: 穿透偏移向量      │
│ - points_vel_w: 点的世界速度           │
│ - points_contact: 接触状态             │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ 奖励函数                                │
│ - volume_points_penetration()           │
│ - step_safety()                         │
└────────────────┬────────────────────────┘
                 │
                 ▼
         ┌────────────────┐
         │ 奖励/惩罚值    │
         └────────────────┘
```

### 组件关系

```
TaskConfig (parkour_env_cfg.py)
    │
    ├─→ SceneConfig
    │   └─→ leg_volume_points (VolumePointsCfg)
    │       ├─→ 传感器类型: VolumePoints
    │       └─→ 点生成器: Grid3dPointsGeneratorCfg
    │
    ├─→ RewardsConfig
    │   └─→ volume_points_penetration (RewTerm)
    │       ├─→ 函数: mdp.volume_points_penetration()
    │       ├─→ 权重: -4.0
    │       └─→ 参数: sensor_cfg="leg_volume_points"
    │
    └─→ MDPModule (instinctlab/envs/mdp)
        └─→ rewards/volume_points.py
            ├─→ volume_points_penetration()
            └─→ step_safety()
```

---

## 详细函数解释

### 1. `volume_points_penetration()` 函数

**文件**: `instinctlab/envs/mdp/rewards/volume_points.py`

#### 函数签名

```python
def volume_points_penetration(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    tolerance: float = 0.0
) -> torch.Tensor:
    """Penalize the penetration of volume points into the environment."""
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `env` | ManagerBasedRLEnv | - | 环境实例 |
| `sensor_cfg` | SceneEntityCfg | - | 体积点传感器配置 |
| `tolerance` | float | 0.0 | 穿透容限 (m)，小于此值不计算惩罚 |

#### 完整代码

```python
def volume_points_penetration(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    tolerance: float = 0.0
) -> torch.Tensor:
    """Penalize the penetration of volume points into the environment."""
    # 获取体积点传感器
    volume_sensor: VolumePoints = env.scene.sensors[sensor_cfg.name]
    
    # 获取穿透偏移向量: (N, B_, P_, 3)
    # N = batch size (环境数量)
    # B_ = 传感器数量 (不同值)
    # P_ = 每个传感器的点数 (不同值)
    # 3 = 3D向量 [x, y, z]
    penetration = volume_sensor.data.penetration_offset
    
    # 展平B_和P_维度: (N, B_*P_, 3)
    penetration = penetration.flatten(1, 2)
    
    # 计算穿透深度(范数): (N, B_*P_)
    penetration_depth = torch.norm(penetration, dim=-1)
    
    # 判断是否穿透(超过容限): (N, B_*P_)
    in_obstacle = (penetration_depth > tolerance).float()
    
    # 获取体积点的世界速度: (N, B_, P_, 3)
    points_vel = volume_sensor.data.points_vel_w
    
    # 展平B_和P_维度: (N, B_*P_, 3)
    points_vel = points_vel.flatten(1, 2)
    
    # 计算速度范数: (N, B_*P_)
    points_vel_norm = torch.norm(points_vel, dim=-1)
    
    # 计算惩罚: 速度 × 穿透深度 × 是否穿透
    # 物理含义: 穿透越深且运动越快 → 惩罚越大
    velocity_times_penetration = in_obstacle * (points_vel_norm + 1e-6) * penetration_depth
    
    # 对所有点求和: (N,)
    return torch.sum(velocity_times_penetration, dim=-1)
```

#### 工作原理详解

##### 步骤1: 获取穿透数据

```python
penetration = volume_sensor.data.penetration_offset  # 形状: [N, B_, P_, 3]
```

- 每个体积点的穿透向量(从碰撞点指向体积点)
- 向量大小 = 穿透深度
- 向量方向 = 穿透方向

#### 步骤2: 计算穿透深度

```python
penetration_depth = torch.norm(penetration, dim=-1)  # 形状: [N, B_*P_]
# 例如: penetration_depth[0, 50] = 0.015 表示第0个环境的第50个点穿透了15mm
```

#### 步骤3: 判断是否穿透

```python
in_obstacle = (penetration_depth > tolerance).float()  # 0 或 1
# tolerance=0 时，所有 > 0 的穿透都被计算
# tolerance=0.01 时，只有 > 0.01m 的穿透才被计算
```

#### 步骤4: 获取点的运动速度

```python
points_vel = volume_sensor.data.points_vel_w  # 形状: [N, B_, P_, 3]
points_vel_norm = torch.norm(points_vel, dim=-1)  # 形状: [N, B_*P_]
# 例如: points_vel_norm[0, 50] = 0.5 表示第0个环境的第50个点以0.5 m/s运动
```

#### 步骤5: 计算复合惩罚

```python
velocity_times_penetration = in_obstacle * (points_vel_norm + 1e-6) * penetration_depth
# 三个因子相乘:
# 1. in_obstacle: 0 或 1 (是否穿透)
# 2. (points_vel_norm + 1e-6): 运动速度 (防止除0)
# 3. penetration_depth: 穿透深度
```

**惩罚公式的物理含义**:

$$\text{Penalty}_i = \begin{cases} 
(v_i + \epsilon) \cdot d_i & \text{if } d_i > \text{tolerance} \\
0 & \text{otherwise}
\end{cases}$$

其中:
- $v_i$ = 第i个点的速度
- $d_i$ = 第i个点的穿透深度
- $\epsilon$ = 1e-6 (数值稳定性)

**关键特性**:
- 穿透越深 → 惩罚越大
- 运动越快 → 惩罚越大
- 静止不动的穿透惩罚很小
- 快速穿透会被严重惩罚

#### 步骤6: 求和得到环境奖励

```python
reward = torch.sum(velocity_times_penetration, dim=-1)  # 形状: [N]
# 对单个环境中所有穿透的点进行求和
# 返回 [N] 形状的张量，每个元素代表一个环境的奖励
```

#### 使用示例

```python
# 在配置中
volume_points_penetration = RewTerm(
    func=mdp.volume_points_penetration,
    weight=-4.0,  # 负权重 = 惩罚
    params={
        "sensor_cfg": SceneEntityCfg("leg_volume_points"),
        "tolerance": 0.0,  # 无容限，任何穿透都被计算
    },
)

# 在奖励计算中
reward = -4.0 * volume_points_penetration(env, sensor_cfg)
# 穿透惩罚乘以权重后加入总奖励
```

---

### 2. `step_safety()` 函数

**文件**: `instinctlab/envs/mdp/rewards/volume_points.py`

#### 函数签名

```python
def step_safety(
    env: ManagerBasedRLEnv,
    volume_points_cfg: SceneEntityCfg,
    contact_forces_cfg: SceneEntityCfg,
    epsilon: float = 1e-5,
    once: bool = False,
) -> torch.Tensor:
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `env` | ManagerBasedRLEnv | - | 环境实例 |
| `volume_points_cfg` | SceneEntityCfg | - | 体积点传感器配置 |
| `contact_forces_cfg` | SceneEntityCfg | - | 接触力传感器配置 |
| `epsilon` | float | 1e-5 | 对数稳定性常数 |
| `once` | bool | False | 仅检测第一次接触还是任何接触 |

#### 完整代码

```python
def step_safety(
    env: ManagerBasedRLEnv,
    volume_points_cfg: SceneEntityCfg,
    contact_forces_cfg: SceneEntityCfg,
    epsilon: float = 1e-5,
    once: bool = False,
) -> torch.Tensor:
    """A log based reward to encourage the robot to make contacts with no penetration to the virtual obstacles.
    
    Inspired by Deep Tracking Control and Robot Parkour Learning and Humanoid Parkour Learning.
    NOTE: make sure the contact forces sensor is selected for that the volume points sensors are interested in.
    aka. The number of selected bodies in the contact forces sensor should be the same as the number of selected bodies
    in all volume points sensors.
    NOTE: Be aware of the body order.
    """
    # 获取体积点传感器
    volume_sensor: VolumePoints = env.scene.sensors[volume_points_cfg.name]
    
    # 获取接触力传感器
    contact_sensor: ContactSensor = env.scene.sensors[contact_forces_cfg.name]
    
    # 获取穿透偏移: (N, B_, P_, 3)
    penetration = volume_sensor.data.penetration_offset
    
    # 计算穿透深度: (N, B_, P_)
    penetration_depth = torch.norm(penetration, dim=-1)
    
    # 获取每个身体的最大穿透深度: (N, B_)
    penetration_depth_max = torch.max(penetration_depth, dim=-1)[0]
    
    # 判断是否接触
    if once:
        # 仅检测第一次接触
        contacts = contact_sensor.compute_first_contact(env.step_dt)[:, contact_forces_cfg.body_ids]  # (N, B_)
    else:
        # 检测任何接触 (通过接触力历史)
        contact_forces = contact_sensor.data.net_forces_w_history[:, :, contact_forces_cfg.body_ids, :]  # (N, T, B_, 3)
        # 如果接触力范数 > 1N，则判定为接触
        contacts = torch.norm(contact_forces, dim=-1).max(dim=1)[0] > 1.0  # (N, B_)
    
    # 计算奖励: -log(penetration_depth + epsilon) * contacts
    # 只有在接触时才给奖励
    # 穿透深度越小 → log值越小 → 奖励越大
    # 穿透深度越大 → log值越大 → 奖励越小(惩罚)
    rewards = -torch.log(penetration_depth_max + epsilon) * contacts  # (N, B_)
    
    # 对所有身体求和
    return torch.sum(rewards, dim=-1)  # (N,)
```

#### 工作原理详解

##### 对数基奖励函数

```
奖励 = -log(穿透深度 + epsilon) × 接触状态
```

**特性分析**:

| 穿透深度 | log值 | -log值 | 接触 | 奖励 |
|---------|-------|--------|------|------|
| 0.001m | -6.91 | +6.91 | 1 | +6.91 |
| 0.01m | -4.61 | +4.61 | 1 | +4.61 |
| 0.1m | -2.30 | +2.30 | 1 | +2.30 |
| 1m | 0 | 0 | 1 | 0 |
| 未接触 | - | - | 0 | 0 |

**关键性质**:
- 穿透越小 → 奖励越大
- 完全不穿透($d \to 0$) → 奖励 → +∞
- 穿透1m → 奖励 = 0
- 未接触 → 奖励 = 0

##### 两种接触检测模式

**模式1: `once=True` (第一次接触)**

```python
contacts = contact_sensor.compute_first_contact(env.step_dt)[:, contact_forces_cfg.body_ids]
# 仅在该步首次接触时返回True
# 适合于奖励"发起新接触"的行为
```

**模式2: `once=False` (任何接触)**

```python
contact_forces = contact_sensor.data.net_forces_w_history[:, :, contact_forces_cfg.body_ids, :]
# 检查所有历史帧的接触力
# 如果任何帧的力 > 1N，则判定为接触
# 适合于奖励"维持接触"的行为
```

##### 完整示例

```python
# 环境1: 足部与地面接触，穿透深度 = 0.005m
penetration_depth[0] = 0.005
contacts[0] = True
reward[0] = -log(0.005 + 1e-5) × 1 = -log(0.005) = 5.30

# 环境2: 足部与地面接触，穿透深度 = 0.05m
penetration_depth[1] = 0.05
contacts[1] = True
reward[1] = -log(0.05 + 1e-5) × 1 = -log(0.05) = 2.99

# 环境3: 足部未接触地面
contacts[2] = False
reward[2] = -log(penetration) × 0 = 0
```

---

## 配置详解

### 1. 体积点传感器配置

**文件**: `tasks/parkour/config/parkour_env_cfg.py` (第322-337行)

```python
leg_volume_points = VolumePointsCfg(
    # 传感器应用位置: 所有环境下两个足部的ankle_roll_link
    prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
    
    # 点生成器: 生成3D网格
    points_generator=Grid3dPointsGeneratorCfg(
        # X轴范围 (前后) [-25mm, +120mm]
        x_min=-0.025,
        x_max=0.12,
        x_num=10,  # X方向10个点
        
        # Y轴范围 (左右) [-30mm, +30mm]
        y_min=-0.03,
        y_max=0.03,
        y_num=5,   # Y方向5个点
        
        # Z轴范围 (上下) [-40mm, 0mm]
        z_min=-0.04,
        z_max=0.0,
        z_num=2,   # Z方向2个点
        
        # 总点数 = 10 × 5 × 2 = 100个点
    ),
    
    # 调试可视化
    debug_vis=False,  # 设置为True时在SimUI中显示点
)
```

### 2. 网格点覆盖范围可视化

```
俯视图 (从上往下看):
    Y (左右)
    ^
    |  y_num=5, y_range=[-30, 30]mm
    |  ●●●●●
    |  ●●●●●
    |  ●●●●●
    |  ●●●●●
    |  ●●●●●
    +-------► X (前后)
       x_num=10, x_range=[-25, 120]mm

侧视图 (从侧面看):
    Z (上下)
    ^
    |  z_num=2, z_range=[-40, 0]mm
    | ●●●●●●●●●●
    | ●●●●●●●●●●
    +------------► X (前后)
```

### 3. 配置参数的物理含义

| 配置 | 值 | 含义 | 用途 |
|------|-----|------|------|
| x_min | -0.025 | 足部向后25mm | 捕捉足部后缘 |
| x_max | 0.12 | 足部向前120mm | 捕捉足部前缘 |
| x_num | 10 | X方向采样密度 | 检测前后边缘 |
| y_min | -0.03 | 足部向左30mm | 捕捉足部左缘 |
| y_max | 0.03 | 足部向右30mm | 捕捉足部右缘 |
| y_num | 5 | Y方向采样密度 | 检测左右边缘 |
| z_min | -0.04 | 足部向下40mm | 捕捉足部底面 |
| z_max | 0.0 | 足部上表面 | 捕捉足部顶面 |
| z_num | 2 | Z方向采样密度 | 2层:底面和中间 |

### 4. 奖励项配置

**文件**: `tasks/parkour/config/parkour_env_cfg.py` (第671-676行)

```python
volume_points_penetration = RewTerm(
    # 使用的奖励函数
    func=mdp.volume_points_penetration,
    
    # 权重 = -4.0 (负数表示惩罚)
    # 权重大小表示此奖励项的重要程度
    # 穿透惩罚在所有奖励中的比重较大
    weight=-4.0,
    
    # 函数参数
    params={
        "sensor_cfg": SceneEntityCfg("leg_volume_points"),
        # "tolerance": 0.0,  # 可选: 穿透容限
    },
)
```

### 5. 在G1Rewards中的位置

```python
class G1Rewards:
    """Reward terms for the MDP."""

    # Task rewards (任务奖励)
    track_lin_vel_xy_exp = ...
    track_ang_vel_z_exp = ...
    
    # Regularization rewards (正则化奖励)
    volume_points_penetration = RewTerm(...)  # ← 在这里
    feet_air_time = ...
    feet_slide = ...
    ...
    
    # Safety rewards (安全奖励)
    dof_pos_limits = ...
    ...
```

---

## 工作原理

### 完整执行流程

```
┌─────────────────────────────────┐
│ Environment.step(action)        │
│ 1. 执行动作                      │
│ 2. 物理模拟                      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 传感器更新                       │
│ - ContactSensor                  │
│ - VolumePoints                   │
│   ├─→ 检测穿透                  │
│   ├─→ 计算穿透深度              │
│   └─→ 获取点速度                │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 奖励计算                         │
│ rewards = ∑ weight_i × term_i() │
│                                 │
│ volume_points_penetration:      │
│ = -4.0 × velocity × penetration │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│ 返回step结果                     │
│ (obs, reward, done, info)       │
└─────────────────────────────────┘
```

### 穿透检测的物理过程

```
1️⃣  生成体积点
    足部(ankle_roll_link)
         │
         ▼
    Grid3D点生成器 (10×5×2)
         │
         ▼
    100个点围绕足部分布

2️⃣  碰撞检测
    物理引擎检测100个点与环境的碰撞
         │
         ▼
    对于每个穿透的点:
    - 计算穿透向量(碰撞点 → 点)
    - 计算穿透深度(向量长度)
    - 记录穿透方向(向量方向)

3️⃣  速度计算
    体积点速度 = 足部速度 + 足部旋转对该点的贡献
    (由于足部运动，点的速度会变化)

4️⃣  惩罚计算
    对于每个穿透的点:
    惩罚_i = 速度_i × 穿透深度_i
    
    总惩罚 = ∑ 惩罚_i

5️⃣  奖励计算
    奖励 = -4.0 × 总惩罚
    (负权重将惩罚转化为奖励)
```

### 与其他足端奖励的关系

```
┌────────────────────────────────────────┐
│ 足部/足端相关奖励项                    │
├────────────────────────────────────────┤
│                                         │
│ 1. feet_air_time (0.5权重)             │
│    └─→ 奖励长步长                      │
│                                         │
│ 2. feet_slide (-0.4权重)               │
│    └─→ 惩罚足部滑动                    │
│                                         │
│ 3. feet_flat_ori (-0.4权重)            │
│    └─→ 惩罚足部倾斜                    │
│                                         │
│ 4. feet_at_plane (-0.1权重)            │
│    └─→ 保持足部高度                    │
│                                         │
│ 5. feet_close_xy (0.4权重)             │
│    └─→ 维持足部间距                    │
│                                         │
│ 6. volume_points_penetration (-4.0权重) ◄── 最大权重！
│    └─→ 防止足部穿透  ✓ 关键约束        │
│                                         │
└────────────────────────────────────────┘
```

---

## 应用场景

### 1. 不规则地形行走

```
场景: 机器人在复杂地形上行走
      /\     /\
     /  \   /  \
    /    \ /    \
   ───────────────

体积点检测:
- 足部接触地形凹陷处时 → 穿透 → 惩罚
- 促使机器人抬高脚或选择更好的踏脚位置
```

### 2. 梯阶攀登

```
场景: 机器人攀登楼梯
    ┌─────┐
    │     │
┌───┘     └───┐
│             │
└─────────────┘

体积点检测:
- 足部进入梯阶空隙时 → 穿透 → 惩罚
- 促使机器人在梯阶表面着陆
```

### 3. 跨越障碍

```
场景: 机器人跨越圆柱形障碍
      ╱╲╱╲╱╲╱╲
     ╱   │   ╲
    ╱    │    ╲

体积点检测:
- 足部与圆柱接触但不穿透 → 接触良好
- 足部穿透圆柱 → 惩罚（不允许穿透）
```

### 4. 精确脚放置

```
场景: 在狭窄平台上精确着陆
      ┌───────┐
      │       │
      │ ●●●  │  (●代表允许着陆区域)
      │       │
      └───────┘

体积点检测:
- 足部点大多数在平台上 → 无穿透 → 奖励
- 足部点穿透边缘 → 穿透 → 惩罚
```

---

## 高级配置

### 自定义穿透容限

```python
# 允许0.5mm的穿透而不产生惩罚
volume_points_penetration = RewTerm(
    func=mdp.volume_points_penetration,
    weight=-4.0,
    params={
        "sensor_cfg": SceneEntityCfg("leg_volume_points"),
        "tolerance": 0.0005,  # 0.5mm
    },
)
```

### 步态安全奖励

```python
# 使用step_safety替代volume_points_penetration
step_safety = RewTerm(
    func=mdp.step_safety,
    weight=1.0,  # 正权重 = 奖励
    params={
        "volume_points_cfg": SceneEntityCfg("leg_volume_points"),
        "contact_forces_cfg": SceneEntityCfg("contact_forces"),
        "epsilon": 1e-5,
        "once": False,
    },
)
```

### 调试可视化

```python
# 在配置中启用可视化
leg_volume_points = VolumePointsCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
    points_generator=Grid3dPointsGeneratorCfg(...),
    debug_vis=True,  # 启用！
)
```

启用后在SimUI中会看到：
- 绿色点 = 未穿透的点
- 红色点 = 穿透的点
- 箭头方向 = 穿透方向
- 箭头长度 = 穿透深度

---

## 常见问题

### Q1: 为什么穿透惩罚权重是-4.0？

**A**: 权重-4.0相对较大，表示穿透是严重的约束。这确保：
- 模型学会避免穿透
- 穿透惩罚足以覆盖其他较小的奖励项
- 在所有奖励项中优先避免穿透

### Q2: 体积点数(10×5×2=100)为什么这么设置？

**A**: 这是计算效率和检测精度的平衡：
- X方向10个点：捕捉前后边缘
- Y方向5个点：捕捉左右边缘  
- Z方向2个点：捕捉上下边缘
- 总共100个点，检测充分但不过度

### Q3: 穿透速度为什么也要考虑？

**A**: 物理含义明确：
- 静止穿透：可能是瞬时的，影响小
- 快速穿透：违反约束更严重，惩罚更大
- 这鼓励机器人小心地放置足部

### Q4: VolumePoints和ContactSensor有什么区别？

| 特性 | VolumePoints | ContactSensor |
|------|-------------|---------------|
| 检测类型 | 穿透深度/偏移 | 接触力/时间 |
| 分辨率 | 多点采样(100点) | 每个body一次 |
| 边缘检测 | 精确 | 粗糙 |
| 用途 | 防止穿透 | 检测接地 |

### Q5: 为什么穿透向量是从碰撞点指向体积点？

**A**: 这样设计的优点：
- 穿透向量方向 = 需要移动的方向
- 穿透深度 = 需要移动的距离
- 可以用来推断运动学反馈

---

## 性能优化建议

### 1. 减少点数以提高性能

```python
Grid3dPointsGeneratorCfg(
    x_min=-0.025, x_max=0.12, x_num=5,   # 减少
    y_min=-0.03,  y_max=0.03,  y_num=3,   # 减少
    z_min=-0.04,  z_max=0.0,   z_num=1,   # 减少
)
# 点数: 5×3×1 = 15 (原来100)
```

### 2. 增加点数以提高精度

```python
Grid3dPointsGeneratorCfg(
    x_min=-0.025, x_max=0.12, x_num=15,  # 增加
    y_min=-0.03,  y_max=0.03,  y_num=8,   # 增加
    z_min=-0.04,  z_max=0.0,   z_num=3,   # 增加
)
# 点数: 15×8×3 = 360 (原来100)
```

### 3. 条件化计算

```python
# 只在足部接近地面时计算穿透
if foot_height < height_threshold:
    penetration_reward = mdp.volume_points_penetration(env, cfg)
else:
    penetration_reward = torch.zeros(env.num_envs)
```

---

## 总结

足端边缘检测系统通过：

1. **体积点传感器**: 在足部周围生成100个采样点
2. **穿透检测**: 检测这些点是否进入障碍物
3. **速度感知**: 考虑穿透时的运动速度
4. **奖励惩罚**: 通过-4.0权重严格约束穿透

**效果**:
✓ 防止足部穿透地形
✓ 提高接触质量
✓ 增强运动稳定性
✓ 提供精确的边缘检测反馈

---

**文档版本**: 1.0  
**最后更新**: 2026年5月13日  
**相关文件**: 
- `instinctlab/envs/mdp/rewards/volume_points.py`
- `tasks/parkour/config/parkour_env_cfg.py`
