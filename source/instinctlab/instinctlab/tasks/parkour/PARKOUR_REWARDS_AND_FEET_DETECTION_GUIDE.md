# Parkour任务 - 奖励和足端检测完整指南

## 目录
1. [奖励定义文件](#奖励定义文件)
2. [足端检测的核心实现](#足端检测的核心实现)
3. [文件导入结构](#文件导入结构)
4. [工作流程图](#工作流程图)
5. [相关配置参数](#相关配置参数)
6. [详细函数说明](#详细函数说明)

---

## 奖励定义文件

### A. 奖励函数实现

**文件**: `mdp/rewards.py`

共定义了 **8个足端相关的奖励函数**：

| 函数名 | 功能 | 关键实现 |
|------|------|---------|
| `feet_air_time()` | 奖励长步长，追踪腾空时间 | 使用 `contact_sensor.data.current_air_time` |
| `feet_orientation_contact()` | 接地时奖励足部垂直朝向 | 使用四元数投影重力向量 |
| `feet_at_plane()` | 奖励足部保持一定高度 | 使用高度扫描仪测量脚-地面距离 |
| `feet_close_xy_gauss()` | 惩罚足部距离过近 | 计算机器人坐标系中的Y向距离 |
| `stand_still()` | 无速度命令时惩罚运动 | 关节位置偏差计算 |
| `dont_wait()` | 有速度命令时不能静止 | 比较实际速度与命令速度 |
| `heading_error()` | 计算航向误差 | 角速度命令差异 |
| `link_orientation()` | 惩罚非平坦身体朝向 | L2范数平方核 |

#### 关键函数详解

##### 1. `feet_air_time()` 函数

```python
def feet_air_time(env, command_name: str, vel_threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.
    
    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.
    
    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
```

**工作原理**:
- 获取接触传感器数据
- 计算每只脚的腾空时间和接触时间
- 判断是否处于单脚支撑阶段 (`single_stance = torch.sum(in_contact.int(), dim=1) == 1`)
- 在单脚阶段奖励较长的步长
- 当速度命令很小时奖励为零

**参数**:
- `command_name`: 速度命令名称 (通常为 "base_velocity")
- `vel_threshold`: 速度阈值，用于判断是否真的在行走
- `sensor_cfg`: 接触传感器配置，指向 ankle_roll_link

---

##### 2. `feet_orientation_contact()` 函数

```python
def feet_orientation_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward feet being oriented vertically when in contact with the ground."""
```

**工作原理**:
- 获取左右脚的四元数
- 将重力向量投影到足部坐标系 (`quat_apply_inverse`)
- 检查是否接地 (通过接触力 `net_forces_w_history`)
- 计算足部与竖直方向的偏差
- 只在接地时给予奖励

**关键代码**:
```python
left_projected_gravity = quat_apply_inverse(left_quat, asset.data.GRAVITY_VEC_W)
# 投影重力向量的XY分量表示偏离竖直的角度
return torch.sum(torch.square(left_projected_gravity[:, :2]), dim=-1) ** 0.5 * is_contact[:, 0]
```

---

##### 3. `feet_at_plane()` 函数

```python
def feet_at_plane(
    env: ManagerBasedRLEnv,
    contact_sensor_cfg: SceneEntityCfg,
    left_height_scanner_cfg: SceneEntityCfg,
    right_height_scanner_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset=0.035,
) -> torch.Tensor:
    """Reward feet being at certain height above the ground plane."""
```

**工作原理**:
- 使用高度扫描仪 (height_scanner) 测量地面高度
- 获取足部实际位置
- 计算足部距离地面的高度
- 当接地时，奖励足部保持在 `height_offset` 高度
- 使用 clamp 限制奖励范围在 [0, 0.3]

**关键代码**:
```python
left_height = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
left_reward = torch.clamp(
    left_height.unsqueeze(-1) - left_sensor_data - height_offset, 
    min=0.0, 
    max=0.3
) * is_contact[:, 0:1]
```

---

##### 4. `feet_close_xy_gauss()` 函数

```python
def feet_close_xy_gauss(
    env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.1
) -> torch.Tensor:
    """Penalize when feet are too close together in the y distance."""
```

**工作原理**:
- 获取左右足在世界坐标系中的XY位置
- 将足部位置转换到机器人坐标系 (考虑机器人朝向)
- 计算两脚在机器人坐标系Y轴方向的距离
- 使用高斯衰减函数惩罚足部过近: `exp(-clamp(threshold - distance) / std²) - 1`

**坐标变换代码**:
```python
cos_heading = torch.cos(heading_w)
sin_heading = torch.sin(heading_w)
# 旋转到机器人坐标系
left_foot_robot_frame = torch.stack([
    cos_heading * left_foot_xy[:, 0] + sin_heading * left_foot_xy[:, 1],
    -sin_heading * left_foot_xy[:, 0] + cos_heading * left_foot_xy[:, 1],
], dim=1)
```

---

### B. 奖励配置

**文件**: `config/parkour_env_cfg.py` (第653-800行)

**`G1Rewards` 类** 配置了所有奖励项，包括：

#### 1. 任务奖励 (Task Rewards)
```python
track_lin_vel_xy_exp = RewTerm(
    func=mdp.track_lin_vel_xy_exp,
    weight=2.0,
    params={"command_name": "base_velocity", "std": 0.5},
)
track_ang_vel_z_exp = RewTerm(
    func=mdp.track_ang_vel_z_exp, 
    weight=2.0, 
    params={"command_name": "base_velocity", "std": 0.5}
)
heading_error = RewTerm(
    func=mdp.heading_error, 
    weight=-1.0, 
    params={"command_name": "base_velocity"}
)
is_alive = RewTerm(func=mdp.is_alive, weight=3.0)
```

#### 2. 正则化奖励 (Regularization Rewards)
```python
feet_air_time = RewTerm(
    func=mdp.feet_air_time,
    weight=0.5,
    params={
        "command_name": "base_velocity",
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        "vel_threshold": 0.15,
    },
)

feet_slide = RewTerm(
    func=mdp.contact_slide,
    weight=-0.4,
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        "threshold": 1.0,
    },
)

feet_flat_ori = RewTerm(
    func=mdp.feet_orientation_contact,
    weight=-0.4,
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    },
)

feet_at_plane = RewTerm(
    func=mdp.feet_at_plane,
    weight=-0.1,
    params={
        "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        "left_height_scanner_cfg": SceneEntityCfg("left_height_scanner"),
        "right_height_scanner_cfg": SceneEntityCfg("right_height_scanner"),
        "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        "height_offset": 0.035,
    },
)

feet_close_xy = RewTerm(
    func=mdp.feet_close_xy_gauss,
    weight=0.4,
    params={
        "threshold": 0.12,
        "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        "std": math.sqrt(0.05),
    },
)
```

#### 3. 安全奖励 (Safety Rewards)
```python
undesired_contacts = RewTerm(
    func=mdp.undesired_contacts,
    weight=-1.0,
    params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names="(?!.*_ankle_roll_link).*"),
        "threshold": 1.0,
    },
)
```

---

## 足端检测的核心实现

### A. 接触传感器定义

**文件**: `config/parkour_env_cfg.py` (第321行)

```python
contact_forces = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*",      # 监测所有机器人部件
    history_length=3,                          # 保存最近3帧历史
    track_air_time=True                        # 自动追踪腾空时间
)
```

**说明**:
- `prim_path`: 使用正则表达式匹配所有Robot下的物体
- `history_length=3`: 保存过去3个时间步的接触力数据，用于计算接触状态变化
- `track_air_time=True`: 启用自动腾空时间追踪，记录每个身体部件的连续接触/腾空时间

---

### B. 足部身体识别

#### 身体ID索引
```python
# 在reward函数中
left_foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :2]   # 左足
right_foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids[1], :2]  # 右足
```

#### 足部匹配模式
```python
# 在配置中使用正则表达式
body_names=".*_ankle_roll_link"
```

这个模式匹配：
- `left_ankle_roll_link`
- `right_ankle_roll_link`

---

### C. 足端检测的关键数据源

#### 1. 接地检测

**从接触力检测**:
```python
contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
net_contact_forces = contact_sensor.data.net_forces_w_history
# net_forces_w_history 形状: [num_envs, history_length, num_bodies, 3]
is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 1
# 如果力的范数 > 1N，则判定为接地
```

#### 2. 腾空/接触时间追踪

```python
air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]           # [num_envs, 2]
contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]   # [num_envs, 2]
in_contact = contact_time > 0.0                                                    # [num_envs, 2]
```

**数据说明**:
- `current_air_time`: 当前连续腾空的时间 (秒)
- `current_contact_time`: 当前连续接地的时间 (秒)
- 在转换时刻会重置

#### 3. 足部位置

```python
left_foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids[0], :2]
right_foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids[1], :2]
# 获取左右足的世界坐标系XY位置

left_foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2]
right_foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids[1], 2]
# 获取足部高度
```

#### 4. 足部朝向

```python
left_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]     # [num_envs, 4]
right_quat = asset.data.body_quat_w[:, asset_cfg.body_ids[1], :]
# 四元数表示: [x, y, z, w]

# 将重力向量投影到足部坐标系
left_projected_gravity = quat_apply_inverse(left_quat, asset.data.GRAVITY_VEC_W)
# 投影后的向量: [num_envs, 3]
# XY分量表示偏离竖直方向的角度

# 计算足部与竖直方向的偏差角
deviation_angle = torch.sum(torch.square(left_projected_gravity[:, :2]), dim=-1) ** 0.5
```

#### 5. 高度扫描数据

```python
left_sensor = env.scene[left_height_scanner_cfg.name]
left_sensor_data = left_sensor.data.ray_hits_w[..., 2]  # Z坐标（高度）
# 形状: [num_envs, num_rays]

right_sensor = env.scene[right_height_scanner_cfg.name]
right_sensor_data = right_sensor.data.ray_hits_w[..., 2]
# 处理无穷值（光线未击中）
right_sensor_data = torch.where(torch.isinf(right_sensor_data), 0.0, right_sensor_data)

# 计算足部与地面的距离
foot_ground_distance = left_height - left_sensor_data - height_offset
```

---

## 文件导入结构

### mdp/__init__.py

```python
from isaaclab.envs.mdp import *  # 导入IsaacLab基础MDP模块

from instinctlab.envs.mdp import *  # 导入InstinctLab自定义MDP模块

from .commands import *  # 导入命令相关函数
from .curriculums import *  # 导入课程学习相关
from .events import *  # 导入事件处理
from .rewards import *  # 导入所有奖励函数 ← 足端奖励在这里
from .terminations import *  # 导入终止条件
```

### 导入使用示例

在 `config/parkour_env_cfg.py` 中：

```python
import instinctlab.tasks.parkour.mdp as mdp

# 在G1Rewards中使用
feet_air_time = RewTerm(
    func=mdp.feet_air_time,  # 直接从mdp模块引用
    ...
)
```

---

## 工作流程图

### 环境运行流程

```
Environment.step(action)
    ↓
┌─────────────────────────────────┐
│ Physics Simulation              │
│ - 更新机器人状态                 │
│ - 检测接触                      │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ ContactSensor采集数据            │
│ - net_forces_w_history          │
│ - current_air_time              │
│ - current_contact_time          │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ 奖励函数查询传感器              │
│ - feet_air_time()               │
│ - feet_orientation_contact()    │
│ - feet_at_plane()               │
│ - feet_close_xy_gauss()         │
│ - ... (其他奖励函数)            │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ 提取足部数据                    │
│ - body_ids[0/1]                 │
│ - ankle_roll_link               │
│ - 位置、朝向、接地状态          │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ 计算奖励值                      │
│ - 应用配置的权重                │
│ - 求和得到总奖励                │
└─────────────────────────────────┘
    ↓
Return: (observation, reward, terminated, truncated, info)
```

### 足端检测数据流

```
ContactSensor (prim_path="Robot/.*")
    │
    ├─→ net_forces_w_history
    │   └─→ 接地状态判定 (is_contact)
    │
    ├─→ current_air_time
    │   └─→ 腾空时间奖励 (feet_air_time)
    │
    └─→ current_contact_time
        └─→ 步长计算
        └─→ 单脚支撑检测

Body Data (body_pos_w, body_quat_w)
    │
    ├─→ 位置 [x, y, z]
    │   ├─→ 足部Y距离 (feet_close_xy_gauss)
    │   └─→ 足部高度 (feet_at_plane)
    │
    └─→ 四元数 [x, y, z, w]
        └─→ 足部朝向 (feet_orientation_contact)

Height Scanner Data
    │
    └─→ 地面高度估计 (feet_at_plane)
```

---

## 相关配置参数

### 足端检测参数

| 参数 | 值 | 位置 | 含义 |
|------|-----|------|------|
| `body_names` | `.*_ankle_roll_link` | rewards.py | 足部身体名称模式 |
| `history_length` | 3 | parkour_env_cfg.py:321 | 接触传感器历史长度 |
| `track_air_time` | True | parkour_env_cfg.py:321 | 启用腾空时间追踪 |

### 奖励权重和参数

| 奖励项 | 权重 | 关键参数 | 含义 |
|------|------|---------|------|
| `feet_air_time` | 0.5 | vel_threshold=0.15 | 鼓励长步长 |
| `feet_slide` | -0.4 | threshold=1.0 | 惩罚足部滑动 |
| `feet_flat_ori` | -0.4 | - | 惩罚足部倾斜 |
| `feet_at_plane` | -0.1 | height_offset=0.035m | 保持足部高度 |
| `feet_close_xy` | 0.4 | threshold=0.12m | 维持足部间距 |

### 传感器配置

| 传感器 | 配置 | 用途 |
|------|------|------|
| `contact_forces` | ContactSensorCfg(all bodies) | 接地检测、接触力、腾空时间 |
| `left_height_scanner` | RayCasterCfg | 左足高度扫描 |
| `right_height_scanner` | RayCasterCfg | 右足高度扫描 |
| `leg_volume_points` | VolumePointsCfg | 腿部体积点检测 |

---

## 详细函数说明

### mdp/rewards.py 完整函数列表

#### 1. `feet_air_time()` ✓
**目的**: 奖励长步长和交替腾空
**关键输出**: 单脚支撑期间的最长腾空/接触时间

#### 2. `stand_still()`
**目的**: 无命令时惩罚运动
**关键输出**: 关节位置偏差 × 无速度命令指示

#### 3. `feet_close_xy_gauss()`
**目的**: 惩罚足部过近
**关键输出**: 高斯衰减惩罚值

#### 4. `heading_error()`
**目的**: 计算航向误差
**关键输出**: 角速度命令的绝对值

#### 5. `dont_wait()`
**目的**: 有速度命令时不能静止
**关键输出**: 基于实际速度的分级惩罚

#### 6. `feet_orientation_contact()` ✓
**目的**: 接地时奖励足部竖直朝向
**关键输出**: 朝向偏差 × 接地指示

#### 7. `feet_at_plane()` ✓
**目的**: 奖励足部保持适当高度
**关键输出**: 高度奖励的求和

#### 8. `link_orientation()`
**目的**: 惩罚非平坦身体朝向
**关键输出**: 投影重力向量的平方和

---

## 常见问题

### Q1: 为什么使用 `ankle_roll_link` 作为足部？
**A**: 脚踝滚动关节是足部最末端的关节，最接近实际接触地面的位置，能准确反映足部的接地状态和朝向。

### Q2: `contact_time > 0.0` 为什么能判定接地？
**A**: IsaacLab的ContactSensor会在有接触力时记录接触时间。`contact_time > 0.0` 意味着该帧有接触，但更精确的判定是通过接触力的范数 `> 1N`。

### Q3: 为什么需要高度扫描仪？
**A**: 高度扫描仪使用射线投射，可以穿过足部获取地面高度，而不会被足部本身遮挡。这使得 `feet_at_plane()` 能准确计算足部到地面的距离。

### Q4: 四元数投影有什么作用？
**A**: 通过 `quat_apply_inverse`，将重力向量从世界坐标系变换到足部局部坐标系。投影后的XY分量大小表示足部偏离竖直方向的角度，便于计算倾斜程度。

### Q5: 为什么 `feet_close_xy_gauss` 要转换到机器人坐标系？
**A**: 使用机器人坐标系使距离计算与机器人朝向无关，只关注腿部在身体方向上的相对位置。

---

## 扩展阅读

### 相关文件
- 命令系统: `mdp/commands/` 
- 事件处理: `mdp/events.py`
- 终止条件: `mdp/terminations.py`
- 课程学习: `mdp/curriculums.py`

### 依赖模块
- IsaacLab: 物理引擎和传感器
- InstinctLab: 自定义MDP和资源

---

**文档生成日期**: 2026年5月13日
**最后更新**: 与 parkour_env_cfg.py 和 mdp/rewards.py 同步
