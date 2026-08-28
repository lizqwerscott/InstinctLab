from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlConv2dHeadCfg,
    InstinctRlMlpCfg,
    InstinctRlActorCriticRecurrentCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@configclass
class DepthEncoderConv2dCfg(InstinctRlConv2dHeadCfg):
    output_size = 128
    channels = [4]
    kernel_sizes = [3]
    strides = [1]
    hidden_sizes = [256, 256]
    paddings = [1]
    nonlinearity = "ReLU"
    use_maxpool = True
    component_names = [
        "depth_image",
    ]


@configclass
class HeightScanEncoderMlpCfg(InstinctRlMlpCfg):
    output_size = 128
    hidden_sizes = [512, 256]
    nonlinearity = "SiLU"
    component_names = [
        "height_scan",
    ]


@configclass
class EncoderConfigs:
    height_scan_encoder = HeightScanEncoderMlpCfg()


@configclass
class RnnPolicyCfg(InstinctRlActorCriticRecurrentCfg):
    init_noise_std = 1.0
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    activation = "elu"
    rnn_type = "gru"


@configclass
class AlgorithmCfg(InstinctRlPpoAlgorithmCfg):
    class_name = "PPO"

    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.006

    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"

    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0


@configclass
class G1ParkourPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 5000

    experiment_name = "g1_parkour_flat_blind"
    resume = False
    load_run = ""

    empirical_normalization = False

    policy = RnnPolicyCfg()
    algorithm = AlgorithmCfg()
