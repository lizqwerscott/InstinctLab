from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlConv2dHeadCfg,
    InstinctRlEncoderMoEActorCriticCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@configclass
class DepthEncoderConv2dCfg(InstinctRlConv2dHeadCfg):
    # 3-layer CNN pyramid: 8x16x16 -> 32x8x8 -> 64x4x4 -> 128x4x4.
    # strides [2,2,1] downsample to 4x4 so the flatten head stays small
    # (128*4*4=2048 vs 16384 with no downsampling) -- keeps encoder ~0.36M params.
    output_size = 128
    channels = [16, 32, 64]
    kernel_sizes = [3, 3, 3]
    strides = [1, 2, 2]
    hidden_sizes = []  
    paddings = [1, 1, 1]
    nonlinearity = "SiLU"
    use_maxpool = False
    final_nonlinearity = False
    component_names = [
        "depth_image",
    ]


@configclass
class EncoderConfigs:
    depth_encoder = DepthEncoderConv2dCfg()


@configclass
class MoEPolicyCfg(InstinctRlEncoderMoEActorCriticCfg):
    init_noise_std = 1.0
    num_moe_experts = 4
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    activation = "elu"
    encoder_configs = EncoderConfigs()
    critic_encoder_configs = EncoderConfigs()


@configclass
class AmpAlgoCfg(InstinctRlPpoAlgorithmCfg):
    class_name = "WasabiPPO"
    discriminator_kwargs = {
        "hidden_sizes": [1024, 512],
        "nonlinearity": "ReLU",
    }

    discriminator_reward_coef = 0.25
    discriminator_reward_type = "quad"
    discriminator_loss_func = "MSELoss"
    discriminator_gradient_penalty_coef = 5.0
    discriminator_optimizer_class_name = "AdamW"
    discriminator_weight_decay_coef = 3e-4
    discriminator_logit_weight_decay_coef = 0.04
    discriminator_optimizer_kwargs = {
        "lr": 1.0e-4,
        "betas": [0.9, 0.999],
    }
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.006
    denoise_loss_coef = 1e-2
    feature_kl_loss_coef = 1e-3
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
    save_interval = 2000
    experiment_name = "g1_parkour"
    resume = False
    load_run = ""
    empirical_normalization = False
    policy = MoEPolicyCfg()
    algorithm = AmpAlgoCfg()
