from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlConv2dHeadCfg,
    InstinctRlMlpCfg,
    InstinctRlEncoderMoEActorCriticCfg,
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
class MoEPolicyCfg(InstinctRlEncoderMoEActorCriticCfg):
    # Recurrent variant: encoder latent + proprio -> GRU -> MoE actor/critic heads.
    # The plain "EncoderMoEActorCritic" silently drops the rnn_* fields below.
    class_name = "EncoderMoEActorCriticRecurrent"
    init_noise_std = 1.0
    num_moe_experts = 4
    moe_gate_hidden_dims = [128]
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"
    rnn_type = "gru"
    rnn_hidden_size = 256
    rnn_num_layers = 1
    encoder_configs = EncoderConfigs()
    critic_encoder_configs = EncoderConfigs()


@configclass
class AmpAlgoCfg(InstinctRlPpoAlgorithmCfg):
    class_name = "WasabiPPO"
    # Multi-AMP: one discriminator per style. Per-env style ids come from the `amp_style`
    # observation group (see AmpStyleObsCfg in parkour_env_cfg.py):
    #   style 0 = flat (stand / walk / turn), style 1 = stairs (up / down).
    num_styles = 2
    discriminator_kwargs = {
        "hidden_sizes": [1024, 512],
        "nonlinearity": "ReLU",
        # Running input normalization (standard AMP practice): the state features have mixed
        # scales, and each style's discriminator keeps its own statistics so the two style
        # rewards stay in comparable operating ranges.
        "normalizer_class_name": "EmpiricalNormalization",
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
        # The explicit discriminator_(logit_)weight_decay losses above are the only intended
        # regularizers; AdamW's default decay (0.01) would silently stack on top of them.
        "weight_decay": 0.0,
    }
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
    save_interval = 2000
    experiment_name = "g1_parkour"
    resume = False
    load_run = ""
    empirical_normalization = False
    policy = MoEPolicyCfg()
    algorithm = AmpAlgoCfg()
