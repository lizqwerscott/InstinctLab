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
class EncoderDepthConfigs:
    depth_image_encoder = DepthEncoderConv2dCfg()

@configclass
class EncoderConfigs:
    height_scan_encoder = HeightScanEncoderMlpCfg()


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
class MoEStudentPolicyCfg(InstinctRlEncoderMoEActorCriticCfg):
    init_noise_std = 1.0
    num_moe_experts = 4
    actor_hidden_dims = [256, 128, 64]
    critic_hidden_dims = [256, 128, 64]
    activation = "elu"
    encoder_configs = EncoderDepthConfigs()
    critic_encoder_configs = EncoderDepthConfigs()


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
    denoise_loss_coef = 0.1
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0


@configclass
class AmpAlgoStudentCfg(InstinctRlPpoAlgorithmCfg):
    class_name = "VaeDistill"

    kl_loss_func = "kl_divergence"
    kl_loss_coef = 1.0
    using_ppo = False

    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1e-3

    teacher_policy_class_name = MoEPolicyCfg().class_name

    teacher_policy: dict = {
        "init_noise_std": 1.0,
        "actor_hidden_dims": [256, 128, 64],
        "critic_hidden_dims": [256, 128, 64],
        "activation": "elu",
        "encoder_configs": {
            "height_scan_encoder": {
                "class_name": "MlpModel",
                "component_names": ["height_scan"],
                "output_size": 128,
                "takeout_input_components": True,
                "hidden_sizes": [512, 256],
                "nonlinearity": "SiLU",
            }
        },
        "critic_encoder_configs": {
            "height_scan_encoder": {
                "class_name": "MlpModel",
                "component_names": ["height_scan"],
                "output_size": 128,
                "takeout_input_components": True,
                "hidden_sizes": [512, 256],
                "nonlinearity": "SiLU",
            }
        },
        "obs_format": {
            "policy": {
                "base_ang_vel": (24,),
                "projected_gravity": (24,),
                "velocity_commands": (24,),
                "joint_pos_rel": (232,),
                "joint_vel_rel": (232,),
                "last_action": (232,),
                "height_scan": (4 * 21 * 33),
            },
            "critic": {
                "base_ang_vel": (24,),
                "projected_gravity": (24,),
                "velocity_commands": (24,),
                "joint_pos_rel": (232,),
                "joint_vel_rel": (232,),
                "last_action": (232,),
                "height_scan": (4 * 21 * 33),
            },
        },
        "num_actions": 29,
        "num_rewards": 1,
    }
    teacher_logdir = os.path.expanduser(
        "~/Data/instinctlab_logs/instinct_rl/g1_perceptive_shadowing/20260111_103654_g1Perceptive_4MotionsKneelClimbStep1_concatMotionBins__GPU0_from20260108_032900"
    )

    # source ppo

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
    denoise_loss_coef = 0.1
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
    experiment_name = "g1_parkour"
    resume = False
    load_run = ""
    empirical_normalization = False
    policy = MoEPolicyCfg()
    algorithm = AmpAlgoCfg()

@configclass
class G1ParkourStudentPPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 5000
    experiment_name = "g1_parkour_student"
    resume = False
    load_run = ""
    empirical_normalization = False
    policy = MoEStudentPolicyCfg()
    algorithm = AmpAlgoStudentCfg()
