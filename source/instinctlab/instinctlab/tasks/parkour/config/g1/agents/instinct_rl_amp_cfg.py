import os

from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlConv2dHeadCfg,
    InstinctRlMlpCfg,
    InstinctRlEncoderMoEActorCriticCfg,
    InstinctRlEncoderMoEActorCriticRecurrentCfg,
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
class MoEPolicyCfg(InstinctRlEncoderMoEActorCriticRecurrentCfg):
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
class MoEStudentPolicyCfg(InstinctRlEncoderMoEActorCriticRecurrentCfg):
    init_noise_std = 0.1
    num_moe_experts = 4
    moe_gate_hidden_dims = [128]
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [256, 128, 64]
    activation = "elu"
    rnn_type = "gru"
    rnn_hidden_size = 256
    rnn_num_layers = 1
    encoder_configs = EncoderDepthConfigs()
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
    class_name = "TPPO"

    # using_ppo=False => pure distillation (DAgger): only the teacher-action imitation loss
    # is optimized, no PPO/GAE/value/discriminator terms.
    using_ppo = False

    # Behavior-cloning loss form: squared L2  ||mu_student - mu_teacher||^2  (paper eq. 8).
    # TPPO default "real" is the (non-squared) L2 norm; "mse_sum" = sum of squared diffs.
    distill_target = "mse_sum"

    # TPPO aggregates each entry in its loss dictionary with the matching
    # ``<loss_name>_coef`` attribute.
    distillation_loss_coef = 1.0
    encoder_distillation_loss_coef = 0.05
    encoder_distillation_student_component = "depth_image_encoder"
    encoder_distillation_teacher_component = "height_scan_encoder"

    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1e-3

    # One-Cycle LR schedule (paper Table XII): initial lr 1e-3, div_factor=10 (peak/init)
    # => max_lr=1e-2, final_div_factor=50 (final/init). TPPO steps the scheduler once per
    # update() (i.e. once per learning iteration), so total_steps MUST match the runner's
    # max_iterations (G1ParkourStudentPPORunnerCfg.max_iterations) to avoid over-stepping.
    lr_scheduler_class_name = "OneCycleLR"
    lr_scheduler: dict = {
        "max_lr": 1.0e-2,
        "total_steps": 30000,
        "div_factor": 10.0,
        "final_div_factor": 50.0,
    }

    teacher_policy_class_name = MoEPolicyCfg().class_name

    teacher_policy: dict = {
        "init_noise_std": 1.0,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
        "rnn_type": "gru",
        "rnn_hidden_size": 256,
        "rnn_num_layers": 1,
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
                "base_ang_vel": (3,),
                "projected_gravity": (3,),
                "velocity_commands": (3,),
                "joint_pos_rel": (29,),
                "joint_vel_rel": (29,),
                "last_action": (29,),
                "height_scan": (21 * 33,),
            },
            "critic": {
                "base_lin_vel": (3,),
                "base_ang_vel": (3,),
                "projected_gravity": (3,),
                "velocity_commands": (3,),
                "joint_pos": (29,),
                "joint_vel": (29,),
                "actions": (29,),
                "height_scan": (21 * 33,),
            },
        },
        "num_moe_experts": 4,
        "moe_gate_hidden_dims": [128],
        "num_actions": 29,
        "num_rewards": 1,
    }
    teacher_logdir = os.path.expanduser(
        "~/MyProject/InstinctLab/logs/20260822_112657_from20260713_091503"
    )
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

@configclass
class G1ParkourStudentFinetunePPORunnerCfg(InstinctRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 5000
    experiment_name = "g1_parkour_student_finetune"
    resume = False
    load_run = ""
    empirical_normalization = False
    policy = MoEStudentPolicyCfg()
    algorithm = AmpAlgoCfg()
