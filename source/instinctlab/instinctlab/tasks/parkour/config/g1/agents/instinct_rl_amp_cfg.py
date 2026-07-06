import os

from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlConv2dHeadCfg,
    InstinctRlCrossAttnHeadCfg,
    InstinctRlMlpCfg,
    InstinctRlEncoderMoEActorCriticCfg,
    InstinctRlEncoderMoEActorCriticRecurrentCfg,
    InstinctRlOnPolicyRunnerCfg,
    InstinctRlPpoAlgorithmCfg,
)


@configclass
class DepthEncoderConv2dCfg(InstinctRlConv2dHeadCfg):
    output_size = 128
    channels = [32, 64, 128]
    kernel_sizes = [3, 3, 3]
    strides = [2, 2, 2]
    hidden_sizes = []
    paddings = [1, 1, 1]
    nonlinearity = "SiLU"
    use_maxpool = False
    final_nonlinearity = False
    component_names = [
        "depth_image",
    ]


@configclass
class DepthCrossAttnCfg(InstinctRlCrossAttnHeadCfg):
    """Proprioception-queried cross-attention depth encoder.

    Replaces the flatten+MLP Conv2d encoder: conv tokenizer -> self-attention over
    depth tokens -> cross-attention with a proprioceptive query. Only depth_image
    is taken out; the proprio components below still flow to the GRU.
    """

    output_size = 128
    channels = [32, 64, 128]  # channels[-1] = d_model, must be divisible by num_heads
    kernel_sizes = [3, 3, 3]
    strides = [2, 2, 2]
    paddings = [1, 1, 1]
    num_heads = 4
    num_self_attn_layers = 1
    ffn_expansion = 2
    # proprio -> query MLP hidden widths (output is always d_model=channels[-1]).
    # None => [channels[-1] * ffn_expansion] = [256]. e.g. [256, 256] for a deeper query MLP.
    info_hidden_sizes = None
    nonlinearity = "ELU"
    use_maxpool = False
    component_names = [
        "depth_image",
    ]
    takeout_input_components = True
    info_component_names = [
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
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
    # A/B switch: DepthEncoderConv2dCfg() for the flatten+MLP baseline,
    # DepthCrossAttnCfg() for the proprioception-queried cross-attention encoder.
    depth_image_encoder = DepthCrossAttnCfg()

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

    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1e-3

    lr_scheduler_class_name = "CosineAnnealingLR"
    lr_scheduler: dict = {
        "T_max": 30000,
        "eta_min": 1.0e-5,
    }

    teacher_act_prob = "linear"
    update_times_scale = 500

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
                "height_scan": (4 * 21 * 33,),
            },
            "critic": {
                "base_lin_vel": (24,),
                "base_ang_vel": (24,),
                "projected_gravity": (24,),
                "velocity_commands": (24,),
                "joint_pos_rel": (232,),
                "joint_vel_rel": (232,),
                "last_action": (232,),
                "height_scan": (4 * 21 * 33,),
            },
        },
        "num_moe_experts": 4,
        "moe_gate_hidden_dims": [],
        "num_actions": 29,
        "num_rewards": 1,
    }
    teacher_logdir = os.path.expanduser(
        "~/Data/20260603_112548"
    )
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.006
    denoise_loss_coef = 0.1
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
