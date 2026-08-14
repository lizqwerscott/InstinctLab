import os

from isaaclab.utils import configclass

from instinctlab.utils.wrappers.instinct_rl import (
    InstinctRlConv2dHeadCfg,
    InstinctRlCrossAttnHeadCfg,
    InstinctRlDistillationAlgorithmCfg,
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


TEACHER_LOGDIR = os.path.expanduser(
    "/home/zhipengli/parkour/InstinctLab/logs/instinct_rl/g1_parkour/20260709_121005"
)

# The teacher's own policy config, shared by the TPPO and Distillation student configs.
TEACHER_POLICY: dict = {
    "init_noise_std": 1.0,
    "actor_hidden_dims": [512, 256, 128],
    "critic_hidden_dims": [512, 256, 128],
    "activation": "elu",
    "num_moe_experts": 4,
    "moe_gate_hidden_dims": [128],
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
            "joint_pos_rel": (29,),
            "joint_vel_rel": (29,),
            "last_action": (29,),
            "height_scan": (21 * 33,),
        },
    },
    "num_actions": 29,
    "num_rewards": 1,
}


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
    # buffer_dilation_ratio = 2.0
    num_mini_batches = 4
    learning_rate = 1e-3

    lr_scheduler_class_name = "CosineAnnealingLR"
    lr_scheduler: dict = {
        "T_max": 30000,
        "eta_min": 1.0e-5,
    }

    teacher_act_prob = "linear"
    update_times_scale = 500

    teacher_policy_class_name = InstinctRlEncoderMoEActorCriticRecurrentCfg().class_name
    teacher_policy: dict = TEACHER_POLICY
    teacher_logdir = TEACHER_LOGDIR
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
class DistillAlgoStudentCfg(InstinctRlDistillationAlgorithmCfg):
    """Pure behaviour cloning, following *Now You See That* (Table XII).

    Differences from `AmpAlgoStudentCfg` (TPPO with `using_ppo=False`) that matter:

    * The environment always executes the *student's* action -- there is no `teacher_act_prob`
      mixing to anneal. The student's RNN therefore learns its carry on exactly the state
      distribution it will meet at deployment.
    * `gradient_length` sets the BPTT depth independently of `num_steps_per_env`, which is what
      makes the paper's 800-step rollout tractable: TPPO's BPTT is the trajectory length after
      splitting at dones, with no way to chunk it.
    * Storage keeps only student observations, teacher action means and dones -- no critic
      observations, values, returns or advantages. At `num_steps_per_env: 800` with a depth
      observation that is the difference between fitting and not.
    """

    # Set explicitly rather than via TEACHER_LOGDIR: every distillation result recorded in
    # this file's comments was measured against this teacher, and the remote runs use it too.
    # Changing the shared constant would silently move the TPPO config's teacher as well.
    teacher_logdir = os.path.expanduser(
        "/home/zhipengli/parkour/InstinctLab/logs/instinct_rl/g1_parkour/20260713_091503"
    )
    teacher_policy_class_name = InstinctRlEncoderMoEActorCriticRecurrentCfg().class_name
    teacher_policy = TEACHER_POLICY
    # The teacher's actor was trained on the `policy` group (proprio + height_scan). In this env
    # that same layout is the `critic` group, which is what feeds it here.
    teacher_obs_source = "critic"

    # The student inherits the teacher's GRU, actor head and MoE gate; only the depth encoder
    # (and the action std) are learned from scratch. Teacher and student are already isomorphic
    # apart from that encoder -- both are EncoderMoEActorCriticRecurrent with rnn_hidden_size
    # 256, actor_hidden_dims [512, 256, 128], 4 MoE experts, and a 128-wide exteroceptive latent
    # -- so the GRU input (proprio 96 + latent 128 = 224) lines up on both sides.
    #
    # Without this the student cold-starts: a student-only rollout from random weights produces
    # nothing but falling, so the only states it ever labels are states the teacher would never
    # visit. Measured on the first attempt: 100% of episodes ended in bad_orientation or
    # root_height, zero timeouts, terrain_levels pinned at 0.
    warm_start_from_teacher = True

    # Paper eq. 8: squared L2 summed over the action dim, meaned over envs and timesteps.
    loss_type = "mse_sum"

    # Table XII: 800 steps per env per iteration, 10 gradient accumulation steps -> 80-step
    # TBPTT chunks (~1.6 s at 50 Hz), one clipped optimizer step per rollout.
    gradient_length = 80
    normalize_accumulated_loss = True
    # `num_steps_per_env == gradient_length` (see the runner cfg), so one chunk per rollout.
    flush_tail = True
    # Accumulation is off: with one chunk per rollout it would only fold that chunk into a
    # single step anyway, and leaving it off keeps `optimizer_step` and `update` scheduler units
    # identical, so `lr_scheduler.total_steps` equals `max_iterations`.
    accumulate_gradients = False

    # Measured, not inherited from the paper. A learning-rate range test (run
    # g1_parkour_student_lrfind/20260813_163652, lr swept 1e-5 -> 1.9e-2) gave:
    #   7.3e-4 - 1.1e-3   healthy band; lowest loss (0.678) and longest episodes (985) here
    #   1.38e-3           first damage: loss x6.4, episode length halved, grad norm 102
    #   4.7e-3            unrecoverable; terrain level to 0, grad norm 300-1300
    # The paper's 1e-2 peak destroyed this task's policy during the *ramp*, at an actual lr of
    # ~2e-3 -- warm start removes One Cycle's premise (a large ramp for from-scratch
    # super-convergence), because the inherited policy already sits near its optimum.
    learning_rate = 7.0e-4
    # Healthy operation reaches 25 in the same sweep, so 30 never binds normally; it exists to
    # cap the excursions (102 and up) that follow the terrain curriculum overshooting the
    # student. The previous 50 let a 30-53 spike through untouched.
    max_grad_norm = 30.0
    optimizer_class_name = "Adam"
    freeze_action_std = True

    # Table XII: EMA decay 0.997 (~330-step horizon). Checkpointed as the deployable model;
    # use `with alg.use_student_weights("ema"):` around any export.
    ema_decay = 0.997

    # Accumulation leaves exactly one optimizer step of carry staleness (the step happens after
    # the replay). This cancels it, at the cost of one extra no_grad pass over the rollout.
    refresh_hidden_after_update = True

    # Table XII: One Cycle, initial lr 1e-3, div_factor 10 -> peak 1e-2, final_div_factor 50
    # -> final 2e-5. Under `accumulate_gradients` there is exactly one optimizer step per
    # iteration, so `total_steps` equals `max_iterations`.
    # Peak 7e-4 sits at the top of the measured healthy band with roughly 2x margin to the
    # 1.38e-3 damage threshold. `div_factor` 10 -> starts at 7e-5; `final_div_factor` 50 ->
    # anneals to 1.4e-6. One optimizer step per iteration, so total_steps == max_iterations.
    lr_scheduler_class_name = "OneCycleLR"
    lr_scheduler: dict = {
        "max_lr": 7.0e-4,
        "total_steps": 6000,
        "div_factor": 10.0,
        "final_div_factor": 50.0,
    }
    lr_scheduler_step_unit = "update"


@configclass
class DistillAlgoLrFindCfg(DistillAlgoStudentCfg):
    """Learning-rate range test (Smith & Topin, the same reference One Cycle comes from).

    Sweeps the learning rate exponentially from 1e-5 upward and lets the run degrade. Plotting
    `Loss/behavior_loss` and `Train/mean_episode_length` against `Loss/learning_rate` gives the
    band where the loss falls fastest and the point where training starts coming apart; a usable
    `max_lr` is roughly a third to a half of that ceiling.

    This exists as a config class rather than a set of CLI overrides because Hydra's struct mode
    refuses to add `gamma` to the One Cycle `lr_scheduler` dict, and the leftover One Cycle keys
    would be rejected by `ExponentialLR` anyway. Being a config also means the sweep parameters
    land in the run's `params/agent.yaml`.

    gamma=1.0095 over 800 iterations sweeps 1e-5 -> ~1.9e-2, crossing 1e-3 (the current
    production peak) at iteration ~487 and 2e-3 (where an earlier run visibly degraded) at ~560.
    """

    learning_rate = 1.0e-5
    lr_scheduler_class_name = "ExponentialLR"
    lr_scheduler: dict = {"gamma": 1.0095}
    lr_scheduler_step_unit = "update"

    # Matches how the production run is actually launched. With num_steps_per_env ==
    # gradient_length there is exactly one chunk per rollout, so this makes no numerical
    # difference here -- it is set so the dumped config states the same thing production does.
    accumulate_gradients = False

    # Left where production has it: 50 is non-binding in normal operation (measured 1.6-11), so
    # it does not mask the divergence this test is meant to locate.
    max_grad_norm = 50.0


@configclass
class G1ParkourStudentLrFindRunnerCfg(InstinctRlOnPolicyRunnerCfg):
    """~90 minutes on a 24 GB card. `num_envs` is reduced from the production 1024 because
    `gradient_length` is deliberately *not* reduced -- the stability ceiling should be measured
    at the BPTT depth it will be used at, and lr limits are far less sensitive to batch size
    than to sequence length."""

    runner_class_name = "DistillationRunner"
    num_steps_per_env = 80
    max_iterations = 800
    init_at_random_ep_len = True
    save_interval = 400
    experiment_name = "g1_parkour_student_lrfind"
    resume = False
    load_run = ""
    empirical_normalization = False
    policy = MoEStudentPolicyCfg()
    algorithm = DistillAlgoLrFindCfg()


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
    init_at_random_ep_len = True
    save_interval = 2000
    experiment_name = "g1_parkour_student"
    resume = False
    load_run = ""
    empirical_normalization = False
    policy = MoEStudentPolicyCfg()
    algorithm = AmpAlgoStudentCfg()

@configclass
class G1ParkourStudentDistillRunnerCfg(InstinctRlOnPolicyRunnerCfg):
    """Pure BC distillation, reproducing *Now You See That* Table XII.

    **Memory.** Peak activation scales as `gradient_length x num_envs x per-step activation`,
    and the depth encoder dominates that. The buffer is a second, independent cost:
    `num_steps_per_env x num_envs x obs_dim`. At the current 64x36 depth (2304 px) the student
    observation is ~2400 floats, so 800 x 4096 would be ~31 GB of buffer alone.

    Downsample the depth to the paper's 24x32 (768 px, ~864-float observation) before running
    this. If it still does not fit, back off in this order: `num_envs` -> `gradient_length`
    (adjusting `lr_scheduler.total_steps` is *not* needed, it tracks iterations) -> only then
    `num_steps_per_env`.

    Do **not** add a `normalizers` entry for the `critic` group: it feeds the teacher, which
    applies its own frozen normalizer from its checkpoint. `DistillationRunner` raises on it.
    """

    runner_class_name = "DistillationRunner"
    # T == gradient_length. `num_steps_per_env` does not change the optimizer-step rate (a
    # rollout yields ceil(T/G) steps and costs time proportional to T) nor the effective batch
    # (G x num_envs), so shrinking it to G costs nothing and buys a 7 s iteration instead of
    # 70 s, plus strictly on-policy data: every chunk is trained on states the current weights
    # collected, rather than states up to nine optimizer steps stale.
    num_steps_per_env = 80
    max_iterations = 6000
    init_at_random_ep_len = True
    save_interval = 200
    experiment_name = "g1_parkour_student_distill"
    resume = False
    load_run = ""
    empirical_normalization = False
    policy = MoEStudentPolicyCfg()
    algorithm = DistillAlgoStudentCfg()


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
