"""Offline ONNX exporter for instinct_rl checkpoints (no Isaac Sim required).

Builds the policy from the run's saved ``params/agent.yaml``, loads the
checkpoint strictly, exports with the *installed* instinct_rl export code
(``EncoderActorCriticMixin.export_as_onnx`` separate-file path), then
numerically verifies the exported encoder+actor chain against PyTorch
``act_inference`` over a recurrent rollout.

The encoder is exported with the legacy tracer at ``--opset 14`` so both files
land on ONNX IR version 7; the dynamo exporter (``--dynamo``) always emits opset
18 / IR 10, which older on-robot onnxruntime builds refuse to load.

NOTE: the observation layout below (``OBS_FORMAT``) matches the G1 parkour
student task (proprio without history + depth_image (1, 18, 32)). Adjust it if
the task's observation config changes. Make sure the instinct_rl checkout
matches the code version the checkpoint was trained with (see the run's
``git/instinct_rl_*.diff``).

Example:
    python scripts/instinct_rl/export_onnx.py \\
        logs/instinct_rl/g1_parkour_student/20260702_164649 \\
        --checkpoint model_25000.pt
"""

import argparse
import glob
import os
from collections import OrderedDict

import numpy as np
import torch
import yaml

# G1 parkour student observation layout (order matters: must match the env's
# policy observation group; depth_image must be last for the deploy-side
# "replace tail with latent" concatenation to hold).
OBS_FORMAT = {
    "policy": OrderedDict(
        [
            ("base_ang_vel", (3,)),
            ("projected_gravity", (3,)),
            ("velocity_commands", (3,)),
            ("joint_pos_rel", (29,)),
            ("joint_vel_rel", (29,)),
            ("last_action", (29,)),
            ("depth_image", (1, 18, 32)),
        ]
    ),
    "critic": OrderedDict(
        [
            ("base_ang_vel", (24,)),
            ("projected_gravity", (24,)),
            ("velocity_commands", (24,)),
            ("joint_pos_rel", (232,)),
            ("joint_vel_rel", (232,)),
            ("last_action", (232,)),
            ("height_scan", (4 * 21 * 33,)),
        ]
    ),
}
NUM_ACTIONS = 29


def build_model(logdir: str, device: str) -> torch.nn.Module:
    import instinct_rl.modules as modules

    with open(os.path.join(logdir, "params", "agent.yaml")) as f:
        agent_cfg = yaml.unsafe_load(f)
    policy_cfg = dict(agent_cfg["policy"])
    class_name = policy_cfg.pop("class_name")
    model = getattr(modules, class_name)(
        obs_format=OBS_FORMAT, num_actions=NUM_ACTIONS, **policy_cfg
    )
    return model.to(device)


def verify(model, export_dir: str, obs_dim: int, num_steps: int) -> float:
    """Chain the exported encoder+actor with onnxruntime and compare against
    the stateful PyTorch act_inference over a recurrent rollout."""
    import onnxruntime as ort

    enc_sess = ort.InferenceSession(os.path.join(export_dir, "0-depth_image_encoder.onnx"))
    act_sess = ort.InferenceSession(os.path.join(export_dir, "actor.onnx"))
    for tag, sess in [("encoder", enc_sess), ("actor", act_sess)]:
        print(f"{tag} inputs:", [(i.name, i.shape) for i in sess.get_inputs()])
        print(f"{tag} outputs:", [(o.name, o.shape) for o in sess.get_outputs()])

    enc_input_names = [i.name for i in enc_sess.get_inputs()]
    act_hidden_name = act_sess.get_inputs()[1].name
    hidden_shape = [int(d) for d in act_sess.get_inputs()[1].shape]
    proprio_dim = obs_dim - int(np.prod(OBS_FORMAT["policy"]["depth_image"]))

    model.reset()
    h = np.zeros(hidden_shape, np.float32)
    max_err = 0.0
    device = next(model.parameters()).device
    with torch.no_grad():
        for _ in range(num_steps):
            obs = torch.rand(1, obs_dim, device=device)
            a_ref = model.act_inference(obs).cpu().numpy()

            o = obs.cpu().numpy()
            img = o[:, proprio_dim:].reshape(1, *OBS_FORMAT["policy"]["depth_image"])
            enc_feed = {enc_input_names[0]: img}
            if len(enc_input_names) > 1:  # cross-attention encoder: proprio query
                enc_feed[enc_input_names[1]] = o[:, :proprio_dim]
            (latent,) = enc_sess.run(None, enc_feed)

            actor_in = np.concatenate([o[:, :proprio_dim], latent], axis=1)
            a_onnx, h = act_sess.run(None, {"input": actor_in, act_hidden_name: h})
            max_err = max(max_err, float(np.abs(a_ref - a_onnx).max()))
    return max_err


def report_versions(export_dir: str) -> None:
    """Print the ONNX IR/opset of every exported file — the on-robot onnxruntime
    rejects files whose IR version is newer than it supports."""
    import onnx

    for path in sorted(glob.glob(os.path.join(export_dir, "*.onnx"))):
        m = onnx.load(path, load_external_data=False)
        opsets = ", ".join(f"{i.domain or 'ai.onnx'}={i.version}" for i in m.opset_import)
        print(f"[export_onnx] {os.path.basename(path)}: ir_version={m.ir_version} opset({opsets})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logdir", help="run directory, e.g. logs/instinct_rl/g1_parkour_student/20260702_164649")
    parser.add_argument("--checkpoint", default=None, help="checkpoint file name; defaults to the latest model_*.pt")
    parser.add_argument("--device", default="cpu", help="device for model/export (default: cpu, matches deployment)")
    parser.add_argument("--verify_steps", type=int, default=30, help="recurrent rollout steps for numeric verification; 0 disables")
    parser.add_argument("--opset", type=int, default=14, help="encoder ONNX opset; 14 -> IR version 7, the widest deploy compatibility (min for attention encoders)")
    parser.add_argument("--dynamo", action="store_true", help="use the dynamo exporter for the encoder (always opset 18 / IR 10 — too new for old onnxruntime)")
    args = parser.parse_args()

    logdir = os.path.abspath(args.logdir)
    if args.checkpoint is None:
        ckpts = sorted(
            glob.glob(os.path.join(logdir, "model_*.pt")),
            key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
        )
        assert ckpts, f"no model_*.pt found in {logdir}"
        ckpt_path = ckpts[-1]
    else:
        ckpt_path = os.path.join(logdir, args.checkpoint)
    print(f"[export_onnx] checkpoint: {ckpt_path}")

    model = build_model(logdir, args.device)
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])  # strict: fails loudly on code/ckpt mismatch
    model.eval()

    export_dir = os.path.join(logdir, "exported")
    os.makedirs(export_dir, exist_ok=True)
    obs_dim = sum(int(np.prod(s)) for s in OBS_FORMAT["policy"].values())
    obs = torch.rand(1, obs_dim, device=args.device)
    model.export_as_onnx(obs, export_dir, opset_version=args.opset, dynamo=args.dynamo)

    report_versions(export_dir)
    if args.verify_steps > 0:
        max_err = verify(model, export_dir, obs_dim, args.verify_steps)
        print(f"[export_onnx] {args.verify_steps}-step recurrent parity max err: {max_err:.3e}")
        assert max_err < 1e-3, "exported ONNX diverges from PyTorch — do NOT deploy"
    print(f"[export_onnx] done -> {export_dir}")


if __name__ == "__main__":
    main()
