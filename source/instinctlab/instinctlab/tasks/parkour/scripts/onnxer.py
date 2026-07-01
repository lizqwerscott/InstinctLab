from __future__ import annotations

import numpy as np
import os
import torch
from typing import TYPE_CHECKING

import onnxruntime as ort

if TYPE_CHECKING:
    from typing import Callable


def load_parkour_onnx_model(
    model_dir: str,
    get_subobs_func: Callable,
    depth_shape: tuple,
    proprio_slice: slice | None = None,
    get_proprio_func: Callable | None = None,
    encoder_name: str | None = None,
) -> Callable:
    """Load the ONNX model as policy, but only for parkour task setting."""
    ort_providers = ort.get_available_providers()
    encoder_filenames = []
    if encoder_name is not None:
        encoder_filenames.append(f"0-{encoder_name}.onnx")
    encoder_filenames.extend(["0-depth_image_encoder.onnx", "0-depth_encoder.onnx"])
    for encoder_filename in dict.fromkeys(encoder_filenames):
        encoder_path = os.path.join(model_dir, encoder_filename)
        if os.path.exists(encoder_path):
            break
    else:
        raise FileNotFoundError(f"Cannot find depth encoder ONNX in {model_dir}: {encoder_filenames}")

    encoder = ort.InferenceSession(encoder_path, providers=ort_providers)
    actor = ort.InferenceSession(os.path.join(model_dir, "actor.onnx"), providers=ort_providers)
    actor_inputs = actor.get_inputs()
    actor_input_name = actor_inputs[0].name
    hidden_input_name = actor_inputs[1].name if len(actor_inputs) > 1 else None
    hidden_state = None
    if hidden_input_name is not None:
        hidden_shape = [dim if isinstance(dim, int) else 1 for dim in actor_inputs[1].shape]
        hidden_state = np.zeros(hidden_shape, dtype=np.float32)

    def policy(obs: torch.Tensor) -> torch.Tensor:
        nonlocal hidden_state
        depth_image_input = get_subobs_func(obs)
        depth_image_input = depth_image_input.cpu().numpy()
        depth_image_input = depth_image_input.reshape((-1, *depth_shape))
        depth_image_output = encoder.run(None, {encoder.get_inputs()[0].name: depth_image_input})[0]
        if get_proprio_func is None:
            if proprio_slice is None:
                raise ValueError("Either proprio_slice or get_proprio_func must be provided.")
            proprio_input = obs.cpu().numpy()[:, proprio_slice]
        else:
            proprio_input = get_proprio_func(obs).cpu().numpy()
        actor_input = np.concatenate(
            [
                proprio_input,
                depth_image_output,
            ],
            axis=1,
        )
        if hidden_input_name is None:
            actor_output = actor.run(None, {actor_input_name: actor_input})[0]
        else:
            if hidden_state.shape[1] != actor_input.shape[0]:
                hidden_shape = list(hidden_state.shape)
                hidden_shape[1] = actor_input.shape[0]
                hidden_state = np.zeros(hidden_shape, dtype=np.float32)
            actor_output, hidden_state = actor.run(
                None,
                {
                    actor_input_name: actor_input,
                    hidden_input_name: hidden_state,
                },
            )
        return torch.from_numpy(actor_output).to(obs.device)

    def reset(dones: torch.Tensor | np.ndarray | None = None) -> None:
        nonlocal hidden_state
        if hidden_state is None:
            return
        if dones is None:
            hidden_state[...] = 0.0
            return
        if isinstance(dones, torch.Tensor):
            dones = dones.detach().cpu().numpy()
        hidden_state[:, np.asarray(dones, dtype=bool), :] = 0.0

    policy.reset = reset
    return policy
