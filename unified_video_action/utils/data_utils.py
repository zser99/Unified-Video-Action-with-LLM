import random
import numpy as np
import sys

sys.path.extend([sys.path[0][:-4], "/app"])
import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
import PIL
from itertools import combinations_with_replacement

"""데이터 전처리 유틸리티 모음.

이 모듈은 dataloader 출력과 MAR 정책/모델 입력 사이의 연결 레이어다.
지원 태스크(libero, umi, toolhang 등)마다 관측 키 구조가 다르기 때문에,
함수 대부분이 태스크 의존 분기 로직을 포함한다.
"""


array = list(range(16))
combinations = list(combinations_with_replacement(array, 4))
combinations = [tem for tem in combinations if tem[-1] == 15]


def resize_image(cfg, x):
    """태스크별 카메라 키를 `obs.image` 중심 포맷으로 통일하고 256으로 리사이즈한다."""
    resize = 256
    if "libero" in cfg.task.name:
        B, T, C, H, W = x["obs"]["agentview_rgb"].shape
        resized_tensor = F.interpolate(
            x["obs"]["agentview_rgb"].float().view(B * T, C, H, W),
            size=(resize, resize),
            mode="bilinear",
            align_corners=False,
        )
        x["obs"]["image"] = resized_tensor.view(
            B, T, C, resize, resize
        )
        del x["obs"]["agentview_rgb"]

    elif "umi" in cfg.task.name:
        B, T, C, H, W = x["obs"]["camera0_rgb"].shape
        resized_tensor = F.interpolate(
            x["obs"]["camera0_rgb"].contiguous().float().view(B * T, C, H, W),
            size=(resize, resize),
            mode="bilinear",
            align_corners=False,
        )
        x["obs"]["image"] = resized_tensor.view(
            B, T, C, resize, resize
        )
        del x["obs"]["camera0_rgb"]

    elif "toolhang" in cfg.task.name:
        B, T, C, H, W = x["obs"]["sideview_image"].shape
        resized_tensor = F.interpolate(
            x["obs"]["sideview_image"].contiguous().float().view(B * T, C, H, W),
            size=(resize, resize),
            mode="bilinear",
            align_corners=False,
        )
        x["obs"]["image"] = resized_tensor.view(
            B, T, C, resize, resize
        )
        del x["obs"]["sideview_image"]

        resized_tensor = F.interpolate(
            x["obs"]["robot0_eye_in_hand_image"]
            .contiguous()
            .float()
            .view(B * T, C, H, W),
            size=(resize, resize),
            mode="bilinear",
            align_corners=False,
        )
        x["obs"]["wrist_image"] = resized_tensor.view(B, T, C, resize, resize)
        del x["obs"]["robot0_eye_in_hand_image"]

    else:
        B, T, C, H, W = x["obs"]["image"].shape
        if resize != H:
            resized_tensor = F.interpolate(
                x["obs"]["image"].contiguous().view(B * T, C, H, W),
                size=(resize, resize),
                mode="bilinear",
                align_corners=False,
            )
            x["obs"]["image"] = resized_tensor.view(B, T, C, resize, resize)

    return x


def resize_image_eval(task_name, obs_dict):
    """추론 시점 관측 딕셔너리에 대해 `resize_image`와 동일한 키 정규화를 수행한다."""
    if "libero" in task_name:
        if "agentview_image" in obs_dict:
            obs_dict["image"] = obs_dict["agentview_image"]
            del obs_dict["agentview_image"]

    elif "toolhang" in task_name:
        if "sideview_image" in obs_dict:
            obs_dict["image"] = obs_dict["sideview_image"]
            del obs_dict["sideview_image"]
        if "robot0_eye_in_hand_image" in obs_dict:
            obs_dict["wrist_image"] = obs_dict["robot0_eye_in_hand_image"]
            del obs_dict["robot0_eye_in_hand_image"]

    elif "umi" in task_name:
        if "camera0_rgb" in obs_dict:
            obs_dict["image"] = obs_dict["camera0_rgb"]
            del obs_dict["camera0_rgb"]

    B, T, C, H, W = obs_dict["image"].shape
    resize = 256
    if H != resize:
        resized_tensor = F.interpolate(
            obs_dict["image"].view(B * T, C, H, W),
            size=(resize, resize),
            mode="bilinear",
            align_corners=False,
        )
        obs_dict["image"] = resized_tensor.view(B, T, C, resize, resize)

        if "wrist_image" in obs_dict:
            resized_tensor = F.interpolate(
                obs_dict["wrist_image"].view(B * T, C, H, W),
                size=(resize, resize),
                mode="bilinear",
                align_corners=False,
            )
            obs_dict["wrist_image"] = resized_tensor.view(B, T, C, resize, resize)

    return obs_dict


def decode_from_sample(vae_model, z):
    """학습 경로에서 사용하는 VAE 래퍼 API로 latent를 복원한다."""
    with torch.no_grad():
        pred = vae_model.model.decode_from_sample(z)
    return pred


def decode_from_sample_autoregressive(vae_model, z):
    """오토리그레시브 경로에서 사용하는 VAE API로 latent를 복원한다."""
    with torch.no_grad():
        pred = vae_model.decode(z)
    return pred


def select_frames(x, T, eval=False, select_timesteps=4, different_history_freq=False):
    """모델 입력으로 사용할 프레임 인덱스를 선택한다.

    - 학습 시: 더 촘촘한 간격으로 뽑은 뒤, 이후 히스토리/예측 절반 분할을 고려한다.
    - 평가 시: 균등 간격의 결정적(deterministic) 인덱스를 사용한다.
    - `different_history_freq` 활성화 시: 사전 계산된 조합에서 히스토리 인덱스를 랜덤 샘플링한다.
    """
    if eval:
        indices = torch.arange(0, T, step=T // select_timesteps) + select_timesteps - 1
    else:
        indices = (
            torch.arange(0, T, step=T // (select_timesteps * 2)) + select_timesteps - 1
        )

        if different_history_freq:
            indices = torch.cat(
                [
                    torch.tensor(random.choice(combinations)),
                    indices[indices.shape[0] // 2 :],
                ]
            )

    x = x[:, indices, :, :, :]

    return x, indices


def normalize_past_action(normalizer, normalizer_type, actions):
    """과거 액션 히스토리를 설정된 정규화 방식에 맞춰 변환한다."""
    if normalizer_type == "all":
        history_nactions = normalizer["action"].normalize(actions)
    elif normalizer_type == "none":
        history_nactions = actions
    return history_nactions


def unnormalize_future_action(normalizer, normalizer_type, actions):
    """모델 출력 액션을 환경 원래 스케일로 되돌린다."""
    if normalizer_type == "all":
        future_nactions = normalizer["action"].unnormalize(actions)
    elif normalizer_type == "none":
        future_nactions = actions
    return future_nactions


def normalize_action(normalizer, normalizer_type, actions):
    """손실 계산에 사용하는 타깃 액션을 정규화한다."""
    if normalizer_type == "all":
        nactions = normalizer["action"].normalize(actions)
    elif normalizer_type == "none":
        nactions = actions
    return nactions


def normalize_obs(normalizer, normalizer_type, batch):
    """이미지 외 관측 채널만 in-place 정규화한다.

    이미지 텐서는 여기서 제외한다. 이미지 스케일링은 `process_data`에서
    [0,255] -> [-1,1] 변환으로 별도 처리되기 때문이다.
    """
    if normalizer_type == "all":
        nobs = {"obs": {}}
        for k, v in batch["obs"].items():
            if "image" in k:
                continue
            nobs["obs"][k] = v

        nobs = normalizer.normalize(nobs["obs"])

        for k, v in batch["obs"].items():
            if "image" in k:
                continue
            batch["obs"][k] = nobs[k]

    elif normalizer_type == "none":
        batch = batch

    return batch


def process_data(batch, task_name="", eval=False, **kwargs):
    """원시 배치를 모델 입력 포맷(영상 + 고유수용감각)으로 변환한다.

    Returns:
        x: 정규화된 이미지 텐서, shape [B, C, T, H, W]
        proprioception_input: 태스크별 상태/보조 이미지 입력(없으면 None)
        indices: 시간축 서브샘플링에 사용한 프레임 인덱스
    """
    train = not eval

    x = batch["obs"]["image"]
    x = x * 255.0
    B, T, C, H, W = x.size()
    device = x.device

    if "umi" in task_name:
        if "img_indices" in batch["obs"]:
            indices: torch.Tensor = batch["obs"]["img_indices"].int().squeeze(2)
            T = T * 4  # Only 8 frames are loaded in the dataloader
        else:
            indices = None
    else:
        x, indices = select_frames(
            x, T, eval=eval, different_history_freq=kwargs["different_history_freq"]
        )

    # VAE/MAR 입력 규약: 픽셀값을 [0,255]에서 [-1,1]로 변환한다.
    x = rearrange(x / 127.5 - 1, "b t c h w -> b c t h w")

    if kwargs["use_proprioception"]:
        if "toolhang" in task_name:
            wrist_image = batch["obs"]["wrist_image"]
            wrist_image = wrist_image * 255.0
            wrist_image = wrist_image[:, indices, :, :, :]
            wrist_image = wrist_image.to(device)
            wrist_image = rearrange(wrist_image / 127.5 - 1, "b t c h w -> b c t h w")

            if train:
                wrist_image, wrist_image_2 = torch.chunk(wrist_image, 2, dim=2)
                robot0_eef_pos, robot0_eef_pos_pred = torch.chunk(
                    batch["obs"]["robot0_eef_pos"], 2, dim=1
                )
                robot0_eef_quat, robot0_eef_quat_pred = torch.chunk(
                    batch["obs"]["robot0_eef_quat"], 2, dim=1
                )
                robot0_gripper_qpos, robot0_gripper_qpos_pred = torch.chunk(
                    batch["obs"]["robot0_gripper_qpos"], 2, dim=1
                )
            else:
                wrist_image = wrist_image
                wrist_image_2 = None
                robot0_eef_pos = batch["obs"]["robot0_eef_pos"]
                robot0_eef_quat = batch["obs"]["robot0_eef_quat"]
                robot0_gripper_qpos = batch["obs"]["robot0_gripper_qpos"]
                robot0_eef_pos_pred = None
                robot0_eef_quat_pred = None
                robot0_gripper_qpos_pred = None

            if kwargs["different_history_freq"]:
                if train:
                    robot0_eef_pos = robot0_eef_pos[:, indices[: indices.shape[0] // 2]]
                    robot0_eef_quat = robot0_eef_quat[
                        :, indices[: indices.shape[0] // 2]
                    ]
                    robot0_gripper_qpos = robot0_gripper_qpos[
                        :, indices[: indices.shape[0] // 2]
                    ]
                else:
                    robot0_eef_pos = robot0_eef_pos[:, indices]
                    robot0_eef_quat = robot0_eef_quat[:, indices]
                    robot0_gripper_qpos = robot0_gripper_qpos[:, indices]

            proprioception_input = {
                "robot0_eef_pos": robot0_eef_pos,
                "robot0_eef_quat": robot0_eef_quat,
                "robot0_gripper_qpos": robot0_gripper_qpos,
                "second_image": wrist_image,
                "pred_second_image": wrist_image_2,
                "robot0_eef_pos_pred": robot0_eef_pos_pred,
                "robot0_eef_quat_pred": robot0_eef_quat_pred,
                "robot0_gripper_qpos_pred": robot0_gripper_qpos_pred,
            }

        elif "pusht" in task_name:
            if train:
                state, state_pred = torch.chunk(batch["obs"]["agent_pos"], 2, dim=1)
            else:
                state = batch["obs"]["agent_pos"]
                state_pred = None

            proprioception_input = {"state": state, "state_pred": state_pred}

        elif "umi" in task_name:
            if train:
                robot0_eef_pos, robot0_eef_pos_pred = torch.chunk(
                    batch["obs"]["robot0_eef_pos"], 2, dim=1
                )
                robot0_eef_rot_axis_angle, robot0_eef_rot_axis_angle_pred = torch.chunk(
                    batch["obs"]["robot0_eef_rot_axis_angle"], 2, dim=1
                )
                robot0_gripper_width, robot0_gripper_width_pred = torch.chunk(
                    batch["obs"]["robot0_gripper_width"], 2, dim=1
                )
                (
                    robot0_eef_rot_axis_angle_wrt_start,
                    robot0_eef_rot_axis_angle_wrt_start_pred,
                ) = torch.chunk(
                    batch["obs"]["robot0_eef_rot_axis_angle_wrt_start"], 2, dim=1
                )
            else:
                robot0_eef_pos = batch["obs"]["robot0_eef_pos"]
                robot0_eef_rot_axis_angle = batch["obs"]["robot0_eef_rot_axis_angle"]
                robot0_gripper_width = batch["obs"]["robot0_gripper_width"]
                robot0_eef_rot_axis_angle_wrt_start = batch["obs"][
                    "robot0_eef_rot_axis_angle_wrt_start"
                ]
                robot0_eef_pos_pred = None
                robot0_eef_rot_axis_angle_pred = None
                robot0_gripper_width_pred = None
                robot0_eef_rot_axis_angle_wrt_start_pred = None

            if "different_history_freq" in kwargs and kwargs["different_history_freq"]:
                
                if indices is not None:
                    # 배치 샘플별 인덱스가 다를 수 있어 행 단위 gather를 사용한다.
                    length = indices.shape[1]  # [bs, 8]
                    if train:
                        length = (
                            length // 2
                        )  # 학습에서는 history 구간(앞 절반) 인덱스만 사용

                    # 고급 인덱싱 시 배치 정렬이 깨지지 않도록 batch 인덱스를 구성한다.
                    batch_indices = (
                        torch.arange(indices.shape[0], device=indices.device)
                        .unsqueeze(-1)
                        .expand(-1, length)
                    )

                    # history 빈도 조건에 맞는 시점만 골라낸다.
                    robot0_eef_pos = robot0_eef_pos[batch_indices, indices[:, :length]]
                    robot0_eef_rot_axis_angle = robot0_eef_rot_axis_angle[
                        batch_indices, indices[:, :length]
                    ]
                    robot0_gripper_width = robot0_gripper_width[
                        batch_indices, indices[:, :length]
                    ]
                    robot0_eef_rot_axis_angle_wrt_start = (
                        robot0_eef_rot_axis_angle_wrt_start[
                            batch_indices, indices[:, :length]
                        ]
                    )

            proprioception_input = {
                "robot0_eef_pos": robot0_eef_pos,
                "robot0_eef_rot_axis_angle": robot0_eef_rot_axis_angle,
                "robot0_gripper_width": robot0_gripper_width,
                "robot0_eef_rot_axis_angle_wrt_start": robot0_eef_rot_axis_angle_wrt_start,
                "robot0_eef_pos_pred": robot0_eef_pos_pred,
                "robot0_eef_rot_axis_angle_pred": robot0_eef_rot_axis_angle_pred,
                "robot0_gripper_width_pred": robot0_gripper_width_pred,
                "robot0_eef_rot_axis_angle_wrt_start_pred": robot0_eef_rot_axis_angle_wrt_start_pred,
            }

    else:
        proprioception_input = None

    return x, proprioception_input, indices


def get_trajectory(nactions, T, shift_action, use_history_action=False):
    """정규화된 액션을 history/future trajectory로 분할한다.

    `shift_action=True`면 관측-액션의 1-step 정렬 오프셋을 반영해 자른다.
    """
    if nactions is not None:
        if use_history_action:
            if shift_action:
                history_trajectory = nactions[:, : T // 2]
                trajectory = nactions[:, T // 2 : -1]
            else:
                history_trajectory, trajectory = torch.chunk(nactions[:, 1:], 2, dim=1)

        else:
            if shift_action:
                trajectory = nactions[:, T // 2 - 1 : -1]
                history_trajectory = None
            else:
                history_trajectory, trajectory = torch.chunk(nactions, 2, dim=1)

    else:
        trajectory = None
        history_trajectory = None

    return history_trajectory, trajectory


def extract_latent_autoregressive(vae_model, x):
    """비디오 텐서를 VAE latent 토큰으로 인코딩하고 고정 스케일을 적용한다."""
    x = x.float()
    B, C, T, H, W = x.size()
    with torch.no_grad():
        posterior = vae_model.encode(rearrange(x, "b c t h w -> (b t) c h w"))
        z = posterior.sample().mul_(0.2325)
        z = rearrange(z, "(b t) c h w -> b t c h w", b=B)
    latent_size = z.size()[2:]
    return z, latent_size


def get_vae_latent(x, vae_model, eval=False, proprioception_input={}):
    """오토리그레시브 학습용 조건/타깃 latent 쌍을 만든다.

    시간축 기준으로 입력을 절반 분할한다.
    - 앞 절반: 조건 latent `c`
    - 뒤 절반: 예측 대상 latent `z`
    """
    train = not eval

    c, x = torch.chunk(x, 2, dim=2)  # 시간축 앞 절반을 조건으로 사용

    if proprioception_input is not None:
        if "second_image" in proprioception_input:
            second_image_z, _ = extract_latent_autoregressive(
                vae_model, proprioception_input["second_image"]
            )
            proprioception_input["second_image_z"] = second_image_z
        if "pred_second_image" in proprioception_input:
            pred_second_image_z, _ = extract_latent_autoregressive(
                vae_model, proprioception_input["pred_second_image"]
            )
            proprioception_input["pred_second_image_z"] = pred_second_image_z

    with torch.no_grad():
        if train:
            z, latent_size = extract_latent_autoregressive(vae_model, x)
        else:
            z, latent_size = extract_latent_autoregressive(vae_model, x)
        c, latent_size = extract_latent_autoregressive(vae_model, c)

    return x, z, c, latent_size, proprioception_input


def save_image_grid(img, fname, drange, grid_size, normalize=True):
    """비디오 텐서를 mp4/gif로 저장해 정성적 결과를 빠르게 확인한다."""
    if normalize:
        lo, hi = drange
        img = np.asarray(img, dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, T, H, W = img.shape
    img = img.reshape(gh, gw, C, T, H, W)
    img = img.transpose(3, 0, 4, 1, 5, 2)
    img = img.reshape(T, gh * H, gw * W, C)

    print(f"Saving Video with {T} frames, img shape {H}, {W}")

    assert C in [3]

    if C == 3:
        torchvision.io.write_video(f"{fname[:-3]}mp4", torch.from_numpy(img), fps=16)
        imgs = [PIL.Image.fromarray(img[i], "RGB") for i in range(len(img))]
        imgs[0].save(
            fname,
            quality=95,
            save_all=True,
            append_images=imgs[1:],
            duration=100,
            loop=0,
        )

    return img
