# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Video / image reading and VAE preprocessing."""

import math
from typing import List, Union

import decord
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


def make_divisible(value: int, stride: int) -> int:
    """Round `value` to the nearest multiple of `stride` (at least `stride`)."""
    return max(stride, int(round(value / stride) * stride))


def _apply_scale(width, height, scale, stride):
    new_width = make_divisible(round(width * scale), stride)
    new_height = make_divisible(round(height * scale), stride)
    return new_width, new_height


class MaxLongEdgeMinShortEdgeResize(torch.nn.Module):
    """Resize so the long edge <= max_size and short edge >= min_size, snapped to `stride`."""

    def __init__(self, max_size, min_size, stride, interpolation=InterpolationMode.BICUBIC, antialias=True):
        super().__init__()
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = _apply_scale(width, height, scale, self.stride)
        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = _apply_scale(new_width, new_height, scale, self.stride)

        # No-op short-circuit: F.resize is never exactly identity, so skip it.
        if (new_width, new_height) == (width, height):
            return img

        # Resize uint8 tensors in float32 for full interpolation precision.
        if isinstance(img, torch.Tensor) and img.dtype == torch.uint8:
            resized = F.resize(img.float(), (new_height, new_width), self.interpolation, antialias=self.antialias)
            return resized.clamp_(0, 255).round_().to(torch.uint8)
        return F.resize(img, (new_height, new_width), self.interpolation, antialias=self.antialias)


class VAEVideoTransform:
    """Resize -> ToTensor -> Normalize to [-1, 1] for VAE input."""

    def __init__(self, max_image_size, min_image_size=1, image_stride=16,
                 image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5)):
        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size, min_size=min_image_size, stride=image_stride
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean=list(image_mean), std=list(image_std), inplace=True)

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img).convert("RGB")
        else:
            img = img.convert("RGB")
        img = self.resize_transform(img)
        img = self.to_tensor_transform(img)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return self.normalize_transform(img)


def smart_video_nframes(
    total_frames: int,
    video_fps: Union[int, float],
    fps: int = 2.0,
    frame_factor: int = None,
    min_frames: int = None,
    max_frames: int = None,
    add_one: bool = False,
) -> List[int]:
    """Pick frame indices so the sampled clip matches the target `fps`/frame count."""
    nframes = total_frames / video_fps * fps
    if frame_factor is not None:
        nframes = math.floor(nframes / frame_factor) * frame_factor + int(add_one)
        nframes = max(nframes, frame_factor + int(add_one))
        if video_fps == fps:
            total_frames = math.floor(total_frames / frame_factor) * frame_factor + int(add_one)
    else:
        nframes = int(nframes + int(add_one))

    idx = torch.linspace(0, total_frames - 1, nframes).round().long().tolist()

    if min_frames is not None:
        if frame_factor is not None:
            min_frames = math.ceil(min_frames / frame_factor) * frame_factor
        nframes = max(min_frames + int(add_one), nframes)
    while len(idx) < int(nframes):
        idx.append(idx[-1])

    if max_frames is not None:
        if frame_factor is not None:
            max_frames = math.floor(max_frames / frame_factor) * frame_factor
        nframes = min(max_frames + int(add_one), nframes)
    if len(idx) > int(nframes):
        idx = idx[: int(nframes)]
    return idx


class VideoFrameReader:
    """Read RGB frames from a local video file by index."""

    def __init__(self, video_path: str):
        self.vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0), fault_tol=1)
        self.vr.seek(0)
        self.fps = self.vr.get_avg_fps()
        self.length = len(self.vr)

    def sample(self, frame_indices: List[int]) -> List[Image.Image]:
        indices = [max(0, min(int(i), self.length - 1)) for i in frame_indices]
        frames = self.vr.get_batch(indices).asnumpy()
        return [Image.fromarray(f).convert("RGB") for f in frames]


def preprocess_video(
    video_path, fps=16, max_image_size=624, min_image_size=1, max_image_num=81,
    torch_dtype=torch.float32, device="cuda",
) -> torch.Tensor:
    """Read a video and return a normalized tensor `[1, C, T, H, W]`."""
    transform = VAEVideoTransform(max_image_size=max_image_size, min_image_size=min_image_size)
    reader = VideoFrameReader(video_path)
    idx = smart_video_nframes(
        total_frames=reader.length, video_fps=reader.fps, fps=fps,
        frame_factor=4, max_frames=max_image_num, add_one=True,
    )
    frames = reader.sample(idx)
    video_tensor = torch.stack([transform(f) for f in frames], dim=1).unsqueeze(0)
    return video_tensor.to(dtype=torch_dtype, device=device)


def _load_image(image) -> Image.Image:
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return image.convert("RGB")


def preprocess_image(image, max_image_size=624, min_image_size=1,
                     torch_dtype=torch.float32, device="cuda") -> torch.Tensor:
    """Return a normalized tensor `[1, C, 1, H, W]` for a single image."""
    transform = VAEVideoTransform(max_image_size=max_image_size, min_image_size=min_image_size)
    img_tensor = transform(_load_image(image)).unsqueeze(0).unsqueeze(2)
    return img_tensor.to(dtype=torch_dtype, device=device)


def preprocess_images(images, max_image_size=624, min_image_size=1,
                      torch_dtype=torch.float32, device="cuda") -> torch.Tensor:
    """Return a normalized tensor `[1, C, N, H, W]` for a list of images."""
    transform = VAEVideoTransform(max_image_size=max_image_size, min_image_size=min_image_size)
    frames = [transform(_load_image(img)) for img in images]
    video_tensor = torch.stack(frames, dim=0).permute(1, 0, 2, 3).unsqueeze(0)
    return video_tensor.to(dtype=torch_dtype, device=device)
