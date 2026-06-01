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

"""Single-GPU Bernini Renderer inference.

Example:
    python infer_single_gpu.py \\
        --high_noise_ckpt <path-or-hf-repo> --low_noise_ckpt <path-or-hf-repo> \\
        --case assets/testcases/v2v/v2v.json
"""

import argparse

import torch

from bernini.cli import (
    add_common_args,
    apply_case_file,
    build_pipeline,
    generation_kwargs,
    load_tasks,
    resolve_system_prompt,
    setup_logging,
)


def main():
    parser = argparse.ArgumentParser(description="Bernini Renderer single-GPU inference")
    add_common_args(parser)
    args = parser.parse_args()
    apply_case_file(args)
    setup_logging()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    pipeline = build_pipeline(args, device)

    rewriter = None
    if args.use_pe:
        from bernini.prompt_enhancer import PromptEnhancer

        rewriter = PromptEnhancer(model=args.pe_model)

    common = generation_kwargs(args)
    for task in load_tasks(args):
        prompt = task["prompt"]
        if rewriter is not None:
            prompt = rewriter(
                task.get("task_type", args.task_type),
                prompt,
                video=task.get("video"),
                image=task.get("image"),
                images=task.get("images"),
            )
        pipeline(
            prompt,
            video=task.get("video"),
            image=task.get("image"),
            images=task.get("images"),
            output_path=task.get("output", args.output),
            system_prompt=resolve_system_prompt(task, args),
            **common,
        )


if __name__ == "__main__":
    main()
