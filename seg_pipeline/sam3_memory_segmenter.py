from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .protocol import RetrievalItem


def _ensure_sam3_importable(sam3_source_path: Optional[str] = None) -> None:
    try:
        import sam3  # noqa: F401
        return
    except Exception:
        pass

    if sam3_source_path and Path(sam3_source_path).is_dir():
        sys.path.insert(0, sam3_source_path)
        return

    sam3_env = os.environ.get("SAM3_SOURCE_PATH", "")
    if sam3_env and Path(sam3_env).is_dir():
        sys.path.insert(0, sam3_env)
        return

    repo_root = Path(__file__).resolve().parents[1]
    for p in [repo_root / "resource" / "sam3", repo_root / "seg_pipeline" / "sam3"]:
        if p.is_dir():
            sys.path.insert(0, str(p))
            return


@dataclass(frozen=True)
class Sam3BuildConfig:
    version: str = "sam3"
    checkpoint_path: Optional[str] = None
    bpe_path: Optional[str] = None
    cuda_device: Optional[int] = None
    gpus_to_use: Optional[List[int]] = None
    compile: bool = False
    warm_up: bool = False
    async_loading_frames: bool = True


class Sam3MemoryAttentionSegmenter:
    def __init__(
        self,
        build_config: Optional[Sam3BuildConfig] = None,
        predictor: Any = None,
        extra_build_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._build_config = build_config or Sam3BuildConfig()
        self._predictor = predictor
        self._extra_build_kwargs = {} if extra_build_kwargs is None else dict(extra_build_kwargs)

    def get_predictor(self):
        if self._predictor is not None:
            return self._predictor

        _ensure_sam3_importable()
        from sam3.model_builder import build_sam3_predictor

        cfg = self._build_config
        if cfg.cuda_device is not None:
            import torch

            torch.cuda.set_device(int(cfg.cuda_device))

        build_kwargs = dict(self._extra_build_kwargs)
        if cfg.gpus_to_use is not None:
            if cfg.version == "sam3":
                build_kwargs["gpus_to_use"] = list(cfg.gpus_to_use)
            else:
                raise ValueError("gpus_to_use 仅对 version='sam3' 生效；sam3.1 请使用 cuda_device 或 CUDA_VISIBLE_DEVICES")
        self._predictor = build_sam3_predictor(
            checkpoint_path=cfg.checkpoint_path,
            bpe_path=cfg.bpe_path,
            version=cfg.version,
            compile=cfg.compile,
            warm_up=cfg.warm_up,
            async_loading_frames=cfg.async_loading_frames,
            **build_kwargs,
        )
        return self._predictor

    def segment(
        self,
        query_image_path: str,
        retrieval_items: Sequence[RetrievalItem],
        *,
        output_prob_thresh: float = 0.5,
        obj_id: int = 0,
        lock_memory: bool = True,
    ) -> Tuple["Image.Image", Dict[str, Any]]:
        predictor = self.get_predictor()

        with tempfile.TemporaryDirectory(prefix="sam3_memvid_") as tmpdir:
            frame_dir = Path(tmpdir)
            frame_map = self._materialize_jpeg_frames(
                frame_dir=frame_dir,
                conditioning_image_paths=[it.image_path for it in retrieval_items],
                query_image_path=query_image_path,
            )

            async_loading = getattr(self._build_config, "async_loading_frames", True)
            if self._build_config.version == "sam3":
                tracker, inference_state = self._init_sam3_tracker_state(
                    predictor=predictor,
                    frame_dir=frame_dir,
                    async_loading_frames=async_loading,
                )
            else:
                tracker = self._extract_tracker(predictor)
                inference_state = self._init_tracker_state_generic(
                    tracker=tracker,
                    frame_dir=frame_dir,
                    async_loading_frames=async_loading,
                )

            for frame_idx, it in enumerate(retrieval_items):
                mask_tensor, _mask_meta = self._load_mask_as_torch(it.mask_path)
                tracker.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    mask=mask_tensor,
                )

            tracker.propagate_in_video_preflight(inference_state, run_mem_encoder=True)

            target_frame_idx = len(retrieval_items)
            run_mem_encoder = not lock_memory

            out_mask_img: Optional[Image.Image] = None
            for (
                out_frame_idx,
                out_obj_ids,
                _out_low_res_masks,
                out_video_res_masks,
                _out_obj_scores,
            ) in tracker.propagate_in_video(
                inference_state=inference_state,
                start_frame_idx=target_frame_idx,
                max_frame_num_to_track=0,
                reverse=False,
                tqdm_disable=True,
                run_mem_encoder=run_mem_encoder,
            ):
                if out_frame_idx != target_frame_idx:
                    continue

                out_mask_img = self._select_and_binarize_mask(
                    out_video_res_masks=out_video_res_masks,
                    out_obj_ids=out_obj_ids,
                    obj_id=obj_id,
                    output_prob_thresh=output_prob_thresh,
                )

            if out_mask_img is None:
                raise RuntimeError("SAM3 未产生目标帧输出掩码")

            meta: Dict[str, Any] = {
                "sam3": {
                    "version": self._build_config.version,
                    "checkpoint_path": self._build_config.checkpoint_path,
                    "bpe_path": self._build_config.bpe_path,
                    "compile": bool(self._build_config.compile),
                    "warm_up": bool(self._build_config.warm_up),
                    "async_loading_frames": bool(self._build_config.async_loading_frames),
                },
                "synthetic_video": {
                    "frame_dir": str(frame_dir),
                    "ephemeral": True,
                    "frames": frame_map,
                },
                "memory_locked": bool(lock_memory),
                "run_mem_encoder_on_target": bool(run_mem_encoder),
                "output_prob_thresh": float(output_prob_thresh),
                "obj_id": int(obj_id),
            }

            return out_mask_img, meta

    def _init_tracker_state_generic(
        self,
        *,
        tracker: Any,
        frame_dir: Path,
        async_loading_frames: bool,
    ):
        import inspect

        sig = inspect.signature(tracker.init_state)
        params = sig.parameters

        if "video_path" in params:
            kwargs: Dict[str, Any] = {
                "video_path": str(frame_dir),
                "async_loading_frames": async_loading_frames,
            }
            if "offload_video_to_cpu" in params:
                kwargs["offload_video_to_cpu"] = False
            if "offload_state_to_cpu" in params:
                kwargs["offload_state_to_cpu"] = False
            return tracker.init_state(**kwargs)

        if "resource_path" in params:
            kwargs = {"resource_path": str(frame_dir), "async_loading_frames": async_loading_frames}
            if "offload_video_to_cpu" in params:
                kwargs["offload_video_to_cpu"] = False
            if "offload_state_to_cpu" in params:
                kwargs["offload_state_to_cpu"] = False
            return tracker.init_state(**kwargs)

        return tracker.init_state(str(frame_dir), False, False, async_loading_frames=async_loading_frames)

    def _init_sam3_tracker_state(
        self,
        *,
        predictor: Any,
        frame_dir: Path,
        async_loading_frames: bool,
    ) -> Tuple[Any, Dict[str, Any]]:
        model = getattr(predictor, "model", predictor)
        tracker = getattr(model, "tracker", None)
        if tracker is None:
            raise RuntimeError("sam3 模式下无法从 predictor.model 获取 tracker")

        if not hasattr(model, "init_state"):
            raise RuntimeError("sam3 模式下 predictor.model 缺少 init_state，无法初始化视频推理状态")
        if not hasattr(model, "run_backbone_and_detection"):
            raise RuntimeError("sam3 模式下 predictor.model 缺少 run_backbone_and_detection，无法预缓存特征")

        video_state = model.init_state(
            resource_path=str(frame_dir),
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            async_loading_frames=async_loading_frames,
        )

        feature_cache = video_state.get("feature_cache")
        input_batch = video_state.get("input_batch")
        constants = video_state.get("constants") or {}
        empty_prompt = constants.get("empty_geometric_prompt")
        num_frames = int(video_state.get("num_frames"))

        if not isinstance(feature_cache, dict):
            raise RuntimeError("sam3 模式下 model.init_state 未返回 feature_cache")
        if input_batch is None:
            raise RuntimeError("sam3 模式下 model.init_state 未返回 input_batch")
        if empty_prompt is None:
            raise RuntimeError("sam3 模式下 model.init_state 未返回 empty_geometric_prompt")

        import torch

        cached_features_all: Dict[int, Any] = {}
        with torch.no_grad():
            for frame_idx in range(num_frames):
                model.run_backbone_and_detection(
                    frame_idx=frame_idx,
                    num_frames=num_frames,
                    input_batch=input_batch,
                    geometric_prompt=empty_prompt,
                    feature_cache=feature_cache,
                    reverse=False,
                    allow_new_detections=False,
                )
                cached = feature_cache.get(frame_idx)
                if cached is not None:
                    cached_features_all[frame_idx] = cached

        inference_state = tracker.init_state(
            cached_features=cached_features_all,
            video_height=video_state.get("orig_height"),
            video_width=video_state.get("orig_width"),
            num_frames=num_frames,
            offload_state_to_cpu=video_state.get("offload_state_to_cpu", False),
        )
        return tracker, inference_state

    def _extract_tracker(self, predictor: Any):
        model = getattr(predictor, "model", predictor)
        tracker = getattr(model, "tracker", None)
        if tracker is None:
            raise RuntimeError("无法从 predictor 提取 tracker（缺少 model.tracker）")
        return tracker

    def _materialize_jpeg_frames(
        self,
        *,
        frame_dir: Path,
        conditioning_image_paths: List[str],
        query_image_path: str,
    ) -> List[Dict[str, Any]]:
        frames: List[Dict[str, Any]] = []
        all_paths = list(conditioning_image_paths) + [query_image_path]
        for idx, src_path in enumerate(all_paths):
            dst_path = frame_dir / f"{idx}.jpg"
            img = Image.open(src_path).convert("RGB")
            img.save(dst_path, format="JPEG", quality=95)
            frames.append(
                {
                    "frame_index": idx,
                    "source_path": src_path,
                    "jpeg_path": str(dst_path),
                }
            )
        return frames

    def _load_mask_as_torch(self, mask_path: str):
        import numpy as np
        import torch

        mask_img = Image.open(mask_path).convert("L")
        mask_np = np.array(mask_img)
        mask_np = (mask_np > 127).astype("float32")
        mask_tensor = torch.from_numpy(mask_np)

        meta = {
            "mask_path": mask_path,
            "height": int(mask_np.shape[0]),
            "width": int(mask_np.shape[1]),
        }
        return mask_tensor, meta

    def _select_and_binarize_mask(
        self,
        *,
        out_video_res_masks,
        out_obj_ids,
        obj_id: int,
        output_prob_thresh: float,
    ) -> Image.Image:
        import math
        import numpy as np
        import torch

        if isinstance(out_obj_ids, torch.Tensor):
            obj_ids_list = out_obj_ids.detach().cpu().tolist()
        else:
            obj_ids_list = list(out_obj_ids)

        if obj_id in obj_ids_list:
            obj_index = obj_ids_list.index(obj_id)
        else:
            obj_index = 0 if len(obj_ids_list) > 0 else None

        if obj_index is None:
            raise RuntimeError("目标帧没有任何对象输出")

        if hasattr(out_video_res_masks, "detach"):
            mask_scores = out_video_res_masks[obj_index, 0].detach().cpu().float().numpy()
        else:
            mask_scores = out_video_res_masks[obj_index, 0].astype("float32")

        p = float(output_prob_thresh)
        if p <= 0.0:
            logit_thresh = -float("inf")
        elif p >= 1.0:
            logit_thresh = float("inf")
        else:
            logit_thresh = math.log(p / (1.0 - p))

        mask_bin = (mask_scores > logit_thresh).astype(np.uint8) * 255
        return Image.fromarray(mask_bin, mode="L")
