import os
import time
import sys
import gc
from typing import List

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.config import config
from backend.inpaint.sttn.auto_sttn import InpaintGenerator
from backend.inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor
from backend.tools.inpaint_tools import (ensure_bgr_uint8, feather_mask,
                                          is_frame_number_in_ab_sections,
                                          normalize_mask)
from backend.tools.video_io import FramePrefetcher
from backend.tools.hardware_accelerator import HardwareAccelerator

# 定义图像预处理方式
_to_tensors = transforms.Compose([
    Stack(),  # 将图像堆叠为序列
    ToTorchFormatTensor()  # 将堆叠的图像转化为PyTorch张量
])

class STTNInpaint:
    def __init__(self, device, model_path):
        self.device = device
        # 1. 创建InpaintGenerator模型实例并装载到选择的设备上
        self.model = InpaintGenerator().to(self.device)
        # 2. 载入预训练模型的权重，转载模型的状态字典
        self.model.load_state_dict(torch.load(model_path, map_location='cpu')['netG'])
        # 3. # 将模型设置为评估模式
        self.model.eval()
        # 模型输入用的宽和高
        self.model_input_width, self.model_input_height = 640, 120
        # 2. 设置相连帧数
        self.neighbor_stride = config.sttnNeighborStride.value
        self.ref_length = config.sttnReferenceLength.value

    def _letterbox(self, image):
        target_w, target_h = self.model_input_width, self.model_input_height
        src_h, src_w = image.shape[:2]
        scale = min(target_w / max(1, src_w), target_h / max(1, src_h))
        resized_w = max(1, min(target_w, int(round(src_w * scale))))
        resized_h = max(1, min(target_h, int(round(src_h * scale))))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)
        left = (target_w - resized_w) // 2
        top = (target_h - resized_h) // 2
        right = target_w - resized_w - left
        bottom = target_h - resized_h - top
        border = cv2.BORDER_REFLECT_101 if min(src_h, src_w) > 1 else cv2.BORDER_REPLICATE
        image_padded = cv2.copyMakeBorder(resized, top, bottom, left, right, border)
        return image_padded, (left, top, resized_w, resized_h)

    @staticmethod
    def _unletterbox(image, metadata, output_size):
        left, top, resized_w, resized_h = metadata
        valid = image[top:top + resized_h, left:left + resized_w]
        return cv2.resize(valid, output_size, interpolation=cv2.INTER_CUBIC)

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        """
        :param input_frames: 原视频帧
        :param mask: 字幕区域mask
        """
        mask = (normalize_mask(input_mask) > 0.5).astype(np.uint8)[:, :, None]
        H_ori, W_ori = mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        # 确定去字幕的垂直高度部分
        split_h = max(32, int(W_ori * 3 / 16))
        inpaint_area = self.build_full_selection_areas(mask, W_ori, H_ori, split_h)
        # 初始化帧存储变量
        # 高分辨率帧存储列表（浅拷贝 + 逐帧 copy，避免 deepcopy 开销）
        frames_hr = [f.copy() for f in input_frames]
        frames_scaled = {}  # 存放缩放后帧的字典
        comps = {}  # 存放补全后帧的字典
        self._area_metadata = {}
        # 存储最终的视频帧
        inpainted_frames = []
        for k in range(len(inpaint_area)):
            frames_scaled[k] = []  # 为每个去除部分初始化一个列表

        # 读取并缩放帧
        for j in range(len(frames_hr)):
            image = frames_hr[j]
            # 对每个去除部分进行切割和缩放
            for k in range(len(inpaint_area)):
                y1, y2, x1, x2 = inpaint_area[k]
                image_crop = image[y1:y2, x1:x2, :]
                image_resize, metadata = self._letterbox(image_crop)
                if j == 0:
                    self._area_metadata[k] = metadata
                frames_scaled[k].append(image_resize)  # 将缩放后的帧添加到对应列表

        # 处理每一个去除部分
        for k in range(len(inpaint_area)):
            # 调用inpaint函数进行处理
            comps[k] = self.inpaint(frames_scaled[k])

        # 如果存在去除部分
        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]  # 取出原始帧
                # 对于模式中的每一个段落
                for k in range(len(inpaint_area)):
                    y1, y2, x1, x2 = inpaint_area[k]
                    crop = frame[y1:y2, x1:x2]
                    comp = self._unletterbox(
                        comps[k][j], self._area_metadata[k], (x2 - x1, y2 - y1))
                    comp = cv2.cvtColor(ensure_bgr_uint8(comp), cv2.COLOR_RGB2BGR)
                    alpha = feather_mask(mask[y1:y2, x1:x2, 0])[:, :, None]
                    frame[y1:y2, x1:x2] = np.clip(
                        comp.astype(np.float32) * alpha +
                        crop.astype(np.float32) * (1.0 - alpha),
                        0, 255).astype(np.uint8)
                # 将最终帧添加到列表
                inpainted_frames.append(frame)
                # print(f'processing frame, {len(frames_hr) - j} left')
        else:
            inpainted_frames = frames_hr
        return inpainted_frames

    @staticmethod
    def build_full_selection_areas(mask, width, height, max_height):
        binary = (normalize_mask(mask, (height, width)) > 0.5).astype(np.uint8)
        if not np.any(binary):
            return []
        ys = np.where(binary > 0)[0]
        selected_top, selected_bottom = int(ys.min()), int(ys.max()) + 1
        context = max(4, min(20, int(max_height * 0.08)))
        top = max(0, selected_top - context)
        bottom = min(height, selected_bottom + context)
        if bottom - top <= max_height:
            return [(top, bottom, 0, width)]
        overlap = max(16, min(20, max_height // 6))
        areas = []
        start = top
        while start < bottom:
            end = min(bottom, start + max_height)
            areas.append((start, end, 0, width))
            if end >= bottom:
                break
            start = max(start + 1, end - overlap)
        return areas

    @staticmethod
    def read_mask(path):
        img = cv2.imread(path, 0)
        # 转为binary mask
        ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        img = img[:, :, None]
        return img

    def get_ref_index(self, neighbor_ids, length):
        """
        采样整个视频的参考帧
        """
        # 初始化参考帧的索引列表
        ref_index = []
        # 在视频长度范围内根据ref_length逐步迭代
        for i in range(0, length, self.ref_length):
            # 如果当前帧不在近邻帧中
            if i not in neighbor_ids:
                # 将它添加到参考帧列表
                ref_index.append(i)
        # 返回参考帧索引列表
        return ref_index

    def inpaint(self, frames: List[np.ndarray]):
        """
        使用STTN完成空洞填充（空洞即被遮罩的区域）
        """
        frame_length = len(frames)
        if frame_length == 0:
            return []
        frames = [ensure_bgr_uint8(frame) for frame in frames]
        # 对帧进行预处理转换为张量，并进行归一化
        feats = _to_tensors(frames).unsqueeze(0) * 2 - 1
        # 把特征张量转移到指定的设备（CPU或GPU）
        feats = feats.to(self.device)
        # 初始化一个与视频长度相同的列表，用于存储处理完成的帧
        comp_frames = [None] * frame_length
        # 统一关闭梯度计算，用于推理阶段节省内存并加速
        with torch.no_grad():
            # 将处理好的帧通过编码器，产生特征表示
            feats = self.model.encoder(feats.view(frame_length, 3, self.model_input_height, self.model_input_width))
            # 获取特征维度信息
            _, c, feat_h, feat_w = feats.size()
            # 调整特征形状以匹配模型的期望输入
            feats = feats.view(1, frame_length, c, feat_h, feat_w)
            # 在设定的邻居帧步幅内循环处理视频
            for f in range(0, frame_length, self.neighbor_stride):
                # 计算邻近帧的ID
                neighbor_ids = [i for i in range(max(0, f - self.neighbor_stride), min(frame_length, f + self.neighbor_stride + 1))]
                # 获取参考帧的索引
                ref_ids = self.get_ref_index(neighbor_ids, frame_length)
                # 通过模型推断特征并传递给解码器以生成完成的帧
                pred_feat = self.model.infer(feats[0, neighbor_ids + ref_ids, :, :, :])
                # 将预测的特征通过解码器生成图片，并应用激活函数tanh
                pred_img = torch.tanh(self.model.decoder(pred_feat[:len(neighbor_ids), :, :, :]))
                # 将结果张量重新缩放到0到255的范围内
                pred_img = (pred_img + 1) / 2
                # 将张量移动回CPU并转为NumPy数组
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                # 遍历邻近帧
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = pred_img[i].astype(np.uint8)
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
        # 返回处理完成的帧序列
        return comp_frames


class STTNAutoInpaint:

    def read_frame_info_from_video(self):
        # 使用opencv读取视频
        reader = cv2.VideoCapture(self.video_path)
        # 获取视频的宽度, 高度, 帧率和帧数信息并存储在frame_info字典中
        frame_info = {
            'W_ori': int(reader.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5),  # 视频的原始宽度
            'H_ori': int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5),  # 视频的原始高度
            'fps': reader.get(cv2.CAP_PROP_FPS),  # 视频的帧率
            'len': int(reader.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)  # 视频的总帧数
        }
        # 返回视频读取对象、帧信息和视频写入对象
        return reader, frame_info

    def __init__(self, device, model_path, video_path, mask_path=None, clip_gap=None):
        # STTNInpaint视频修复实例初始化
        self.sttn_inpaint = STTNInpaint(device, model_path)
        # 视频和掩码路径
        self.video_path = video_path
        self.mask_path = mask_path
        # 设置输出视频文件的路径
        self.video_out_path = os.path.join(
            os.path.dirname(os.path.abspath(self.video_path)),
            f"{os.path.basename(self.video_path).rsplit('.', 1)[0]}_no_sub.mp4"
        )
        # 配置可在一次处理中加载的最大帧数
        if clip_gap is None:
            self.clip_gap = config.getSttnMaxLoadNum()
        else:
            self.clip_gap = clip_gap

    @staticmethod
    def _scene_cut_index(frames, threshold=32.0):
        for index in range(1, len(frames)):
            first = cv2.cvtColor(frames[index - 1], cv2.COLOR_BGR2GRAY)
            second = cv2.cvtColor(frames[index], cv2.COLOR_BGR2GRAY)
            if float(np.mean(cv2.absdiff(first, second))) >= threshold:
                return index
        return None

    def __call__(self, input_mask=None, input_sub_remover=None, tbar=None):
        reader = None
        writer = None
        try:
            # 读取视频帧信息
            reader, frame_info = self.read_frame_info_from_video()
            # 使用帧预读取，I/O 与推理重叠
            prefetcher = FramePrefetcher(reader)
            if input_sub_remover is not None:
                ab_sections = input_sub_remover.ab_sections
                
                writer = input_sub_remover.video_writer
            else:
                ab_sections = None
                # 创建视频写入对象，用于输出修复后的视频
                writer = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_info['fps'], (frame_info['W_ori'], frame_info['H_ori']))
            
            split_h = max(32, int(frame_info['W_ori'] * 3 / 16))

            if input_mask is None:
                # 读取掩码
                mask = self.sttn_inpaint.read_mask(self.mask_path)
            else:
                mask = (normalize_mask(input_mask) > 0.5).astype(np.uint8)[:, :, None]

            inpaint_area = STTNInpaint.build_full_selection_areas(
                mask, frame_info['W_ori'], frame_info['H_ori'], split_h)
            if not inpaint_area:
                tqdm.write('STTN fast erase: empty mask, copying original frames')
                while True:
                    success, image = prefetcher.read()
                    if not success:
                        break
                    writer.write(ensure_bgr_uint8(image))
                    if input_sub_remover is not None:
                        input_sub_remover.update_progress(tbar, increment=1)
                return
            # 根据可用显存动态调整 clip_gap，避免 OOM
            effective_clip_gap = self.clip_gap
            vram_mb = HardwareAccelerator.instance().get_available_vram_mb()
            if vram_mb > 0:
                max_frames_by_vram = int(max(512.0, vram_mb - 1536.0) / 128.0)
                max_frames_by_vram = max(max_frames_by_vram, 4)
                effective_clip_gap = min(self.clip_gap, max_frames_by_vram)
                if effective_clip_gap < self.clip_gap:
                    tqdm.write(f'GPU VRAM: {vram_mb:.0f}MB, adjusting clip_gap: {self.clip_gap} -> {effective_clip_gap}')
            context_size = min(config.sttnReferenceLength.value, max(1, effective_clip_gap // 4))
            pending = []
            history = []
            next_frame_index = 0
            while pending or next_frame_index < frame_info['len']:
                core = []
                while len(core) < effective_clip_gap:
                    if pending:
                        core.append(pending.pop(0))
                        continue
                    if next_frame_index >= frame_info['len']:
                        break
                    success, image = prefetcher.read()
                    if not success:
                        next_frame_index = frame_info['len']
                        break
                    core.append((next_frame_index, ensure_bgr_uint8(image)))
                    next_frame_index += 1
                if not core:
                    break
                while len(pending) < context_size and next_frame_index < frame_info['len']:
                    success, image = prefetcher.read()
                    if not success:
                        next_frame_index = frame_info['len']
                        break
                    pending.append((next_frame_index, ensure_bgr_uint8(image)))
                    next_frame_index += 1
                prefix = list(history[-context_size:])
                lookahead = list(pending[:context_size])
                inference_records = prefix + core + lookahead
                cut_index = self._scene_cut_index([item[1] for item in inference_records])
                core_prefix_len = len(prefix)
                if cut_index is not None and cut_index < core_prefix_len:
                    history = []
                    prefix = []
                    inference_records = core + list(pending[:context_size])
                elif cut_index is not None and cut_index < core_prefix_len + len(core):
                    split_at = max(0, cut_index - core_prefix_len)
                    if split_at < len(core):
                        pending[0:0] = core[split_at:]
                        core = core[:split_at]
                        history = []
                        prefix = []
                        inference_records = core + list(pending[:context_size])
                elif cut_index is not None and cut_index >= core_prefix_len + len(core):
                    lookahead = list(pending[:max(0, cut_index - core_prefix_len - len(core))])
                    inference_records = prefix + core + lookahead
                    history = []
                if not core:
                    continue
                tqdm.write(
                    f"Processing: {core[0][0] + 1} - {core[-1][0] + 1} / "
                    f"Total: {frame_info['len']} (context {len(prefix)}+{len(lookahead)})")
                frames_by_area = {k: [] for k in range(len(inpaint_area))}
                area_metadata = {}
                for _, image in inference_records:
                    for k, (y1, y2, x1, x2) in enumerate(inpaint_area):
                        crop = image[y1:y2, x1:x2]
                        resized, metadata = self.sttn_inpaint._letterbox(crop)
                        frames_by_area[k].append(resized)
                        area_metadata[k] = metadata
                try:
                    comps = {k: self.sttn_inpaint.inpaint(value)
                             for k, value in frames_by_area.items()}
                except Exception as error:
                    tqdm.write(f'STTN fast erase: block failed, fallback to original frames: {error}')
                    comps = {}
                output_lengths = [len(value) for value in comps.values()]
                if output_lengths and any(length != len(inference_records) for length in output_lengths):
                    tqdm.write('STTN fast erase: output count mismatch, fallback to original core frames')
                    comps = {}
                core_ids = {item[0]: index for index, item in enumerate(inference_records)}
                for frame_index, original in core:
                    frame = original.copy()
                    if comps and is_frame_number_in_ab_sections(frame_index, ab_sections):
                        output_index = core_ids.get(frame_index)
                        for k, (y1, y2, x1, x2) in enumerate(inpaint_area):
                            if output_index is None or output_index >= len(comps.get(k, [])):
                                continue
                            comp = self.sttn_inpaint._unletterbox(
                                comps[k][output_index], area_metadata[k], (x2 - x1, y2 - y1))
                            comp = cv2.cvtColor(ensure_bgr_uint8(comp), cv2.COLOR_RGB2BGR)
                            alpha = feather_mask(normalize_mask(mask[y1:y2, x1:x2]))[:, :, None]
                            crop = frame[y1:y2, x1:x2].astype(np.float32)
                            frame[y1:y2, x1:x2] = np.clip(
                                comp.astype(np.float32) * alpha + crop * (1.0 - alpha),
                                0, 255).astype(np.uint8)
                    writer.write(frame)
                    if input_sub_remover is not None:
                        if tbar is not None:
                            input_sub_remover.update_progress(tbar, increment=1)
                        input_sub_remover.emit_preview(original, frame)
                history = core[-context_size:]
                del frames_by_area, comps, inference_records
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error during video processing: {str(e)}")
            # 不抛出异常，允许程序继续执行
        finally:
            if reader:
                prefetcher.release()
            if writer:
                writer.release()


if __name__ == '__main__':
    mask_path = '../../test/test.png'
    video_path = '../../test/test.mp4'
    # 记录开始时间
    start = time.time()
    sttn_video_inpaint = STTNAutoInpaint(video_path, mask_path, clip_gap=config.getSttnMaxLoadNum())
    sttn_video_inpaint()
    print(f'video generated at {sttn_video_inpaint.video_out_path}')
    print(f'time cost: {time.time() - start}')
