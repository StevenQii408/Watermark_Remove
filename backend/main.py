import gc
import torch
import shutil
import traceback
import subprocess
import os
from pathlib import Path
import threading
import cv2
import sys
from functools import cached_property

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.config import *
from backend.tools.hardware_accelerator import HardwareAccelerator
from backend.tools.common_tools import is_video_or_image, is_image_file, get_readable_path, read_image
from backend.inpaint.sttn_auto_inpaint import STTNAutoInpaint
from backend.inpaint.sttn_det_inpaint import STTNDetInpaint
from backend.inpaint.lama_inpaint import LamaInpaint
from backend.inpaint.opencv_inpaint import OpenCVInpaint
from backend.inpaint.pure_background_inpaint import PureBackgroundInpaint
from backend.inpaint.propainter_inpaint import PropainterInpaint
from backend.tools.inpaint_tools import (create_mask, batch_generator, ensure_bgr_uint8,
                                          create_subtitle_masks, expand_frame_ranges,
                                          expand_solid_background_mask, mask_edge_density)
from backend.tools.model_config import ModelConfig
from backend.tools.ffmpeg_cli import FFmpegCLI
from backend.tools.subtitle_detect import SubtitleDetect
from backend.tools.video_io import FramePrefetcher, FFmpegVideoWriter
import tempfile
import multiprocessing
import time
from tqdm import tqdm
import numpy as np

class SubtitleRemover:
    def __init__(self, vd_path, gui_mode=False):
        # 线程锁
        self.lock = threading.RLock()
        # 用户指定的字幕区域位置
        self.sub_areas = []
        # 是否为gui运行，gui运行需要显示预览
        self.gui_mode = gui_mode
        self.hardware_accelerator = HardwareAccelerator.instance()
        # 是否使用硬件加速
        self.hardware_accelerator.set_enabled(config.hardwareAcceleration.value)
        self.model_config = ModelConfig()
        # 判断是否为图片
        self.is_picture = is_image_file(str(vd_path))
        # 视频路径
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(get_readable_path(vd_path))
        # 通过视频路径获取视频名称
        self.vd_name = Path(self.video_path).stem
        # 视频帧总数
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        # 视频帧率
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # 视频尺寸
        self.size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.mask_size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 创建视频临时对象，windows下delete=True会有permission denied的报错
        self.video_temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        # 创建视频写对象（使用 FFmpeg libx264 编码，比 mp4v 质量更好、文件更小）
        try:
            self.video_writer = FFmpegVideoWriter(get_readable_path(self.video_temp_file.name), self.fps, self.size)
        except Exception:
            self.video_writer = cv2.VideoWriter(get_readable_path(self.video_temp_file.name), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
        self.video_out_path = os.path.abspath(os.path.join(os.path.dirname(self.video_path), f'{self.vd_name}_no_sub.mp4'))
        self.propainter_inpaint = None
        self.ext = os.path.splitext(vd_path)[-1]
        if self.is_picture:
            pic_dir = os.path.join(os.path.dirname(self.video_path), 'no_sub')
            if not os.path.exists(pic_dir):
                os.makedirs(pic_dir)
            self.video_out_path = os.path.join(pic_dir, f'{self.vd_name}{self.ext}')

        # 总处理进度
        self.progress_total = 0
        self.progress_remover = 0
        self.isFinished = False
        # 是否将原音频嵌入到去除字幕后的视频
        self.is_successful_merged = False
        # 进度监听器列表
        self.progress_listeners = []
        # inpaint的frame_no区域列表, 默认为inpaint所有帧
        self.ab_sections = None
        self._preview_interval = 6
        self._preview_counter = 0

    @staticmethod
    def is_current_frame_no_start(frame_no, continuous_frame_no_list):
        """
        判断给定的帧号是否为开头，是的话返回结束帧号，不是的话返回-1
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no == frame_no:
                return True
        return False

    @staticmethod
    def find_frame_no_end(frame_no, continuous_frame_no_list):
        """
        判断给定的帧号是否为开头，是的话返回结束帧号，不是的话返回-1
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no <= frame_no <= end_no:
                return end_no
        return -1

    def update_progress(self, tbar, increment):
        tbar.update(increment)
        current_percentage = (tbar.n / tbar.total) * 100
        self.progress_remover = int(current_percentage)
        self.progress_total = self.progress_remover
        self.notify_progress_listeners()

    def append_output(self, *args):
        """输出信息到控制台
        Args:
            *args: 要输出的内容，多个参数将用空格连接
        """
        print(*args)

    def log_inpaint_fallback(self, inpaint_model):
        reason = getattr(inpaint_model, "last_fallback_reason", None)
        if reason:
            self.append_output(tr['Main']['InpaintFallback'].format(reason))
            inpaint_model.last_fallback_reason = None
    
    def add_progress_listener(self, listener):
        """
        添加进度监听器
        
        Args:
            listener: 一个回调函数，接收参数 (progress_total, isFinished)
        """
        if listener not in self.progress_listeners:
            self.progress_listeners.append(listener)
    
    def remove_progress_listener(self, listener):
        """
        移除进度监听器
        
        Args:
            listener: 要移除的监听器函数
        """
        if listener in self.progress_listeners:
            self.progress_listeners.remove(listener)
            
    def notify_progress_listeners(self):
        """
        通知所有进度监听器当前进度
        """
        for listener in self.progress_listeners:
            try:
                listener(self.progress_total, self.isFinished)
            except Exception as e:
                traceback.print_exc()

    def update_preview_with_comp(self, frame_ori, frame_comp):
        """
        更新预览
        """
        pass

    def emit_preview(self, frame_ori, frame_comp, mask=None, force=False):
        """Throttle GUI preview work without affecting video processing."""
        if not self.gui_mode:
            return
        self._preview_counter += 1
        if not force and self._preview_counter % self._preview_interval:
            return
        preview_frame = ensure_bgr_uint8(frame_ori).copy()
        self.update_preview_with_comp(preview_frame, ensure_bgr_uint8(frame_comp))

    @staticmethod
    def enrich_solid_background_mask(frame, mask):
        if config.pureBackgroundMode.value == "off":
            return mask
        return expand_solid_background_mask(frame, mask)

    @staticmethod
    def interpolate_subtitle_boxes(sub_list, frame_no, start_frame, end_frame):
        """Interpolate OCR boxes for frames that were skipped during sampling."""
        if frame_no in sub_list and sub_list[frame_no]:
            return list(sub_list[frame_no])
        known = sorted(
            key for key, boxes in sub_list.items()
            if start_frame <= key <= end_frame and boxes)
        if not known:
            return []
        before = max((key for key in known if key < frame_no), default=None)
        after = min((key for key in known if key > frame_no), default=None)
        if before is None:
            return list(sub_list[after])
        if after is None:
            return list(sub_list[before])
        previous = sorted(sub_list[before], key=lambda box: (box[0], box[2]))
        following = sorted(sub_list[after], key=lambda box: (box[0], box[2]))
        ratio = (frame_no - before) / max(1, after - before)
        boxes = []
        for index, first in enumerate(previous):
            second = following[min(index, len(following) - 1)]
            if len(first) != 4 or len(second) != 4:
                boxes.append(first)
                continue
            boxes.append(tuple(int(round(a + (b - a) * ratio)) for a, b in zip(first, second)))
        return boxes

    @staticmethod
    def is_fine_text_clip(frames, mask):
        """Identify short, fine text clips where LaMa is less destructive."""
        binary = (mask > 0).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        heights = [stats[index, cv2.CC_STAT_HEIGHT] for index in range(1, count)
                   if stats[index, cv2.CC_STAT_AREA] >= 4]
        if not heights or max(heights) > config.smallSubtitlePixelThreshold.value:
            return False
        return mask_edge_density(frames[0], mask) >= 0.08

    @staticmethod
    def has_moving_subtitles(sub_list):
        """Return whether adjacent detected subtitle frames show meaningful motion."""
        previous_boxes = None
        for frame_no in sorted(sub_list):
            boxes = sub_list[frame_no]
            if previous_boxes and boxes:
                first = previous_boxes[0]
                second = boxes[0]
                first_center = ((first[0] + first[1]) / 2.0, (first[2] + first[3]) / 2.0)
                second_center = ((second[0] + second[1]) / 2.0, (second[2] + second[3]) / 2.0)
                width = max(1.0, first[1] - first[0])
                height = max(1.0, first[3] - first[2])
                if (abs(second_center[0] - first_center[0]) > max(2.0, width * 0.03) or
                        abs(second_center[1] - first_center[1]) > max(2.0, height * 0.12)):
                    return True
            previous_boxes = boxes
        return False

    def basic_sttn_batch_limit(self):
        limit = config.getSttnMaxLoadNum()
        free_vram = HardwareAccelerator.instance().get_available_vram_mb()
        if free_vram > 0:
            limit = min(limit, max(1, int(max(512.0, free_vram - 1024.0) / 96.0)))
        return max(1, limit), free_vram

    def write_original_video(self, tbar):
        """Pass through the source frames when no text was detected."""
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.video_cap.read()
            if not ret:
                break
            self.video_writer.write(frame)
            self.update_progress(tbar, increment=1)

    def propainter_mode(self, tbar):
        sub_detector = SubtitleDetect(self.video_path, self.sub_areas)
        sub_list = sub_detector.find_subtitle_frame_no(sub_remover=self)
        if len(sub_list) == 0:
            self.append_output(tr['Main']['NoTextInAreaHint'])
            self.write_original_video(tbar)
            return
        self.append_output(tr['Main']['DetectedSubtitleBoxes'].format(
            sum(len(boxes) for boxes in sub_list.values())))
        continuous_frame_no_list = sub_detector.find_continuous_ranges_with_same_mask(sub_list)
        scene_div_points = sub_detector.get_scene_div_frame_no(self.video_path)
        continuous_frame_no_list = sub_detector.split_range_by_scene(continuous_frame_no_list,
                                                                          scene_div_points)
        del sub_detector
        gc.collect()        
        device = self.hardware_accelerator.device if self.hardware_accelerator.has_cuda() else torch.device("cpu")
        propainter_inpaint = PropainterInpaint(device, self.model_config.PROPAINTER_MODEL_DIR, config.propainterMaxLoadNum.value)
        self.append_output(tr['Main']['ProcessingStartRemovingSubtitles'])
        index = 0
        # 使用帧预读取，I/O 与推理重叠
        reader = FramePrefetcher(self.video_cap)
        while True:
            ret, frame = reader.read()
            if not ret:
                break
            index += 1
            # 如果当前帧没有水印/文本则直接写
            if index not in sub_list.keys():
                self.video_writer.write(frame)
                # self.append_output(f'write frame: {index}')
                self.update_progress(tbar, increment=1)
                self.emit_preview(frame, frame)
                continue
            # 如果有水印，判断该帧是不是开头帧
            else:
                # 如果是开头帧，则批推理到尾帧
                if self.is_current_frame_no_start(index, continuous_frame_no_list):
                    # self.append_output(f'No 1 Current index: {index}')
                    start_frame_no = index
                    # self.append_output(f'find start: {start_frame_no}')
                    # 找到结束帧
                    end_frame_no = self.find_frame_no_end(index, continuous_frame_no_list)
                    # 判断当前帧号是不是字幕起始位置
                    # 如果获取的结束帧号不为-1则说明
                    if end_frame_no != -1:
                        # self.append_output(f'find end: {end_frame_no}')
                        # ************ 读取该区间所有帧 start ************
                        temp_frames = list()
                        # 将头帧加入处理列表
                        temp_frames.append(frame)
                        inner_index = 0
                        # 一直读取到尾帧
                        while index < end_frame_no:
                            ret, frame = reader.read()
                            if not ret:
                                break
                            index += 1
                            temp_frames.append(frame)
                        # ************ 读取该区间所有帧 end ************
                        if len(temp_frames) < 1:
                            # 没有待处理，直接跳过
                            continue
                        elif len(temp_frames) == 1:
                            inner_index += 1
                            single_mask, _ = create_subtitle_masks(
                                self.mask_size, sub_list[start_frame_no])
                            single_mask = self.enrich_solid_background_mask(temp_frames[0], single_mask)
                            inpainted_frame = self.lama_inpaint.inpaint(temp_frames[0], single_mask)
                            self.video_writer.write(inpainted_frame)
                            self.emit_preview(temp_frames[0], inpainted_frame, single_mask, force=True)
                            # self.append_output(f'write frame: {start_frame_no + inner_index} with mask {sub_list[start_frame_no]}')
                            self.update_progress(tbar, increment=1)
                            continue
                        else:
                            # 将读取的视频帧分批处理
                            # 1. 获取当前批次使用的mask
                            mask = create_mask(self.mask_size, sub_list[start_frame_no])
                            mask = self.enrich_solid_background_mask(temp_frames[0], mask)
                            self.append_output(tr['Main']['InpaintMaskStats'].format(
                                int(np.count_nonzero(mask)), mask.shape[1], mask.shape[0]))
                            if not np.any(mask):
                                self.append_output(tr['Main']['NoTextInAreaHint'])
                                for original_frame in temp_frames:
                                    self.video_writer.write(original_frame)
                                self.update_progress(tbar, increment=len(temp_frames))
                                continue
                            pure_mode = config.pureBackgroundMode.value
                            if pure_mode != "off" and config.qualityProfile.value != "speed":
                                pure_inpaint = PureBackgroundInpaint(
                                    config.pureBackgroundVarianceThreshold.value,
                                    config.pureBackgroundTemporalWindow.value)
                                pure_frames = pure_inpaint(temp_frames, mask)
                                if pure_frames is not None or pure_mode == "force":
                                    output_frames = (pure_frames if pure_frames is not None
                                                     else pure_inpaint.force(temp_frames, mask))
                                    for output_index, output_frame in enumerate(output_frames):
                                        self.video_writer.write(output_frame)
                                        self.emit_preview(
                                            temp_frames[output_index], output_frame, mask,
                                            force=output_index == len(output_frames) - 1)
                                    self.update_progress(tbar, increment=len(output_frames))
                                    continue
                            for batch in batch_generator(temp_frames, config.propainterMaxLoadNum.value):
                                # 2. 调用批推理
                                if len(batch) == 1:
                                    single_mask = create_mask(self.mask_size, sub_list[start_frame_no])
                                    single_mask = self.enrich_solid_background_mask(batch[0], single_mask)
                                    inpainted_frame = self.lama_inpaint.inpaint(batch[0], single_mask)
                                    self.video_writer.write(inpainted_frame)
                                    self.emit_preview(batch[0], inpainted_frame, single_mask,
                                                      force=True)
                                    # self.append_output(f'write frame: {start_frame_no + inner_index} with mask {sub_list[start_frame_no]}')
                                    inner_index += 1
                                    self.update_progress(tbar, increment=1)
                                elif len(batch) > 1:
                                    inpainted_frames = propainter_inpaint(batch, mask)
                                    self.log_inpaint_fallback(propainter_inpaint)
                                    for i, inpainted_frame in enumerate(inpainted_frames):
                                        self.video_writer.write(inpainted_frame)
                                        # self.append_output(f'write frame: {start_frame_no + inner_index} with mask {sub_list[index]}')
                                        inner_index += 1
                                        self.emit_preview(batch[i], inpainted_frame, mask,
                                                          force=i == len(batch) - 1)
                                self.update_progress(tbar, increment=len(batch))

    def sttn_auto_mode(self, tbar):
        """
        使用sttn对选中区域进行重绘，不进行字幕检测
        """
        self.append_output(tr['Main']['ProcessingStartRemovingSubtitles'])
        mask_area_coordinates = []
        for sub_area in self.sub_areas:
            ymin, ymax, xmin, xmax = sub_area
            mask_area_coordinates.append((xmin, xmax, ymin, ymax))
        mask = create_mask(self.mask_size, mask_area_coordinates)
        sttn_video_inpaint = STTNAutoInpaint(self.hardware_accelerator.device, self.model_config.STTN_AUTO_MODEL_PATH, self.video_path)
        sttn_video_inpaint(input_mask=mask, input_sub_remover=self, tbar=tbar)

    def video_inpaint(self, tbar, model):
        sub_detector = SubtitleDetect(self.video_path, self.sub_areas)
        is_sttn_det = isinstance(model, STTNDetInpaint)
        sub_list = sub_detector.find_subtitle_frame_no(
            sub_remover=self, tracking_mode="sttn_det" if is_sttn_det else "default")
        if len(sub_list) == 0:
            self.append_output(tr['Main']['NoTextInAreaHint'])
            self.write_original_video(tbar)
            return
        self.append_output(tr['Main']['DetectedSubtitleBoxes'].format(
            sum(len(boxes) for boxes in sub_list.values())))
        continuous_frame_no_list = (
            sub_detector.find_continuous_ranges(sub_list)
            if is_sttn_det else sub_detector.find_continuous_ranges_with_same_mask(sub_list))
        moving_subtitles = is_sttn_det and self.has_moving_subtitles(sub_list)
        if is_sttn_det:
            tbar.write(f"Basic continuous subtitle ranges: {continuous_frame_no_list}")
        else:
            tbar.write(f"Subtitle detected: {continuous_frame_no_list}")
        backward_padding = (min(2, config.subtitleTimelineBackwardFrameCount.value)
                            if moving_subtitles else config.subtitleTimelineBackwardFrameCount.value)
        forward_padding = (min(2, config.subtitleTimelineForwardFrameCount.value)
                           if moving_subtitles else config.subtitleTimelineForwardFrameCount.value)
        continuous_frame_no_list = expand_frame_ranges(
            continuous_frame_no_list, backward_padding, forward_padding)
        if moving_subtitles:
            self.append_output('基础版：检测到移动字幕，缩短时间扩展并启用运动轨迹')
        tbar.write(
            f"Subtitle timeline expand ({backward_padding} <- -> {forward_padding}): "
            f"{continuous_frame_no_list}")
        continuous_frame_no_list = sub_detector.filter_and_merge_intervals(
            continuous_frame_no_list, config.sttnReferenceLength.value,
            min_frame=1, max_frame=self.frame_count)
        tbar.write(f'Subtitle filter_and_merge_intervals: {continuous_frame_no_list}')
        scene_div_points = sub_detector.get_scene_div_frame_no(self.video_path)
        continuous_frame_no_list = sub_detector.split_range_by_scene(
            continuous_frame_no_list, scene_div_points)
        continuous_frame_no_list = sub_detector.clamp_frame_ranges(
            continuous_frame_no_list, 1, self.frame_count, merge_overlaps=False)
        tbar.write(f'Subtitle split by scene: {continuous_frame_no_list}')
        del sub_detector
        gc.collect()
        start_end_map = dict()
        for start, end in continuous_frame_no_list:
            if not 1 <= start <= end <= self.frame_count:
                self.append_output(f'基础版：跳过非法处理区间 ({start}, {end})')
                continue
            start_end_map[start] = end
        current_frame_index = 0
        self.append_output(tr['Main']['ProcessingStartRemovingSubtitles'])
        # 使用帧预读取，I/O 与推理重叠
        reader = FramePrefetcher(self.video_cap)
        while True:
            ret, frame = reader.read()
            # 如果读取到为，则结束
            if not ret:
                break
            current_frame_index += 1
            # 判断当前帧号是不是字幕区间开始, 如果不是，则直接写
            if current_frame_index not in start_end_map.keys():
                self.video_writer.write(frame)
                # self.append_output(f'write frame: {current_frame_index}')
                self.update_progress(tbar, increment=1)
                self.emit_preview(frame, frame)
            # 如果是区间开始，则找到尾巴
            else:
                start_frame_index = current_frame_index
                end_frame_index = start_end_map[current_frame_index]
                tbar.write(f'processing frame {start_frame_index} to {end_frame_index}')
                # 用于存储需要去字幕的视频帧
                frames_need_inpaint = list()
                frames_need_inpaint.append(frame)
                inner_index = 0
                # 接着往下读，直到读取到尾巴
                for j in range(end_frame_index - start_frame_index):
                    ret, frame = reader.read()
                    if not ret:
                        break
                    current_frame_index += 1
                    frames_need_inpaint.append(frame)
                frame_blend_masks = []
                frame_model_masks = []
                mask_area_coordinates = []
                fine_text_candidate = False
                for offset in range(len(frames_need_inpaint)):
                    frame_no = start_frame_index + offset
                    boxes = self.interpolate_subtitle_boxes(
                        sub_list, frame_no, start_frame_index, end_frame_index)
                    valid_boxes = []
                    for area in boxes:
                        if len(area) == 4:
                            xmin, xmax, ymin, ymax = area
                            if (ymax - ymin) - (xmax - xmin) > config.subtitleYXAxisDifferencePixel.value:
                                continue
                            if ymax - ymin <= config.smallSubtitlePixelThreshold.value:
                                fine_text_candidate = True
                        valid_boxes.append(area)
                        if area not in mask_area_coordinates:
                            mask_area_coordinates.append(area)
                    blend_mask, model_mask = create_subtitle_masks(
                        self.mask_size, valid_boxes, prefer_polygon=is_sttn_det)
                    frame_blend_masks.append(blend_mask)
                    frame_model_masks.append(model_mask)

                blend_mask = np.maximum.reduce(frame_blend_masks)
                model_mask = np.maximum.reduce(frame_model_masks)
                pure_masks = ([self.enrich_solid_background_mask(
                    frames_need_inpaint[output_index], frame_blend_masks[output_index])
                    for output_index in range(len(frames_need_inpaint))]
                    if is_sttn_det else self.enrich_solid_background_mask(
                        frames_need_inpaint[0], blend_mask))
                self.append_output(tr['Main']['InpaintMaskStats'].format(
                    int(np.count_nonzero(blend_mask)), blend_mask.shape[1], blend_mask.shape[0]))
                if not np.any(blend_mask):
                    self.append_output(tr['Main']['NoTextInAreaHint'])
                    for original_frame in frames_need_inpaint:
                        self.video_writer.write(original_frame)
                    self.update_progress(tbar, increment=len(frames_need_inpaint))
                    continue
                # Flat-background path: deterministic color/gradient reconstruction is
                # faster and more stable than generative inpainting for these clips.
                pure_mode = config.pureBackgroundMode.value
                if pure_mode != "off" and config.qualityProfile.value != "speed":
                    pure_inpaint = PureBackgroundInpaint(
                        config.pureBackgroundVarianceThreshold.value,
                        config.pureBackgroundTemporalWindow.value)
                    pure_frames = pure_inpaint(frames_need_inpaint, pure_masks)
                    if pure_frames is not None or pure_mode == "force":
                        output_frames = pure_frames if pure_frames is not None else pure_inpaint.force(frames_need_inpaint, pure_masks)
                        if is_sttn_det:
                            self.append_output('基础版：命中纯色背景确定性填充')
                        for output_index, output_frame in enumerate(output_frames):
                            self.video_writer.write(output_frame)
                            self.emit_preview(
                                frames_need_inpaint[output_index], output_frame,
                                frame_blend_masks[output_index],
                                force=output_index == len(output_frames) - 1)
                        self.update_progress(tbar, increment=len(output_frames))
                        continue
                if (isinstance(model, STTNDetInpaint) and fine_text_candidate
                        and len(frames_need_inpaint) <= 6
                        and self.is_fine_text_clip(frames_need_inpaint, blend_mask)):
                    self.append_output(tr['Main'].get(
                        'FineTextFallback',
                        '短细字幕片段使用 LaMa 精细修复'))
                    try:
                        fine_outputs = [
                            self.lama_inpaint.inpaint(
                                original_frame, frame_blend_masks[output_index])
                            for output_index, original_frame in enumerate(frames_need_inpaint)]
                    except RuntimeError as error:
                        if 'out of memory' not in str(error).lower():
                            raise
                        self.append_output('LaMa 显存不足，回退 STTN 基础修复')
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        for output_index, output_frame in enumerate(fine_outputs):
                            self.video_writer.write(output_frame)
                            self.emit_preview(
                                frames_need_inpaint[output_index], output_frame,
                                frame_blend_masks[output_index],
                                force=output_index == len(frames_need_inpaint) - 1)
                        self.update_progress(tbar, increment=len(fine_outputs))
                        continue
                # self.append_output(f'inpaint with mask: {mask_area_coordinates}')
                batch_offset = 0
                batch_limit, free_vram = (self.basic_sttn_batch_limit()
                                          if isinstance(model, STTNDetInpaint)
                                          else (config.getSttnMaxLoadNum(), 0))
                if isinstance(model, STTNDetInpaint):
                    self.append_output(f'基础版：安全批次 {batch_limit}，可用显存 {free_vram:.0f} MB')
                pending_batches = list(batch_generator(frames_need_inpaint, batch_limit))
                while pending_batches:
                    batch = pending_batches.pop(0)
                    # 2. 调用批推理
                    if len(batch) >= 1:
                        batch_blend_masks = frame_blend_masks[batch_offset:batch_offset + len(batch)]
                        batch_model_masks = frame_model_masks[batch_offset:batch_offset + len(batch)]
                        try:
                            if isinstance(model, STTNDetInpaint):
                                inpainted_frames = model(
                                    batch, batch_model_masks, blend_masks=batch_blend_masks)
                                if len(inpainted_frames) != len(batch):
                                    self.append_output(
                                        f'基础版：模型输出帧数异常（输入 {len(batch)}，输出 '
                                        f'{len(inpainted_frames)}），当前批次保留原帧')
                                    inpainted_frames = batch
                                if model.last_wide_slice_count:
                                    self.append_output(
                                        f'基础版：宽字幕增加 {model.last_wide_slice_count} 个局部切片')
                            else:
                                inpainted_frames = model(batch, model_mask)
                        except RuntimeError as error:
                            if not isinstance(model, STTNDetInpaint) or 'out of memory' not in str(error).lower():
                                raise
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if len(batch) > 1:
                                middle = max(1, len(batch) // 2)
                                pending_batches[0:0] = [batch[:middle], batch[middle:]]
                                self.append_output(
                                    f'基础版：STTN 显存不足，批次 {len(batch)} 二分为 {len(batch[:middle])}+{len(batch[middle:])}')
                                continue
                            self.append_output('基础版：单帧仍显存不足，保留原帧')
                            inpainted_frames = [batch[0]]
                            self.log_inpaint_fallback(model)
                        self.log_inpaint_fallback(model)
                        for i, inpainted_frame in enumerate(inpainted_frames):
                            self.video_writer.write(inpainted_frame)
                            # self.append_output(f'write frame: {start_frame_index + inner_index} with mask')
                            inner_index += 1
                            preview_mask = batch_blend_masks[i] if isinstance(model, STTNDetInpaint) else model_mask
                            self.emit_preview(batch[i], inpainted_frame, preview_mask,
                                              force=i == len(batch) - 1)
                        batch_offset += len(batch)
                        if isinstance(model, STTNDetInpaint) and torch.cuda.is_available():
                            # 基础版按批次释放缓存，避免长视频逐批累积保留块。
                            torch.cuda.empty_cache()
                    self.update_progress(tbar, increment=len(batch))
        reader.stop()

    def run(self):
        # 记录开始时间
        start_time = time.time()
        if len(self.sub_areas) == 0:
            self.append_output(tr['Main']['FullScreenProcessingNote'])
            self.sub_areas.append((0, self.frame_height, 0, self.frame_width))
        self.append_output(tr['Main']['SubtitleArea'].format(self.sub_areas))
        self.append_output(tr['Main']['ABSection'].format(str(self.ab_sections).replace("range", "") if self.ab_sections is not None and len(self.ab_sections) > 0 else tr['Main']['ABSectionAll']))
        # 如果使用GPU加速，则打印GPU加速提示
        if self.hardware_accelerator.has_accelerator():
            accelerator_name = self.hardware_accelerator.accelerator_name
            if accelerator_name == 'DirectML' and config.inpaintMode.value not in [InpaintMode.STTN_AUTO, InpaintMode.STTN_DET]:
                self.append_output(tr['Main']['DirectMLWarning'])
        os.makedirs(os.path.dirname(self.video_out_path), exist_ok=True)
        # 重置进度条
        self.progress_total = 0
        tbar = tqdm(total=int(self.frame_count), unit='frame', position=0, file=sys.__stdout__,
                    desc='Subtitle Removing')
        if self.is_picture:
            original_frame = read_image(self.video_path)
            if original_frame is None:
                self.append_output(tr['Main']['ReadImageFailed'].format(self.video_path))
                return
            sub_detector = SubtitleDetect(self.video_path, self.sub_areas)
            sub_list = sub_detector.detect_subtitle(original_frame)
            del sub_detector
            gc.collect()
            if len(sub_list):
                mask = create_mask(original_frame.shape[0:2], sub_list)
                mask = self.enrich_solid_background_mask(original_frame, mask)
                inpainted_frame = self.lama_inpaint.inpaint(original_frame, mask)
                self.emit_preview(original_frame, inpainted_frame, mask, force=True)
            else:
                inpainted_frame = original_frame
                self.emit_preview(original_frame, inpainted_frame, force=True)
            cv2.imencode(self.ext, inpainted_frame)[1].tofile(self.video_out_path)
            tbar.update(1)
            self.progress_total = 100
        else:
            # 精准模式下，获取场景分割的帧号，进一步切割
            self.log_model()
            if config.inpaintMode.value == InpaintMode.PROPAINTER:
                self.propainter_mode(tbar)
            elif config.inpaintMode.value == InpaintMode.STTN_AUTO:
                self.sttn_auto_mode(tbar)
            elif config.inpaintMode.value == InpaintMode.STTN_DET:
                self.video_inpaint(tbar, self.sttn_det_inpaint)
            elif config.inpaintMode.value == InpaintMode.LAMA:
                self.video_inpaint(tbar, self.lama_inpaint)
            elif config.inpaintMode.value == InpaintMode.OPENCV:
                self.video_inpaint(tbar, OpenCVInpaint())
            else:
                raise Exception(f'inpaint mode: {config.inpaintMode.value} not implemented')

        self.video_cap.release()
        self.video_writer.release()
        if not self.is_picture:
            # 将原音频合并到新生成的视频文件中
            self.merge_audio_to_video()
        self.append_output(tr['Main']['FinishedProcessing'].format(self.video_out_path))
        self.append_output(tr['Main']['ProcessingTime'].format(round(time.time() - start_time)))
        self.isFinished = True
        self.progress_total = 100
        if os.path.exists(self.video_temp_file.name):
            try:
                os.remove(self.video_temp_file.name)
            except Exception:
                pass #ignore

    def log_model(self):
        model_key = {
            InpaintMode.STTN_DET: 'Basic',
            InpaintMode.PROPAINTER: 'Enhanced',
            InpaintMode.STTN_AUTO: 'SttnFast',
        }.get(config.inpaintMode.value, 'Basic')
        model_friendly_name = tr['ModelProfile'].get(model_key, tr['ModelProfile']['Basic'])
        model_device = 'CPU'
        if config.inpaintMode.value != InpaintMode.OPENCV and self.hardware_accelerator.has_accelerator():
            accelerator_name = self.hardware_accelerator.accelerator_name
            if accelerator_name == 'DirectML' and config.inpaintMode.value in [InpaintMode.STTN_AUTO, InpaintMode.STTN_DET]:
                model_device = 'DirectML'
            if self.hardware_accelerator.has_cuda() or self.hardware_accelerator.has_mps():
                model_device = accelerator_name
        self.append_output(tr['Main']['SubtitleRemoverModel'].format(f"{model_friendly_name} ({model_device})"))
        providers = ", ".join(self.hardware_accelerator.onnx_providers)
        providers_str = f" ({providers})" if providers else ""
        if config.inpaintMode.value == InpaintMode.STTN_AUTO:
            self.append_output(tr['Main'].get(
                'FastEraseMode', '字幕检测：快速擦除模式（未启用 OCR）'))
        else:
            self.append_output(tr['Main']['SubtitleDetectionModel'].format(
                tr['SubtitleExtractorGUI']['PreciseDetectionEnabled']))

    def merge_audio_to_video(self):
        # 创建音频临时对象，windows下delete=True会有permission denied的报错
        temp = tempfile.NamedTemporaryFile(suffix='.aac', delete=False)
        audio_extract_command = [FFmpegCLI.instance().ffmpeg_path,
                                 "-y", "-i", self.video_path,
                                 "-acodec", "copy",
                                 "-vn", "-loglevel", "error", temp.name]
        use_shell = True if os.name == "nt" else False
        try:
            subprocess.check_output(audio_extract_command, stdin=open(os.devnull), shell=use_shell, timeout=600)
        except Exception as e:
            traceback.print_exc()
            self.append_output(tr['Main']['FailToExtractAudio'].format(str(e)))
            return
        else:
            if os.path.exists(self.video_temp_file.name):
                audio_merge_command = [FFmpegCLI.instance().ffmpeg_path,
                                       "-y", "-i", self.video_temp_file.name,
                                       "-i", temp.name,
                                       "-vcodec", "copy",
                                       "-acodec", "copy",
                                       "-loglevel", "error", self.video_out_path]
                try:
                    subprocess.check_output(audio_merge_command, stdin=open(os.devnull), shell=use_shell, timeout=600)
                except Exception as e:
                    traceback.print_exc()
                    self.append_output(tr['Main']['FailToMergeAudio'].format(str(e)))
                    return
            if os.path.exists(temp.name):
                try:
                    os.remove(temp.name)
                except Exception:
                    #ignore
                    pass
            self.is_successful_merged = True
        finally:
            temp.close()
            if not self.is_successful_merged:
                try:
                    shutil.copy2(self.video_temp_file.name, self.video_out_path)
                except IOError as e:
                    self.append_output(tr['Main']['CopyFileFailed'].format(self.video_temp_file.name, self.video_out_path, str(e)))
            self.video_temp_file.close()

    @cached_property
    def lama_inpaint(self):
        model_path = os.path.join(self.model_config.LAMA_MODEL_DIR, 'big-lama.pt')
        device = self.hardware_accelerator.device if self.hardware_accelerator.has_cuda() or self.hardware_accelerator.has_mps() else torch.device("cpu")
        return LamaInpaint(device, model_path)

    @cached_property
    def sttn_det_inpaint(self):
        return STTNDetInpaint(self.hardware_accelerator.device, self.model_config.STTN_DET_MODEL_PATH)


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    from backend.tools.args_handler import parse_args
    args = parse_args()
    # force english
    config.set(config.interface, 'en')
    TRANSLATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interface', f"{config.interface.value}.ini")
    tr.read(TRANSLATION_FILE, encoding='utf-8')
    sr = SubtitleRemover(args.input)
    if not is_video_or_image(args.input):
        sr.append_output(f'Error: {video_path} is not supported not corrupted.')
        exit(-1)
    sr.sub_areas = args.subtitle_area_coords
    sr.video_out_path = args.output
    config.set(config.inpaintMode, InpaintMode(args.inpaint_mode))
    sr.run()
        
