import sys
from functools import cached_property

import cv2
from tqdm import tqdm

from .model_config import ModelConfig
from .hardware_accelerator import HardwareAccelerator
from .common_tools import get_readable_path
from .ocr import get_coordinates
from backend.config import config, tr
from backend.scenedetect import scene_detect
from backend.scenedetect.detectors import ContentDetector
from backend.tools.inpaint_tools import is_frame_number_in_ab_sections

class SubtitleDetect:
    """
    文本框检测类，用于检测视频帧中是否存在文本框
    """

    # 采样间隔，根据视频帧率在 _init_sample_step 中自适应设置
    SAMPLE_STEP = 3

    def __init__(self, video_path, sub_areas=[]):
        self.video_path = video_path
        self.sub_areas = sub_areas
        self._scene_div_points = None
        self.last_tracking_fill_count = 0
        self._init_sample_step()

    def _init_sample_step(self):
        """根据视频帧率自适应设置采样间隔，保持每秒至少采样8帧"""
        cap = cv2.VideoCapture(get_readable_path(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps >= 60:
            self.SAMPLE_STEP = 4
        elif fps >= 30:
            self.SAMPLE_STEP = 3
        else:
            self.SAMPLE_STEP = 2

    @cached_property
    def text_detector(self):
        import paddle
        paddle.disable_signal_handler()
        from paddleocr import TextDetection
        hardware_accelerator = HardwareAccelerator.instance()
        onnx_providers = hardware_accelerator.onnx_providers
        model_config = ModelConfig()
        return TextDetection(
            model_name=model_config.DET_MODEL_NAME,
            model_dir=model_config.DET_MODEL_DIR,
            device="cpu",
            enable_hpi=len(onnx_providers) > 0,
        )

    def detect_subtitle(self, img):
        temp_list = self._detect_once(img)
        small_trigger = (not temp_list or min((ymax - ymin for _, _, ymin, ymax in temp_list), default=999) <=
                         config.smallSubtitlePixelThreshold.value)
        if config.smallSubtitleEnhance.value and config.qualityProfile.value != "speed" and small_trigger:
            height, width = img.shape[:2]
            regions = self.sub_areas or [(0, height, 0, width)]
            for ymin, ymax, xmin, xmax in regions:
                x1, x2 = max(0, int(xmin)), min(width, int(xmax))
                y1, y2 = max(0, int(ymin)), min(height, int(ymax))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img[y1:y2, x1:x2]
                scale = max(2, min(4, int(config.smallSubtitleScale.value)))
                enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
                enhanced = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
                blur = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
                sharpened = cv2.addWeighted(enhanced, 1.35, blur, -0.35, 0)
                tile_size = config.smallSubtitleTileSize.value * scale
                overlap = min(config.smallSubtitleTileOverlap.value * scale, tile_size // 2)
                step = max(1, tile_size - overlap)
                variants = (enlarged, sharpened) if config.qualityProfile.value == "quality" else (enlarged,)
                for variant in variants:
                    for offset_x in range(0, variant.shape[1], step):
                        tile = variant[:, offset_x:min(variant.shape[1], offset_x + tile_size)]
                        if tile.size == 0:
                            continue
                        for box in self._detect_once(tile):
                            temp_list.append(((box[0] + offset_x) // scale + x1,
                                              (box[1] + offset_x) // scale + x1,
                                              box[2] // scale + y1, box[3] // scale + y1))
                        if offset_x + tile_size >= variant.shape[1]:
                            break
        temp_list = self._filter_and_merge(temp_list, img.shape[1], img.shape[0])
        return self._filter_to_sub_areas(temp_list)

    def _detect_once(self, img):
        temp_list = []
        results = self.text_detector.predict(img)
        sub_areas = self.sub_areas
        for res in results:
            dt_polys = res['dt_polys']
            if dt_polys is None or len(dt_polys) == 0:
                continue
            coordinate_list = get_coordinates(dt_polys.tolist())
            if not coordinate_list:
                continue
            temp_list.extend(coordinate_list)
        return temp_list

    def _filter_to_sub_areas(self, boxes):
        if not self.sub_areas:
            return boxes
        filtered = []
        for original_box in boxes:
            xmin, xmax, ymin, ymax = original_box
            for s_ymin, s_ymax, s_xmin, s_xmax in self.sub_areas:
                inter_w = max(0, min(xmax, s_xmax) - max(xmin, s_xmin))
                inter_h = max(0, min(ymax, s_ymax) - max(ymin, s_ymin))
                if inter_w * inter_h >= 0.25 * max(1, (xmax - xmin) * (ymax - ymin)):
                    filtered.append(original_box)
                    break
        return filtered

    @staticmethod
    def _filter_and_merge(boxes, width, height):
        valid = []
        for original_box in boxes:
            xmin, xmax, ymin, ymax = original_box
            xmin, xmax = max(0, int(xmin)), min(width - 1, int(xmax))
            ymin, ymax = max(0, int(ymin)), min(height - 1, int(ymax))
            if xmax > xmin and ymax > ymin:
                normalized_box = (xmin, xmax, ymin, ymax)
                valid.append(original_box if normalized_box == tuple(original_box) else normalized_box)
        valid.sort(key=lambda box: (box[2], box[0]))
        merged = []
        for box in valid:
            bx1, bx2, by1, by2 = box
            merged_into = False
            for index, current in enumerate(merged):
                cx1, cx2, cy1, cy2 = current
                inter = max(0, min(bx2, cx2) - max(bx1, cx1)) * max(0, min(by2, cy2) - max(by1, cy1))
                union = (bx2 - bx1) * (by2 - by1) + (cx2 - cx1) * (cy2 - cy1) - inter
                close = abs(((by1 + by2) - (cy1 + cy2)) / 2) <= config.subtitleAreaYAxisDifferencePixel.value
                overlap = union > 0 and inter / union >= 0.25
                adjacent = close and bx1 <= cx2 + max(3, (by2 - by1) // 2)
                if overlap or adjacent:
                    merged[index] = (min(bx1, cx1), max(bx2, cx2), min(by1, cy1), max(by2, cy2))
                    merged_into = True
                    break
            if not merged_into:
                merged.append(box)
        return merged

    def _build_sttn_det_tracks(self, sampled_results, scene_points=()):
        """Build motion-aware tracks for the basic STTN detector path."""
        if not sampled_results:
            return {}
        self.last_tracking_fill_count = 0
        detected_nos = sorted(sampled_results)
        tracked = {}
        previous_frame = detected_nos[0]
        previous_boxes = list(sampled_results[previous_frame])
        previous_state = [tuple(float(value) for value in box) for box in previous_boxes]
        velocities = [(0.0, 0.0, 0.0, 0.0) for _ in previous_state]
        tracked[previous_frame] = previous_boxes

        for frame_no in detected_nos[1:]:
            gap = frame_no - previous_frame
            current_boxes = list(sampled_results[frame_no])
            if any(previous_frame < point <= frame_no for point in scene_points):
                tracked[frame_no] = current_boxes
                previous_frame = frame_no
                previous_state = [tuple(float(value) for value in box) for box in current_boxes]
                velocities = [(0.0, 0.0, 0.0, 0.0) for _ in previous_state]
                continue
            matches = []
            candidates = []
            for previous_index, previous_box in enumerate(previous_state):
                velocity = velocities[previous_index]
                predicted = tuple(previous_box[index] + velocity[index] * gap for index in range(4))
                for current_index, current_box in enumerate(current_boxes):
                    iou = self._box_iou(predicted, current_box)
                    first_width = max(1.0, previous_box[1] - previous_box[0])
                    first_height = max(1.0, previous_box[3] - previous_box[2])
                    second_width = max(1.0, current_box[1] - current_box[0])
                    second_height = max(1.0, current_box[3] - current_box[2])
                    center_distance = (
                        abs((predicted[0] + predicted[1]) - (current_box[0] + current_box[1])) /
                        max(1.0, first_width + second_width) +
                        abs((predicted[2] + predicted[3]) - (current_box[2] + current_box[3])) /
                        max(1.0, first_height + second_height))
                    size_distance = abs(second_width - first_width) / first_width + abs(second_height - first_height) / first_height
                    score = (1.0 - iou) + 0.35 * center_distance + 0.15 * size_distance
                    if iou >= 0.05 or center_distance <= 1.25:
                        candidates.append((score, previous_index, current_index, predicted))
                    elif center_distance <= 2.0 and size_distance <= 1.2:
                        candidates.append((score + 0.5, previous_index, current_index, predicted))
            used_previous = set()
            used_current = set()
            for _, previous_index, current_index, predicted in sorted(candidates):
                if previous_index in used_previous or current_index in used_current:
                    continue
                used_previous.add(previous_index)
                used_current.add(current_index)
                current_box = current_boxes[current_index]
                measured = tuple((float(current_box[index]) - previous_state[previous_index][index]) /
                                 max(1, gap) for index in range(4))
                velocities[previous_index] = tuple(
                    0.65 * velocities[previous_index][index] + 0.35 * measured[index]
                    for index in range(4))
                matches.append((previous_index, current_index, predicted))

            if gap <= self.SAMPLE_STEP * 2:
                for fill_frame in range(previous_frame + 1, frame_no):
                    ratio = (fill_frame - previous_frame) / max(1, gap)
                    interpolated = []
                    for previous_index, current_index, predicted in matches:
                        first = previous_state[previous_index]
                        second = current_boxes[current_index]
                        interpolated.append(tuple(int(round(first[index] + (second[index] - first[index]) * ratio))
                                                 for index in range(4)))
                    if interpolated:
                        tracked[fill_frame] = interpolated
                        self.last_tracking_fill_count += 1

            tracked[frame_no] = current_boxes
            next_state = [tuple(float(value) for value in box) for box in current_boxes]
            next_velocities = [(0.0, 0.0, 0.0, 0.0) for _ in next_state]
            for previous_index, current_index, _ in matches:
                next_velocities[current_index] = velocities[previous_index]
            previous_frame = frame_no
            previous_state = next_state
            velocities = next_velocities
        return tracked

    def find_subtitle_frame_no(self, sub_remover=None, tracking_mode="default"):
        video_cap = cv2.VideoCapture(get_readable_path(self.video_path))
        frame_count = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        tbar = tqdm(total=int(frame_count), unit='frame', position=0, file=sys.__stdout__, desc='Subtitle Finding')
        current_frame_no = 0
        # 阶段1：采样检测，仅对每隔 sample_step 帧执行 OCR
        sampled_results = {}  # frame_no -> temp_list
        if sub_remover:
            sub_remover.append_output(tr['Main']['ProcessingStartFindingSubtitles'])
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            # 如果读取视频帧失败（视频读到最后一帧）
            if not ret:
                break
            # 读取视频帧成功
            current_frame_no += 1
            ab_sections = sub_remover.ab_sections if sub_remover else None
            if not is_frame_number_in_ab_sections(current_frame_no - 1, ab_sections):
                tbar.update(1)
                continue
            # 仅对采样帧执行 OCR 推理
            if (current_frame_no - 1) % self.SAMPLE_STEP == 0 or self.SAMPLE_STEP <= 1:
                temp_list = self.detect_subtitle(frame)
                if len(temp_list) > 0:
                    sampled_results[current_frame_no] = temp_list
            tbar.update(1)
            if sub_remover:
                sub_remover.progress_total = (100 * float(current_frame_no) / float(frame_count)) // 2
        video_cap.release()
        if tracking_mode == "sttn_det":
            subtitle_frame_no_box_dict = self._build_sttn_det_tracks(
                sampled_results, self.get_scene_div_frame_no(self.video_path))
            if sub_remover:
                if self.last_tracking_fill_count:
                    sub_remover.append_output(
                        f'基础版：轨迹补齐短暂漏检 {self.last_tracking_fill_count} 帧')
                sub_remover.append_output(tr['Main']['FinishedFindingSubtitles'])
            return {key: value for key, value in subtitle_frame_no_box_dict.items() if value}

        # 阶段2：按 IoU/中心距离匹配轨迹，并插值填充采样间的漏检帧
        subtitle_frame_no_box_dict = {}
        detected_nos = sorted(sampled_results.keys())
        max_gap = self.SAMPLE_STEP * 2
        for f, next_f in zip(detected_nos, detected_nos[1:]):
            subtitle_frame_no_box_dict[f] = sampled_results[f]
            if next_f - f <= max_gap:
                for fill_f in range(f + 1, next_f):
                    ratio = (fill_f - f) / (next_f - f)
                    current = sampled_results[f]
                    following = sampled_results[next_f]
                    interpolated = []
                    for box in current:
                        match = min(following, key=lambda other: self._box_distance(box, other), default=None)
                        if match is None or self._box_iou(box, match) < 0.05:
                            interpolated.append(box)
                            continue
                        interpolated.append(tuple(int(a + (b - a) * ratio) for a, b in zip(box, match)))
                    subtitle_frame_no_box_dict[fill_f] = interpolated
        # 添加最后一个检测帧
        if detected_nos:
            subtitle_frame_no_box_dict[detected_nos[-1]] = sampled_results[detected_nos[-1]]
        subtitle_frame_no_box_dict = self.unify_regions(subtitle_frame_no_box_dict)
        if sub_remover:
            sub_remover.append_output(tr['Main']['FinishedFindingSubtitles'])
        new_subtitle_frame_no_box_dict = dict()
        for key in subtitle_frame_no_box_dict.keys():
            if len(subtitle_frame_no_box_dict[key]) > 0:
                new_subtitle_frame_no_box_dict[key] = subtitle_frame_no_box_dict[key]
        return new_subtitle_frame_no_box_dict

    @staticmethod
    def _box_iou(first, second):
        x1 = max(first[0], second[0]); x2 = min(first[1], second[1])
        y1 = max(first[2], second[2]); y2 = min(first[3], second[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = max(1, (first[1] - first[0]) * (first[3] - first[2]))
        area_b = max(1, (second[1] - second[0]) * (second[3] - second[2]))
        return inter / max(1, area_a + area_b - inter)

    @staticmethod
    def _box_distance(first, second):
        return abs((first[0] + first[1]) - (second[0] + second[1])) + abs((first[2] + first[3]) - (second[2] + second[3]))

    @staticmethod
    def split_range_by_scene(intervals, points):
        # 确保离散值列表是有序的
        points.sort()
        # 用于存储结果区间的列表
        result_intervals = []
        # 遍历区间
        for start, end in intervals:
            # 在当前区间内的点
            current_points = [p for p in points if start <= p <= end]

            # 遍历当前区间内的离散点
            for p in current_points:
                # 如果当前离散点不是区间的起始点，添加从区间开始到离散点前一个数字的区间
                if start < p:
                    result_intervals.append((start, p - 1))
                # 更新区间开始为当前离散点
                start = p
            # 添加从最后一个离散点或区间开始到区间结束的区间
            result_intervals.append((start, end))
        # 输出结果
        return result_intervals

    def get_scene_div_frame_no(self, v_path=None):
        """
        获取发生场景切换的帧号
        """
        if self._scene_div_points is not None:
            return list(self._scene_div_points)
        v_path = v_path or self.video_path
        scene_div_frame_no_list = []
        scene_list = scene_detect(v_path, ContentDetector())
        for scene in scene_list:
            start, end = scene
            if start.frame_num == 0:
                pass
            else:
                scene_div_frame_no_list.append(start.frame_num + 1)
        self._scene_div_points = scene_div_frame_no_list
        return list(scene_div_frame_no_list)

    @staticmethod
    def are_similar(region1, region2):
        """判断两个区域是否相似。"""
        xmin1, xmax1, ymin1, ymax1 = region1
        xmin2, xmax2, ymin2, ymax2 = region2

        return abs(xmin1 - xmin2) <= config.subtitleAreaPixelToleranceXPixel.value and abs(xmax1 - xmax2) <= config.subtitleAreaPixelToleranceXPixel.value and \
            abs(ymin1 - ymin2) <= config.subtitleAreaPixelToleranceYPixel.value and abs(ymax1 - ymax2) <= config.subtitleAreaPixelToleranceYPixel.value

    def unify_regions(self, raw_regions):
        """将连续相似的区域统一，保持列表结构。"""
        if len(raw_regions) > 0:
            keys = sorted(raw_regions.keys())  # 对键进行排序以确保它们是连续的
            unified_regions = {}

            # 初始化
            last_key = keys[0]
            unify_value_map = {last_key: raw_regions[last_key]}

            for key in keys[1:]:
                current_regions = raw_regions[key]

                # 新增一个列表来存放匹配过的标准区间
                new_unify_values = []

                previous_regions = unify_value_map[last_key]
                unmatched_regions = list(previous_regions)
                for region in current_regions:
                    last_standard_region = min(
                        unmatched_regions,
                        key=lambda previous: self._box_distance(region, previous),
                        default=None)

                    # 如果当前的区间与前一个键的对应区间相似，我们统一它们
                    if last_standard_region and (self.are_similar(region, last_standard_region) or
                                                 self._box_iou(region, last_standard_region) >= 0.25):
                        new_unify_values.append(last_standard_region)
                        unmatched_regions.remove(last_standard_region)
                    else:
                        new_unify_values.append(region)

                # 更新unify_value_map为最新的区间值
                unify_value_map[key] = new_unify_values
                last_key = key

            # 将最终统一后的结果传递给unified_regions
            for key in keys:
                unified_regions[key] = unify_value_map[key]
            return unified_regions
        else:
            return raw_regions

    @staticmethod
    def find_continuous_ranges(subtitle_frame_no_box_dict):
        """
        获取字幕出现的起始帧号与结束帧号
        """
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值

        for i in range(1, len(numbers)):
            # 如果当前数字与前一个数字间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def find_continuous_ranges_with_same_mask(subtitle_frame_no_box_dict):
        numbers = sorted(list(subtitle_frame_no_box_dict.keys()))
        ranges = []
        start = numbers[0]  # 初始区间开始值
        for i in range(1, len(numbers)):
            # 如果当前帧号与前一个帧号间隔超过1，
            # 则上一个区间结束，记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] != 1:
                end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                ranges.append((start, end))
                start = numbers[i]  # 开始下一个连续区间
            # 如果当前帧号与前一个帧号间隔为1，且当前帧号对应的坐标点与上一帧号对应的坐标点不一致
            # 记录当前区间的开始与结束
            if numbers[i] - numbers[i - 1] == 1:
                if subtitle_frame_no_box_dict[numbers[i]] != subtitle_frame_no_box_dict[numbers[i - 1]]:
                    end = numbers[i - 1]  # 则该数字是当前连续区间的终点
                    ranges.append((start, end))
                    start = numbers[i]  # 开始下一个连续区间
        # 添加最后一个区间
        ranges.append((start, numbers[-1]))
        return ranges

    @staticmethod
    def clamp_frame_ranges(intervals, min_frame, max_frame, merge_overlaps=False):
        """Clamp frame intervals to valid video bounds and drop empty ranges."""
        normalized = []
        for start, end in intervals:
            start = max(min_frame, int(start))
            end = min(max_frame, int(end))
            if start <= end:
                normalized.append((start, end))
        normalized.sort()
        if not merge_overlaps:
            return normalized
        merged = []
        for start, end in normalized:
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def filter_and_merge_intervals(intervals, target_length, min_frame=1, max_frame=None):
        """
        合并传入的字幕起始区间，确保区间大小最低为STTN_REFERENCE_LENGTH
        复杂度 O(n log n)
        """
        if not intervals:
            return []
        max_frame = int(max_frame) if max_frame is not None else max(
            int(end) for _, end in intervals)
        intervals = SubtitleDetect.clamp_frame_ranges(
            intervals, int(min_frame), max_frame, merge_overlaps=False)
        if not intervals:
            return []
        target_length = max(1, int(target_length))
        # 一次遍历：扩展单点区间，利用排序后的相邻关系 O(n)
        expanded = []
        for i, (start, end) in enumerate(intervals):
            if start == end:  # 单点区间
                prev_end = expanded[-1][1] if expanded else min_frame - 1
                next_start = intervals[i + 1][0] if i + 1 < len(intervals) else float('inf')
                half = (target_length - 1) // 2
                new_start = max(min_frame, start - half, prev_end + 1)
                new_end = min(max_frame, start + half, next_start - 1)
                if new_end < new_start:
                    new_start, new_end = start, start
                expanded.append((new_start, new_end))
            else:
                expanded.append((start, end))
        # 一次遍历：合并重叠或相邻的短区间 O(n)
        merged = [expanded[0]]
        for start, end in expanded[1:]:
            last_start, last_end = merged[-1]
            last_len = last_end - last_start + 1
            cur_len = end - start + 1
            if (start <= last_end or start == last_end + 1) and (cur_len < target_length or last_len < target_length):
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return SubtitleDetect.clamp_frame_ranges(
            merged, int(min_frame), max_frame, merge_overlaps=False)
