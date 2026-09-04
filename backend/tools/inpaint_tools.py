import multiprocessing
import cv2
import numpy as np

from backend.config import config


def ensure_bgr_uint8(image, shape=None):
    """Normalize an image to a contiguous BGR uint8 array."""
    array = np.asarray(image)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    elif array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if shape is not None and array.shape[:2] != tuple(shape):
        array = cv2.resize(array, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(array)


def binary_mask_uint8(mask, shape=None):
    """Normalize a mask to a single-channel binary uint8 array (0 or 255)."""
    normalized = normalize_mask(mask, shape)
    return np.ascontiguousarray((normalized > 0.5).astype(np.uint8) * 255)


def normalize_mask(mask, shape=None):
    """Return a clipped single-channel float mask in the range [0, 1]."""
    array = np.asarray(mask)
    if array.ndim == 3:
        if array.shape[2] == 1:
            array = array[:, :, 0]
        else:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    array = array.astype(np.float32, copy=False)
    if array.size and float(array.max()) > 1.0:
        array = array / 255.0
    array = np.clip(array, 0.0, 1.0)
    if shape is not None and array.shape != tuple(shape):
        array = cv2.resize(array, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_NEAREST)
    return array


def alpha_blend(original, generated, mask):
    """Blend generated pixels only where the normalized mask is active."""
    source = ensure_bgr_uint8(original)
    replacement = ensure_bgr_uint8(generated, source.shape[:2])
    alpha = normalize_mask(mask, source.shape[:2])[:, :, None]
    if not np.isfinite(replacement).all():
        return source.copy()
    result = replacement.astype(np.float32) * alpha + source.astype(np.float32) * (1.0 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def feather_mask(mask, radius=1):
    """Return a small soft alpha transition around a binary mask."""
    normalized = normalize_mask(mask)
    if radius <= 0 or not np.any(normalized):
        return normalized
    kernel_size = max(3, int(radius) * 2 + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return np.clip(
        cv2.GaussianBlur(normalized, (kernel_size, kernel_size), 0.6), 0.0, 1.0)


def create_subtitle_masks(size, coords_list, context_pixels=3, prefer_polygon=False):
    """Create a tight compositing mask and a slightly wider model mask.

    The tight mask limits the pixels that can be changed in the output. The
    model mask provides a small amount of context around thin glyphs without
    turning the whole subtitle band into a hole.
    """
    height, width = int(size[0]), int(size[1])
    blend_mask = np.zeros((height, width), dtype=np.uint8)
    if coords_list:
        for coords in coords_list:
            values = tuple(coords)
            if len(values) == 4:
                xmin, xmax, ymin, ymax = [int(round(value)) for value in values]
                box_height = max(1, ymax - ymin)
                horizontal_pad = max(1, min(4, int(round(box_height * 0.08))))
                vertical_pad = max(1, min(5, int(round(box_height * 0.12))))
                x1 = max(0, xmin - horizontal_pad)
                y1 = max(0, ymin - vertical_pad)
                x2 = min(width - 1, xmax + horizontal_pad)
                y2 = min(height - 1, ymax + vertical_pad)
                polygon = getattr(coords, "polygon", None) if prefer_polygon else None
                if polygon is not None and len(polygon) == 4:
                    points = np.asarray(polygon, dtype=np.int32)
                    polygon_mask = np.zeros_like(blend_mask)
                    cv2.fillPoly(polygon_mask, [points], 255)
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (2 * horizontal_pad + 1, 2 * vertical_pad + 1))
                    blend_mask |= cv2.dilate(polygon_mask, kernel)
                elif x2 > x1 and y2 > y1:
                    cv2.rectangle(blend_mask, (x1, y1), (x2, y2), 255, thickness=-1)
            elif len(values) == 8:
                points = np.asarray(values, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(blend_mask, [points], 255)
    if not np.any(blend_mask):
        return blend_mask, blend_mask.copy()
    blend_mask = cv2.morphologyEx(
        blend_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    context = max(1, min(4, int(context_pixels)))
    model_mask = cv2.dilate(
        blend_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * context + 1, 2 * context + 1)),
    )
    return blend_mask, model_mask


def mask_edge_density(frame, mask, dilation=5):
    """Measure texture around a mask, used to avoid false solid-panel expansion."""
    image = ensure_bgr_uint8(frame)
    binary = (normalize_mask(mask, image.shape[:2]) > 0.5).astype(np.uint8)
    if not np.any(binary):
        return 0.0
    ring = cv2.dilate(binary, np.ones((2 * dilation + 1, 2 * dilation + 1), np.uint8))
    ring = (ring > 0) & (binary == 0)
    pixels = int(ring.sum())
    if pixels == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    return float(np.count_nonzero(edges[ring])) / pixels


def expand_solid_background_mask(frame, mask, max_area_ratio=8.0):
    """Include a locally uniform subtitle panel around OCR text when reliable."""
    image = ensure_bgr_uint8(frame)
    binary = (normalize_mask(mask) > 0.5).astype(np.uint8)
    if not np.any(binary):
        return binary_mask_uint8(binary)
    height, width = binary.shape
    result = binary.copy()
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    for index in range(1, count):
        x, y, box_width, box_height, area = stats[index]
        if area < 10:
            continue
        pad_x = max(16, int(box_width * 1.6))
        pad_y = max(10, int(box_height * 1.4))
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(width, x + box_width + pad_x), min(height, y + box_height + pad_y)
        roi_mask = binary[y1:y2, x1:x2]
        roi_lab = lab_image[y1:y2, x1:x2]
        roi_gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        roi_edges = cv2.Canny(roi_gray, 60, 150)
        roi_edge_density = float(np.count_nonzero(roi_edges)) / max(1, roi_edges.size)
        if roi_edge_density > 0.12:
            continue
        ring = cv2.dilate(roi_mask, np.ones((5, 5), np.uint8)) - roi_mask
        ring_pixels = roi_lab[ring > 0]
        if len(ring_pixels) < 20:
            continue
        reference = np.median(ring_pixels, axis=0)
        distance = np.linalg.norm(roi_lab - reference, axis=2)
        local_variance = float(np.mean(np.var(ring_pixels, axis=0)))
        threshold = min(42.0, max(14.0, local_variance * 1.8 + 10.0))
        candidate = (distance <= threshold).astype(np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        candidate[roi_mask > 0] = 1
        candidate_area = int(candidate.sum())
        roi_area = max(1, candidate.shape[0] * candidate.shape[1])
        if candidate_area < area * 1.4 or candidate_area > roi_area * 0.85:
            continue
        candidate_count, candidate_labels, candidate_stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
        selected = None
        text_center_x = x - x1 + box_width / 2.0
        text_center_y = y - y1 + box_height / 2.0
        for candidate_index in range(1, candidate_count):
            cx, cy, cw, ch, carea = candidate_stats[candidate_index]
            if (cx <= text_center_x <= cx + cw and cy <= text_center_y <= cy + ch):
                if carea <= area * max_area_ratio:
                    selected = candidate_labels == candidate_index
                    break
        if selected is not None:
            result[y1:y2, x1:x2] |= selected.astype(np.uint8)
    return result.astype(np.uint8) * 255


def get_local_inpaint_areas(mask, context_x=0.35, context_y=1.0, multiple=1):
    """Build bounded local crop areas around connected mask components."""
    binary = (normalize_mask(mask) > 0.5).astype(np.uint8)
    if not np.any(binary):
        return []
    height, width = binary.shape[:2]
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = []
    for index in range(1, count):
        x, y, box_width, box_height, area = stats[index]
        if area < 10:
            continue
        pad_x = max(12, int(box_width * context_x))
        pad_y = max(12, int(box_height * context_y))
        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y))
        x2 = min(width, int(x + box_width + pad_x))
        y2 = min(height, int(y + box_height + pad_y))
        if multiple > 1:
            crop_width = x2 - x1
            crop_height = y2 - y1
            width_remainder = crop_width % multiple
            height_remainder = crop_height % multiple
            x2 = min(width, x2 + (multiple - width_remainder) % multiple)
            y2 = min(height, y2 + (multiple - height_remainder) % multiple)
            if x2 - x1 < multiple:
                x1 = max(0, x2 - multiple)
            if y2 - y1 < multiple:
                y1 = max(0, y2 - multiple)
        areas.append((y1, y2, x1, x2))
    areas.sort()
    merged = []
    for area in areas:
        y1, y2, x1, x2 = area
        merged_index = None
        for index, current in enumerate(merged):
            cy1, cy2, cx1, cx2 = current
            intersects = min(x2, cx2) > max(x1, cx1) and min(y2, cy2) > max(y1, cy1)
            if intersects or (x1 <= cx2 and cx1 <= x2 and y1 <= cy2 and cy1 <= y2):
                merged_index = index
                merged[index] = (min(y1, cy1), max(y2, cy2), min(x1, cx1), max(x2, cx2))
                break
        if merged_index is None:
            merged.append(area)
    if multiple <= 1:
        return merged

    def align_area(area):
        y1, y2, x1, x2 = area
        crop_height = y2 - y1
        crop_width = x2 - x1
        target_height = ((crop_height + multiple - 1) // multiple) * multiple
        target_width = ((crop_width + multiple - 1) // multiple) * multiple
        target_height = min(target_height, height)
        target_width = min(target_width, width)
        if target_height < multiple:
            target_height = min(height, multiple)
        if target_width < multiple:
            target_width = min(width, multiple)

        extra_height = max(0, target_height - crop_height)
        extra_width = max(0, target_width - crop_width)
        top = min(extra_height // 2, y1)
        left = min(extra_width // 2, x1)
        y1 -= top
        x1 -= left
        y2 = min(height, y1 + target_height)
        x2 = min(width, x1 + target_width)
        y1 = max(0, y2 - target_height)
        x1 = max(0, x2 - target_width)
        return (y1, y2, x1, x2)

    return [align_area(area) for area in merged]

def batch_generator(data, max_batch_size):
    """
    根据data大小，生成最大长度不超过max_batch_size的均匀批次数据
    """
    n_samples = len(data)
    # 尝试找到一个比MAX_BATCH_SIZE小的batch_size，以使得所有的批次数量尽量接近
    batch_size = max_batch_size
    num_batches = n_samples // batch_size

    # 处理最后一批可能不足batch_size的情况
    # 如果最后一批少于其他批次，则减小batch_size尝试平衡每批的数量
    while n_samples % batch_size < batch_size / 2.0 and batch_size > 1:
        batch_size -= 1  # 减小批次大小
        num_batches = n_samples // batch_size

    # 生成前num_batches个批次
    for i in range(num_batches):
        yield data[i * batch_size:(i + 1) * batch_size]

    # 将剩余的数据作为最后一个批次
    last_batch_start = num_batches * batch_size
    if last_batch_start < n_samples:
        yield data[last_batch_start:]

def create_mask(size, coords_list):
    mask = np.zeros(size, dtype="uint8")
    if coords_list:
        for coords in coords_list:
            if len(coords) == 4:
                xmin, xmax, ymin, ymax = coords
                polygon = getattr(coords, "polygon", None)
                box_height = max(1, ymax - ymin)
                horizontal_pad = max(2, min(config.subtitleAreaDeviationPixel.value, box_height // 2))
                vertical_pad = max(3, config.subtitleAreaDeviationPixel.value)
                x1 = max(0, int(xmin - horizontal_pad))
                y1 = max(0, int(ymin - vertical_pad))
                x2 = min(size[1] - 1, int(xmax + horizontal_pad))
                y2 = min(size[0] - 1, int(ymax + vertical_pad))
                if polygon:
                    points = np.asarray(polygon, dtype=np.int32)
                    cv2.fillPoly(mask, [points], 255)
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (2 * horizontal_pad + 1, 2 * vertical_pad + 1))
                    mask = cv2.dilate(mask, kernel)
                else:
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
            elif len(coords) == 8:
                points = np.asarray(coords, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [points], 255)
    if np.any(mask):
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def get_inpaint_area_by_mask(W, H, h, mask, multiple=1):
    """
    获取字幕去除区域，根据mask来确定需要填补的区域和高度，
    并根据模型要求调整区域大小为指定倍数
    
    Args:
        W: 图像宽度
        H: 图像高度
        h: 检测区域高度
        mask: 遮罩图像
        multiple: 区域尺寸需要满足的倍数，默认为1
    
    Returns:
        调整后的绘画区域列表，格式为[(ymin, ymax, xmin, xmax), ...]
    """
    # 存储绘画区域的列表
    inpaint_area = []
    
    # 如果mask全为0，直接返回空列表
    if np.all(mask == 0):
        return inpaint_area
    
    # 使用连通组件分析找出mask中的所有孤岛
    # 首先确保mask是二值图像
    binary_mask = (mask > 0).astype(np.uint8) * 255
    
    # 查找连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # 跳过背景（标签0）
    island_info = []
    for i in range(1, num_labels):
        # 获取当前孤岛的统计信息
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # 忽略太小的区域（可能是噪点）
        if area < 10:
            continue
        
        # 保存孤岛信息：顶部y坐标，底部y坐标，中心点y坐标，面积，标签
        center_y = int(centroids[i][1])
        island_info.append((y, y + height, center_y, area, i))
    
    # 如果没有有效孤岛，返回空列表
    if not island_info:
        return inpaint_area
    
    # 按中心点y坐标排序孤岛
    island_info.sort(key=lambda x: x[2])
    
    # 尝试合并孤岛
    merged_islands = []
    current_group = [island_info[0]]
    
    for i in range(1, len(island_info)):
        # 当前组的范围
        min_y = min([island[0] for island in current_group])
        max_y = max([island[1] for island in current_group])
        
        # 当前孤岛
        top_y, bottom_y, center_y, _, _ = island_info[i]
        
        # 计算如果添加当前孤岛，新组的范围
        new_min_y = min(min_y, top_y)
        new_max_y = max(max_y, bottom_y)
        
        # 检查是否有mask连接当前组和新孤岛
        has_connection = False
        if max_y < top_y:  # 只有当前组在新孤岛上方时才需要检查连接
            # 检查两个区域之间是否有mask像素
            middle_region = binary_mask[max_y:top_y, :]
            if np.any(middle_region > 0):
                has_connection = True
        else:  # 重叠或相邻
            has_connection = True
        
        # 检查合并后的高度是否在h范围内，并且有连接
        if new_max_y - new_min_y <= h and has_connection:
            # 可以合并
            current_group.append(island_info[i])
        else:
            # 无法合并，保存当前组并开始新组
            merged_islands.append(current_group)
            current_group = [island_info[i]]
    
    # 添加最后一个组
    merged_islands.append(current_group)
    
    # 为每个合并后的组创建区域
    for group in merged_islands:
        # 获取组内所有孤岛的范围
        min_y = min([island[0] for island in group])
        max_y = max([island[1] for island in group])
        
        # 计算组的中心点
        center_y = sum([island[2] for island in group]) // len(group)
        
        # 确保区域高度精确等于h
        half_h = h // 2
        
        # 从中心点向上下扩展，确保高度为h
        ymin = max(0, center_y - half_h)
        ymax = ymin + h  # 确保高度精确等于h
        
        # 如果超出图像底部，从底部向上调整
        if ymax > H:
            ymax = H
            ymin = max(0, H - h)  # 确保高度为h
        
        # 检查是否包含了所有孤岛
        if ymin > min_y or ymax < max_y:
            # 如果区域不能完全包含所有孤岛，尝试调整位置但保持高度为h
            if max_y - min_y <= h:
                # 孤岛总高度不超过h，可以调整位置使其完全包含
                ymin = min_y
                ymax = ymin + h
                # 如果超出底部，从底部向上调整
                if ymax > H:
                    ymax = H
                    ymin = max(0, H - h)
            else:
                # 孤岛总高度超过h，无法完全包含，优先包含中心区域
                # 计算孤岛的中心
                island_center = (min_y + max_y) // 2
                ymin = max(0, island_center - half_h)
                ymax = ymin + h
                # 如果超出底部，从底部向上调整
                if ymax > H:
                    ymax = H
                    ymin = max(0, H - h)
        
        # 使用完整宽度
        xmin = 0
        xmax = W
        
        # 调整区域大小为指定倍数
        if multiple > 1:
            # 计算区域高度
            height = ymax - ymin
            # 计算需要调整的高度，使其成为multiple的倍数
            remainder = height % multiple
            
            if remainder != 0:
                # 需要调整的像素数
                adjust_pixels = multiple - remainder
                
                # 计算区域中心点
                center_y = (ymin + ymax) / 2
                
                # 优先对称扩展
                if ymin - adjust_pixels/2 >= 0 and ymax + adjust_pixels/2 <= H:
                    # 对称扩展
                    ymin = int(center_y - height/2 - adjust_pixels/2)
                    ymax = int(center_y + height/2 + adjust_pixels/2)
                # 如果对称扩展会超出边界，尝试对称缩小
                elif height > multiple:  # 确保缩小后高度至少为multiple
                    # 对称缩小
                    ymin = int(center_y - (height - remainder)/2)
                    ymax = int(center_y + (height - remainder)/2)
                # 如果无法对称调整，则尝试单边调整
                else:
                    # 向下扩展
                    if ymax + adjust_pixels <= H:
                        ymax += adjust_pixels
                    # 向上扩展
                    elif ymin - adjust_pixels >= 0:
                        ymin -= adjust_pixels
                    # 如果都不行，则尝试缩小区域
                    elif height > multiple:
                        ymax = ymin + height - remainder
            
            # 调整宽度，确保是multiple的倍数
            width = xmax - xmin
            remainder_w = width % multiple
            
            if remainder_w != 0:
                # 需要调整的像素数
                adjust_pixels_w = multiple - remainder_w
                
                # 计算中心点，对称缩小
                center_x = (xmin + xmax) / 2
                xmin = int(center_x - (width - remainder_w)/2)
                xmax = int(center_x + (width - remainder_w)/2)
        
        # 将该区域添加到列表中，格式为(ymin, ymax, xmin, xmax)
        area = (int(ymin), int(ymax), int(xmin), int(xmax))
        if area not in inpaint_area:
            inpaint_area.append(area)
    
    return inpaint_area  # 返回绘画区域列表，格式为[(ymin, ymax, xmin, xmax), ...]
    
def expand_frame_ranges(frame_ranges, backward_frame_count, forward_frame_count):
    """
    扩展帧区间列表，向前和向后扩展指定的帧数，并确保区间连续性
    
    Args:
        frame_ranges: 帧区间列表，格式为[(start1, end1), (start2, end2), ...]
        backward_frame_count: 向前扩展的帧数
        forward_frame_count: 向后扩展的帧数
        
    Returns:
        扩展后的帧区间列表，保证连续性
    """
    if not frame_ranges:
        return []
    
    # 按起始帧排序
    sorted_ranges = sorted(frame_ranges)
    expanded_ranges = []
    
    for i, (start, end) in enumerate(sorted_ranges):
        # 向前扩展，但不能小于1
        new_start = max(1, start - backward_frame_count)
        
        # 向后扩展
        new_end = end + forward_frame_count
        
        # 检查是否与下一个区间重叠
        if i < len(sorted_ranges) - 1:
            next_start = sorted_ranges[i + 1][0]
            
            # 如果扩展后的结束帧超过了下一个区间的起始帧
            if new_end >= next_start:
                # 计算中点
                mid_point = (end + next_start) // 2
                
                # 如果区间是连续的(相差1)，则对半平分
                if next_start - end == 1:
                    new_end = end  # 保持原结束帧
                else:
                    # 非连续区间，限制扩展到下一个区间起始帧减去backward_frame_count
                    max_expand = next_start - 1  # 确保不会与下一个区间重叠
                    new_end = min(new_end, max_expand)
        
        # 确保与前一个区间不重叠
        if expanded_ranges:
            prev_end = expanded_ranges[-1][1]
            if new_start <= prev_end:
                # 如果新区间的开始小于等于前一个区间的结束，调整开始位置
                new_start = prev_end + 1
        
        # 确保区间有效（开始不大于结束）
        if new_start <= new_end:
            expanded_ranges.append((new_start, new_end))
        else:
            # 如果调整后区间无效，保留原始区间
            expanded_ranges.append((start, end))
    
    return expanded_ranges

def is_frame_number_in_ab_sections(frame_no, ab_sections):
    """
    检查给定的帧号是否在指定的A/B区间内。

    Args:
        frame_no: 要检查的帧号
        ab_sections: 包含A/B区间的列表，格式为[range(start, end), ...]

    Returns:
        如果帧号在A/B区间内，返回True；否则返回False。
    """
    if ab_sections is None:
        return True
    if len(ab_sections) <= 0:
        return True
    for section in ab_sections:
        if frame_no in section:
            return True
    return False

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
