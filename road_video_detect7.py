import os
import json
import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Optional


class VideoLaneDetector:
    def __init__(self):
        """初始化视频车道线检测器（含画面静止检测和稳定线段输出）"""
        # 基础参数设置
        self.gaussian_kernel_size = 5
        self.gaussian_sigma = 1.5
        self.canny_high_threshold = 0.9
        self.canny_low_ratio = 0.4
        self.hough_threshold = 80
        self.hough_min_line_length = 100
        self.hough_max_line_gap = 5

        # ROI区域参数
        self.roi_top_ratio = 0.3
        self.roi_bottom_ratio = 0.98
        self.roi_left_ratio = 0.05
        self.roi_right_ratio = 0.95

        # 聚类与延伸参数
        self.solid_angle_range = (-80, -10) + (10, 80)
        self.cluster_angle_threshold = 5.0  # 角度相似度阈值（度）
        self.max_clusters = 4  # 保留最大的4个聚类
        self.extension_tolerance = 5  # 边界扩展容差（像素）

        # 画面静止检测参数
        self.frame_diff_threshold = 5.0  # 帧差异阈值（低于此值认为无变化）
        self.static_frame_threshold = 5  # 连续静止帧数阈值（达到此值判定为静止）
        self.recent_frames = []  # 存储最近帧用于差异计算
        self.max_recent_frames = 2  # 用于差异计算的最近帧数
        self.is_static = False  # 当前是否为静止状态
        self.consecutive_static_count = 0  # 连续静止帧数

        # 稳定线段融合参数
        self.stable_line_buffer = []  # 静止期间收集的线段
        self.stable_buffer_size = 5  # 融合所需的静止帧数
        self.final_stable_lines = None  # 最终输出的稳定线段
        self.stable_output_saved = False  # 是否已保存稳定输出

        # 存储检测结果
        self.detection_results = []

        # 图像尺寸
        self.image_width = 0
        self.image_height = 0

    def gaussian_filter(self, src: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(
            src,
            (self.gaussian_kernel_size, self.gaussian_kernel_size),
            self.gaussian_sigma
        )

    def canny_edge_detection(self, src: np.ndarray) -> np.ndarray:
        mean_val = np.mean(src)
        high_threshold = min(255, max(50, mean_val * self.canny_high_threshold))
        low_threshold = high_threshold * self.canny_low_ratio
        return cv2.Canny(src, low_threshold, high_threshold)

    def roi_mask(self, src: np.ndarray) -> np.ndarray:
        rows, cols = src.shape[:2]
        top_y = int(rows * self.roi_top_ratio)
        bottom_y = int(rows * self.roi_bottom_ratio)
        top_left_x = int(cols * self.roi_left_ratio)
        top_right_x = int(cols * self.roi_right_ratio)

        mask = np.zeros_like(src)
        roi_vertices = np.array([[
            (0, bottom_y),
            (top_left_x, top_y),
            (top_right_x, top_y),
            (cols - 1, bottom_y)
        ]], dtype=np.int32)

        cv2.fillPoly(mask, roi_vertices, 255)
        return cv2.bitwise_and(src, mask)

    def _calculate_frame_diff(self, frame: np.ndarray) -> float:
        """计算当前帧与最近帧的差异，判断画面是否静止"""
        # 转换为灰度图并缩小尺寸以提高效率
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 240))

        # 初始化最近帧列表
        if len(self.recent_frames) < self.max_recent_frames:
            self.recent_frames.append(gray)
            return float('inf')  # 帧数不足时返回无穷大

        # 计算与最近帧的平均差异
        diff_sum = 0.0
        for prev_frame in self.recent_frames:
            diff = cv2.absdiff(gray, prev_frame)
            diff_sum += np.mean(diff)

        avg_diff = diff_sum / self.max_recent_frames

        # 更新最近帧列表（保持固定大小）
        self.recent_frames.pop(0)
        self.recent_frames.append(gray)

        return avg_diff

    def _update_static_state(self, frame: np.ndarray) -> None:
        """更新画面静止状态"""
        frame_diff = self._calculate_frame_diff(frame)

        if frame_diff < self.frame_diff_threshold:
            self.consecutive_static_count += 1
            # 达到连续静止帧数阈值，标记为静止状态
            if self.consecutive_static_count >= self.static_frame_threshold:
                self.is_static = True
        else:
            # 画面变动，重置状态
            self.is_static = False
            self.consecutive_static_count = 0
            self.stable_line_buffer = []
            self.stable_output_saved = False
            self.final_stable_lines = None

    def _fuse_stable_lines(self) -> List[List[int]]:
        """融合多帧线段，得到稳定的线段坐标"""
        if len(self.stable_line_buffer) < self.stable_buffer_size:
            return []

        # 收集所有缓冲的线段
        all_lines = []
        for lines in self.stable_line_buffer:
            all_lines.extend(lines)

        # 对所有线段进行最终聚类，得到稳定线段
        return self.lines_cluster(all_lines)

    def _calculate_line_angle(self, line: List[int]) -> float:
        """计算直线的角度（度）"""
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            return 90.0 if dy > 0 else -90.0  # 垂直直线
        return np.arctan2(dy, dx) * 180 / np.pi

    def _calculate_line_length(self, line: List[int]) -> float:
        """计算线段长度"""
        x1, y1, x2, y2 = line
        return np.hypot(x2 - x1, y2 - y1)

    def _calculate_line_parameters(self, line: List[int]) -> Tuple[float, float]:
        """计算直线的斜率和截距 (y = mx + b)"""
        x1, y1, x2, y2 = line
        if x2 == x1:  # 垂直直线
            return (np.inf, x1)  # 用无穷大表示垂直斜率，x = b

        m = (y2 - y1) / (x2 - x1)  # 斜率
        b = y1 - m * x1  # 截距
        return (m, b)

    def _calculate_cluster_bounds(self, lines: List[List[int]]) -> Dict:
        """计算聚类的边界范围（所有线段覆盖的最大/最小坐标）"""
        all_x = []
        all_y = []

        for line in lines:
            x1, y1, x2, y2 = line
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])

        # 计算边界（增加容差以确保覆盖完整）
        return {
            'min_x': min(all_x) - self.extension_tolerance,
            'max_x': max(all_x) + self.extension_tolerance,
            'min_y': min(all_y) - self.extension_tolerance,
            'max_y': max(all_y) + self.extension_tolerance
        }

    def _extend_line_to_bounds(self, line: List[int], bounds: Dict) -> List[int]:
        """将线段延伸至聚类边界范围内"""
        x1, y1, x2, y2 = line
        m, b = self._calculate_line_parameters(line)
        min_x, max_x = bounds['min_x'], bounds['max_x']
        min_y, max_y = bounds['min_y'], bounds['max_y']

        # 垂直直线 (x = b)
        if m == np.inf:
            x = int(b)
            # 确保x在聚类x边界内
            x_clamped = max(min_x, min(max_x, x))
            # 计算与y边界的交点
            y_start = min_y
            y_end = max_y
            return [x_clamped, y_start, x_clamped, y_end]

        # 非垂直直线 (y = mx + b)
        # 计算与x边界的交点
        y_at_min_x = m * min_x + b
        y_at_max_x = m * max_x + b

        # 计算与y边界的交点
        x_at_min_y = (min_y - b) / m if m != 0 else (min_x if y1 < y2 else max_x)
        x_at_max_y = (max_y - b) / m if m != 0 else (max_x if y1 < y2 else min_x)

        # 收集所有可能的边界交点
        candidates = []
        if min_y <= y_at_min_x <= max_y:
            candidates.append((min_x, y_at_min_x))
        if min_y <= y_at_max_x <= max_y:
            candidates.append((max_x, y_at_max_x))
        if min_x <= x_at_min_y <= max_x:
            candidates.append((x_at_min_y, min_y))
        if min_x <= x_at_max_y <= max_x:
            candidates.append((x_at_max_y, max_y))

        # 如果没有交点（线段完全在边界内），使用原始线段
        if not candidates:
            return line

        # 找到距离最远的两个点作为延伸后的端点
        candidates = [(int(x), int(y)) for x, y in candidates]
        max_dist = 0
        best_pair = None

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                dist = np.hypot(
                    candidates[i][0] - candidates[j][0],
                    candidates[i][1] - candidates[j][1]
                )
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (candidates[i], candidates[j])

        return [best_pair[0][0], best_pair[0][1], best_pair[1][0], best_pair[1][1]]

    def is_similar(self, angle1: float, angle2: float) -> bool:
        """相似度函数：判断两个角度是否相似"""
        angle_diff = abs(angle1 - angle2)
        angle_diff = min(angle_diff, 360 - angle_diff)  # 考虑角度周期性
        return angle_diff < self.cluster_angle_threshold

    def update_cluster(self, line: List[int], cluster: Dict) -> None:
        """更新聚类信息：追踪最长线段和边界范围"""
        line_angle = self._calculate_line_angle(line)
        line_length = self._calculate_line_length(line)

        # 更新角度（加权平均）
        new_angle = (cluster['center_angle'] * cluster['size'] + line_angle) / (cluster['size'] + 1)
        cluster['center_angle'] = new_angle

        # 更新最长线段
        if line_length > cluster['max_length']:
            cluster['max_length'] = line_length
            cluster['longest_line'] = line.copy()

        # 添加线段到聚类并更新边界
        cluster['all_lines'].append(line)
        cluster['bounds'] = self._calculate_cluster_bounds(cluster['all_lines'])
        cluster['size'] += 1

    def lines_cluster(self, lines: List[List[int]]) -> List[List[int]]:
        """直线聚类：每个聚类保留最长线段并延伸至该类边界"""
        if not lines:
            return []

        # 初始化聚类列表
        clusters = []
        for line in lines:
            line_angle = self._calculate_line_angle(line)
            flag = True  # 标记是否需要新建聚类

            for cluster in clusters:
                if self.is_similar(line_angle, cluster['center_angle']):
                    self.update_cluster(line, cluster)
                    flag = False
                    break

            if flag:
                # 新建聚类
                line_length = self._calculate_line_length(line)
                all_lines = [line]
                clusters.append({
                    'center_angle': line_angle,
                    'size': 1,
                    'max_length': line_length,
                    'longest_line': line.copy(),
                    'all_lines': all_lines,
                    'bounds': self._calculate_cluster_bounds(all_lines)
                })

        # 按聚类规模排序，取前4个最大的聚类
        clusters.sort(key=lambda x: x['size'], reverse=True)
        selected_clusters = clusters[:self.max_clusters]

        # 提取每个聚类的最长线段并延伸至该类边界
        result_lines = []
        for cluster in selected_clusters:
            extended_line = self._extend_line_to_bounds(
                cluster['longest_line'],
                cluster['bounds']
            )
            result_lines.append(extended_line)

        return result_lines

    def filter_solid_lines(self, lines: np.ndarray) -> List[List[int]]:
        """过滤出实线并准备聚类"""
        solid_lines = []
        if lines is None:
            return solid_lines

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.hypot(x2 - x1, y2 - y1)
            if length < self.hough_min_line_length * 0.8:
                continue

            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 and dy == 0:  # 避免无效线段
                continue

            angle = np.arctan2(dy, dx) * 180 / np.pi

            in_range = (self.solid_angle_range[0] < angle < self.solid_angle_range[1]) or \
                       (self.solid_angle_range[2] < angle < self.solid_angle_range[3])
            if not in_range:
                continue

            solid_lines.append([int(x1), int(y1), int(x2), int(y2)])

        return solid_lines

    def hough_transform(self, src: np.ndarray) -> List[List[int]]:
        """霍夫变换检测直线 -> 过滤实线 -> 聚类 -> 延伸至类别边界"""
        lines = cv2.HoughLinesP(
            src,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap
        )

        # 过滤实线
        solid_lines = self.filter_solid_lines(lines)
        if not solid_lines:
            return []

        # 直线聚类并延伸至类别边界
        clustered_lines = self.lines_cluster(solid_lines)
        return clustered_lines

    def draw_solid_lanes(self, frame: np.ndarray, solid_lines: List[List[int]],
                         is_static: bool = False, is_stable: bool = False) -> np.ndarray:
        """绘制线段，并标记静止状态和稳定线段"""
        result = frame.copy()
        # 为不同的聚类使用不同颜色
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0)]

        # 绘制状态指示器
        state_text = "State: Moving"
        state_color = (0, 0, 255)  # 红色表示运动
        if is_static:
            state_text = "State: Static"
            state_color = (0, 255, 0)  # 绿色表示静止
            if is_stable:
                state_text += " (Stable Lines)"
                state_color = (255, 0, 255)  # 紫色表示已生成稳定线段

        cv2.putText(
            result, state_text, (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1, state_color, 2
        )

        # 绘制线段
        for i, line in enumerate(solid_lines):
            x1, y1, x2, y2 = line
            # 稳定线段使用更粗的线宽和不同样式
            line_width = 6 if is_stable else 4
            color = (0, 255, 255) if is_stable else colors[i % len(colors)]

            cv2.line(result, (x1, y1), (x2, y2), color, line_width)
            cv2.circle(result, (x1, y1), 6, color, -1)
            cv2.circle(result, (x2, y2), 6, color, -1)

        return result

    def _normalize_coordinates(self, line: List[int]) -> List[float]:
        """将线段坐标归一化到0-1范围"""
        x1, y1, x2, y2 = line
        return [
            x1 / self.image_width, y1 / self.image_height,
            x2 / self.image_width, y2 / self.image_height
        ]

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, Dict]:
        start_time = time.time()

        # 更新画面静止状态
        self._update_static_state(frame)

        # 图像处理流程
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = self.gaussian_filter(gray)
        edges = self.canny_edge_detection(blurred)
        roi_edges = self.roi_mask(edges)
        current_lines = self.hough_transform(roi_edges)

        # 静止状态下收集线段用于稳定输出
        if self.is_static:
            self.stable_line_buffer.append(current_lines)
            # 限制缓冲大小
            if len(self.stable_line_buffer) > self.stable_buffer_size:
                self.stable_line_buffer.pop(0)

            # 缓冲足够时计算稳定线段
            if len(self.stable_line_buffer) >= self.stable_buffer_size:
                self.final_stable_lines = self._fuse_stable_lines()

        # 确定当前显示的线段
        display_lines = self.final_stable_lines if (self.is_static and self.final_stable_lines) else current_lines
        is_stable = self.is_static and self.final_stable_lines is not None

        # 绘制结果
        result_frame = self.draw_solid_lanes(
            frame,
            display_lines,
            is_static=self.is_static,
            is_stable=is_stable
        )

        # 准备归一化坐标
        normalized_current_lines = [self._normalize_coordinates(line) for line in current_lines]
        normalized_stable_lines = [self._normalize_coordinates(line) for line in
                                   self.final_stable_lines] if self.final_stable_lines else None
        normalized_display_lines = [self._normalize_coordinates(line) for line in display_lines]

        # 记录检测数据（包含原始坐标和归一化坐标）
        frame_data = {
            "frame_idx": frame_idx,
            "is_static": self.is_static,
            "current_lines": current_lines,
            "normalized_current_lines": normalized_current_lines,
            "stable_lines": self.final_stable_lines if is_stable else None,
            "normalized_stable_lines": normalized_stable_lines,
            "display_lines": display_lines,
            "normalized_display_lines": normalized_display_lines,
            "line_count": len(display_lines),
            "process_time": float(time.time() - start_time)
        }
        self.detection_results.append(frame_data)

        # 显示帧信息
        cv2.putText(
            result_frame,
            f"Frame: {frame_idx} | Lines: {len(display_lines)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 2
        )

        return result_frame, frame_data

    def process_video(self, input_path: str, output_path: str, json_path: str = None,
                      stable_output_path: str = None) -> None:
        cap = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(input_path, cv2.CAP_ANY)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {input_path}")

        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.image_width = width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.image_height = height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频信息: {width}x{height} | {fps:.1f} FPS | 共 {total_frames} 帧")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        start_total = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                result_frame, frame_data = self.process_frame(frame, frame_idx)
                out.write(result_frame)

                # 当检测到稳定线段且未保存时，保存稳定线段结果
                if (self.is_static and self.final_stable_lines is not None and
                        not self.stable_output_saved):
                    self._save_stable_results(stable_output_path, frame_idx)
                    self.stable_output_saved = True

            except Exception as e:
                print(f"处理第 {frame_idx} 帧出错: {str(e)}")
                out.write(frame)

            if frame_idx % 50 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"进度: {progress:.1f}% ({frame_idx}/{total_frames})")

            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # 保存整体检测结果
        if not json_path:
            json_path = os.path.splitext(output_path)[0] + "_results.json"

        # 准备归一化后的稳定线段
        normalized_stable_lines = None
        if self.final_stable_lines:
            normalized_stable_lines = [self._normalize_coordinates(line) for line in self.final_stable_lines]

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "video_path": input_path,
                "output_path": output_path,
                "total_frames": frame_idx,
                "fps": float(fps),
                "resolution": f"{width}x{height}",
                "total_time": float(time.time() - start_total),
                "final_stable_lines": self.final_stable_lines,
                "normalized_final_stable_lines": normalized_stable_lines,
                "frame_data": self.detection_results
            }, f, ensure_ascii=False, indent=2)

        print(f"处理完成！输出视频: {output_path} | 检测数据: {json_path}")
        if self.final_stable_lines and self.stable_output_saved:
            print(f"稳定线段已保存至: {stable_output_path}")

    def _save_stable_results(self, stable_output_path: str, frame_idx: int) -> None:
        """保存稳定线段结果到文件（包含原始坐标和归一化坐标）"""
        if not stable_output_path:
            stable_output_path = "stable_lane_lines.json"

        # 准备归一化后的稳定线段
        normalized_stable_lines = None
        if self.final_stable_lines:
            normalized_stable_lines = [self._normalize_coordinates(line) for line in self.final_stable_lines]

        # 准备稳定线段数据
        stable_data = {
            "detected_frame_idx": frame_idx,
            "stable_lines_count": len(self.final_stable_lines),
            "stable_lines": self.final_stable_lines,
            "normalized_stable_lines": normalized_stable_lines,
            "line_format": "每个线段格式: [x1, y1, x2, y2] (原始坐标)",
            "normalized_line_format": "每个线段格式: [x1, y1, x2, y2] (归一化坐标，范围[0,1])",
            "image_width": self.image_width,
            "image_height": self.image_height,
            "detection_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(stable_output_path, "w", encoding="utf-8") as f:
            json.dump(stable_data, f, ensure_ascii=False, indent=2)


def main():
    input_video = "tunnel.mp4"  # 输入视频路径
    output_video = "output_stable_lanes.mp4"  # 输出带检测结果的视频
    json_output = "detection_113.json"  # 完整检测数据
    stable_output = "data/detect_labels/detect_labels_113.json"  # 稳定线段结果

    output_dir = os.path.dirname(output_video)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        detector = VideoLaneDetector()
        detector.process_video(input_video, output_video, json_output, stable_output)
    except Exception as e:
        print(f"处理失败: {str(e)}")


if __name__ == "__main__":
    main()