import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import json

@dataclass
class PoseData:
    keypoints: np.ndarray  # (33, 3) - x, y, confidence
    timestamp: float
    frame_size: Tuple[int, int]
    world_keypoints: Optional[np.ndarray] = None  # (33, 4) - x, y, z

@dataclass
class PostureScore:
    score: float
    quality: str  # "good", "warning", "bad"
    problems: List[str]
    angles: dict
    view: str = "unknown"
    view_confidence: float = 0.0
    alignment_used: bool = False
    baseline_view: str = "none"
    features: Dict[str, float] = field(default_factory=dict)
    baseline_features: Dict[str, float] = field(default_factory=dict)

class PoseEstimator:
    def __init__(self, width=320, height=240):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.width, self.height = width, height
        #TODO json output of results
        self.json = []
        self.last_sec = -1

    def detect(self, frame: np.ndarray) -> Optional[PoseData]:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (self.width, self.height)) # Reduce computation by resizing
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if not results.pose_landmarks:
            return None

        keypoints = np.zeros((33, 3), dtype=np.float32)
        for i, lm in enumerate(results.pose_landmarks.landmark):
            keypoints[i] = [lm.x * w, lm.y * h, lm.visibility] # Normalized coordinates to pixels

        # 3D pose estimation relative to camera 
        world_keypoints = None
        if getattr(results, "pose_world_landmarks", None):
            world_keypoints = np.zeros((33, 4), dtype=np.float32)
            for i, lm in enumerate(results.pose_world_landmarks.landmark):
                world_keypoints[i] = [lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)]

        return PoseData(keypoints=keypoints, timestamp=time.time(), frame_size=(w, h), world_keypoints=world_keypoints)

    def close(self):
        self.pose.close()

# Posture quality based on feature alignment, symmetry as well as angles for neck, spine and shoulders. 
class PostureAnalyzer:
    def __init__(self, good_thresh=80, warn_thresh=60):
        self.good_thresh = good_thresh
        self.warn_thresh = warn_thresh
        self.baseline: Optional[np.ndarray] = None
        self.baseline_normalized: Optional[np.ndarray] = None
        self.baseline_view: Optional[str] = None
        self.baseline_features: Dict[str, float] = {}
        self.baseline_world_normalized: Optional[np.ndarray] = None

    def set_baseline(self, pose: PoseData):
        self.baseline = pose.keypoints.copy()
        self.baseline_normalized, _, _ = self._normalize(pose.keypoints)
        self.baseline_view, _ = self._estimate_view(pose.keypoints)
        self.baseline_features = self._extract_features(pose.keypoints, pose.world_keypoints)
        self.baseline_world_normalized = self._normalize_world(pose.world_keypoints)

    def _normalize(self, kp: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        pairs = [(11, 23), (12, 24)]
        lengths = []
        for s, h in pairs:
            if kp[s, 2] > 0.3 and kp[h, 2] > 0.3:
                lengths.append(np.linalg.norm(kp[s, :2] - kp[h, :2]))

        torso_len = float(np.mean(lengths)) if lengths else 1.0

        if kp[23, 2] > 0.3 and kp[24, 2] > 0.3:
            center = (kp[23, :2] + kp[24, :2]) / 2
        else:
            valid = kp[kp[:, 2] > 0.3, :2]
            center = np.mean(valid, axis=0) if len(valid) > 0 else np.array([0.0, 0.0])

        normalized = kp.copy()
        normalized[:, 0] = (kp[:, 0] - center[0]) / max(torso_len, 1)
        normalized[:, 1] = (kp[:, 1] - center[1]) / max(torso_len, 1)
        return normalized, torso_len, (float(center[0]), float(center[1]))

    # Joint angle computation
    def _compute_angles(self, kp: np.ndarray) -> dict:
        angles = {}

        def angle_to_vertical(p1, p2):
            vec = np.array([p1[0] - p2[0], p1[1] - p2[1]], dtype=np.float32)
            vertical = np.array([0.0, -1.0], dtype=np.float32)
            denom = np.linalg.norm(vec) + 1e-6
            cos = np.dot(vec, vertical) / denom
            return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

        def line_angle(p1, p2):
            raw = float(np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])))
            # Normalize to horizontal so level shoulders/hips are near 0
            return min(abs(raw), abs(raw - 180.0), abs(raw + 180.0))

        if kp[7, 2] > 0.3 and kp[11, 2] > 0.3:
            angles["neck"] = angle_to_vertical(kp[7, :2], kp[11, :2])

        if kp[11, 2] > 0.3 and kp[12, 2] > 0.3:
            angles["shoulders"] = line_angle(kp[11, :2], kp[12, :2])

        if kp[23, 2] > 0.3 and kp[24, 2] > 0.3:
            angles["hips"] = line_angle(kp[23, :2], kp[24, :2])

        if all(kp[i, 2] > 0.3 for i in [11, 12, 23, 24]):
            shoulder_mid = (kp[11, :2] + kp[12, :2]) / 2
            hip_mid = (kp[23, :2] + kp[24, :2]) / 2
            angles["spine"] = angle_to_vertical(shoulder_mid, hip_mid)

        return angles

    def _normalize_world(self, world_kp: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if world_kp is None or len(world_kp) < 25:
            return None
        valid = world_kp[:, 3] > 0.3
        if not (valid[11] and valid[12] and valid[23] and valid[24]):
            return None

        pts = world_kp[:, :3].astype(np.float32).copy()
        hip_mid = 0.5 * (pts[23] + pts[24])
        torso_len = 0.5 * (np.linalg.norm(pts[11] - pts[23]) + np.linalg.norm(pts[12] - pts[24]))
        torso_len = float(max(torso_len, 1e-6))
        pts = (pts - hip_mid) / torso_len
        out = np.zeros((len(world_kp), 4), dtype=np.float32)
        out[:, :3] = pts
        out[:, 3] = world_kp[:, 3]
        return out

    def _rigid_alignment_score(self, current_world: Optional[np.ndarray], baseline_world: Optional[np.ndarray]) -> float:
        if current_world is None or baseline_world is None:
            return 50.0

        candidate_idx = [0, 7, 8, 11, 12, 23, 24]
        valid_idx = [i for i in candidate_idx if current_world[i, 3] > 0.3 and baseline_world[i, 3] > 0.3]
        if len(valid_idx) < 4:
            return 50.0

        X = current_world[valid_idx, :3].astype(np.float64)
        Y = baseline_world[valid_idx, :3].astype(np.float64)

        Xc = X - X.mean(axis=0, keepdims=True)
        Yc = Y - Y.mean(axis=0, keepdims=True)

        # SVD 
        H = Xc.T @ Yc
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        X_aligned = Xc @ R
        residual = np.linalg.norm(X_aligned - Yc, axis=1)
        rms = float(np.sqrt(np.mean(residual ** 2)))

        # Nonlinear mapping: low residuals stay high, larger deviations fall faster.
        tolerance = 0.18
        return float(100.0 * np.exp(-0.5 * (rms / tolerance) ** 2))

    def _estimate_view(self, kp: np.ndarray) -> Tuple[str, float]:
        if not (kp[11, 2] > 0.3 and kp[12, 2] > 0.3):
            return "unknown", 0.0

        left = kp[11, :2]
        right = kp[12, :2]
        shoulder_width = abs(float(right[0] - left[0]))

        torso_lengths = []
        if kp[23, 2] > 0.3:
            torso_lengths.append(np.linalg.norm(kp[11, :2] - kp[23, :2]))
        if kp[24, 2] > 0.3:
            torso_lengths.append(np.linalg.norm(kp[12, :2] - kp[24, :2]))
        torso_len = float(np.mean(torso_lengths)) if torso_lengths else 1e-6
        torso_len = max(torso_len, 1e-6)

        ratio = shoulder_width / torso_len

        offset = 0.0
        if kp[0, 2] > 0.3:
            mid = (left[0] + right[0]) / 2
            offset = float(kp[0, 0] - mid)

        if ratio > 0.75:
            view = "front"
        elif ratio > 0.45:
            view = "three_quarter"
        else:
            view = "side"

        if view != "front":
            if offset > 10:
                view = "right_" + view
            elif offset < -10:
                view = "left_" + view

        confidence = float(np.clip(ratio, 0.0, 1.0))
        return view, confidence

    def _extract_features(self, kp: np.ndarray, world_kp: Optional[np.ndarray] = None) -> Dict[str, float]:
        feats: Dict[str, float] = {}
        _, torso_len, _ = self._normalize(kp) # Remove distance effect via normalization
        torso_len = max(torso_len, 1e-6)

        if all(kp[i, 2] > 0.3 for i in [11, 12]):
            left_sh = kp[11, :2]
            right_sh = kp[12, :2]
            shoulder_mid = (left_sh + right_sh) / 2
            feats["shoulder_tilt"] = float((left_sh[1] - right_sh[1]) / torso_len)
            feats["shoulder_width_ratio"] = float(np.linalg.norm(left_sh - right_sh) / torso_len)
        else:
            shoulder_mid = None

        if all(kp[i, 2] > 0.3 for i in [23, 24]):
            left_hip = kp[23, :2]
            right_hip = kp[24, :2]
            hip_mid = (left_hip + right_hip) / 2
            feats["hip_tilt"] = float((left_hip[1] - right_hip[1]) / torso_len)
            feats["hip_width_ratio"] = float(np.linalg.norm(left_hip - right_hip) / torso_len)
        else:
            hip_mid = None

        if shoulder_mid is not None and hip_mid is not None:
            feats["spine_shift"] = float((shoulder_mid[0] - hip_mid[0]) / torso_len)
            feats["torso_stack"] = float((kp[0, 0] - hip_mid[0]) / torso_len) if kp[0, 2] > 0.3 else 0.0
            feats["shoulder_hip_ratio"] = float(feats.get("shoulder_width_ratio", 0.0) / max(feats.get("hip_width_ratio", 1e-6), 1e-6))

        if shoulder_mid is not None and kp[0, 2] > 0.3:
            feats["head_center"] = float((kp[0, 0] - shoulder_mid[0]) / torso_len)

        if kp[7, 2] > 0.3 and kp[11, 2] > 0.3:
            feats["left_head_forward"] = float((kp[7, 0] - kp[11, 0]) / torso_len)
        if kp[8, 2] > 0.3 and kp[12, 2] > 0.3:
            feats["right_head_forward"] = float((kp[8, 0] - kp[12, 0]) / torso_len)

        if "left_head_forward" in feats and "right_head_forward" in feats:
            feats["head_forward"] = 0.5 * (feats["left_head_forward"] + feats["right_head_forward"])
        elif "left_head_forward" in feats:
            feats["head_forward"] = feats["left_head_forward"]
        elif "right_head_forward" in feats:
            feats["head_forward"] = feats["right_head_forward"]

        if world_kp is not None and len(world_kp) >= 25:
            wvis = world_kp[:, 3]
            if all(wvis[i] > 0.3 for i in [11, 12, 23, 24]):
                left_sh_w = world_kp[11, :3]
                right_sh_w = world_kp[12, :3]
                left_hip_w = world_kp[23, :3]
                right_hip_w = world_kp[24, :3]
                shoulder_mid_w = 0.5 * (left_sh_w + right_sh_w)
                hip_mid_w = 0.5 * (left_hip_w + right_hip_w)
                torso_len_w = 0.5 * (
                    np.linalg.norm(left_sh_w - left_hip_w) + np.linalg.norm(right_sh_w - right_hip_w)
                )
                torso_len_w = float(max(torso_len_w, 1e-6))
                feats["shoulder_depth"] = float((left_sh_w[2] - right_sh_w[2]) / torso_len_w)
                feats["hip_depth"] = float((left_hip_w[2] - right_hip_w[2]) / torso_len_w)
                feats["torso_stack_z"] = float((shoulder_mid_w[2] - hip_mid_w[2]) / torso_len_w)
                feats["shoulder_hip_depth_align"] = float(
                    ((left_sh_w[2] - left_hip_w[2]) + (right_sh_w[2] - right_hip_w[2])) / (2.0 * torso_len_w)
                )
            if wvis[0] > 0.3 and wvis[11] > 0.3 and wvis[12] > 0.3:
                shoulder_mid_w = 0.5 * (world_kp[11, :3] + world_kp[12, :3])
                torso_len_w = float(max(np.linalg.norm(world_kp[11, :3] - world_kp[23, :3]) if wvis[23] > 0.3 else 1.0, 1e-6))
                feats["head_forward_z"] = float((world_kp[0, 2] - shoulder_mid_w[2]) / torso_len_w)

        return feats

    def _feature_config(self, view: str) -> Dict[str, Tuple[float, float]]:
        if view == "front":
            return {
                "head_center": (0.10, 1.4),
                "shoulder_tilt": (0.08, 1.2),
                "hip_tilt": (0.10, 1.0),
                "spine_shift": (0.18, 1.2),
                "shoulder_depth": (0.2, 1.0),
                "hip_depth": (0.2, 0.8),
                "shoulder_hip_depth_align": (0.2, 1.3),
                "shoulder_hip_ratio": (0.20, 0.6),
            }
        if "side" in view:
            return {
                "head_forward_z": (0.2, 1.8),
                "torso_stack_z": (0.2, 1.8),
                "shoulder_hip_depth_align": (0.22, 1.6),
                "head_forward": (0.18, 0.7),
            }
        return {
            "head_center": (0.16, 1.0),
            "spine_shift": (0.16, 1.0),
            "shoulder_depth": (0.20, 1.0),
            "hip_depth": (0.20, 0.8),
            "torso_stack_z": (0.2, 1.4),
            "shoulder_hip_depth_align": (0.2, 1.2),
            "head_forward_z": (0.20, 1.0),
        }

    @staticmethod 
    def _soft_score(deviation: float, tolerance: float) -> float:
        # Smooth, nonlinear falloff: large errors drop faster.
        z = deviation / max(tolerance, 1e-6)
        return float(100.0 * np.exp(-0.5 * z * z))

    def analyze(self, pose: PoseData) -> PostureScore:
        kp = pose.keypoints
        view, view_conf = self._estimate_view(kp)
        angles = self._compute_angles(kp)
        current_features = self._extract_features(kp, pose.world_keypoints) 
        baseline_features = self.baseline_features.copy()

        problems = []
        weighted_sum = 0.0
        total_weight = 0.0

        config = self._feature_config(view)

        for name, (tol, weight) in config.items():
            if name not in current_features:
                continue

            target = baseline_features.get(name, 0.0)
            deviation = abs(current_features[name] - target)

            # Reduce trust if the relevant landmarks are weakly visible.
            visibility_scale = 1.0
            if "head" in name:
                visibility_scale = 0.9 if (kp[0, 2] > 0.3 or kp[7, 2] > 0.3 or kp[8, 2] > 0.3) else 0.4
            elif "shoulder" in name and not (kp[11, 2] > 0.3 and kp[12, 2] > 0.3):
                visibility_scale = 0.4
            elif "hip" in name and not (kp[23, 2] > 0.3 and kp[24, 2] > 0.3):
                visibility_scale = 0.4

            feature_score = self._soft_score(deviation, tol)
            weighted_sum += feature_score * weight * visibility_scale
            total_weight += weight * visibility_scale
            if feature_score < 65:
                problems.append(name)

        # Angle terms remain as a light secondary check.
        angle_cfg = {"neck": (25.0, 0.5), "shoulders": (12.0, 0.5), "hips": (12.0, 0.4), "spine": (18.0, 0.6)}
        for name, value in angles.items():
            if name not in angle_cfg:
                continue
            tol, weight = angle_cfg[name]
            angle_score = self._soft_score(abs(value), tol)
            weighted_sum += angle_score * weight
            total_weight += weight

        use_alignment = False
        alignment_score = None
        current_world_norm = self._normalize_world(pose.world_keypoints)
        if self.baseline_world_normalized is not None and current_world_norm is not None and view_conf > 0.35:
            alignment_score = self._rigid_alignment_score(current_world_norm, self.baseline_world_normalized)
            use_alignment = True
            weighted_sum += alignment_score * 0.9
            total_weight += 0.9

        total_score = weighted_sum / max(total_weight, 1e-6)
        quality = "good" if total_score >= self.good_thresh else "warning" if total_score >= self.warn_thresh else "bad"

        return PostureScore(
            score=total_score,
            quality=quality,
            problems=sorted(set(problems)),
            angles=angles,
            view=view,
            view_confidence=view_conf,
            alignment_used=use_alignment,
            baseline_view=self.baseline_view if self.baseline_view is not None else "none",
            features=current_features,
            baseline_features=baseline_features,
        )

class KalmanFilter:
    def __init__(self, process_noise=1e-2, measurement_noise=5e-1):
        self.filters = []
        self.initialized = False
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def reset(self):
        self.filters = []
        self.initialized = False

    def _create_filter(self, x: float, y: float):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.process_noise
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.measurement_noise
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        return kf

    def apply(self, keypoints: np.ndarray) -> np.ndarray:
        smoothed = keypoints.copy()
        if not self.initialized or len(self.filters) != len(keypoints):
            self.filters = [self._create_filter(float(kp[0]), float(kp[1])) for kp in keypoints]
            self.initialized = True
            return smoothed

        for i, kp in enumerate(keypoints):
            pred = self.filters[i].predict()
            pred_xy = pred[:2].ravel()
            conf = float(kp[2])
            if conf > 0.3:
                meas = np.array([[np.float32(kp[0])], [np.float32(kp[1])]])
                corrected = self.filters[i].correct(meas)
                smoothed[i, 0] = float(corrected[0, 0])
                smoothed[i, 1] = float(corrected[1, 0])
            else:
                smoothed[i, 0] = float(pred_xy[0])
                smoothed[i, 1] = float(pred_xy[1])
                smoothed[i, 2] = conf * 0.9
        return smoothed

# Lucas - Kanade or we can try something else
class OpticalFlowTracker:
    def __init__(self):
        self.prev_gray = None
        self.prev_keypoints = None
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def reset(self):
        self.prev_gray = None
        self.prev_keypoints = None

    def track(self, frame: np.ndarray, new_keypoints: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if new_keypoints is not None:
            self.prev_gray = gray
            self.prev_keypoints = new_keypoints.copy()
            return new_keypoints

        if self.prev_gray is None or self.prev_keypoints is None:
            return None

        valid = self.prev_keypoints[:, 2] > 0.3
        if np.sum(valid) < 4:
            self.prev_gray = gray
            return self.prev_keypoints

        # LK method for optical flow
        pts = self.prev_keypoints[valid, :2].reshape(-1, 1, 2).astype(np.float32)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts, None, **self.lk_params)

        tracked = self.prev_keypoints.copy()
        valid_idx = np.where(valid)[0]
        for i, idx in enumerate(valid_idx):
            if int(status[i, 0]) == 1:
                tracked[idx, :2] = next_pts[i].ravel()
            else:
                tracked[idx, 2] *= 0.5

        self.prev_gray = gray
        self.prev_keypoints = tracked.copy()
        return tracked

# UI Overlay
class OverlayRenderer:
    COLORS = {"good": (0, 255, 0), "warning": (0, 165, 255), "bad": (0, 0, 255)}
    CONNECTIONS = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]

    def draw(self, frame: np.ndarray, keypoints: Optional[np.ndarray], score: Optional[PostureScore],
             fps: float, is_calibrating: bool = False) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        color = self.COLORS.get(score.quality, (255, 255, 255)) if score else (255, 255, 255)

        if keypoints is not None:
            for i, j in self.CONNECTIONS:
                if keypoints[i, 2] > 0.3 and keypoints[j, 2] > 0.3:
                    p1 = tuple(keypoints[i, :2].astype(int))
                    p2 = tuple(keypoints[j, :2].astype(int))
                    cv2.line(out, p1, p2, color, 2)

            for kp in keypoints:
                if kp[2] > 0.3:
                    cv2.circle(out, tuple(kp[:2].astype(int)), 4, (0, 255, 255), -1)

        if score:
            cv2.rectangle(out, (w - 220, 10), (w - 10, 95), (0, 0, 0), -1)
            cv2.putText(out, f"Score: {score.score:.0f}", (w - 205, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(out, score.quality.upper(), (w - 205, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(out, f"3D Align: {'ON' if score.alignment_used else 'OFF'}", (w - 235, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

            if score.quality == "good":
                banner_color = (0, 255, 0)  # Green
                banner_text = "Good Posture Detected"
            else:
                banner_color = (255, 0, 0)  # Blue (OpenCV uses BGR instead of RGB)
                banner_text = "Stand up Straight"


            (text_w, text_h), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            text_x = w // 2 - text_w // 2
            text_y = h - 40
            cv2.rectangle(out, (w // 2 - 160, h - 70), (w // 2 + 160, h - 25), banner_color, -1)
            cv2.putText(out, banner_text, (w // 2 - 135, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.putText(out, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if score:
            cv2.putText(out, f"View: {score.view}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2)
            cv2.putText(out, f"Baseline: {score.baseline_view}", (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2)
            cv2.putText(out, f"Conf: {score.view_confidence:.2f}", (10, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2)

            # Show current vs baseline feature targets for active features
            y = 148
            cv2.putText(out, "Feature Cur Base", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,0), 1)
            y += 18
            shown = 0
            for name in score.features:
                if shown >= 7:
                    break
                cur = score.features[name]
                base = score.baseline_features.get(name, 0.0)
                line = f"{name[:12]:12} {cur:+.2f} {base:+.2f}"
                cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1)
                y += 18
                shown += 1

        if is_calibrating:
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)
            cv2.putText(out, "CALIBRATING - Hold Still!!!", (w // 2 - 180, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        return out

# Main App
class PostureFixApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PostureFix")
        self.root.geometry("960x620")

        self.estimator = PoseEstimator(320, 240)
        self.analyzer = PostureAnalyzer()
        self.tracker = OpticalFlowTracker()
        self.kalman = KalmanFilter()
        self.renderer = OverlayRenderer()

        self.running = False
        self.cap = None
        self.current_pose = None
        self.current_score = None
        self.fps = 0.0
        self.frame_count = 0
        self.frame_index = 1
        self.last_fps_time = time.time()
        self.calibrating = False
        self.calib_frames = []

        # Variables for json output
        self.json = []
        self.last_sec = -1

        # start time instead of universal time stamp to keep output consistent
        self.start_time = None
    
        self._build_gui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_gui(self): 
        self.canvas = tk.Canvas(self.root, bg="black", width=640, height=480)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        panel = ttk.Frame(self.root, width=250)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        panel.pack_propagate(False)

        ttk.Label(panel, text="Video Source:").pack(anchor=tk.W, pady=(10,0)) 
        self.source_var = tk.StringVar(value="0")
        ttk.Entry(panel, textvariable=self.source_var).pack(fill=tk.X)
        ttk.Button(panel, text="Browse File", command=self._browse).pack(fill=tk.X, pady=2)

        ttk.Label(panel, text="Inference Resolution:").pack(anchor=tk.W, pady=(10,0))
        self.res_var = tk.IntVar(value=320)
        ttk.Scale(panel, from_=160, to=480, variable=self.res_var, orient=tk.HORIZONTAL).pack(fill=tk.X)

        ttk.Separator(panel).pack(fill=tk.X, pady=10)
        self.start_btn = ttk.Button(panel, text="Start", command=self._start)
        self.start_btn.pack(fill=tk.X, pady=2)
        self.stop_btn = ttk.Button(panel, text="Stop", command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)

        ttk.Separator(panel).pack(fill=tk.X, pady=10)
        ttk.Button(panel, text="Calibrate", command=self._calibrate).pack(fill=tk.X, pady=2)
        self.calib_label = ttk.Label(panel, text="No baseline set", wraplength=230)
        self.calib_label.pack(anchor=tk.W)


    def _browse(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov"), ("All", "*.*")])
        if path:
            self.source_var.set(path)

    def _start(self):
        source = self.source_var.get()
        try:
            source = int(source)
        except Exception: # Doesn't really matter let's just pass through
            pass

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open video source")
            return

        self.estimator.width = self.res_var.get()
        self.estimator.height = int(self.res_var.get() * 0.75)

        self.current_pose = None
        self.current_score = None
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.tracker.reset()
        self.kalman.reset()


        # Reset report on restart
        self.start_time = time.time()
        self.last_sec = -1
        self.json = []
        self.frame_index = 1

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._update()



    def _stop(self):
        self.running = False
        self._save_json() # Json for Confusion matrix 
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.tracker.reset()
        self.kalman.reset()
        self.canvas.delete("all")

    def _calibrate(self):
        self.calibrating = True
        self.calib_frames = []
        self.calib_label.config(text="Calibrating...")
        self.root.after(3000, self._finish_calibrate)

    def _finish_calibrate(self):
        self.calibrating = False
        if len(self.calib_frames) >= 5:
            avg_kp = np.mean([f.keypoints for f in self.calib_frames], axis=0)
            avg_pose = PoseData(
                keypoints=avg_kp.astype(np.float32),
                timestamp=time.time(),
                frame_size=self.calib_frames[0].frame_size,
            )
            self.analyzer.set_baseline(avg_pose)
            self.calib_label.config(
                text=f"Baseline set ({len(self.calib_frames)} frames) - {self.analyzer.baseline_view}"
            )
        else:
            self.calib_label.config(text="Calibration failed")
        self.calib_frames = []

    def _update(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        self.frame_index +=1
        if not ret:
            self._stop()
            return

        # output video size
        frame = cv2.resize(frame, (640, 480))

        pose = self.estimator.detect(frame)
        if pose:
            smoothed_kp = self.kalman.apply(pose.keypoints)
            pose = PoseData(keypoints=smoothed_kp, timestamp=pose.timestamp, frame_size=pose.frame_size, world_keypoints=pose.world_keypoints)
            self.current_pose = pose
            self.tracker.track(frame, pose.keypoints) 
            self.current_score = self.analyzer.analyze(pose)
            # JSOn part for the confusion matrix 
            elasped = time.time() - self.start_time
            if self.current_score:
                self.last_sec = elasped # We'll use frame instead of second

                self.json.append({
                    "label": 1 if self.current_score.quality == "good" else 0,
                    "frame": self.frame_index,
                })
            if self.calibrating:
                self.calib_frames.append(pose)

        else: # Fallback for when subject is moving quickly or rotating
            tracked = self.tracker.track(frame) # Motion based tracking
            if tracked is not None and self.current_pose is not None:
                tracked = self.kalman.apply(tracked)
                world_kp = self.current_pose.world_keypoints if self.current_pose is not None else None
                self.current_pose = PoseData(keypoints=tracked, timestamp=time.time(), frame_size=(640, 480), world_keypoints=world_kp)
                self.current_score = self.analyzer.analyze(self.current_pose)
                # JSOn part for the confusion matrix 
                elasped = time.time() - self.start_time
                if self.current_score:
                    self.last_sec = elasped

                    self.json.append({
                        "label": 1 if self.current_score.quality == "good" else 0,
                        "frame": self.frame_index,
                    })
            else:
                self.current_pose = None
                self.current_score = None

        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = now

        kp = self.current_pose.keypoints if self.current_pose else None
        display = self.renderer.draw(frame, kp, self.current_score, self.fps, self.calibrating)

        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display)
        photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        self.canvas.create_image(
            canvas_w // 2,
            canvas_h // 2,
            anchor=tk.CENTER,
            image=photo
        )

        self.canvas.image = photo

        if self.running:
            self.root.after(1, self._update) # Set MS

    def _on_close(self):
        self.running = False
        self._save_json()
        if self.cap:
            self.cap.release()
        self.estimator.close()
        self.root.destroy()
    
    def _save_json(self):
        if not self.json:
            return

        data = {
            "source": str(self.source_var.get()),
            "results": self.json
        }

        try:
            with open("posture_result.json", "w") as f:
                json.dump(data, f, indent=2)
            print("Results saved")
        except Exception as e:
            print("Failed to save json")

if __name__ == "__main__":
    root = tk.Tk()
    app = PostureFixApp(root)
    root.mainloop()
