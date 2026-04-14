import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp
import time
import threading
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path

# === DATA CLASSES ===
@dataclass
class PoseData:
    keypoints: np.ndarray  # (33, 3) - x, y, confidence
    timestamp: float
    frame_size: Tuple[int, int]

@dataclass
class PostureScore:
    score: float
    quality: str
    problems: List[str]
    angles: dict
    view: str = "unknown"
    view_confidence: float = 0.0
    homography_used: bool = False

# === POSE ESTIMATOR ===
class PoseEstimator:
    CONNECTIONS = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]
    
    def __init__(self, width=320, height=240):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1,
                                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.width, self.height = width, height
    
    def detect(self, frame: np.ndarray) -> Optional[PoseData]:
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (self.width, self.height))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if not results.pose_landmarks:
            return None
        
        keypoints = np.zeros((33, 3), dtype=np.float32)
        for i, lm in enumerate(results.pose_landmarks.landmark):
            keypoints[i] = [lm.x * w, lm.y * h, lm.visibility]
        
        return PoseData(keypoints=keypoints, timestamp=time.time(), frame_size=(w, h))
    
    def close(self):
        self.pose.close()

# === POSTURE ANALYZER ===
class PostureAnalyzer:
    def __init__(self, good_thresh=80, warn_thresh=60):
        self.good_thresh = good_thresh
        self.warn_thresh = warn_thresh
        self.baseline: Optional[np.ndarray] = None
        self.baseline_normalized: Optional[np.ndarray] = None
        self.baseline_view = None
    
    def set_baseline(self, pose: PoseData):
        self.baseline = pose.keypoints.copy()
        self.baseline_normalized, _, _ = self._normalize(pose.keypoints)
        self.baseline_view, _ = self._estimate_view(pose.keypoints)
    
    def _normalize(self, kp: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Use Torso Length to normalize"""
        # Torso = avg of shoulder-to-hip distances
        pairs = [(11, 23), (12, 24)]  # shoulder to hip
        lengths = []
        for s, h in pairs:
            if kp[s, 2] > 0.3 and kp[h, 2] > 0.3:
                lengths.append(np.linalg.norm(kp[s, :2] - kp[h, :2]))
        
        torso_len = np.mean(lengths) if lengths else 1.0
        
        # Center = midpoint of hips
        if kp[23, 2] > 0.3 and kp[24, 2] > 0.3:
            center = (kp[23, :2] + kp[24, :2]) / 2
        else:
            valid = kp[kp[:, 2] > 0.3, :2]
            center = np.mean(valid, axis=0) if len(valid) > 0 else np.array([0, 0])
        
        normalized = kp.copy()
        normalized[:, 0] = (kp[:, 0] - center[0]) / max(torso_len, 1)
        normalized[:, 1] = (kp[:, 1] - center[1]) / max(torso_len, 1)
        
        return normalized, torso_len, tuple(center)
    
    def _compute_angles(self, kp: np.ndarray) -> dict:
        """Compute posture angles."""
        angles = {}
        
        def angle_to_vertical(p1, p2):
            vec = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            vertical = np.array([0, -1])
            cos = np.dot(vec, vertical) / (np.linalg.norm(vec) + 1e-6)
            return np.degrees(np.arccos(np.clip(cos, -1, 1)))
        
        def line_angle(p1, p2):
            return np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
        
        # Neck angle (ear to shoulder vs vertical)
        if kp[7, 2] > 0.3 and kp[11, 2] > 0.3:
            angles['neck'] = angle_to_vertical(kp[7, :2], kp[11, :2])
        
        # Shoulder alignment
        if kp[11, 2] > 0.3 and kp[12, 2] > 0.3:
            angles['shoulders'] = abs(line_angle(kp[11, :2], kp[12, :2]))
        
        # Hip alignment
        if kp[23, 2] > 0.3 and kp[24, 2] > 0.3:
            angles['hips'] = abs(line_angle(kp[23, :2], kp[24, :2]))
        
        # Spine angle
        if all(kp[i, 2] > 0.3 for i in [11, 12, 23, 24]):
            shoulder_mid = (kp[11, :2] + kp[12, :2]) / 2
            hip_mid = (kp[23, :2] + kp[24, :2]) / 2
            angles['spine'] = angle_to_vertical(shoulder_mid, hip_mid)
        
        return angles
    
    def _homography_score(self, current: np.ndarray, baseline: np.ndarray) -> float:
        """Compare poses using homography with RANSAC."""
        valid = (current[:, 2] > 0.3) & (baseline[:, 2] > 0.3)
        if np.sum(valid) < 4:
            return 100  # Not enough points
        
        pts1 = baseline[valid, :2].astype(np.float32)
        pts2 = current[valid, :2].astype(np.float32)
        
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        if H is None:
            return 100
        
        # Compute reprojection error
        pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
        projected = (H @ pts1_h.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        
        error = np.mean(np.linalg.norm(projected - pts2, axis=1))
        return max(0, 100 - error * 2)  # Scale error to score
    

    def _estimate_view(self, kp):
        if not (kp[11,2] > 0.3 and kp[12,2] > 0.3):
            return "unknown", 0.0

        left = kp[11,:2]
        right = kp[12,:2]

        shoulder_width = abs(right[0] - left[0])

        torso_len = np.mean([
            np.linalg.norm(kp[11,:2] - kp[23,:2]) if kp[23,2]>0.3 else 0,
            np.linalg.norm(kp[12,:2] - kp[24,:2]) if kp[24,2]>0.3 else 0
        ])
        torso_len = max(torso_len, 1e-6)

        ratio = shoulder_width / torso_len

        # nose offset for direction
        if kp[0,2] > 0.3:
            mid = (left[0] + right[0]) / 2
            offset = kp[0,0] - mid
        else:
            offset = 0

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

        confidence = min(1.0, ratio)

        return view, confidence

    def analyze(self, pose: PoseData) -> PostureScore:
        kp = pose.keypoints

        view, view_conf = self._estimate_view(kp)

        angles = self._compute_angles(kp)
        problems = []
        weighted_sum = 0
        total_weight = 0

        if view == "front":
            thresholds = {
                'neck': (30, 1.2),
                'shoulders': (20, 1.3),
                'hips': (20, 1.1),
                'spine': (25, 1.5),
            }
        elif "side" in view:
            thresholds = {
                'neck': (20, 1.8),
                'shoulders': (20, 0.4),
                'hips': (20, 0.4),
                'spine': (15, 1.8),
            }
        else:  # three_quarter / unknown
            thresholds = {
                'neck': (20, 1.3),
                'shoulders': (12, 0.8),
                'hips': (12, 0.8),
                'spine': (14, 1.4),
            }

        for name, value in angles.items():
            tol, weight = thresholds.get(name, (10, 1.0))
            score = max(0, 100 - (abs(value)/tol)*20)
            weighted_sum += score * weight
            total_weight += weight
            if score < 60:
                problems.append(name)

        # Homography gates
        use_homo = False
        if self.baseline_normalized is not None:
            if self.baseline_view == view and view_conf > 0.5:
                use_homo = True

        if use_homo:
            current_norm, _, _ = self._normalize(kp)
            homo_score = self._homography_score(current_norm, self.baseline_normalized)
            weighted_sum += homo_score * 1.0
            total_weight += 1.0

        total_score = weighted_sum / max(total_weight, 1e-6)

        quality = "good" if total_score >= self.good_thresh else \
                  "warning" if total_score >= self.warn_thresh else "bad"

        return PostureScore(
            score=total_score,
            quality=quality,
            problems=problems,
            angles=angles,
            view=view,
            view_confidence=view_conf,
            homography_used=use_homo,
        )

""" Track optical flow """
class OpticalFlowTracker:
    def __init__(self):
        self.prev_gray = None
        self.prev_keypoints = None
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    def track(self, frame: np.ndarray, new_keypoints: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if new_keypoints is not None:
            self.prev_gray = gray
            self.prev_keypoints = new_keypoints
            return new_keypoints
        
        if self.prev_gray is None or self.prev_keypoints is None:
            return None
        
        valid = self.prev_keypoints[:, 2] > 0.3
        if np.sum(valid) < 4:
            self.prev_gray = gray
            return self.prev_keypoints
        
        pts = self.prev_keypoints[valid, :2].reshape(-1, 1, 2).astype(np.float32)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts, None, **self.lk_params)
        
        tracked = self.prev_keypoints.copy()
        valid_idx = np.where(valid)[0]
        for i, idx in enumerate(valid_idx):
            if status[i]:
                tracked[idx, :2] = next_pts[i].ravel()
            else:
                tracked[idx, 2] *= 0.5
        
        self.prev_gray = gray
        self.prev_keypoints = tracked
        return tracked


# Render overlays
class OverlayRenderer:
    COLORS = {'good': (0, 255, 0), 'warning': (0, 165, 255), 'bad': (0, 0, 255)}
    CONNECTIONS = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28)]
    
    def draw(self, frame: np.ndarray, keypoints: Optional[np.ndarray], 
             score: Optional[PostureScore], fps: float, is_calibrating: bool = False) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        
        color = self.COLORS.get(score.quality, (255, 255, 255)) if score else (255, 255, 255)
        
        # Draw skeleton
        if keypoints is not None:
            for i, j in self.CONNECTIONS:
                if keypoints[i, 2] > 0.3 and keypoints[j, 2] > 0.3:
                    p1 = tuple(keypoints[i, :2].astype(int))
                    p2 = tuple(keypoints[j, :2].astype(int))
                    cv2.line(out, p1, p2, color, 2)
            
            for kp in keypoints:
                if kp[2] > 0.3:
                    cv2.circle(out, tuple(kp[:2].astype(int)), 4, (0, 255, 255), -1)
            
            # Highlight problems
            if score and score.problems:
                problem_kps = {'neck': [7, 8, 11, 12], 'shoulders': [11, 12], 
                               'hips': [23, 24], 'spine': [11, 12, 23, 24]}
                for prob in score.problems:
                    for idx in problem_kps.get(prob, []):
                        if keypoints[idx, 2] > 0.3:
                            cv2.circle(out, tuple(keypoints[idx, :2].astype(int)), 10, (0, 0, 255), 2)
        
        # Score panel
        if score:
            cv2.rectangle(out, (w-180, 10), (w-10, 90), (0, 0, 0), -1)
            cv2.putText(out, f"Score: {score.score:.0f}", (w-170, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(out, f"{score.quality.upper()}", (w-170, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # FPS
        cv2.putText(out, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # View info
        if score:
            cv2.putText(out, f"View: {score.view}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.putText(out, f"Conf: {score.view_confidence:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            cv2.putText(out, f"Homo: {'ON' if score.homography_used else 'OFF'}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


        # Calibration overlay
        if is_calibrating:
            overlay = out.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)
            cv2.putText(out, "CALIBRATING - Hold Still", (w//2-180, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        return out

# === MAIN APP ===
class PostureFixApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PostureFix Minimal")
        self.root.geometry("900x600")
        
        # Components
        self.estimator = PoseEstimator(320, 240)
        self.analyzer = PostureAnalyzer()
        self.tracker = OpticalFlowTracker()
        self.recorder = SessionRecorder()
        self.renderer = OverlayRenderer()
        
        # State
        self.running = False
        self.cap = None
        self.current_pose = None
        self.current_score = None
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.calibrating = False
        self.calib_frames = []
        
        self._build_gui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _build_gui(self):
        # Video canvas
        self.canvas = tk.Canvas(self.root, bg="black", width=640, height=480)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Control panel
        panel = ttk.Frame(self.root, width=220)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        panel.pack_propagate(False)
        
        # Source
        ttk.Label(panel, text="Video Source:").pack(anchor=tk.W, pady=(10,0))
        self.source_var = tk.StringVar(value="0")
        ttk.Entry(panel, textvariable=self.source_var).pack(fill=tk.X)
        ttk.Button(panel, text="Browse File", command=self._browse).pack(fill=tk.X, pady=2)
        
        # Resolution
        ttk.Label(panel, text="Inference Resolution:").pack(anchor=tk.W, pady=(10,0))
        self.res_var = tk.IntVar(value=320)
        ttk.Scale(panel, from_=160, to=480, variable=self.res_var, orient=tk.HORIZONTAL).pack(fill=tk.X)
        
        # Controls
        ttk.Separator(panel).pack(fill=tk.X, pady=10)
        self.start_btn = ttk.Button(panel, text="Start", command=self._start)
        self.start_btn.pack(fill=tk.X, pady=2)
        self.stop_btn = ttk.Button(panel, text="Stop", command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)
        
        # Calibration
        ttk.Separator(panel).pack(fill=tk.X, pady=10)
        ttk.Button(panel, text="Calibrate (3s)", command=self._calibrate).pack(fill=tk.X, pady=2)
        self.calib_label = ttk.Label(panel, text="No baseline set")
        self.calib_label.pack(anchor=tk.W)
        
        # Recording
        ttk.Separator(panel).pack(fill=tk.X, pady=10)
        self.rec_btn = ttk.Button(panel, text="Start Recording", command=self._toggle_record)
        self.rec_btn.pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Save Report", command=self._save_report).pack(fill=tk.X, pady=2)
        self.rec_label = ttk.Label(panel, text="Not recording")
        self.rec_label.pack(anchor=tk.W)
    
    def _browse(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov"), ("All", "*.*")])
        if path:
            self.source_var.set(path)
    
    def _start(self):
        source = self.source_var.get()
        try:
            source = int(source)
        except ValueError:
            pass
        
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open video source")
            return
        
        self.estimator.width = self.res_var.get()
        self.estimator.height = int(self.res_var.get() * 0.75)
        
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._update()
    
    def _stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.canvas.delete("all")
    
    def _calibrate(self):
        if not self.running:
            messagebox.showwarning("Warning", "Start video first")
            return
        self.calibrating = True
        self.calib_frames = []
        self.calib_label.config(text="Calibrating...")
        self.root.after(3000, self._finish_calibrate)
    
    def _finish_calibrate(self):
        self.calibrating = False
        if len(self.calib_frames) >= 5:
            avg_kp = np.mean([f.keypoints for f in self.calib_frames], axis=0)
            avg_pose = PoseData(keypoints=avg_kp, timestamp=time.time(), frame_size=self.calib_frames[0].frame_size)
            self.analyzer.set_baseline(avg_pose)
            self.calib_label.config(text=f"Baseline set ({len(self.calib_frames)} frames)")
        else:
            self.calib_label.config(text="Calibration failed")
        self.calib_frames = []
    
    def _toggle_record(self):
        if self.recorder.recording:
            self.recorder.stop()
            self.rec_btn.config(text="Start Recording")
            self.rec_label.config(text="Stopped")
        else:
            self.recorder.start()
            self.rec_btn.config(text="Stop Recording")
    
    def _save_report(self):
        if not self.recorder.data:
            messagebox.showwarning("Warning", "No data recorded")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text", "*.txt")])
        if path:
            self.recorder.save_report(path)
            messagebox.showinfo("Saved", f"Report saved to {path}")
    
    def _update(self):
        if not self.running or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self._stop()
            return
        
        frame = cv2.resize(frame, (640, 480))
        
        # Pose detection
        pose = self.estimator.detect(frame)
        
        if pose:
            self.current_pose = pose
            self.tracker.track(frame, pose.keypoints)
            self.current_score = self.analyzer.analyze(pose)
            
            if self.calibrating:
                self.calib_frames.append(pose)
            
            if self.recorder.recording:
                self.recorder.record(self.current_score.score, self.current_score.quality, self.current_score.problems)
        else:
            # Use tracking
            tracked = self.tracker.track(frame)
            if tracked is not None and self.current_pose:
                self.current_pose = PoseData(keypoints=tracked, timestamp=time.time(), frame_size=(640, 480))
        
        # FPS calculation
        self.frame_count += 1
        now = time.time()
        if now - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = now
        
        # Recording label
        if self.recorder.recording:
            elapsed = time.time() - self.recorder.start_time
            self.rec_label.config(text=f"Recording: {elapsed:.0f}s")
        
        # Render
        kp = self.current_pose.keypoints if self.current_pose else None
        display = self.renderer.draw(frame, kp, self.current_score, self.fps, self.calibrating)
        
        # Convert to Tkinter
        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display)
        photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo
        
        if self.running:
            self.root.after(1, self._update)
    
    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.estimator.close()
        self.root.destroy()
        
class SessionRecorder:
    def __init__(self):
        self.data: List[Tuple[float, float, str, List[str]]] = []
        self.start_time = None
        self.recording = False
    
    def start(self):
        self.data = []
        self.start_time = time.time()
        self.recording = True
    
    def stop(self):
        self.recording = False
    
    def record(self, score: float, quality: str, problems: List[str]):
        if self.recording:
            self.data.append((time.time(), score, quality, problems.copy()))
#  ENTRY
if __name__ == "__main__":
    root = tk.Tk()
    app = PostureFixApp(root)
    root.mainloop()
