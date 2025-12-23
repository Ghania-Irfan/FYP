"""
Improved Eye-Controlled Cursor System
Uses relative gaze estimation for more accurate and stable cursor control
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

class EyeCursorController:
    """Enhanced eye-controlled cursor with smoothing and relative positioning"""
    
    def __init__(self):
        # MediaPipe setup
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except AttributeError:
            print("❌ Error: MediaPipe 'solutions' module not found.")
            print("   Please install the correct version:")
            print("   pip uninstall mediapipe")
            print("   pip install mediapipe==0.10.8")
            raise
        
        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Cursor smoothing (Exponential Moving Average)
        self.smoothing_factor = 0.5  # Higher = more smoothing (0-1) (reduced for more responsive movement)
        self.last_cursor_x = self.screen_w // 2
        self.last_cursor_y = self.screen_h // 2
        
        # Movement sensitivity
        self.sensitivity = 50  # Higher = faster cursor movement (increased for better responsiveness)
        
        # Dead zone (ignore small movements to prevent jitter)
        self.dead_zone = 0.01  # Minimum gaze ratio to trigger movement (reduced for more sensitivity)
        
        # Blink detection
        self.blink_threshold = 0.25  # Eye Aspect Ratio threshold
        self.blink_counter = 0
        self.blink_frames = 3  # Frames to detect blink
        self.last_blink_time = 0
        self.blink_cooldown = 0.5  # Seconds between clicks
        
        # Calibration
        self.calibrated = False
        self.calibration_points = []
        self.gaze_center = None
        
        # Eye landmark indices (MediaPipe Face Mesh)
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382]
        self.LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
        self.RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
        
        # Eye state tracking
        self.ear_history = []  # Eye Aspect Ratio history
        
    def calculate_eye_center(self, landmarks, eye_indices):
        """Calculate the center point of an eye"""
        eye_points = []
        for idx in eye_indices:
            pt = landmarks.landmark[idx]
            eye_points.append([pt.x, pt.y])
        return np.mean(eye_points, axis=0)
    
    def calculate_iris_center(self, landmarks, iris_indices):
        """Calculate the center point of the iris"""
        iris_points = []
        for idx in iris_indices:
            pt = landmarks.landmark[idx]
            iris_points.append([pt.x, pt.y])
        return np.mean(iris_points, axis=0)
    
    def calculate_eye_aspect_ratio(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        # Get eye landmark points
        eye_points = np.array([
            [landmarks.landmark[i].x, landmarks.landmark[i].y]
            for i in eye_indices
        ])
        
        # Calculate vertical distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Calculate horizontal distance
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def calculate_gaze_ratio(self, iris_center, eye_center, eye_width, eye_height):
        """
        Calculate gaze ratio: how far the iris is from eye center
        Returns normalized values between -1 and 1
        """
        # Calculate offset
        offset_x = iris_center[0] - eye_center[0]
        offset_y = iris_center[1] - eye_center[1]
        
        # Normalize by eye dimensions
        gaze_ratio_x = offset_x / eye_width if eye_width > 0 else 0
        gaze_ratio_y = offset_y / eye_height if eye_height > 0 else 0
        
        # Apply non-linear scaling for better responsiveness (cubic curve)
        # This makes small eye movements translate to larger cursor movements
        gaze_ratio_x = np.sign(gaze_ratio_x) * (abs(gaze_ratio_x) ** 0.7) if gaze_ratio_x != 0 else 0
        gaze_ratio_y = np.sign(gaze_ratio_y) * (abs(gaze_ratio_y) ** 0.7) if gaze_ratio_y != 0 else 0
        
        # Scale up the ratio for better sensitivity
        gaze_ratio_x *= 2.0  # Amplify horizontal movement
        gaze_ratio_y *= 2.0  # Amplify vertical movement
        
        # Clamp to reasonable range
        gaze_ratio_x = np.clip(gaze_ratio_x, -1.0, 1.0)
        gaze_ratio_y = np.clip(gaze_ratio_y, -1.0, 1.0)
        
        # Apply dead zone
        if abs(gaze_ratio_x) < self.dead_zone:
            gaze_ratio_x = 0
        if abs(gaze_ratio_y) < self.dead_zone:
            gaze_ratio_y = 0
        
        return gaze_ratio_x, gaze_ratio_y
    
    def calculate_eye_dimensions(self, landmarks, eye_indices):
        """Calculate width and height of the eye"""
        eye_points = np.array([
            [landmarks.landmark[i].x, landmarks.landmark[i].y]
            for i in eye_indices
        ])
        
        # Eye width (horizontal distance)
        eye_width = np.max(eye_points[:, 0]) - np.min(eye_points[:, 0])
        
        # Eye height (vertical distance)
        eye_height = np.max(eye_points[:, 1]) - np.min(eye_points[:, 1])
        
        return eye_width, eye_height
    
    def detect_blink(self, left_ear, right_ear):
        """Detect if user is blinking"""
        avg_ear = (left_ear + right_ear) / 2.0
        current_time = time.time()
        
        # Track EAR history
        self.ear_history.append(avg_ear)
        if len(self.ear_history) > 10:
            self.ear_history.pop(0)
        
        # Detect blink (EAR drops below threshold)
        if avg_ear < self.blink_threshold:
            self.blink_counter += 1
        else:
            if self.blink_counter >= self.blink_frames:
                # Blink detected
                if current_time - self.last_blink_time > self.blink_cooldown:
                    self.last_blink_time = current_time
                    self.blink_counter = 0
                    return True
            self.blink_counter = 0
        
        return False
    
    def move_cursor(self, gaze_ratio_x, gaze_ratio_y):
        """Move cursor based on gaze ratio"""
        # Calculate movement delta with enhanced sensitivity
        # Use screen-relative movement for better control
        delta_x = gaze_ratio_x * self.sensitivity
        delta_y = gaze_ratio_y * self.sensitivity
        
        # For larger gaze movements, increase speed (acceleration)
        if abs(gaze_ratio_x) > 0.3:
            delta_x *= 1.5
        if abs(gaze_ratio_y) > 0.3:
            delta_y *= 1.5
        
        # Calculate new cursor position
        new_x = self.last_cursor_x + delta_x
        new_y = self.last_cursor_y + delta_y
        
        # Clamp to screen boundaries
        new_x = np.clip(new_x, 0, self.screen_w)
        new_y = np.clip(new_y, 0, self.screen_h)
        
        # Apply smoothing (less aggressive for more responsive movement)
        smoothed_x = self.smoothing_factor * self.last_cursor_x + (1 - self.smoothing_factor) * new_x
        smoothed_y = self.smoothing_factor * self.last_cursor_y + (1 - self.smoothing_factor) * new_y
        
        # Only move if there's significant change (reduce jitter)
        if abs(smoothed_x - self.last_cursor_x) > 0.5 or abs(smoothed_y - self.last_cursor_y) > 0.5:
            # Update cursor position
            pyautogui.moveTo(int(smoothed_x), int(smoothed_y))
        
        # Update last position
        self.last_cursor_x = smoothed_x
        self.last_cursor_y = smoothed_y
    
    def calibrate(self, landmarks):
        """Simple calibration: set current gaze as center"""
        # Calculate average eye centers
        left_eye_center = self.calculate_eye_center(landmarks, self.LEFT_EYE_INDICES)
        right_eye_center = self.calculate_eye_center(landmarks, self.RIGHT_EYE_INDICES)
        
        # Calculate average iris centers
        left_iris_center = self.calculate_iris_center(landmarks, self.LEFT_IRIS_INDICES)
        right_iris_center = self.calculate_iris_center(landmarks, self.RIGHT_IRIS_INDICES)
        
        # Store calibration data
        self.gaze_center = {
            'left_eye': left_eye_center,
            'right_eye': right_eye_center,
            'left_iris': left_iris_center,
            'right_iris': right_iris_center
        }
        self.calibrated = True
        print("✅ Calibration complete! Look at the center of the screen.")
    
    def process_frame(self, frame):
        """Process a single frame and update cursor"""
        frame = cv2.flip(frame, 1)  # Mirror for natural feel
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        frame_h, frame_w, _ = frame.shape
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Calibrate on 'c' key press (handled in main loop)
            
            # Calculate eye centers
            left_eye_center = self.calculate_eye_center(landmarks, self.LEFT_EYE_INDICES)
            right_eye_center = self.calculate_eye_center(landmarks, self.RIGHT_EYE_INDICES)
            
            # Calculate iris centers
            left_iris_center = self.calculate_iris_center(landmarks, self.LEFT_IRIS_INDICES)
            right_iris_center = self.calculate_iris_center(landmarks, self.RIGHT_IRIS_INDICES)
            
            # Calculate eye dimensions
            left_eye_width, left_eye_height = self.calculate_eye_dimensions(landmarks, self.LEFT_EYE_INDICES)
            right_eye_width, right_eye_height = self.calculate_eye_dimensions(landmarks, self.RIGHT_EYE_INDICES)
            
            # Calculate gaze ratios for both eyes
            left_gaze_x, left_gaze_y = self.calculate_gaze_ratio(
                left_iris_center, left_eye_center, left_eye_width, left_eye_height
            )
            right_gaze_x, right_gaze_y = self.calculate_gaze_ratio(
                right_iris_center, right_eye_center, right_eye_width, right_eye_height
            )
            
            # Average both eyes for better accuracy
            avg_gaze_x = (left_gaze_x + right_gaze_x) / 2.0
            avg_gaze_y = (left_gaze_y + right_gaze_y) / 2.0
            
            # Move cursor
            self.move_cursor(avg_gaze_x, avg_gaze_y)
            
            # Blink detection
            left_ear = self.calculate_eye_aspect_ratio(landmarks, self.LEFT_EYE_INDICES)
            right_ear = self.calculate_eye_aspect_ratio(landmarks, self.RIGHT_EYE_INDICES)
            
            if self.detect_blink(left_ear, right_ear):
                pyautogui.click()
                print("👆 Click!")
            
            # Draw visualization
            frame = self.draw_visualization(
                frame, landmarks, left_eye_center, right_eye_center,
                left_iris_center, right_iris_center, avg_gaze_x, avg_gaze_y,
                left_ear, right_ear
            )
        
        return frame
    
    def draw_visualization(self, frame, landmarks, left_eye_center, right_eye_center,
                          left_iris_center, right_iris_center, gaze_x, gaze_y,
                          left_ear, right_ear):
        """Draw visualization on frame"""
        frame_h, frame_w, _ = frame.shape
        
        # Draw eye centers
        left_eye_px = (int(left_eye_center[0] * frame_w), int(left_eye_center[1] * frame_h))
        right_eye_px = (int(right_eye_center[0] * frame_w), int(right_eye_center[1] * frame_h))
        cv2.circle(frame, left_eye_px, 5, (255, 0, 0), -1)
        cv2.circle(frame, right_eye_px, 5, (255, 0, 0), -1)
        
        # Draw iris centers
        left_iris_px = (int(left_iris_center[0] * frame_w), int(left_iris_center[1] * frame_h))
        right_iris_px = (int(right_iris_center[0] * frame_w), int(right_iris_center[1] * frame_h))
        cv2.circle(frame, left_iris_px, 8, (0, 255, 0), 2)
        cv2.circle(frame, right_iris_px, 8, (0, 255, 0), 2)
        
        # Draw line from eye center to iris center (gaze direction)
        cv2.line(frame, left_eye_px, left_iris_px, (0, 255, 255), 2)
        cv2.line(frame, right_eye_px, right_iris_px, (0, 255, 255), 2)
        
        # Draw eye contours
        left_eye_points = np.array([
            [int(landmarks.landmark[i].x * frame_w), int(landmarks.landmark[i].y * frame_h)]
            for i in self.LEFT_EYE_INDICES
        ])
        right_eye_points = np.array([
            [int(landmarks.landmark[i].x * frame_w), int(landmarks.landmark[i].y * frame_h)]
            for i in self.RIGHT_EYE_INDICES
        ])
        cv2.polylines(frame, [left_eye_points], True, (255, 255, 0), 1)
        cv2.polylines(frame, [right_eye_points], True, (255, 255, 0), 1)
        
        # Display information
        info_y = 30
        cv2.putText(frame, f"Gaze: X={gaze_x:.3f}, Y={gaze_y:.3f}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"EAR: L={left_ear:.2f}, R={right_ear:.2f}", 
                   (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Sensitivity: {self.sensitivity} (Press +/- to adjust)", 
                   (10, info_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        cv2.putText(frame, "Press 'C' to calibrate | 'Q' to quit", 
                   (10, info_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        
        # Show cursor position on screen
        cursor_info = f"Cursor: ({int(self.last_cursor_x)}, {int(self.last_cursor_y)})"
        cv2.putText(frame, cursor_info, 
                   (10, info_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw gaze direction indicator
        center_x, center_y = frame_w // 2, frame_h // 2
        indicator_x = int(center_x + gaze_x * 100)
        indicator_y = int(center_y + gaze_y * 100)
        cv2.arrowedLine(frame, (center_x, center_y), (indicator_x, indicator_y), 
                       (0, 0, 255), 3, tipLength=0.3)
        
        return frame


def main():
    """Main application loop"""
    print("=" * 60)
    print("👁️  IMPROVED EYE-CONTROLLED CURSOR SYSTEM")
    print("=" * 60)
    print("\n📋 Instructions:")
    print("   • Look at the center of the screen")
    print("   • Press 'C' to calibrate (center your gaze)")
    print("   • Move your eyes to control the cursor")
    print("   • Blink to click")
    print("   • Press 'Q' to quit")
    print("\n💡 Tips:")
    print("   • Sit 50-100 cm from camera")
    print("   • Ensure good lighting")
    print("   • Keep your head relatively still")
    print("   • Move only your eyes, not your head")
    print("=" * 60 + "\n")
    
    # Initialize controller
    controller = EyeCursorController()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Webcam initialized")
    print("⏳ Starting in 2 seconds...\n")
    time.sleep(2)
    
    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame")
            break
        
        # Process frame
        frame = controller.process_frame(frame)
        
        # Display frame
        cv2.imshow('Eye Controlled Cursor', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n👋 Shutting down...")
            break
        elif key == ord('c'):
            # Calibrate
            rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = controller.face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                controller.calibrate(results.multi_face_landmarks[0])
            else:
                print("❌ No face detected. Please look at the camera.")
        elif key == ord('+') or key == ord('='):
            # Increase sensitivity
            controller.sensitivity = min(controller.sensitivity + 5, 200)
            print(f"📈 Sensitivity: {controller.sensitivity}")
        elif key == ord('-') or key == ord('_'):
            # Decrease sensitivity
            controller.sensitivity = max(controller.sensitivity - 5, 10)
            print(f"📉 Sensitivity: {controller.sensitivity}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Application closed")


if __name__ == "__main__":
    main()

