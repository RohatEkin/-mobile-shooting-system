#!/usr/bin/env python3


import cv2
import numpy as np
import configparser
import os

# ===================== CONFIG YÜKLE =====================
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'config', 'tracker_config.ini')

cfg = configparser.ConfigParser()
if os.path.exists(CONFIG_PATH):
    cfg.read(CONFIG_PATH)
    print(f"[BİLGİ] Config yüklendi: {CONFIG_PATH}")
else:
    print(f"[UYARI] Config bulunamadı: {CONFIG_PATH} — varsayılan değerler kullanılıyor.")

def cget(section, key, fallback, typ=str):
    val = cfg.get(section, key, fallback=str(fallback))
    if typ == int:    return int(val)
    if typ == float:  return float(val)
    if typ == bool:   return val.lower() in ('true', '1', 'yes')
    return val

# Camera
CAMERA_SOURCE  = cget('camera', 'source', '', str)
CAMERA_INDEX   = cget('camera', 'index', 0, int)
FRAME_WIDTH    = cget('camera', 'width', 640, int)
FRAME_HEIGHT   = cget('camera', 'height', 480, int)
FPS            = cget('camera', 'fps', 30, int)

# Servo
servo_x        = cget('servo', 'start_x', 90, int)
servo_y        = cget('servo', 'start_y', 90, int)
SERVO_MIN      = cget('servo', 'min_angle', 0, int)
SERVO_MAX      = cget('servo', 'max_angle', 180, int)

# Tracking
DEAD_ZONE      = cget('tracking', 'dead_zone', 30, int)

# Detection
HSV_H_MAX      = cget('detection', 'hsv_h_max', 180, int)
HSV_S_MAX      = cget('detection', 'hsv_s_max', 80, int)
HSV_V_MAX      = cget('detection', 'hsv_v_max', 60, int)
MIN_RADIUS     = cget('detection', 'min_radius', 50, int)
MAX_RADIUS     = cget('detection', 'max_radius', 300, int)
BLUR_KERNEL    = cget('detection', 'blur_kernel', 7, int)
MIN_CIRC       = cget('detection', 'min_circularity', 0.65, float)
MIN_RINGS      = cget('detection', 'min_nested_rings', 2, int)
PATTERN_CHECK  = cget('detection', 'pattern_check', True, bool)
PATTERN_DIST   = cget('detection', 'pattern_check_distance', 40, int)
MIN_TRANSITIONS= cget('detection', 'min_transitions', 2, int)

# Kalman
PROC_NOISE     = cget('kalman', 'process_noise', 0.03, float)
MEAS_NOISE     = cget('kalman', 'measurement_noise', 0.5, float)
PREDICT_FRAMES = cget('kalman', 'predict_frames', 15, int)
MAX_JUMP       = cget('kalman', 'max_jump_distance', 100, int)

# TTL
TTL_PORT = '/dev/ttyUSB0'  # <-- BURADAN DEĞİŞTİR (örn: '/dev/ttyACM0', 'COM3' vb.)
TTL_BAUD = 9600             # <-- Baud rate
ser = None
try:
    import serial
    ser = serial.Serial(TTL_PORT, TTL_BAUD, timeout=1)
    print(f"[BİLGİ] TTL bağlantısı kuruldu: {TTL_PORT}")
except:
    ser = None

def send_data(cmd_x, cmd_y, servo_x_val, servo_y_val):
    """TTL üzerinden Arduino'ya komut gönder. Format: X<cmd_x>Y<cmd_y>\\n"""
    if ser is not None and ser.is_open:
        try:
            msg = f"X{cmd_x}Y{cmd_y}\n"
            ser.write(msg.encode())
        except:
            pass


def open_camera():
    camera_candidates = []

    if CAMERA_SOURCE:
        camera_candidates.append(CAMERA_SOURCE)

    camera_candidates.append(CAMERA_INDEX)

    if CAMERA_INDEX != 0:
        camera_candidates.append(0)
    if CAMERA_INDEX != 1:
        camera_candidates.append(1)

    tried = []
    for candidate in camera_candidates:
        if candidate in tried:
            continue
        tried.append(candidate)

        cap = cv2.VideoCapture(candidate)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, FPS)
            print(f"[BİLGİ] Kamera açıldı: {candidate}")
            return cap

        cap.release()

    print(f"[HATA] Kamera açılamadı! Denenen kaynaklar: {tried}")
    return None


# ===================== KALMAN FİLTRESİ =====================
class TargetKalman:
    def __init__(self):
        # 4 state (x, y, vx, vy), 2 measurement (x, y)
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * PROC_NOISE
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEAS_NOISE
        self.initialized = False
        self.frames_lost = 0
        self.last_radius = 0

    def update(self, cx, cy, radius):
        meas = np.array([[np.float32(cx)], [np.float32(cy)]])
        if not self.initialized:
            self.kf.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
            self.kf.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
            self.initialized = True
        self.kf.correct(meas)
        pred = self.kf.predict()
        self.frames_lost = 0
        self.last_radius = radius
        return int(pred[0][0]), int(pred[1][0])

    def predict_only(self):
        if not self.initialized:
            return None
        self.frames_lost += 1
        if self.frames_lost > PREDICT_FRAMES:
            # Çok uzun süredir kayıp — Kalman'ı resetle
            self.initialized = False
            self.frames_lost = 0
            return None
        pred = self.kf.predict()
        return int(pred[0][0]), int(pred[1][0])

    def distance_to(self, cx, cy):
        if not self.initialized:
            return 0
        sx = self.kf.statePost[0][0]
        sy = self.kf.statePost[1][0]
        return np.sqrt((cx - sx)**2 + (cy - sy)**2)


# ===================== DESEN KONTROLÜ =====================
def check_bullseye_pattern(gray, cx, cy, radius):
    
    h, w = gray.shape
    dist = min(PATTERN_DIST, radius, cx, cy, w - cx - 1, h - cy - 1)
    if dist < 10:
        return False

    transitions = 0
    # 4 yönde kontrol (sağ, sol, yukarı, aşağı)
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        prev_bright = gray[cy, cx] > 128
        dir_transitions = 0
        for step in range(3, dist, 2):
            px = cx + dx * step
            py = cy + dy * step
            if 0 <= px < w and 0 <= py < h:
                bright = gray[py, px] > 128
                if bright != prev_bright:
                    dir_transitions += 1
                    prev_bright = bright
        transitions = max(transitions, dir_transitions)

    return transitions >= MIN_TRANSITIONS


# ===================== YÜZ TESPİTİ (MASKELEME İÇİN) =====================
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)


def mask_faces(gray, frame_bgr):
    """Yüz+saç bölgelerini siyahla maskele, böylece kontür bulamasın."""
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
    mask = np.ones(gray.shape, dtype=np.uint8) * 255
    for (x, y, w, h) in faces:
        # Yüz + saç bölgesi (üstten %50 fazla al ki saçı da kapatsın)
        top = max(0, y - int(h * 0.5))
        cv2.rectangle(mask, (x, top), (x + w, y + h), 0, -1)
    return mask


# ===================== HEDEF TESPİTİ =====================
def detect_bullseye(frame_bgr, gray, v_max):
    # Yüz bölgelerini maskele
    face_mask = mask_faces(gray, frame_bgr)

    # Gri threshold (basit ve etkili)
    blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL, BLUR_KERNEL), 0)
    _, thresh = cv2.threshold(blurred, v_max, 255, cv2.THRESH_BINARY_INV)

    # Yüz maskesini uygula — yüz/saç bölgesi temizlenir
    thresh = cv2.bitwise_and(thresh, face_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, thresh

    best = None
    best_score = 0

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0 or area < np.pi * MIN_RADIUS**2:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRC:
            continue

        (ex, ey), radius = cv2.minEnclosingCircle(cnt)
        if radius < MIN_RADIUS or radius > MAX_RADIUS:
            continue

        child_count = 0
        if hierarchy is not None:
            child_idx = hierarchy[0][i][2]
            while child_idx != -1:
                child_count += 1
                child_idx = hierarchy[0][child_idx][0]

        if child_count < MIN_RINGS:
            continue

        # Moments ile merkez
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            mcx = int(M["m10"] / M["m00"])
            mcy = int(M["m01"] / M["m00"])
        else:
            mcx, mcy = int(ex), int(ey)

        # Siyah-beyaz desen kontrolü
        if PATTERN_CHECK:
            if not check_bullseye_pattern(gray, mcx, mcy, int(radius)):
                continue

        score = circularity * area * (1 + child_count * 0.5)
        if score > best_score:
            best_score = score
            best = (mcx, mcy, int(radius), circularity, child_count)

    return best, thresh


# ===================== ANA DÖNGÜ =====================
def main():
    global servo_x, servo_y, HSV_V_MAX

    cap = open_camera()
    if cap is None:
        return

    print("[BİLGİ] Bullseye Takip + Kalman Filtre başlatıldı.")
    print("[BİLGİ] 'q'=çık  't'=threshold  '+'/'-'=HSV V ayarla")

    kalman = TargetKalman()
    show_thresh = False

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        img_h, img_w = frame.shape[:2]
        cam_cx, cam_cy = img_w // 2, img_h // 2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        result, thresh = detect_bullseye(frame, gray, HSV_V_MAX)

        # Kamera merkezi
        cv2.drawMarker(frame, (cam_cx, cam_cy), (0, 0, 255), cv2.MARKER_CROSS, 30, 2)

        cmd_x = 0
        cmd_y = 0
        tracking = False

        if result is not None:
            cx, cy, radius, circ, children = result

            # Kalman: çok büyük sıçrama varsa yoksay (saç atlama)
            jump = kalman.distance_to(cx, cy)
            if kalman.initialized and jump > MAX_JUMP:
                # Yanlış pozitif — Kalman tahminini kullan
                pred = kalman.predict_only()
                if pred:
                    cx, cy = pred
                    radius = kalman.last_radius
                    tracking = True
            else:
                cx, cy = kalman.update(cx, cy, radius)
                tracking = True
        else:
            # Hedef kayıp — Kalman tahmini ile devam et
            pred = kalman.predict_only()
            if pred:
                cx, cy = pred
                radius = kalman.last_radius
                tracking = True

        if tracking:
            offset_x = cx - cam_cx
            offset_y = cam_cy - cy

            if abs(offset_x) > DEAD_ZONE:
                cmd_x = +1 if offset_x > 0 else -1
            if abs(offset_y) > DEAD_ZONE:
                cmd_y = -1 if offset_y > 0 else +1

            servo_x = max(SERVO_MIN, min(SERVO_MAX, servo_x - cmd_x))
            servo_y = max(SERVO_MIN, min(SERVO_MAX, servo_y + cmd_y))
            send_data(cmd_x, cmd_y, servo_x, servo_y)

            # Çizimler
            color = (0, 255, 0) if result is not None else (0, 165, 255)  # turuncu=tahmin
            cv2.circle(frame, (cx, cy), radius, color, 2)
            cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)
            cv2.line(frame, (cam_cx, cam_cy), (cx, cy), (255, 0, 255), 2)

            label = "HEDEF" if result is not None else "TAHMIN"
            cv2.putText(frame, f"{label} (r={radius})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Offset X:{offset_x:+d} Y:{offset_y:+d}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Cmd X:{cmd_x:+d} Y:{cmd_y:+d}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, f"Servo X:{servo_x} Y:{servo_y}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)

            if cmd_x == 0 and cmd_y == 0:
                cv2.putText(frame, "KILITLENDI!", (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            print(f"{label} r={radius} | Offset X:{offset_x:+d} Y:{offset_y:+d} | "
                  f"Cmd X:{cmd_x:+d} Y:{cmd_y:+d} | Servo X:{servo_x} Y:{servo_y}")
        else:
            cv2.putText(frame, "HEDEF ARANIYOR...", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.putText(frame, f"HSV V:{HSV_V_MAX} S:{HSV_S_MAX}", (10, img_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Bullseye Takip', frame)
        if show_thresh and thresh is not None:
            cv2.imshow('Threshold', thresh)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            show_thresh = not show_thresh
            if not show_thresh:
                cv2.destroyWindow('Threshold')
        elif key in (ord('+'), ord('=')):
            HSV_V_MAX = min(255, HSV_V_MAX + 5)
            print(f"HSV V Max: {HSV_V_MAX}")
        elif key == ord('-'):
            HSV_V_MAX = max(0, HSV_V_MAX - 5)
            print(f"HSV V Max: {HSV_V_MAX}")

    cap.release()
    cv2.destroyAllWindows()
    if ser is not None:
        try:
            ser.close()
        except:
            pass
    print("[BİLGİ] Sistem kapatıldı.")


if __name__ == '__main__':
    main()
