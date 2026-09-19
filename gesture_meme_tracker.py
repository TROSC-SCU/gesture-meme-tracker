"""
Gesture Meme Tracker - Mahmoud's custom 12-gesture version.
Removed 'thumbs_up' and 'cheek' gestures.
Adjusted Achievement Side Panel to 2x5 grid.
Added Frame Smoothing (Debounce) to prevent jittery transitions.
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import math


# ---------------------------------------------------------------------------
# Media / filenames
# ---------------------------------------------------------------------------
GESTURE_MEDIA = {
    "index_finger": ["index_up.png"],
    "heart": ["heart.png"],
    "shy": ["shy.mp4"],
    "chin": ["thinking.png"],
    "salute": ["salute.png"],
    "eyebrow": ["raised_eyebrow.png"],
    "mouth_open": ["open_mouth.png"],
    "smile": ["closed_smile.png"],
    "fists": ["double_fist.mp4"],
    "hide_face": ["hide_face.png"],
    "none": ["default.png"],
}

GESTURE_KEYWORDS = {
    "index_finger": ["index", "mafroor", "mafror"],
    "heart": ["heart", "mahboub"],
    "shy": ["shy", "maksouf"],
    "chin": ["chin", "thinking", "metgyer", "mtgyer", "metghayer"],
    "salute": ["salute", "taheya"],
    "eyebrow": ["eyebrow", "raised_eyebrow", "ancelotti"],
    "mouth_open": ["mouth_open", "open_mouth", "objection", "e3terad"],
    "smile": ["smile", "accepted", "ma2bool", "maqbool"],
    "fists": ["fists", "double_fist", "celebration", "far7a", "farha"],
    "hide_face": ["hide_face", "killed", "ma2tool", "maktol"],
}

# قائمة الحركات التي ستظهر في اللوحة الجانبية (نستبعد none)
PANEL_GESTURES = [g for g in GESTURE_MEDIA.keys() if g != "none"]

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def point_xy(lm):
    return np.array([lm.x, lm.y], dtype=np.float32)

def hand_center(hand):
    pts = [point_xy(p) for p in hand.landmark]
    return np.mean(pts, axis=0)

def get_face_scale(face):
    return dist(face.landmark[10], face.landmark[152])

def get_face_width(face):
    return dist(face.landmark[234], face.landmark[454])

def face_bbox(face):
    xs = [p.x for p in face.landmark]
    ys = [p.y for p in face.landmark]
    return min(xs), min(ys), max(xs), max(ys)

def finger_extended(hand, tip, pip, mcp):
    lm = hand.landmark
    return lm[tip].y < lm[pip].y and lm[pip].y < lm[mcp].y

def thumb_extended(hand):
    lm = hand.landmark
    
    fingers_folded = all(
        lm[tip].y > lm[pip].y
        for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]
    )
    
    thumb_high = lm[4].y < lm[3].y - 0.02
    
    thumb_higher_than_index = lm[4].y < lm[8].y
    
    return fingers_folded and thumb_high and thumb_higher_than_index

def extended_fingers(hand):
    checks = {
        "index": finger_extended(hand, 8, 6, 5),
        "middle": finger_extended(hand, 12, 10, 9),
        "ring": finger_extended(hand, 16, 14, 13),
        "pinky": finger_extended(hand, 20, 18, 17),
    }
    return [name for name, value in checks.items() if value]


# ---------------------------------------------------------------------------
# Face-only detectors
# ---------------------------------------------------------------------------
def is_mouth_open(face):
    face_h = get_face_scale(face)
    if face_h == 0: return False
    mouth_h = abs(face.landmark[13].y - face.landmark[14].y)
    return (mouth_h / face_h) > 0.15

def is_closed_mouth_smile(face):
    lm = face.landmark
    face_w = get_face_width(face)
    face_h = get_face_scale(face)
    if face_w == 0 or face_h == 0: return False
    
    mouth_w = dist(lm[61], lm[291])
    mouth_h = abs(lm[13].y - lm[14].y)
    
    corner_y = (lm[61].y + lm[291].y) / 2
    center_y = (lm[13].y + lm[14].y) / 2
    smile_curve = center_y - corner_y
    
    return (mouth_w / face_w) > 0.38 and (mouth_h / face_h) < 0.1 and (smile_curve / face_h) > 0.015

def are_eyes_closed(face):
    lm = face.landmark
    face_h = get_face_scale(face)
    if face_h == 0: return False
    left_eye_h = dist(lm[159], lm[145])
    right_eye_h = dist(lm[386], lm[374])
    return (left_eye_h / face_h) < 0.025 and (right_eye_h / face_h) < 0.025

def is_eyebrow_raised(face):
    lm = face.landmark
    face_h = get_face_scale(face)
    if face_h == 0: return False
    
    left_brow = np.mean([lm[65].y, lm[66].y, lm[70].y])
    left_eye = np.mean([lm[159].y, lm[160].y])
    right_brow = np.mean([lm[295].y, lm[296].y, lm[300].y])
    right_eye = np.mean([lm[386].y, lm[385].y])
    
    left_dist = left_eye - left_brow
    right_dist = right_eye - right_brow
    
    return (left_dist / face_h) > 0.12 or (right_dist / face_h) > 0.12


# ---------------------------------------------------------------------------
# Hand + face detectors
# ---------------------------------------------------------------------------
def count_hands_on_mouth(hands, face):
    if not hands or not face: return 0
    lm = face.landmark
    mouth = np.array([(lm[61].x + lm[291].x) / 2, (lm[61].y + lm[291].y) / 2])
    count = 0
    for hand in hands:
        center = hand_center(hand)
        near = np.linalg.norm(center - mouth) < 0.28
        near_points = sum(np.linalg.norm(point_xy(p) - mouth) < 0.25 for p in hand.landmark)
        if near or near_points >= 5:
            count += 1
    return count

def is_salute(hand, face):
    lm = face.landmark
    forehead = point_xy(lm[10])
    center = hand_center(hand)
    face_h = get_face_scale(face)
    fingers = extended_fingers(hand)
    return (len(fingers) >= 3 and center[1] < forehead[1] + (face_h * 0.2) and np.linalg.norm(center - forehead) < face_h * 0.8)

def is_hand_under_chin(hand, face):
    lm = face.landmark
    chin = point_xy(lm[152])
    center = hand_center(hand)
    face_h = get_face_scale(face)
    return np.linalg.norm(center - chin) < face_h * 0.5 and center[1] >= chin[1] - (face_h * 0.1)

def is_heart(hands):
    if not hands or len(hands) != 2: 
        return False
    a, b = hands[0].landmark, hands[1].landmark
    
    index_gap = dist(a[8], b[8])
    thumb_gap = dist(a[4], b[4])
    
    return (index_gap < 0.15 and thumb_gap < 0.15)

def is_both_fists(hands):
    if not hands or len(hands) != 2: return False
    for hand in hands:
        if len(extended_fingers(hand)) != 0: return False
        lm = hand.landmark
        folded = (lm[8].y > lm[6].y and lm[12].y > lm[10].y and lm[16].y > lm[14].y and lm[20].y > lm[18].y)
        if not folded: return False
    return True


# ---------------------------------------------------------------------------
# Main gesture classifier
# ---------------------------------------------------------------------------
def detect_gesture(hands, face):
    if face and are_eyes_closed(face):
        return "hide_face"

    if len(hands) == 2:
        if is_heart(hands): return "heart"
        if is_both_fists(hands): return "fists"

    if len(hands) >= 1:
        for hand in hands:
            if extended_fingers(hand) == ["index"]:
                return "index_finger"

    if face and len(hands) >= 1:
        if any(is_salute(h, face) for h in hands):
            return "salute"
            
        if any(is_hand_under_chin(h, face) for h in hands):
            return "chin"
            
        if count_hands_on_mouth(hands, face) >= 1:
            return "shy"

    if len(hands) >= 1:
        for hand in hands:
            fingers = extended_fingers(hand)
            if "index" in fingers and (len(fingers) >= 2 or not thumb_extended(hand)):
                return "fists"

    if face:
        if is_eyebrow_raised(face): return "eyebrow"
        if is_mouth_open(face): return "mouth_open"
        if is_closed_mouth_smile(face): return "smile"

    return "none"

# ---------------------------------------------------------------------------
# Media loading & Application Loop
# ---------------------------------------------------------------------------
def resolve_media_path(images_folder, candidates, keywords):
    files = {
        name.lower(): name
        for name in os.listdir(images_folder)
        if os.path.isfile(os.path.join(images_folder, name))
    }
    for candidate in candidates:
        actual = files.get(candidate.lower())
        if actual:
            return os.path.join(images_folder, actual)
    for name in files.values():
        stem = os.path.splitext(name)[0].lower()
        if any(keyword.lower() in stem for keyword in keywords):
            return os.path.join(images_folder, name)
    return None

def create_placeholder_image(gesture_name):
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.putText(img, gesture_name.upper(), (35, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img

def load_meme_media(images_folder):
    meme_images = {}
    video_caps = {}
    is_video = {}

    for gesture, candidates in GESTURE_MEDIA.items():
        path = resolve_media_path(images_folder, candidates, GESTURE_KEYWORDS.get(gesture, []))
        if path is None:
            meme_images[gesture] = create_placeholder_image(gesture)
            is_video[gesture] = False
            continue

        ext = os.path.splitext(path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                is_video[gesture] = True
                video_caps[gesture] = cap
                ret, frame = cap.read()
                if ret:
                    meme_images[gesture] = frame
                else:
                    meme_images[gesture] = create_placeholder_image(gesture)
            else:
                is_video[gesture] = False
                meme_images[gesture] = create_placeholder_image(gesture)
        else:
            img = cv2.imread(path)
            is_video[gesture] = False
            if img is not None:
                meme_images[gesture] = img
            else:
                meme_images[gesture] = create_placeholder_image(gesture)
    return meme_images, video_caps, is_video

def resize_meme(meme_image, target_height):
    h, w = meme_image.shape[:2]
    if h <= 0 or w <= 0: return create_placeholder_image("invalid_media")
    new_width = max(1, int(target_height * (w / h)))
    return cv2.resize(meme_image, (new_width, target_height), interpolation=cv2.INTER_AREA)

# ---------------------------------------------------------------------------
# UI Helpers for Achievement Panel
# ---------------------------------------------------------------------------
def draw_check_mark(img, x, y, size):
    """رسم علامة صح خضراء بسيطة"""
    color = (0, 255, 0) # أخضر BGR
    thickness = max(2, int(size * 0.1))
    
    pt1 = (int(x + size * 0.2), int(y + size * 0.5))
    pt2 = (int(x + size * 0.45), int(y + size * 0.75))
    pt3 = (int(x + size * 0.8), int(y + size * 0.2))
    
    cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
    cv2.line(img, pt2, pt3, color, thickness, cv2.LINE_AA)

def create_side_panel(height, meme_images, unlocked_gestures):
    """إنشاء اللوحة الجانبية للإنجازات"""
    panel_width = 300 
    panel = np.zeros((height, panel_width, 3), dtype=np.uint8)
    
    panel[:] = (30, 30, 30) 
    
    cv2.putText(panel, "Achievements", (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    cols = 2
    rows = 5 
    margin = 10
    start_y = 50
    
    avail_width = panel_width - (margin * (cols + 1))
    avail_height = height - start_y - (margin * (rows + 1))
    
    thumb_w = avail_width // cols
    thumb_h = avail_height // rows
    
    for i, gesture in enumerate(PANEL_GESTURES):
        if i >= 10: break 
        
        row = i // cols
        col = i % cols
        
        x = margin + col * (thumb_w + margin)
        y = start_y + row * (thumb_h + margin)
        
        img = meme_images.get(gesture, create_placeholder_image(gesture))
        
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        
        is_unlocked = gesture in unlocked_gestures
        
        if not is_unlocked:
            thumb = cv2.convertScaleAbs(thumb, alpha=0.3, beta=0)
            
        panel[y:y+thumb_h, x:x+thumb_w] = thumb
        
        border_color = (0, 255, 0) if is_unlocked else (100, 100, 100)
        cv2.rectangle(panel, (x, y), (x+thumb_w, y+thumb_h), border_color, 2)
        
        if is_unlocked:
            check_size = int(min(thumb_w, thumb_h) * 0.4)
            check_x = x + (thumb_w - check_size) // 2
            check_y = y + (thumb_h - check_size) // 2
            draw_check_mark(panel, check_x, check_y, check_size)
            
    return panel


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".webm")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")

def main():
    images_folder = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(images_folder, exist_ok=True)

    meme_images, video_caps, is_video = load_meme_media(images_folder)
    
    unlocked_gestures = set()
    
    # ---------------------------------------------------------
    # متغيرات التثبيت (Smoothing / Debounce variables)
    # ---------------------------------------------------------
    last_detected_gesture = "none"
    consecutive_frames = 0
    # يمكنك زيادة أو تقليل هذا الرقم حسب رغبتك 
    # (5 فريمات = استقرار جيد جداً واستجابة سريعة في نفس الوقت)
    REQUIRED_FRAMES_TO_CHANGE = 5 
    current_stable_gesture = "none"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5, # تم رفع الدقة قليلاً لمزيد من الاستقرار
        min_tracking_confidence=0.5,
    ) as hands_model, mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as face_model:

        while True:
            success, frame = cap.read()
            if not success: break

            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands_model.process(rgb)
            face_results = face_model.process(rgb)

            detected_hands = hand_results.multi_hand_landmarks if hand_results.multi_hand_landmarks else []
            face_landmarks = face_results.multi_face_landmarks[0] if face_results.multi_face_landmarks else None

            # 1. الحصول على الحركة الحالية (التي قد تكون متذبذبة)
            raw_gesture = detect_gesture(detected_hands, face_landmarks)
            
            # 2. تطبيق نظام الاستقرار (Debounce)
            if raw_gesture == last_detected_gesture:
                consecutive_frames += 1
            else:
                consecutive_frames = 1
                last_detected_gesture = raw_gesture
                
            # لو الحركة استمرت لعدد الفريمات المطلوب، نعتمدها كالحركة النهائية
            if consecutive_frames >= REQUIRED_FRAMES_TO_CHANGE:
                current_stable_gesture = raw_gesture
            
            # 3. تسجيل الإنجازات بناءً على الحركة المستقرة
            if current_stable_gesture != "none" and current_stable_gesture in PANEL_GESTURES:
                unlocked_gestures.add(current_stable_gesture)

            # 4. تشغيل الفيديو بسلاسة
            if is_video.get(current_stable_gesture, False):
                video_cap = video_caps.get(current_stable_gesture)
                if video_cap is not None and video_cap.isOpened():
                    ret, video_frame = video_cap.read()
                    if ret:
                        meme_images[current_stable_gesture] = video_frame
                    else:
                        video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, video_frame = video_cap.read()
                        if ret:
                            meme_images[current_stable_gesture] = video_frame

            # 5. تجهيز صورة الميم
            meme = meme_images.get(current_stable_gesture)
            if meme is None: meme = meme_images.get("none", create_placeholder_image("none"))

            meme_resized = resize_meme(meme, frame_height)
            if meme_resized.shape[1] > frame_width:
                meme_resized = cv2.resize(meme_resized, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            meme_width = meme_resized.shape[1]

            # 6. تجهيز اللوحة الجانبية
            side_panel = create_side_panel(frame_height, meme_images, unlocked_gestures)
            panel_width = side_panel.shape[1]

            # 7. دمج الشاشات
            total_width = frame_width + meme_width + panel_width
            combined = np.zeros((frame_height, total_width, 3), dtype=np.uint8)

            combined[:, :frame_width] = frame 
            combined[:, frame_width:frame_width+meme_width] = meme_resized 
            combined[:, frame_width+meme_width:] = side_panel 

            cv2.imshow("Gesture Meme Tracker", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    for video_cap in video_caps.values():
        if video_cap.isOpened(): video_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()