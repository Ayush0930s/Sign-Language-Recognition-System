import os
import pickle
import cv2
import mediapipe as mp
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ================== MEDIAPIPE SETUP ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    min_detection_confidence=0.5,
    max_num_hands=1
)

# ================== CONFIG ==================
DATA_DIR = "./data"
EXPECTED_FEATURES = 63   # 21 landmarks × (x, y, z)

data = []
labels = []

total_images = 0
used_images = 0

# ================== DATASET CREATION ==================
for label in sorted(os.listdir(DATA_DIR)):
    label_path = os.path.join(DATA_DIR, label)

    if not os.path.isdir(label_path):
        continue

    for img_name in os.listdir(label_path):
        print("Processing:", img_name) #---------------------------
        img_path = os.path.join(label_path, img_name)
        total_images += 1

        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            continue

        for hand_landmarks in results.multi_hand_landmarks:
            x_list = []
            y_list = []
            data_aux = []

            for lm in hand_landmarks.landmark:
                x_list.append(lm.x)
                y_list.append(lm.y)

            min_x = min(x_list)
            min_y = min(y_list)

            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min_x)
                data_aux.append(lm.y - min_y)
                data_aux.append(lm.z)

            # Force exact feature length
            if len(data_aux) < EXPECTED_FEATURES:
                data_aux.extend([0] * (EXPECTED_FEATURES - len(data_aux)))
            elif len(data_aux) > EXPECTED_FEATURES:
                data_aux = data_aux[:EXPECTED_FEATURES]

            data.append(data_aux)
            labels.append(int(label))
            used_images += 1

# ================== SAVE DATASET ==================
with open("data.pickle", "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print("DATASET CREATION COMPLETED")
print(f"Total images scanned : {total_images}")
print(f" Samples used         : {used_images}")
print(f"Saved as             : data.pickle")
