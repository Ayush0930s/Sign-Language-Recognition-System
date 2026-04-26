import os
import cv2
import time

# ================== CONFIG ==================
DATA_DIR = "./data"
NUMBER_OF_CLASSES = 38     # A-Z, 0-9, space, dot
DATASET_SIZE = 500        # images per class
CAMERA_INDEX = 0

# ================== CREATE DATA FOLDER ==================
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ================== OPEN CAMERA ==================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(3, 1280)
cap.set(4, 720)

if not cap.isOpened():
    print("Camera not opening")
    exit()

print("Camera opened successfully")

# ================== DATA COLLECTION ==================
for class_id in range(4, NUMBER_OF_CLASSES):

    class_dir = os.path.join(DATA_DIR, str(class_id))
    os.makedirs(class_dir, exist_ok=True)

    print(f"\n Collecting data for class {class_id}")
    print(" Press 'Q' when ready to start capturing images")

    # ---------- WAIT FOR USER ----------
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(
            frame,
            f"Class {class_id} | Press Q to start",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(25)
        if key == ord('q'):
            break
        elif key == 27:  # ESC to exit
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # ---------- CAPTURE IMAGES ----------
    counter = 0
    time.sleep(1)  # small delay before capture

    while counter < DATASET_SIZE:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.putText(
            frame,
            f"Class {class_id} | Image {counter+1}/{DATASET_SIZE}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2
        )

        cv2.imshow("Data Collection", frame)

        # Save image
        img_path = os.path.join(class_dir, f"{counter}.jpg")
        cv2.imwrite(img_path, frame)

        counter += 1
        cv2.waitKey(50)   # small delay (better quality)

    print(f" Done class {class_id}")

# ================== CLEANUP ==================
cap.release()
cv2.destroyAllWindows()

print("\n DATA COLLECTION COMPLETED SUCCESSFULLY")

