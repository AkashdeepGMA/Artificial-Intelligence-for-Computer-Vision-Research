import subprocess
import sys

def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install_dependencies():
    # Force reinstall numpy 1.26.4 for compatibility issues
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--force-reinstall"])
    install_and_import("ultralytics")
    install_and_import("opencv-python")

try:
    from ultralytics import YOLO
    import cv2
except ImportError:
    print("Dependencies missing, installing now...")
    install_dependencies()
    print("Please restart the script after installation.")
    sys.exit()

def detect_vehicles(input_video='traffic1.mp4', output_video='traffic1_boxed.mp4'):
    model = YOLO('yolov8n.pt')  # This will auto-download if not present

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    names = model.model.names if hasattr(model.model, "names") else model.names

    print("Processing video for vehicle detection...")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        boxes = results.boxes.data.cpu().numpy() if hasattr(results.boxes, 'data') else results.boxes.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2, conf, cls = box[:6]
            label = names[int(cls)]
            if label in ['car', 'truck', 'bus', 'motorcycle']:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out.write(frame)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Finished processing. Output saved as {output_video}")

if __name__ == '__main__':
    detect_vehicles()
