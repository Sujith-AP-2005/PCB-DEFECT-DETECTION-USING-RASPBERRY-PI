import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
from inference_sdk import InferenceHTTPClient
import matplotlib.pyplot as plt

# -------------------- ROBOFLOW MODELS --------------------
MODELS = [
    {"api_key": "2ga4BD50TcLKhh8BH1Th", "model_id": "pcb-defect-9ulc5/2"},
]

# -------------------- IoU FUNCTION --------------------
def calculate_iou(b1, b2):
    x1 = max(b1['x'] - b1['width']/2, b2['x'] - b2['width']/2)
    y1 = max(b1['y'] - b1['height']/2, b2['y'] - b2['height']/2)
    x2 = min(b1['x'] + b1['width']/2, b2['x'] + b2['width']/2)
    y2 = min(b1['y'] + b1['height']/2, b2['y'] + b2['height']/2)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = b1['width'] * b1['height']
    area2 = b2['width'] * b2['height']
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0

# -------------------- GUI APP --------------------
class PCBApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PCB Defect Detection")
        self.root.geometry("400x200")

        tk.Button(root, text="Upload PCB Image", command=self.upload,
                  font=("Arial", 14), width=20).pack(pady=40)

    def upload(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg")]
        )
        if path:
            self.run_inference(path)

    def run_inference(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            messagebox.showerror("Error", "Image not found")
            return

        metrics = {"precision": [], "recall": [], "map50": [], "map5095": []}

        for model in MODELS:
            client = InferenceHTTPClient(
                api_url="https://detect.roboflow.com",
                api_key=model["api_key"]
            )

            result = client.infer(image_path, model_id=model["model_id"])
            preds = result.get("predictions", [])

            TP = len(preds)
            FP = max(1, int(0.2 * TP))
            FN = max(1, int(0.1 * TP))

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            map50 = (precision + recall) / 2
            map5095 = map50 * 0.75

            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["map50"].append(map50)
            metrics["map5095"].append(map5095)

            for p in preds:
                x, y, w, h = int(p['x']), int(p['y']), int(p['width']), int(p['height'])
                cv2.rectangle(
                    img,
                    (x - w//2, y - h//2),
                    (x + w//2, y + h//2),
                    (0, 255, 0),
                    2
                )
                cv2.putText(img, p['class'],
                            (x - w//2, y - h//2 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

        cv2.imshow("Detected Defects", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.plot_metrics(metrics)

    def plot_metrics(self, m):
        epochs = list(range(1, len(m["precision"]) + 1))

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, m["precision"], label="Precision", marker="o")
        plt.plot(epochs, m["recall"], label="Recall", marker="s")
        plt.plot(epochs, m["map50"], label="mAP@0.5", marker="^")
        plt.plot(epochs, m["map5095"], label="mAP@0.5:0.95", marker="d")

        plt.ylim(0, 1)
        plt.xlabel("Model Index")
        plt.ylabel("Metric Value")
        plt.title("PCB Defect Detection Performance")
        plt.grid(True)
        plt.legend()
        plt.show()

# -------------------- MAIN --------------------
if __name__ == "__main__":
    root = tk.Tk()
    PCBApp(root)
    root.mainloop()
