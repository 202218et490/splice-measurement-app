from flask import Flask, request, jsonify, send_file, render_template
import os
import json
import cv2
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

UPLOAD_FOLDER = "static"
os.makedirs(os.path.join(BASE_DIR, UPLOAD_FOLDER), exist_ok=True)

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- UPLOAD ----------------
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]
    path = os.path.join(BASE_DIR, UPLOAD_FOLDER, "test.jpg")
    file.save(path)
    return jsonify({"status": "uploaded"})


# ---------------- RECTIFY ----------------
@app.route("/rectify", methods=["POST"])
def rectify():
    data = request.json
    corners = data["corners"]
    length = float(data["length"])
    width = float(data["width"])

    image = cv2.imread(os.path.join(BASE_DIR, "static", "test.jpg"))

    pts = np.array([[p["x"], p["y"]] for p in corners], dtype="float32")

    # order points
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect = np.zeros((4,2), dtype="float32")
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    PIXELS_PER_MM = 3
    max_w = int(length * PIXELS_PER_MM)
    max_h = int(width * PIXELS_PER_MM)

    dst = np.array([[0,0],[max_w-1,0],[max_w-1,max_h-1],[0,max_h-1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_w, max_h))

    rectified_path = os.path.join(BASE_DIR, "rectified.jpg")
    cv2.imwrite(rectified_path, warped)

    return send_file(rectified_path, mimetype="image/jpeg")


# ---------------- PROCESS ----------------
@app.route("/process", methods=["POST"])
def process():

    data = request.json
    points = data["centers"]

    # Load rectified image
    img = cv2.imread(os.path.join(BASE_DIR, "rectified.jpg"))
    H, W = img.shape[:2]

    PIXELS_PER_MM = 3
    mm_per_pixel = 1 / PIXELS_PER_MM

    # Convert points exactly like your original code
    centers = np.array([[p["x"], p["y"]] for p in points])

    # =========================
    # SAME LOGIC FROM YOUR CODE
    # =========================

    centers = centers[centers[:,1].argsort()]

    rows = []
    row = [centers[0]]

    for pt in centers[1:]:
        if abs(pt[1] - row[-1][1]) < 20 * PIXELS_PER_MM:
            row.append(pt)
        else:
            rows.append(sorted(row, key=lambda x: x[0]))
            row = [pt]
    rows.append(sorted(row, key=lambda x: x[0]))

    FONT = cv2.FONT_HERSHEY_SIMPLEX

    for i, r in enumerate(rows):
        for j, (x, y) in enumerate(r):
            x, y = int(x), int(y)

            cv2.circle(img, (x, y), 6, (0, 0, 255), -1)

            # Horizontal CTC
            if j > 0:
                px, py = r[j-1]
                d = abs(x - px) * mm_per_pixel
                cv2.line(img, (int(px), int(py)), (x, y), (255, 0, 0), 2)
                cv2.putText(img, str(int(d)),
                            ((x + int(px)) // 2, (y + int(py)) // 2 - 10),
                            FONT, 0.6, (255, 0, 0), 2)

            # Vertical CTC
            if i > 0 and j < len(rows[i-1]):
                px, py = rows[i-1][j]
                d = abs(y - py) * mm_per_pixel
                cv2.line(img, (int(px), int(py)), (x, y), (0, 255, 0), 2)
                cv2.putText(img, str(int(d)),
                            ((x + int(px)) // 2 + 10, (y + int(py)) // 2),
                            FONT, 0.6, (0, 255, 0), 2)

        # Edge distances
        xs = [pt[0] for pt in r]
        left_x, right_x = min(xs), max(xs)

        for (x, y) in r:
            x, y = int(x), int(y)

            if abs(x - left_x) < 5:
                d = x * mm_per_pixel
                cv2.line(img, (0, y), (x, y), (0, 255, 255), 2)
                cv2.putText(img, str(int(d)),
                            (x + 10, y + 25),
                            FONT, 0.6, (0, 255, 255), 2)

            if abs(x - right_x) < 5:
                d = (W - x) * mm_per_pixel
                cv2.line(img, (x, y), (W, y), (0, 165, 255), 2)
                cv2.putText(img, str(int(d)),
                            (x - 80, y + 25),
                            FONT, 0.6, (0, 165, 255), 2)

    # Save result
    final_path = os.path.join(BASE_DIR, "annotated_final.jpg")

    cv2.imwrite(final_path, img)
    return send_file(final_path, mimetype="image/jpeg")


if __name__ == "__main__":
       app.run()