

# ERPAttendanceFlask.py
import os
import cv2
import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime, date
import uuid
import smtplib
import shutil
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from flask import Flask, request, redirect, url_for, render_template_string, send_file

# --------------------
# CONFIG
# --------------------
smtp_user = "2025studattendanceda@gmail.com"
smtp_pass = "iedicabertsusqpt"  # 16-char App Password
email_to_default = "harsh29012006@gmail.com"

DB_PATH = "attendance.db"
DATASET_DIR = "dataset"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "trainer.yml")
LABEL_MAP_FILE = os.path.join(MODEL_DIR, "label_map.json")
ATTENDANCE_EXPORT_DIR = "exports"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ATTENDANCE_EXPORT_DIR, exist_ok=True)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
DEFAULT_CONF_THRESHOLD = 55.0  # lower = stricter

app = Flask(__name__)

# ---------------------------
# Database helpers
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT UNIQUE,
            name TEXT,
            phone TEXT,
            class TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            code TEXT,
            date TEXT,
            time TEXT,
            status TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("[DB] Initialized:", DB_PATH)

def add_student_db(student_id, name, phone="", klass=""):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO students (student_id, name, phone, class, created_at)
        VALUES (?, ?, ?, ?, ?)
    """, (student_id, name, phone, klass, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_all_students():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT student_id, name, phone FROM students")
    rows = cur.fetchall()
    conn.close()
    return rows

def get_name_by_student_id(student_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM students WHERE student_id = ?", (student_id,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else ""

def mark_attendance_db(student_id, target_date=None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    if target_date is None:
        target_date = date.today().isoformat()
    cur.execute("SELECT COUNT(*) FROM attendance WHERE student_id = ? AND date = ?", (student_id, target_date))
    if cur.fetchone()[0] == 0:
        code = str(uuid.uuid4())[:8]
        cur.execute("INSERT INTO attendance (student_id, code, date, time, status) VALUES (?, ?, ?, ?, ?)",
                    (student_id, code, target_date, datetime.now().strftime("%H:%M:%S"), "present"))
        conn.commit()
        print(f"[ATTENDANCE] {student_id} marked present on {target_date} with code {code}")
    conn.close()

def get_present_student_ids(target_date=None):
    if target_date is None:
        target_date = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT student_id FROM attendance WHERE date = ?", (target_date,))
    rows = cur.fetchall()
    conn.close()
    return {r[0] for r in rows}

def export_attendance_csv(target_date=None):
    if target_date is None:
        target_date = date.today().isoformat()
    students = get_all_students()
    present_ids = get_present_student_ids(target_date)
    records = []
    for sid, name, phone in students:
        status = "Present" if sid in present_ids else "Absent"
        records.append([sid, name, phone, target_date, status])
    df = pd.DataFrame(records, columns=["Student ID", "Name", "Phone", "Date", "Status"])
    path = os.path.join(ATTENDANCE_EXPORT_DIR, f"attendance_{target_date}.csv")
    df.to_csv(path, index=False)
    print("[EXPORT] Attendance exported to", path)
    return path

# ---------------------------
# Email helper (attach CSV)
# ---------------------------
def send_email_with_attachment(subject, body, to_addr=email_to_default, attachment_path=None):
    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    if attachment_path and os.path.exists(attachment_path):
        part = MIMEBase('application', "octet-stream")
        with open(attachment_path, "rb") as f:
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment_path))
        msg.attach(part)
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        print(f"[EMAIL] Sent to {to_addr}")
    except Exception as e:
        print("[EMAIL] Error:", e)
        raise

def send_attendance_email(target_date=None, to_addr=email_to_default):
    if target_date is None:
        target_date = date.today().isoformat()
    students = get_all_students()
    present_ids = get_present_student_ids(target_date)
    body = f"Attendance Report for {target_date}:\n\n"
    for sid, name, phone in students:
        status = "Present" if sid in present_ids else "Absent"
        body += f"- {sid} | {name} | {phone} | {status}\n"
    csv_path = export_attendance_csv(target_date)
    send_email_with_attachment(f"Attendance Report {target_date}", body, to_addr=to_addr, attachment_path=csv_path)
    return csv_path

# ---------------------------
# Face preprocessing + train + enroll + recognize
# ---------------------------
def preprocess_face(face):
    face = cv2.resize(face, (200, 200))
    face = cv2.equalizeHist(face)
    face = cv2.GaussianBlur(face, (3, 3), 0)
    return face

def gather_training_data():
    faces, labels, label_map = [], [], {}
    next_label = 0
    for student_id in sorted(os.listdir(DATASET_DIR)):
        student_path = os.path.join(DATASET_DIR, student_id)
        if not os.path.isdir(student_path): continue
        label_map[next_label] = student_id
        for file in os.listdir(student_path):
            path = os.path.join(student_path, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: 
                continue
            img = preprocess_face(img)
            faces.append(img)
            labels.append(next_label)
        next_label += 1
    return faces, labels, label_map

def train_model():
    faces, labels, label_map = gather_training_data()
    if not faces:
        print("[TRAIN] No faces found. Enroll students first.")
        return False
    # create LBPH recognizer (ensure OpenCV has face module)
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_FILE)
    with open(LABEL_MAP_FILE, "w") as f:
        json.dump(label_map, f)
    print("[TRAIN] Model trained and saved.")
    return True

def enroll_student_interactive(student_id, name, phone="", klass=""):
    """OpenCV-based enrollment: captures ~100 images and saves to dataset/<student_id>/"""
    add_student_db(student_id, name, phone, klass)
    student_dir = os.path.join(DATASET_DIR, student_id)
    os.makedirs(student_dir, exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ENROLL] ERROR: Could not open camera.")
        return 0
    detector = cv2.CascadeClassifier(CASCADE_PATH)
    print("[ENROLL] Look at the camera. Capturing images... Press 'q' to stop early.")
    count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ENROLL] Camera read failed.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = preprocess_face(face)
            count += 1
            filename = os.path.join(student_dir, f"{student_id}_{count}.jpg")
            cv2.imwrite(filename, face)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(frame, str(count), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Enroll - press q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= 100:
            break
    cam.release()
    cv2.destroyAllWindows()
    print(f"[ENROLL] Captured {count} images for {student_id}")
    return count

def recognize_and_mark(conf_threshold=DEFAULT_CONF_THRESHOLD, target_date=None):
    if not os.path.exists(MODEL_FILE) or not os.path.exists(LABEL_MAP_FILE):
        print("[RECOGNIZE] Model or label map missing. Train first.")
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)
    with open(LABEL_MAP_FILE, "r") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    detector = cv2.CascadeClassifier(CASCADE_PATH)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[RECOGNIZE] Could not open camera.")
        return
    print("[RECOGNIZE] Recognition started. Press 'q' to stop.")
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = preprocess_face(face)
            try:
                label, conf = recognizer.predict(face)
            except Exception:
                continue
            if conf < conf_threshold and label in label_map:
                student_id = label_map[label]
                name = get_name_by_student_id(student_id)
                mark_attendance_db(student_id, target_date)
                text = f"{student_id} - {name} ({int(conf)})"
                color = (0,255,0)
            else:
                text = f"Unknown ({int(conf)})"
                color = (0,0,255)
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.imshow("Recognition - press q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
    print("[RECOGNIZE] Stopped.")

# ---------------------------
# Remove student and retrain
# ---------------------------
def remove_student(student_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
    cur.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
    conn.commit()
    conn.close()
    student_dir = os.path.join(DATASET_DIR, student_id)
    if os.path.exists(student_dir):
        shutil.rmtree(student_dir)
        print(f"[REMOVE] Deleted dataset for {student_id}")
    print(f"[REMOVE] Student {student_id} removed.")
    # retrain the model to forget the removed student (if dataset left)
    try:
        train_model()
    except Exception as e:
        print("[REMOVE] Retrain failed:", e)

# ---------------------------
# Flask UI (single-file HTML + CSS)
# ---------------------------
PAGE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Face Recognition Attendance System</title>
  <style>
    body { font-family: Arial, sans-serif; background: #f4f7f9; margin:0; padding:0; }
    .container { width: 950px; margin: 30px auto; background:#fff; padding:24px; border-radius:12px; box-shadow:0 6px 18px rgba(0,0,0,0.08); }
    h1 { text-align:center; color:#2c3e50; }
    .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:12px; }
    .card { padding:12px; border-radius:8px; background:#fafafa; border:1px solid #eee; flex:1; min-width:220px; }
    .btn { padding:8px 12px; border:none; border-radius:8px; background:#3498db; color:white; cursor:pointer; text-decoration:none; }
    .btn:hover{ background:#2980b9; }
    .danger { background:#e74c3c; }
    .danger:hover{ background:#c0392b; }
    input[type=text], input[type=date], input[type=email] { padding:8px; border-radius:6px; border:1px solid #ccc; }
    table { width:100%; border-collapse:collapse; margin-top:16px; }
    th, td { border:1px solid #e6e6e6; padding:10px; text-align:center; }
    th { background:#3498db; color:white; }
    .present { background:#e8f8f5; color:#27ae60; }
    .absent { background:#fdecea; color:#e74c3c; }
    .small { font-size:13px; color:#555; }
    .success { color: #27ae60; }
    .error { color: #e74c3c; }
    .note { margin-top:8px; font-size:13px; color:#666; }
  </style>
</head>
<body>
  <div class="container">
    <h1>🎓 Face Recognition Attendance System</h1>

    <div class="row">
      <div class="card">
        <form method="post" action="/initdb" style="display:flex; gap:8px; align-items:center;">
          <button class="btn" type="submit">Initialize / Reset DB</button>
          <span class="small">Creates tables if missing</span>
        </form>
      </div>

      <div class="card">
        <form method="post" action="/enroll">
          <div style="display:flex; gap:8px; flex-wrap:wrap;">
            <input name="student_id" type="text" placeholder="Student ID (S001)" required>
            <input name="name" type="text" placeholder="Full Name" required>
            <input name="phone" type="text" placeholder="Phone (optional)">
            <button class="btn" type="submit">➕ Enroll (camera)</button>
          </div>
          <div class="note">Opens camera and captures ~100 face images. Press 'q' to stop early.</div>
        </form>
      </div>

      <div class="card">
        <form method="post" action="/train" style="display:flex; gap:8px; align-items:center;">
          <button class="btn" type="submit">🧠 Train Model</button>
          <span class="small">Trains LBPH on dataset/</span>
        </form>
      </div>
    </div>

    <div class="row">
      <div class="card">
        <form method="post" action="/recognize" style="display:flex; gap:8px; align-items:center;">
          <input name="date" type="date" value="{{ selected_date }}">
          <button class="btn" type="submit">📸 Start Recognition (camera)</button>
        </form>
        <div class="note">Recognizes faces using webcam and marks attendance for the chosen date.</div>
      </div>

      <div class="card">
        <form method="post" action="/export" style="display:flex; gap:8px; align-items:center;">
          <input name="date" type="date" value="{{ selected_date }}">
          <button class="btn" type="submit">📂 Export CSV</button>
          <a href="/download?date={{ selected_date }}" class="btn">⬇️ Download Latest</a>
        </form>
        <div class="note">Export creates exports/attendance_YYYY-MM-DD.csv and allows download.</div>
      </div>

      <div class="card">
        <form method="post" action="/email" style="display:flex; gap:8px; align-items:center;">
          <input name="date" type="date" value="{{ selected_date }}">
          <input name="email_to" type="email" placeholder="Recipient email" value="{{ email_to }}">
          <button class="btn" type="submit">✉️ Send Email (attach CSV)</button>
        </form>
        <div class="note">Sends CSV as attachment to recipient.</div>
      </div>
    </div>

    <h2>Students & Attendance ({{ selected_date }})</h2>
    <table>
      <thead>
        <tr>
          <th>Student ID</th>
          <th>Name</th>
          <th>Phone</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody>
        {% for sid, name, phone, status in records %}
        <tr class="{{ 'present' if status=='present' else 'absent' }}">
          <td>{{ sid }}</td>
          <td>{{ name }}</td>
          <td>{{ phone }}</td>
          <td>{{ status.capitalize() }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>

    <div style="margin-top:16px; display:flex; gap:8px; align-items:center;">
      <form method="post" action="/remove" style="display:flex; gap:8px; align-items:center;">
        <input name="student_id" type="text" placeholder="Student ID to remove">
        <button class="btn danger" type="submit">❌ Remove Student</button>
      </form>
      <form method="get" action="/students" style="display:inline;">
        <button class="btn" type="submit">View All Students</button>
      </form>
    </div>

    <div style="margin-top:12px;">
      {% if message %}
        <div class="{{ 'success' if success else 'error' }}">{{ message }}</div>
      {% endif %}
    </div>

  </div>
</body>
</html>
"""

# ---------------------------
# Flask routes / wiring
# ---------------------------
def get_attendance_records_for_date(selected_date=None):
    if selected_date is None:
        selected_date = date.today().isoformat()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT s.student_id, s.name, s.phone,
               CASE WHEN a.status IS NULL THEN 'absent' ELSE a.status END as status
        FROM students s
        LEFT JOIN attendance a
          ON s.student_id = a.student_id AND a.date = ?
        ORDER BY s.student_id
    """, (selected_date,))
    rows = cur.fetchall()
    conn.close()
    return rows

@app.route("/", methods=["GET"])
def index_redirect():
    return redirect(url_for("home"))

@app.route("/home", methods=["GET", "POST"])
def home():
    selected_date = request.form.get("date") if request.method == "POST" else date.today().isoformat()
    records = get_attendance_records_for_date(selected_date)
    return render_template_string(PAGE_HTML,
                                  records=records,
                                  selected_date=selected_date,
                                  email_to=email_to_default,
                                  message=None,
                                  success=True)

@app.route("/initdb", methods=["POST"])
def route_initdb():
    init_db()
    return redirect(url_for("home"))

@app.route("/enroll", methods=["POST"])
def route_enroll():
    sid = request.form.get("student_id")
    name = request.form.get("name")
    phone = request.form.get("phone", "")
    if not sid or not name:
        return redirect(url_for("home"))
    # This will open camera window and block until finished.
    count = enroll_student_interactive(sid, name, phone, "")
    # optionally retrain automatically after enrollment
    try:
        train_model()
    except Exception as e:
        print("[ENROLL] Auto-train failed:", e)
    return redirect(url_for("home"))

@app.route("/train", methods=["POST"])
def route_train():
    success = train_model()
    return redirect(url_for("home"))

@app.route("/recognize", methods=["POST"])
def route_recognize():
    d = request.form.get("date") or None
    # This opens camera window and runs recognition; marks attendance with the given date.
    recognize_and_mark(conf_threshold=DEFAULT_CONF_THRESHOLD, target_date=d)
    return redirect(url_for("home"))

@app.route("/export", methods=["POST"])
def route_export():
    d = request.form.get("date") or None
    export_attendance_csv(d)
    return redirect(url_for("home"))

@app.route("/download", methods=["GET"])
def route_download():
    d = request.args.get("date") or date.today().isoformat()
    path = os.path.join(ATTENDANCE_EXPORT_DIR, f"attendance_{d}.csv")
    if not os.path.exists(path):
        export_attendance_csv(d)
    return send_file(path, as_attachment=True)

@app.route("/email", methods=["POST"])
def route_email():
    d = request.form.get("date") or None
    to_addr = request.form.get("email_to") or email_to_default
    try:
        csv_path = send_attendance_email(d, to_addr=to_addr)
        msg = f"Email sent to {to_addr}. CSV: {os.path.basename(csv_path)}"
        records = get_attendance_records_for_date(d or date.today().isoformat())
        return render_template_string(PAGE_HTML, records=records, selected_date=(d or date.today().isoformat()),
                                      email_to=to_addr, message=msg, success=True)
    except Exception as e:
        records = get_attendance_records_for_date(d or date.today().isoformat())
        return render_template_string(PAGE_HTML, records=records, selected_date=(d or date.today().isoformat()),
                                      email_to=to_addr, message=str(e), success=False)

@app.route("/remove", methods=["POST"])
def route_remove():
    sid = request.form.get("student_id")
    if sid:
        remove_student(sid)
    return redirect(url_for("home"))

@app.route("/students", methods=["GET"])
def route_students():
    rows = get_all_students()
    out = "<h2>Registered students</h2><ul>"
    for sid, name, phone in rows:
        out += f"<li>{sid} | {name} | {phone}</li>"
    out += "</ul><a href='/home'>Back</a>"
    return out

# ---------------------------
# Start
# ---------------------------
if __name__ == "__main__":
    init_db()
    # run Flask app; camera-based actions will open OpenCV windows when called
    app.run(host="127.0.0.1", port=5000, debug=True)
