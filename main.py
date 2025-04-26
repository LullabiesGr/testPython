from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np
from io import BytesIO
from typing import List
import base64
import mediapipe as mp
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Enhancement Functions -----------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

def decode_image(contents):
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img

def is_blurry(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold

def get_blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def auto_enhance(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = result[:, :, 1].mean()
    avg_b = result[:, :, 2].mean()
    result[:, :, 1] -= ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] -= ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

def sharpen_image(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def auto_crop_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    (x, y, w, h) = faces[0]
    return img[y:y+h, x:x+w]

def denoise_image(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

def straighten_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    angle = 0
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle += (theta - np.pi/2)
        angle /= len(lines)
    center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle * 180/np.pi, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

def cartoonify_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    return cv2.bitwise_and(color, color, mask=edges)

def calculate_ear(landmarks, eye_indices):
    p1 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p2 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p5 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p6 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])

    vertical1 = np.linalg.norm(p2 - p4)
    vertical2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p6)
    return (vertical1 + vertical2) / (2.0 * horizontal)

def is_facing_camera(landmarks):
    left_shoulder = np.array([landmarks[11].x, landmarks[11].y])
    right_shoulder = np.array([landmarks[12].x, landmarks[12].y])
    distance = np.linalg.norm(left_shoulder - right_shoulder)
    return distance < 0.4

def is_smiling(landmarks):
    mouth_left = np.array([landmarks[61].x, landmarks[61].y])
    mouth_right = np.array([landmarks[291].x, landmarks[291].y])
    mouth_top = np.array([landmarks[13].x, landmarks[13].y])
    mouth_bottom = np.array([landmarks[14].x, landmarks[14].y])

    horizontal = np.linalg.norm(mouth_left - mouth_right)
    vertical = np.linalg.norm(mouth_top - mouth_bottom)
    return (vertical / horizontal) > 0.25

# ----------------- API Endpoints -----------------

@app.post("/check_blur_and_enhance")
async def check_blur_and_enhance(file: UploadFile = File(...)):
    contents = await file.read()
    img = decode_image(contents)

    if is_blurry(img):
        return {"status": "rejected", "reason": "Image is too blurry."}

    enhanced = auto_enhance(img)
    _, buffer = cv2.imencode('.jpg', enhanced)
    return StreamingResponse(BytesIO(buffer), media_type="image/jpeg")

@app.post("/batch_enhance")
async def batch_enhance(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        contents = await file.read()
        img = decode_image(contents)
        score = get_blur_score(img)

        if score < 100:
            results.append({
                "filename": file.filename,
                "status": "rejected",
                "reason": "Image is too blurry.",
                "blur_score": score
            })
        else:
            enhanced = auto_enhance(img)
            _, buffer = cv2.imencode('.jpg', enhanced)
            encoded = base64.b64encode(buffer).decode('utf-8')
            results.append({
                "filename": file.filename,
                "status": "enhanced",
                "blur_score": score,
                "image_base64": encoded
            })
    return JSONResponse(content={"results": results})

@app.post("/white_balance_fix")
async def white_balance_fix(file: UploadFile = File(...)):
    contents = await file.read()
    img = decode_image(contents)
    corrected = white_balance(img)
    _, buffer = cv2.imencode('.jpg', corrected)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return {"image_base64": encoded}

@app.post("/sharpen_image")
async def sharpen_image_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    img = decode_image(contents)
    sharpened = sharpen_image(img)
    _, buffer = cv2.imencode('.jpg', sharpened)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return {"image_base64": encoded}

@app.post("/blur_score")
async def blur_score(file: UploadFile = File(...)):
    contents = await file.read()
    img = decode_image(contents)
    score = get_blur_score(img)
    return {"blur_score": score}

@app.post("/auto_crop_face")
async def auto_crop_face_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    img = decode_image(contents)
    cropped = auto_crop_face(img)
    if cropped is None:
        return {"status": "no_face_detected"}
    _, buffer = cv2.imencode('.jpg', cropped)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return {"image_base64": encoded}

@app.post("/denoise_image")
async def denoise_image_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    img = decode_image(contents)
    denoised = denoise_image(img)
    _, buffer = cv2.imencode('.jpg', denoised)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return {"image_base64": encoded}

@app.post("/straighten_image")
async def straighten_image_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    img = decode_image(contents)
    straightened = straighten_image(img)
    _, buffer = cv2.imencode('.jpg', straightened)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return {"image_base64": encoded}

@app.post("/cartoonify_image")
async def cartoonify_image_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    img = decode_image(contents)
    cartoon = cartoonify_image(img)
    _, buffer = cv2.imencode('.jpg', cartoon)
    encoded = base64.b64encode(buffer).decode('utf-8')
    return {"image_base64": encoded}

@app.post("/detect_closed_eyes")
async def detect_closed_eyes(file: UploadFile = File(...)):
    contents = await file.read()
    image = decode_image(contents)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
        result = face_mesh.process(rgb_image)
        if not result.multi_face_landmarks:
            return {"status": "no_face_detected"}
        face_landmarks = result.multi_face_landmarks[0].landmark
        LEFT_EYE = [362, 385, 387, 263, 373, 380]
        RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        left_ear = calculate_ear(face_landmarks, LEFT_EYE)
        right_ear = calculate_ear(face_landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2
        eyes_closed = avg_ear < 0.21
        return {"closed_eyes": eyes_closed, "average_ear": avg_ear}

@app.post("/detect_pose_smile")
async def detect_pose_smile(file: UploadFile = File(...)):
    contents = await file.read()
    image = decode_image(contents)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pose = mp_pose.Pose(static_image_mode=True)
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
    pose_result = pose.process(rgb_image)
    face_result = face_mesh.process(rgb_image)

    if not pose_result.pose_landmarks or not face_result.multi_face_landmarks:
        return {"status": "no_person_detected"}

    pose_landmarks = pose_result.pose_landmarks.landmark
    face_landmarks = face_result.multi_face_landmarks[0].landmark
    facing = is_facing_camera(pose_landmarks)
    smiling = is_smiling(face_landmarks)
    return {"facing_camera": facing, "smiling": smiling}

        
