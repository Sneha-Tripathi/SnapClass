
import dlib
import numpy as np
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st

from src.database.db import get_all_students


@st.cache_resource
def load_dlib_models():
    detector = dlib.get_frontal_face_detector()

    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, sp, facerec


def get_face_embeddings(image_np):
    detector, sp, facerec = load_dlib_models()

    # Ensure RGB image is converted to grayscale for face detection
    if len(image_np.shape) == 3:
        gray = np.mean(image_np[:, :, :3], axis=2).astype(np.uint8)
    else:
        gray = image_np

    faces = detector(gray, 1)
    encodings = []

    for face in faces:
        shape = sp(image_np, face)
        descriptor = facerec.compute_face_descriptor(
            image_np, shape, 1
        )
        encodings.append(np.array(descriptor, dtype=np.float64))

    return encodings


def get_trained_model():
    # Fetch current registered students every time; do not cache this data.
    student_db = get_all_students()

    if not student_db:
        return None

    X = []
    y = []

    for student in student_db:
        embedding = student.get("face_embedding")
        student_id = student.get("student_id")

        if embedding is None or student_id is None:
            continue

        try:
            vector = np.asarray(embedding, dtype=np.float64)

            if vector.shape == (128,) and np.all(np.isfinite(vector)):
                X.append(vector)
                y.append(student_id)
        except (TypeError, ValueError):
            continue

    if not X:
        return None

    return {"X": X, "y": y}


def train_classifier():
    # Kept for compatibility with existing registration code.
    # The matching function now reads fresh student embeddings directly.
    return get_trained_model() is not None


def predict_attendance(class_image_np):
    detected_student = {}
    encodings = get_face_embeddings(class_image_np)

    model_data = get_trained_model()

    if model_data is None:
        return detected_student, [], len(encodings)

    X_train = model_data["X"]
    y_train = model_data["y"]

    all_students = list(dict.fromkeys(y_train))

    # Require exactly one face for login/attendance identification.
    if len(encodings) != 1:
        return detected_student, all_students, len(encodings)

    encoding = encodings[0]

    # Calculate distance from this face to every registered embedding.
    distances = [
        float(np.linalg.norm(known_embedding - encoding))
        for known_embedding in X_train
    ]

    ranked_indices = np.argsort(distances)
    best_index = int(ranked_indices[0])
    best_distance = distances[best_index]
    best_student_id = y_train[best_index]

    # Conservative starting threshold; tune using genuine test photos.
    threshold = 0.50

    if best_distance > threshold:
        return detected_student, all_students, len(encodings)

    # Reject ambiguous matches when another student's face is nearly as close.
    if len(ranked_indices) > 1:
        second_index = int(ranked_indices[1])
        second_distance = distances[second_index]
        second_student_id = y_train[second_index]

        if (
            second_student_id != best_student_id
            and second_distance - best_distance < 0.05
        ):
            return detected_student, all_students, len(encodings)

    detected_student[best_student_id] = True

    return detected_student, all_students, len(encodings)
