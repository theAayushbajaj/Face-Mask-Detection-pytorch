""" Face detection using neural network
"""
from pathlib import Path

import numpy as np
from cv2 import resize
from cv2.dnn import blobFromImage, readNetFromCaffe, readNetFromTorch
import pickle

class FaceRecognizerException(Exception):
    """ generic default exception
    """


class FaceRecognizer:
    """ Face Detector class
    """
    def __init__(self, prototype: Path=None, model: Path=None,
                embedder: Path=None, labelencoder: Path=None, recognizer: Path=None,
                 confidenceThreshold: float=0.8):
        self.prototype = prototype
        self.model = model
        self.confidenceThreshold = confidenceThreshold
        if self.prototype is None:
            raise FaceRecognizerException("must specify prototype '.prototxt.txt' file "
                                        "path")
        if self.model is None:
            raise FaceRecognizerException("must specify model '.caffemodel' file path")
        self.classifier = readNetFromCaffe(str(prototype), str(model))

        if embedder is None:
            raise FaceRecognizerException("must specify Face Embedding '.t7' file "
                                        "path")

        if recognizer is None:
            raise FaceRecognizerException("must specify face recognizor")

        if labelencoder is None:
            raise FaceRecognizerException("must specify Classifier")

        self.embedder = readNetFromTorch(str(embedder))
        # load the actual face recognition model along with the label encoder
        self.recognizer = pickle.loads(open(str(recognizer), "rb").read())
        self.le = pickle.loads(open(str(labelencoder), "rb").read())
    
    def detect_faces(self, image):
        """ detect faces in image
        """
        net = self.classifier
        height, width = image.shape[:2]
        blob = blobFromImage(resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidenceThreshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            startX, startY, endX, endY = box.astype("int")
            faces.append(np.array([startX, startY, endX-startX, endY-startY]))
        return faces

    def recognize_face(self, face):
        # load the serialized face embedding model from disk
        # construct a blob for the face ROI, then pass the blob
        # through our face embedding model to obtain the 128-d
        # quantification of the face
        faceBlob = blobFromImage(face, 1.0 / 255, (96, 96),
            (0, 0, 0), swapRB=True, crop=False)
        self.embedder.setInput(faceBlob)
        vector = self.embedder.forward()

        # perform classification to recognize the face
        face_recognizer_preds = self.recognizer.predict_proba(vector)[0]
        proba = face_recognizer_preds[np.argmax(face_recognizer_preds)]
        name = self.le.classes_[np.argmax(face_recognizer_preds)] if proba > self.confidenceThreshold else 'Unknown'
        return name

        
