# Usage
# python video.py --model models/v1_torch.pt
# 

from pathlib import Path

import argparse
from threading import Thread
import cv2
import torch
from skvideo.io import FFmpegWriter
from torchvision import transforms

from facerecognizer import FaceRecognizer
from train import TransferLearningModel
import datetime

# NOTE: FPS obtained on:
# Powersave ~ 9fps
# Performance ~ 11fps
class FPS:
	def __init__(self):
		# store the start time, end time, and total number of frames
		# that were examined between the start and end intervals
		self._start = None
		self._end = None
		self._numFrames = 0

	def start(self):
		# start the timer
		self._start = datetime.datetime.now()
		return self

	def stop(self):
		# stop the timer
		self._end = datetime.datetime.now()

	def update(self):
		# increment the total number of frames examined during the
		# start and end intervals
		self._numFrames += 1

	def elapsed(self):
		# return the total number of seconds between the start and
		# end interval
		return (self._end - self._start).total_seconds()

	def fps(self):
		# compute the (approximate) frames per second
		return self._numFrames / self.elapsed()

class WebcamVideoStream:
	def __init__(self, src=0, name="WebcamVideoStream"):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()

		# initialize the thread name
		self.name = name

		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		t = Thread(target=self.update, name=self.name, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True

@torch.no_grad()
def tagVideo(modelpath, outputPath=None) -> None:
	""" detect if persons in video are wearing masks or not
	"""
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	model = TransferLearningModel.resnet18()
	model.load_state_dict(torch.load(modelpath, map_location=device)['model_state_dict'], strict=False)
	
	model = model.to(device)
	model.eval()
	
	faceRecognizer = FaceRecognizer(
		prototype='face-recognition/face_models/deploy.prototxt.txt',
		model='face-recognition/face_models/res10_300x300_ssd_iter_140000.caffemodel',
		embedder=args['embedding_model'],
		labelencoder=args['label_encoder'],
		recognizer=args['recognizer']
	)

	
	transformations = transforms.Compose([
			transforms.ToPILImage(),
			transforms.Resize(224),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])

	if outputPath:
		writer = FFmpegWriter(str(outputPath))
	
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.namedWindow('main', cv2.WINDOW_NORMAL)
	labels = ['No mask', 'Mask']
	labelColor = [(0, 0, 255), (0, 255, 0)]

	vs = WebcamVideoStream(src=1).start()
	fps = FPS().start()
	process_this_frame = True
	start_time = datetime.datetime.now()

	while True:
		frame = vs.read()
		#small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
		#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		faces = faceRecognizer.detect_faces(frame)
		for face in faces:
			xStart, yStart, width, height = face
			
			# clamp coordinates that are outside of the image
			xStart, yStart = max(xStart, 0), max(yStart, 0)
			
			# predict mask label on extracted face
			faceImg = frame[yStart:yStart+height, xStart:xStart+width]

			if process_this_frame:
				outputs = model(transformations(faceImg).unsqueeze(0).to(device))
				predicted = torch.sigmoid(outputs)
				predicted = predicted>0.5
			
			process_this_frame = not process_this_frame

			name = faceRecognizer.recognize_face(faceImg) if predicted==0 else 'Unknown'
			
			# draw face frame
			cv2.rectangle(frame,
						  (xStart, yStart),
						  (xStart + width, yStart + height),
						  labelColor[predicted],
						  thickness=2)
			
			# center text according to the face frame
			textSize = cv2.getTextSize(labels[predicted], font, 1, 2)[0]
			textX = xStart + width // 2 - textSize[0] // 2
			
			# draw prediction label
			cv2.putText(frame,
						'{}, {}'.format(labels[predicted],name),
						(textX, yStart-20),
						font, 1, labelColor[predicted], 2)
		if outputPath:
			writer.writeFrame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
		cv2.imshow('main', frame)
		fps.update()
		if cv2.waitKey(1) & 0xFF == ord('q') or (datetime.datetime.now() - start_time).total_seconds()>900.0:
			break
	if outputPath:
		writer.close()
	
	fps.stop()
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	cv2.destroyAllWindows()

# pylint: disable=no-value-for-parameter
if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--face", type=str,
		default="face_detector",
		help="path to face detector model directory")
	ap.add_argument("-m", "--model", type=str,
		help="path to trained face mask detector model")
	ap.add_argument("-em", "--embedding-model",
		default='face-recognition/face_models/openface_nn4.small2.v1.t7',
		help="path to OpenCV's deep learning face embedding model")
	ap.add_argument("-r", "--recognizer",
		default='face-recognition/output/recognizer.pickle',
		help="path to model trained to recognize faces")
	ap.add_argument("-l", "--label_encoder",
		default='face-recognition/output/le.pickle',
		help="path to label encoder")
	ap.add_argument("-o", "--output", type=str,
		help="path to video output")	
	ap.add_argument("-c", "--confidence", type=float, default=0.7,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	tagVideo(modelpath=args['model'])
