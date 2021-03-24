import sys

if not len(sys.argv) == 3:
    print('Improper Usage.')
    exit(1)

from cv2 import cv2
import face_recognition
import os
from PIL import Image, ImageDraw


image_name = sys.argv[1]
video_name = sys.argv[2]

# check if the provided files exist

if not os.path.exists(video_name):
    print("Video doesn't exist.")
    exit(1)
if not os.path.exists(image_name):
    print("Image doesn't exist.")
    exit(1)

# get face-encoding

face_image = face_recognition.load_image_file(image_name)
face_encodings = face_recognition.face_encodings(face_image)
face_encoding = None

if len(face_encodings) > 1:
    print("Multiple faces in the provided image.")
    exit(1)
elif len(face_encodings) == 0:
    print("No face found in the image.")
    exit(1)
else:
    face_encoding = face_encodings[0]


# load video 

vid_src = cv2.VideoCapture(video_name)
color = (0,0,255)

while (vid_src.isOpened()):
    face_locations = None
    ret, frame = vid_src.read()
    # change frame to RGB color mode
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    encodings = face_recognition.face_encodings(rgb_frame)
    matches = face_recognition.compare_faces(encodings,face_encoding)
    if True in matches:
        color = (0,255,0)
    face_locations = face_recognition.face_locations(rgb_frame)
    print(f'{len(face_locations)} faces found.')

    if not face_locations:
        continue

    # Draw rectangle on face
    for location in face_locations:
        top, right, bottom, left = location
        pil_image = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_image)
        draw.rectangle(((left,top),(right,bottom)),outline=color)
        #cv2.rectangle(frame,(top,left),(bottom,right),color,2) 
        #top, right, bottom, left = (None,None,None,None)
    
    pil_image.show()

    """cv2.imshow('Looking for match...',pil_image)
    if cv2.waitKey(1) == ord('q'):
       break"""