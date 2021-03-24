import sys

if not len(sys.argv) == 3:
    print('Improper Usage.')
    exit(1)

import face_recognition
import os

image_name = sys.argv[1]
lookup_path = sys.argv[2]

if not os.path.exists(image_name):
    print("Image don't exist!")
    exit(1)
if not os.path.exists(lookup_path):
    print("Directory not found.")

face_encoding = None
IMAGE_EXTENSIONS = ('jpg','jpeg','png')
matches_found_in = []

image = face_recognition.load_image_file(image_name)
face_encodings = face_recognition.face_encodings(image)

if len(face_encodings) > 1:
    print("Multiple faces in the provided image.")
    exit(1)
elif len(face_encodings) == 0:
    print("No face found in the image.")
    exit(1)
else:
    face_encoding = face_encodings[0]


image_list = [x for x in os.listdir(lookup_path) if x[-3:] in IMAGE_EXTENSIONS]

for file in image_list:
    img = face_recognition.load_image_file(os.path.join(lookup_path,file))
    encodings = face_recognition.face_encodings(img)
    if len(encodings) == 0:
        print(f"No face found in {file}")
        continue
    print(f"Looking for a match in {file}")
    match = face_recognition.compare_faces(encodings,face_encoding)
    if True in match:
        matches_found_in.append(file)


if not matches_found_in:
    print("No matches found.")
    exit(0)

print("Matches found in:")
for file in matches_found_in:
    print(f"\t {file}")