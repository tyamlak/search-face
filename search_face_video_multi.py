import argparse
import os


def file_exists(name):
    if not os.path.exists(name):
        raise argparse.ArgumentTypeError(f"{name} doesn't exist.")
    return name

parser = argparse.ArgumentParser(description='Search face in a Video.')
parser.add_argument("image_name",help="path of image with a single face.",type=file_exists)
parser.add_argument("video_name",help="path of video to search the provided face.",type=file_exists)
parser.add_argument("--process",help="Number of process,default is 2",type=int,default=2)

args = parser.parse_args()


from multiprocessing import Queue, Process, Value
from cv2 import cv2
from PIL import Image, ImageDraw
import face_recognition

image = face_recognition.load_image_file(args.image_name)
video = cv2.VideoCapture(args.video_name)
frames = Queue()
matched_frames = Queue()
no_of_frames = Value('i')
processed_frames = Value('i')
no_of_process = args.process

face_encoding = face_recognition.face_encodings(image)[0]

def display_frame(frame,face_locations=None):
    image = Image.fromarray(frame)
    if face_locations is None:   
        draw = ImageDraw.Draw(image)
        for (top,right,bottom,left) in face_locations:
            draw.rectangle(((left,top),(right,bottom)), outline=(0,255,0))
        del draw
    image.show()

def load_frames(frames:Queue,no_of_frames):
    print("Started loading frames.")
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            # change to rgb color model
            rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frames.put(rgb_frame)
            no_of_frames.value += 1
        else:
            print(f'Finished loading {no_of_frames.value} frames.')
            return

def match_frames(frames:Queue,matched_frames:Queue,face_encoding,pf):
    while True:
        try:
            frame = frames.get_nowait()
        except:
            break
        face_locations = face_recognition.face_locations(frame)
        if len(face_locations):
            encodings = face_recognition.face_encodings(frame,face_locations)
            matches = face_recognition.compare_faces(encodings,face_encoding)
            if True in matches:
                matched_frames.put(frame)
                display_frame(frame,face_locations)
        pf.value += 1
    print(f"Process {os.getpid()} exited.")
        

if __name__ == '__main__':
    load_frames(frames,no_of_frames)
    import time
    process_list = []
    start = time.time()
    for i in range(no_of_process):
        print(f'Starting process {i}')
        _ = Process(target=match_frames,args=(frames,matched_frames,face_encoding,processed_frames))
        _.start()
        process_list.append(_)
    while True:
        print(f'Currently processed frames...{processed_frames.value}')
        time.sleep(3)
        if 75 <= processed_frames.value:
            break
    for px in process_list:
        px.terminate()
    end = time.time()
    print(f'There are {matched_frames.qsize()} matched frames.')
    for i in range(matched_frames.qsize()):
        try:
            frame = matched_frames.get_nowait()
        except:
            print('Failed to get frame.')
            continue
        display_frame(frame)
    print(f'It takes {end-start} seconds.')