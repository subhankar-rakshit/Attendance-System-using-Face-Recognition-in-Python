import face_recognition as fr
from threading import Thread
import numpy as np
import time
import os
import cv2

exclude_names = ['Unknown', 'HOD', 'Principal']

class VideoStream:
    def __init__(self, stream):
        self.video = cv2.VideoCapture(stream)
        # Setting the FPS for the video stream
        self.video.set(cv2.CAP_PROP_FPS, 60)

        if self.video.isOpened() is False:
            print("Can't accessing the webcam stream.")
            exit(0)

        self.grabbed , self.frame = self.video.read()

        self.stopped = True
        
        self.thread = Thread(target=self.update)
        self.thread.daemon = True
    
    def start(self):
        self.stopped = False
        self.thread.start()

    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.video.read()

        self.video.release()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

        
def encode_faces():
    encoded_data = {}

    for dirpath, dnames, fnames in os.walk("./Images"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("Images/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded_data[f.split(".")[0]] = encoding

    # return encoded data of images
    return encoded_data

def Attendance(name):
    # It will get the current date
    today = time.strftime('%d_%m_%Y')
    # To create a file if it doesn't exists
    f = open(f'Records/record_{today}.csv', 'a')
    f.close()

    # It will read the CSV file and check if the name
    # is already present there or not.
    # If the name doesn't exist there, it'll be added
    # to a list called 'names'
    with open(f'Records/record_{today}.csv', 'r') as f:
        data = f.readlines()
        names = []
        for line in data:
            entry = line.split(',')
            names.append(entry[0])

    # It will check it the name is in the list 'names'
    # or not. If not then, the name will be added to
    # the CSV file along with the entering time
    with open(f'Records/record_{today}.csv', 'a') as fs:
        if name not in names:
            current_time = time.strftime('%H:%M:%S')
            if name not in exclude_names:
                fs.write(f"\n{name}, {current_time}")


if __name__ == "__main__":
    faces = encode_faces()
    encoded_faces = list(faces.values())
    faces_name = list(faces.keys())
    video_frame = True

    # Initialize and start multi-thread video input
    # stream from the WebCam.
    # 0 refers to the default WebCam
    video_stream = VideoStream(stream=0)
    video_stream.start()

    while True:
        if video_stream.stopped is True:
            break
        else :
            frame = video_stream.read()

            if video_frame:
                face_locations = fr.face_locations(frame)
                unknown_face_encodings = fr.face_encodings(frame, \
                face_locations)

                face_names = []
                for face_encoding in unknown_face_encodings:
                    # Comapring the faces
                    matches = fr.compare_faces(encoded_faces, \
                    face_encoding)
                    name = "Unknown"

                    face_distances = fr.face_distance(encoded_faces,\
                    face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = faces_name[best_match_index]

                    face_names.append(name)

            video_frame = not video_frame

            for (top, right, bottom, left), name in zip(face_locations,\
            face_names):
                # Draw a rectangular box around the face
                cv2.rectangle(frame, (left-20, top-20), (right+20, \
                bottom+20), (0, 255, 0), 2)
                # Draw a Label for showing the name of the person
                cv2.rectangle(frame, (left-20, bottom -15), \
                (right+20, bottom+20), (0, 255, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                # Showing the name of the detected person through 
                # the WebCam
                cv2.putText(frame, name, (left -20, bottom + 15), \
                font, 0.85, (255, 255, 255), 2)
                
                # Call the function for attendance
                Attendance(name)

        # delay for processing a frame 
        delay = 0.04
        time.sleep(delay)

        cv2.imshow('frame' , frame)
        key = cv2.waitKey(1)
        # Press 'q' for stop the executing of the program
        if key == ord('q'):
            break

    video_stream.stop()

    # closing all windows 
    cv2.destroyAllWindows()
