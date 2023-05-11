import face_recognition
import numpy as np
from PIL import Image, ImageDraw
from IPython.display import display
import os
import sys 

known_face_encodings = []
known_face_names = []

print('Learning all the known faces')
for filename in os.listdir("images"):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join("images", filename)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]
        name = filename.split(".")[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)
print('found faces:', len(known_face_encodings))

# Load an image with an unknown face
print('Loading unknown face...')
file_name_unknown_person = sys.argv[1]
unknown_image = face_recognition.load_image_file(file_name_unknown_person)
#finds the location for the unknwon image. there can be more than one face as well
unknown_face_locations = face_recognition.face_locations(unknown_image)
#find the encoding for the unknonwn image
unknown_face_encodings = face_recognition.face_encodings(unknown_image, unknown_face_locations)
# Loop through each face to find a given unknwon image
for (top, right, bottom, left), face_en_comparing in zip(unknown_face_locations, unknown_face_encodings):
    # See if the face is a match for the known face(s) using the distance
    name = "Unknown"
    index=-1
    # This way we are sure about the person and is more accurate. It will provide the smallest index of images that it matches with
    faceDistances = face_recognition.face_distance(known_face_encodings, face_en_comparing)
    index = np.argmin(faceDistances)
    if index!=-1:
        name = known_face_names[index]
    # print the names
    print('In this picture the person is : ', name)
