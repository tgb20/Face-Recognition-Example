import face_recognition
import cv2
import numpy as np

# Open Default Video Camera
cap = cv2.VideoCapture(0)
# Set width to 640
cap.set(3, 640)
# Set height to 480
cap.set(4, 480)

# We need to append our custom face to the face_recongition model
# Load image of persons face
jacob_image = face_recognition.load_image_file("faces/Jacob.jpg")
# Encode the image for face_recognition
jacob_encoding = face_recognition.face_encodings(jacob_image)[0]

# List of all possible face encodings
known_face_encodings = [ 
    jacob_encoding
]

# List of all possible names (should be indexed the same as known_face_encodings)
known_face_names = [
    "Jacob"
]

# Temp variables to store positions and names
face_locations = []
face_encodings = []
face_names = []

# We are only going to scan every other image
process_this_img = True

while True:
    # Grab the image from the camera
    _, img = cap.read()

    # Flip it so it feels more natural, 1 is horizontal, 0 would be vertical
    img = cv2.flip(img, 0)

    # Running the face model on a large image is taxing to the computer, so we will scale it down for processing
    small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    # OpenCV uses BGR, face_recognition uses RGB, the image is a matrix, so we can invert the color positions like this BGR -> RGB
    rgb_img = small_img[:, :, ::-1]

    # If this is a frame we want to process
    if process_this_img:
        # It doesn't make sense to run the model on the entire image
        # So first we grab the faces in the image
        face_locations = face_recognition.face_locations(rgb_img)
        # Then we run the model on just the faces
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        # Temp list for face names
        face_names = []

        # Once we have all the encodings of the faces in our image we need to compare them to our custom faces
        for face_encoding in face_encodings:
            # We first need to compare each face found in the image to our custom image, this will give us a distance between faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            # If we don't know their name we will leave it as unknown
            name = "Unknown"

            # For each face we need to find the distance it is to our custom faces
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            # We will then find the face that is the closest to one of our custom faces (min)
            best_match_index = np.argmin(face_distance)
            # If we have a face for that index then...
            if matches[best_match_index]:
                # This is the name of that person
                name = known_face_names[best_match_index]
            # And we can add their name to our list of names
            face_names.append(name)

    # This inverts the process_this_img variable, so we only run our face matching every other frame
    process_this_img = not process_this_img

    # Now we need to draw out the faces on the screen
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # We made our processed image 1/4 of the original capture, so when we draw squares around faces they need to be 4x larger
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # This draws a red rectangle around each face (BGR)
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

        # And we can draw a name tag with the name of the person we found
        cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # This will show the completed image with the rectangles around faces
    cv2.imshow('Faces', img)

    # How you quit out of a simple OpenCV program (wait for Q to be pressed)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
