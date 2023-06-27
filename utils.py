import cv2
import numpy as np


def get_faces(image_data):
    """
    This function takes in image data as a parameter and then uses OpenCV
    functions to identify faces in the image. Then it returns all those faces as
    separate images in a list.

    Parameters:
    image_data - Image data in bytes format.
    """
    # Convert the image data to a numpy array
    nparr = np.frombuffer(image_data, np.uint8)

    # Decode the image data
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the image from BGR to RGB color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the face cascade classifier
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    found_faces = []
    print("Found {0} Faces!".format(len(faces)))

    for (x, y, w, h) in faces:
        # Extract the face region of interest from the image
        roi_color = image[y:y + h, x:x + w]

        # Add the face to the list of found faces
        found_faces.append(roi_color)

    return found_faces
