import face_recognition
import cv2

# Blurs the face locations within an image
def blur_image(image, face_locations):
    # Go through each face location and blur
    for top, right, bottom, left in face_locations:

        # Extract the region of the image that contains the face
        face_image = image[top:bottom, left:right]

        # Blur the face image proportionally to face size
        blur_size = round(min(bottom - top, right - left) / 5)
        face_image = cv2.blur(face_image, (blur_size, blur_size), 30)

        # Put the blurred face region back into the frame image
        image[top:bottom, left:right] = face_image


#image = face_recognition.load_image_file("obama.jpeg")
#image = face_recognition.load_image_file("team.png")
image = face_recognition.load_image_file("test.JPG")

# Initialize some variables
face_locations = []

# Find all the faces locations in the image
face_locations = face_recognition.face_locations(image)

blur_image(image, face_locations)

# CV2 uses BGR, so convert back to RGB for display
converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow('result', converted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


