import face_recognition
import cv2

#image = face_recognition.load_image_file("obama.jpeg")
image = face_recognition.load_image_file("team.png")

# Read in the sunglasses image
shades = cv2.imread("shades.png")
shades_height = shades.shape[0]
shades_width = shades.shape[1]

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

# Apply sunglasses to all faces in the image
for face_landmarks in face_landmarks_list:
    # Scale the width and height based on eyebrow size
    new_shades_width = face_landmarks['right_eyebrow'][3][0] - face_landmarks['left_eyebrow'][0][0]
    new_shades_height = round(shades_height * (new_shades_width / shades_width))
    resized_shades = cv2.resize(shades, (new_shades_width, new_shades_height))
    # Place the sunglasses starting at the left eyebrow
    image[face_landmarks['left_eyebrow'][0][1]:face_landmarks['left_eyebrow'][0][1]+new_shades_height,
          face_landmarks['left_eyebrow'][0][0]:face_landmarks['left_eyebrow'][0][0]+new_shades_width] = resized_shades


# CV2 uses BGR, so convert back to RGB for display
converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow('result', converted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


