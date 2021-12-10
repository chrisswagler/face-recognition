import face_recognition
import cv2
import cvzone
import math

#image = face_recognition.load_image_file("obama.jpeg")
image = face_recognition.load_image_file("team.png")

# Read in the sunglasses image
shades = cv2.imread("shades.png", cv2.IMREAD_UNCHANGED)
shades_height = shades.shape[0]
shades_width = shades.shape[1]

# Find all facial features in all the faces in the image
face_landmarks_list = face_recognition.face_landmarks(image)

# Apply sunglasses to all faces in the image
for face_landmarks in face_landmarks_list:
    # Link for definition of each point index:
        # https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png

        # Find angle of rotation based on eyebrow edges
        brow_deltax = face_landmarks['right_eyebrow'][4][0] - face_landmarks['left_eyebrow'][0][0]
        brow_deltay = face_landmarks['right_eyebrow'][4][1] - face_landmarks['left_eyebrow'][0][1]
        brow_angle_r = math.atan(brow_deltay / brow_deltax)
        brow_width = math.sqrt((brow_deltax * brow_deltax) + (brow_deltay * brow_deltay))
        
        # Scale the width and height based on eyebrow size
        scale_factor = 1.2
        new_shades_width = round(brow_width * scale_factor)
        new_shades_height = round(shades_height * (new_shades_width / shades_width) * scale_factor)
        resized_shades = cv2.resize(shades, (new_shades_width, new_shades_height))

        # Rotate the shades based on eyebrow angle
        resized_shades = cvzone.rotateImage(resized_shades, -(math.degrees(brow_angle_r)))

        # Calculate offset of where exactly the glasses begin
        # Find the change in x y coordinates of where top left of shades are relative to top left of image
        deltax = (new_shades_width - (new_shades_width * math.cos(brow_angle_r))) / 2
        deltay = (new_shades_width * math.sin(brow_angle_r)) / 2

        # Find the displacement between eyebrow and eye to adjust glasses
        eye_eyebrow_x = face_landmarks['left_eyebrow'][2][0] - face_landmarks['left_eye'][1][0]
        eye_eyebrow_y = face_landmarks['left_eyebrow'][2][1] - face_landmarks['left_eye'][1][1]
        eye_eyebrow_dist = math.sqrt((eye_eyebrow_x * eye_eyebrow_x) + (eye_eyebrow_y * eye_eyebrow_y))
        s_offsetx = deltax + 0.5 * eye_eyebrow_dist
        s_offsety = (new_shades_height / 2) - deltay - eye_eyebrow_dist

        # Apply the offset relative to corner of left eyebrow
        tot_offset = (round(face_landmarks['left_eyebrow'][0][0] - s_offsetx),
                      round(face_landmarks['left_eyebrow'][0][1] - s_offsety))
        
        # Overlay the rotated and translated glasses
        image = cvzone.overlayPNG(image, resized_shades, tot_offset)

# CV2 uses BGR, so convert back to RGB for display
converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow('result', converted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


