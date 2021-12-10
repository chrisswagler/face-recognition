# Import the opencv and face recognition library
import cv2
import face_recognition
import cvzone
import math

def distance(num1, num2):
    return math.sqrt((num1 * num1) + (num2 * num2))

# Read in the sunglasses image
shades = cv2.imread("shades.png", cv2.IMREAD_UNCHANGED)
shades_height = shades.shape[0]
shades_width = shades.shape[1]

# Define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    ret, frame = vid.read()

    # Resize frame of video to 1/4 size for faster face detection processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(small_frame)

    # Apply sunglasses to all faces in the image
    for face_landmarks in face_landmarks_list:
        # Link for definition of each point index:
        # https://cdn-images-1.medium.com/max/1600/1*AbEg31EgkbXSQehuNJBlWg.png

        # Find angle of rotation based on eyebrow edges
        brow_deltax = 4*(face_landmarks['right_eyebrow'][4][0] - face_landmarks['left_eyebrow'][0][0])
        brow_deltay = 4*(face_landmarks['right_eyebrow'][4][1] - face_landmarks['left_eyebrow'][0][1])
        brow_angle_r = math.atan(brow_deltay / brow_deltax)
        brow_width = distance(brow_deltax, brow_deltay)
        
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

        # Find the displacement between center of glasses to top of nose
        eye_eyebrow_x = 4*(face_landmarks['left_eyebrow'][2][0] - face_landmarks['left_eye'][1][0])
        eye_eyebrow_y = 4*(face_landmarks['left_eyebrow'][2][1] - face_landmarks['left_eye'][1][1])
        eye_eyebrow_dist = distance(eye_eyebrow_x, eye_eyebrow_y)
        s_offsetx = deltax + 0.5 * eye_eyebrow_dist
        s_offsety = (new_shades_height / 2) - deltay - eye_eyebrow_dist

        # Apply the offset relative to corner of left eyebrow
        tot_offset = (round(4*face_landmarks['left_eyebrow'][0][0] - s_offsetx),
                      round(4*face_landmarks['left_eyebrow'][0][1] - s_offsety))
        
        # Overlay the rotated and translated glasses
        frame = cvzone.overlayPNG(frame, resized_shades, tot_offset)

  
    # Display the resulting frame
    cv2.imshow('nice shades', frame)
      
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

