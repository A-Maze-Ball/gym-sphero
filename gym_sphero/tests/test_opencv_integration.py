import cv2
import cv2.aruco
import numpy as np

# NOTE: Generate tags using http://chev.me/arucogen/

def main():
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off
    webcam.set(3, 1280) # set the Horizontal resolution
    webcam.set(4, 720) # Set the Vertical resolution

    aruco_params = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

    while webcam.isOpened():
        ret, frame = webcam.read()
        if not ret:
            # This is an error
            break

        # TODO: we probably want to grayscale this first.
        corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
        if ids is not None:
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, borderColor=(0, 0, 255))
        
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__": main()