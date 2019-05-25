import cv2
import cv2.aruco
import numpy as np

# NOTE: Generate tags using http://chev.me/arucogen/

ARUCO_PARAMS = cv2.aruco.DetectorParameters_create()
ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)


def main():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # turn the autofocus off
    cam.set(3, 1280)  # set the Horizontal resolution
    cam.set(4, 720)  # Set the Vertical resolution

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            # This is an error
            break

        detect_and_draw_markers(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # crop_and_rotate_to_env(frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        scaled_frame = scale_and_reduce(frame)
        cv2.imshow('scaled_frame', scaled_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()


def detect_and_draw_markers(frame):
    global ARUCO_DICT
    global ARUCO_PARAMS
    corners, ids, _ = cv2.aruco.detectMarkers(
        frame, ARUCO_DICT, parameters=ARUCO_PARAMS)
    if ids is not None:
        marker_frame = cv2.aruco.drawDetectedMarkers(
            frame, corners, borderColor=(0, 0, 255))
        cv2.imshow('marker_frame', marker_frame)


def scale_and_reduce(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray_frame, (64, 64))


def crop_and_rotate_to_env(frame):
    # TODO: Get this working...
    # I'm now not confident that this will be able to find the right box.
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # extreme black filter
    _, thresh_gray_frame = cv2.threshold(gray_frame, 50, 255, 0)
    #thresh_gray_frame = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(
        thresh_gray_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(contours[0])

    cv2.imshow('thresh_gray_frame', thresh_gray_frame)

    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cnt_frame = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    cv2.imshow('cnt_frame', cnt_frame)

    # # rotate img
    # angle = rect[2]
    # rows, cols = frame.shape[0], frame.shape[1]
    # M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    # rot_frame = cv2.warpAffine(frame, M, (cols, rows))

    # # rotate bounding box
    # rect0 = (rect[0], rect[1], 0.0)
    # box = cv2.boxPoints(rect0)
    # pts = np.int0(cv2.transform(np.array([box]), M))[0]
    # pts[pts < 0] = 0

    # # crop
    # crop_rot_frame = rot_frame[pts[1][1]:pts[0][1],
    #                            pts[1][0]:pts[2][0]]

    # cv2.imshow('crop_rot_frame', crop_rot_frame)


if __name__ == "__main__":
    main()
