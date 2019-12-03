import cv2
from configparser import ConfigParser

config = ConfigParser()
config.read("config.ini")	

LIVE_WEBCAM = False
if LIVE_WEBCAM:
    video_capture = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # test it works
        cv2.imshow('salut johnule', frame)

        # predict and plot for the image `frame`
        # the plot function has to be called with `video=True`  

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    video_capture = cv2.VideoCapture(config["DATASET"]["video"])
    while not video_capture.isOpened():
        video_capture = cv2.VideoCapture(config["DATASET"]["video"])
        cv2.waitKey(100)

    frame_index = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        if ret:
            # The frame is ready and already captured

            # test it works
            cv2.imshow('sal din nou jhonule', frame)

            # predict and plot for the image `frame`
            # the plot function has to be called with `video=True`
            
            frame_index = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            # The next frame is not ready, so we try to read it again
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
            cv2.waitKey(100)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if video_capture.get(cv2.CAP_PROP_POS_FRAMES) == video_capture.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames, stop
            break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
