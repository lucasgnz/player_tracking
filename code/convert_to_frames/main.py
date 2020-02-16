import cv2
print(cv2.__version__)
vidcap = cv2.VideoCapture('../data/openpose/100.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  cv2.imwrite("../data/frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1