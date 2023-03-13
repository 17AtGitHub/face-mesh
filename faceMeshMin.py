import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('videos/simple.mp4')

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=3)
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=1)
#detection is always more heavy than tracking, so we set a somewhat low min_detection conf

pTime = 0
while True:
    success, img = cap.read()

    #resizing the image
    scaleFactor = 30
    nw = int(img.shape[1]*scaleFactor/100)
    nh = int(img.shape[0]*scaleFactor/100)
    dim = (nw, nh)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fm_results = faceMesh.process(imgRGB)

    if fm_results.multi_face_landmarks:
        for face_lms in fm_results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, face_lms, mpFaceMesh.FACEMESH_CONTOURS,
                                  drawSpec, drawSpec)
            ih,iw,ic = img.shape
            for id, lm in enumerate(face_lms.landmark):
                x,y=int(lm.x*iw), int(lm.y*ih)
                print(id,x,y)
                # print(lm)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {str(int(fps))}', (100,100), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
