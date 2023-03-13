import cv2
import mediapipe as mp
import time


class faceMesh():
    def __init__(self, mode=False, faces=1, ref_lms=False,
                 min_det_conf=0.5, min_track_conf=0.5):

        self.mode = mode
        self.faces = faces
        self.ref_lms = ref_lms
        self.min_det_conf = min_det_conf
        self.min_track_conf = min_track_conf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=1)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1,circle_radius=1)
        #detection is always more heavy than tracking, so we set a somewhat low min_detection conf

    def getFaceMesh(self,img,draw=True):
        #resizing the image
        scaleFactor = 100
        nw = int(img.shape[1]*scaleFactor/100)
        nh = int(img.shape[0]*scaleFactor/100)
        dim = (nw, nh)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.fm_results = self.faceMesh.process(imgRGB)

        if self.fm_results.multi_face_landmarks:
            faceLandmarks = []
            ih, iw, ic = img.shape
            for face_id,face_lms in enumerate(self.fm_results.multi_face_landmarks):
                if draw:
                    self.mpDraw.draw_landmarks(img, face_lms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                          self.drawSpec, self.drawSpec)
                    # iterating over all the landmarks in the landmarks list
                for id, lm in enumerate(face_lms.landmark):
                    x,y=int(lm.x*iw), int(lm.y*ih)
                    faceLandmarks.append([id,x,y])

        return img, faceLandmarks

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = faceMesh()
    while True:
        success, img = cap.read()
        img, lmList = detector.getFaceMesh(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {str(int(fps))}', (100, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()