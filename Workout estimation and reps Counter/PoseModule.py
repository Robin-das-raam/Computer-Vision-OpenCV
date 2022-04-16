import cv2
import mediapipe as mp
import time
import math

class PoseDetector():
    
    def __init__(self,mode = False,complexity = 1,smooth = True,segmentation = False,
                 smooth_segmentation = True, detectionCon = 0.5, trackCon = 0.5):
        
        self.mode = mode
        self.smooth = smooth
        self.complexity = complexity
        self.segmentation = segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.smooth,self.complexity,
                                     self.segmentation,self.smooth_segmentation,self.detectionCon,self.trackCon)
        
        
    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
                
        return img
    
    def getPosition(self, img, draw = True):
        
        self.lmList = []
        
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                
                h, w , c = img.shape
                cx,cy = int(lm.x * w) , int(lm.y * h)
                self.lmList.append([id,cx,cy])
                
                if draw:
                    cv2.circle(img, (cx,cy), 5,(255,0,0),cv2.FILLED)
        
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw = True):
        #get the landmark
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        
        #calculate the angle
        angle = math.degrees((math.atan2(y3-y2, x3-x2))- (math.atan2(y1-y2, x1-x2)))
        
        if angle <0:
            angle += 360
        
        print(angle)
        
        if draw:
            cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
            cv2.line(img,(x3,y3),(x2,y2),(255,255,255),3)
            cv2.circle(img,(x1,y1),10,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x1,y1),10,(0,0,255),2)
            cv2.circle(img,(x2,y2),10,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x2,y2),10,(0,0,255),2)
            cv2.circle(img,(x3,y3),10,(0,0,255),cv2.FILLED)
            cv2.circle(img,(x3,y3),10,(0,0,255),2)
            cv2.putText(img, str(int(angle)), (x2-60, y2+60),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
            
        
        return angle
            
    
    
    
def main():
    cap = cv2.VideoCapture("KneeBendVideo.mp4")
    pTime = 0
    detector = PoseDetector(0.6)
    
    while True:
        success, img = cap.read()
        img = detector.findPose(img,draw=False)
        lmList = detector.getPosition(img,draw=False)
        if len(lmList) != 0:
            detector.findAngle(img,11,13,15)
            detector.findAngle(img,12,14,16)
        #angle = detector.findAngle(img,11,13,15)
        #print(lmList[14])
        
#         cv2.circle(img, (lmList[13][1],lmList[13][2]),10,(0,0,250),cv2.FILLED)
#         cv2.circle(img, (lmList[14][1],lmList[14][2]),10,(0,0,250),cv2.FILLED)
        
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3 , (0,0,222),4)
    
        cv2.imshow("input", img)
        if cv2.waitKey(1) == 13:
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()