import cv2
import mediapipe as mp
import time



class handDetector():
    def __init__(self,mode = False, maxHands = 2, detectionCon =0.5,trackCon = 0.5):
        
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]
        
        
    def findHands(self,img,draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.res = self.hands.process(imgRGB)
        
        if self.res.multi_hand_landmarks:
            for handLms in self.res.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
                    
        return img
    
    
    def findPositions(self,img,handNo = 0, draw = True):
        self.lmList = []
        
        if self.res.multi_hand_landmarks:
            myHand = self.res.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                self.lmList.append([id,cx,cy])
                
                if draw:
                    cv2.circle(img,(cx,cy),7,(222,100,22),cv2.FILLED)
                    
        return self.lmList
    
    def fingersUp(self):
        fingers = []
        
        #thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        
        else:
            fingers.append(0)
            
            
        #for other fingers
        for id in range(1,5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
                
            else:
                fingers.append(0)
                
        return fingers



def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    
    
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmlist = detector.findPostions(img)
        if len(lmlist) !=0:
            print(lmlist[4])
        
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        
        cv2.putText(img,str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 3 ,(0,0,222),3)
        cv2.imshow("IMAGE",img)
        if cv2.waitKey(1) == 13:
            break
            
if __name__ == "__main__":
    main()