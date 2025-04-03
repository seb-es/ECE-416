import cv2
import mediapipe as mp
import time
import math as math


class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):     
            # custom constructor made for objects of the HandTrackingDynamic class. The four parameters are attributes which are set to defaults as shown within the parantheses. 
        self.__mode__   =  mode                                                     
        self.__maxHands__   =  maxHands                                             
        self.__detectionCon__   =   detectionCon
        self.__trackCon__   =   trackCon
            # these four attributes are created either by the defaults assigned to the parameters above or by manual assignment.
            # the fact that the self.(attribute) isn't exactly the same as the atrribute name (given the underscores) tells us that these are just ways to access the attributes. 
       
        self.tipIds = [4, 8, 12, 16, 20]
            # this attribute serves to tell us which landmarks (out of the 21) are the finger tips. 
                                                        
        self.handsMp = mp.solutions.hands
                # Assigns the hand detection AI model from mediapipe wiht given confidence paramters to attribute handsMp                                         
        self.hands = self.handsMp.Hands()
        self.mpDraw= mp.solutions.drawing_utils
            # these three come from the mediapipe library mostly. As a reminder, the mediapipe library is a pre-trained computer vision AI model. 

    # A quick note about the coordinate system: (0,0) is intially located at the TOP RIGHT of the screen and the x and y values respectively get higher as go to the left and down. 
    # Not sure why the coordinate system is like this, it just is. 
    # The flip function applied to the frame in the drawHandLandmarks method below flips the x axis. 
    # For consistenty and convenience, we will also flip the y axis below to make (0,0 the bottom left)

    def processAndCorrectView(self, frame): 

        frame = cv2.flip(frame, 1)
            #flips frame to match user's hands.

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
            # Changes color scheme, runs the model on the image.
            # Reasoning for this color switch is that mediapipe uses RGB while cv2 operates on BGR, so you need to change it.
        
        return frame
    
    def drawHandLandmarks(self, frame, draw=True):  
        
        if self.results.multi_hand_landmarks: 
            # if there is a hand on screen, then...
            for handLms in self.results.multi_hand_landmarks:
                # for each each set of landmark lists corresponding to each detected hand...
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms,self.handsMp.HAND_CONNECTIONS)
                    # draw the red dots at each knuckle/wrist detected and interconnecting white lines.

        return frame

    def findAndMark_Positions( self, frame, handNo=0, draw=True):
        xList = []
        yList = []
        zList = []
        bbox =  []
        self.lmsList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
                # because the handNo parameter is set to 0 by default, this variable refers to ONLY the first hand detected
            for id, lm in enumerate(myHand.landmark):
                # for each landmark on the detected hand...
                h, w, c = frame.shape
                    # the .shape method from the mediapipe library assigns the height and width of the screen, in pixels. c is color channel. 
                
                z = lm.z
                zScaled = z * 10 * -1
                #Scales the Z value so that it falls roughly within the 0-1 range as opposed to whatever arbitary value it uses now. 
                # If these values continue to be unpredicatable and you keep needing to adjust coefficients, use some pre-built mapping function. 
                
                def clamp(z, zMin, zMax):
                    return max(min(zMax, z), zMin)
                #Prevents z value from going outside of a set range, 0 to 1 in this case. 
                
                cz = (2 * w) - (clamp(zScaled, 0, 1) * (2 * w))
                #sets cz to pixel values by first ensuring the 0 - 1 range and then multiplying by twice the width of the screen, which is approximately the max distance from the screen the hand will relistically move away. 
                #Also makes it such that cz increases with distance from the camera instead of decreasing. 
                #Unlike cx and cy (below), cz is very much relative. Z will increase/decrease based on comparison to position of other landmarks, even if objective z location didnt change. 

                cx, cy = int(lm.x * w), int(lm.y * h)
                    # the raxnge of the coordinate values of each landmark gets converted into the possible range of pixel values instead of the arbitrary 0-1 range. 

                xList.append(cx)
                yList.append(cy)
                zList.append(cz)

                cyDraw = cy
                #Defines seperate y coordinates specifically for drawing because the flip applied below flips the orientation of drawings if used as is.

                cy = int((1 - lm.y) * h) 
                #Flips the y axis to go from bottom to top as described earlier, causing (0,0) to be at the bottom right.
                self.lmsList.append([id, cx, cy, cz, cyDraw])

                
                if draw:
                    # As shown in paramater declaration above, by default, draw = true. 
                    cv2.circle(frame,  (cx, cyDraw), 5, (255, 0, 255), cv2.FILLED)
                        # Draw purple circles on each landmark, overtop the normal red ones.
                
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
                #No need to flip values in xList and yList for rectangle, works as is. 

                #print( "Hands Keypoint")
                #print(bbox)
            if draw:
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 200 , 0), 2)
                # draw a green rectangle that is 20 pixels larger than the hand in all four directions. 
                

        return self.lmsList, bbox
    

    def drawMarkers(self, p1, p2, color, frame, r=5, t=3):

        if color == "red":
            lineColor = (74,26,200)
            dotColor = (37,13,100)
        elif color == "green": 
            lineColor = (50,200,50)
            dotColor = (25,120,25)
        elif color == "blue": 
            lineColor = (200,74,26)
            dotColor = (100,37,13)
        elif color == "magenta":
            lineColor = (200,50,200)
            dotColor = (100,25,100)
        elif color == "cyan":
            lineColor = (255,255,0)
            dotColor = (125,125,0)
        elif color == "yellow":
            lineColor = (0,255,255)
            dotColor = (0,125,125)

        x1, x2 = self.lmsList[p1][1], self.lmsList[p2][1]
            #Assigns the x coords of the first and second target landmart to x1 and x2, respectively.
        xMid = (x1+x2)//2  

        y1Draw, y2Draw = self.lmsList[p1][4], self.lmsList[p2][4]
        yMidDraw = (y1Draw + y2Draw)//2

        cv2.line(frame,(x1, y1Draw),(x2, y2Draw) ,(lineColor), t)
            # Draws line between target landmark in bright color.
        cv2.circle(frame,(x1, y1Draw),r,(dotColor),cv2.FILLED)
        cv2.circle(frame,(x2, y2Draw),r, (dotColor),cv2.FILLED)
        cv2.circle(frame,(xMid,yMidDraw ), int(r*0.65) ,(dotColor),cv2.FILLED)
            # Draws dark colored circles on each target landmark + midpoint. Made midpoint circle smaller. 
            # 74,26,200 is a bright rose red color in case you want to use that.

        return frame


    def defineDistanceAndOrientation(self, p1, p2):
         
        x1 , y1, z1 = self.lmsList[p1][1:4]
            #Assigns the x and y coords of the first target landmart to x1 and y1. 
        x2, y2, z2 = self.lmsList[p2][1:4]
            #Assigns the x and y coords of the second  target landmart to x2 and y2. 
        xDist, yDist, zDist = (x2-x1), (y2-y1), (z2-z1)
            #assigns the difference between x and y coords to xDist and yDist, respectively. 
        xMid , yMid, zMid = (x1+x2)//2 , (y1 + y2)//2, (z1 + z2)//2
            #Finds midpoint components. 
 
        absoluteDistXY = math.hypot(xDist,yDist)
        absoluteDistXZ = math.hypot(xDist,zDist)
        absoluteDistZY = math.hypot(zDist,yDist)
            #finds absolute value of distance between target landmarks.

        absoluteDist3D = math.hypot(xDist,yDist,zDist)

        angleXY = math.atan2(yDist, xDist)
        angleXZ = math.atan2(zDist, xDist)
        angleZY = math.atan2(yDist, zDist)
            # Finds angle (in radians) between yDist and xDist vectors using arctan. 

        horizontalnessXY = math.cos(angleXY)
        horizontalnessXZ = math.cos(angleXZ)
        horizontalnessZY = math.cos(angleZY)
            #Takes the cosine of this angle to determine an horizontal orientation coeffecient for the chosen section. 

        uprightnessXY = math.sin(angleXY)
        uprightnessXZ = math.sin(angleXZ)
        uprightnessZY = math.sin(angleZY)
            #Takes the sine of this angle to determine how upright the chosen section is. 
        
        #print(xDist, yDist, angle, uprightnessXY) 
        
        return  [absoluteDistXY, absoluteDistXZ, absoluteDistZY, absoluteDist3D], \
                [horizontalnessXY, horizontalnessXZ, horizontalnessZY], \
                [uprightnessXY,uprightnessXZ,uprightnessZY], \
                [x1, y1, z1, x2, y2, z2, xMid, yMid, zMid, xDist, yDist, zDist, angleXY, angleXZ, angleZY]
                    #Explicit line breaks used to condense return statement. 
                    #Might need to create a graphic for XY, XZ and ZY at some point. 

    def findOrientation(self):

        _, handHorizontalOrientation, _, _ = self.defineDistanceAndOrientation(self.tipIds[0] - 3, 0)
                # Measures the horizontalnessXY from the first landmark along the thumb to the wrist. 
        handHorizontalOrientationXY = handHorizontalOrientation[0]    

        if handHorizontalOrientationXY > 0: 
            thumbOnLeft = True
        else: 
            thumbOnLeft = False
            #This if statement solves the issue of not being able to distinguish between left and right hand being up as well as if a hand is flipped. 

        _, _, handVerticalOrientation, _ = self.defineDistanceAndOrientation(0, self.tipIds[2] - 3)
            # Measures the verticality from the wrist to the first knuckle of the middle finger. 
        handVerticalOrientationXY = handVerticalOrientation[0]

        if handVerticalOrientationXY > 0: 
            handIsUpright = True
        else: 
            handIsUpright = False

        return handIsUpright, thumbOnLeft, handVerticalOrientationXY, handHorizontalOrientationXY
    
    def markOrientation(self, frame): 

        frame = self.drawMarkers(0, self.tipIds[2] - 3, "green", frame)
            #Draw line from wrist to middle finger base. 
        frame = self.drawMarkers(0, self.tipIds[0] - 3, "green", frame)
            #Draw line from wrist to thumb base.

        return frame 
    

    def findRotation(self, frame): 
        pointerBaseKnuckleZ, pinkieBaseKnuckleZ, WristZ = self.lmsList[self.tipIds[1] - 3][3], self.lmsList[self.tipIds[4] - 3][3], self.lmsList[0][3]

        palmLength, _, _, _ = self.defineDistanceAndOrientation(self.tipIds[1] - 3, self.tipIds[4] - 3)
        palmLengthXY = palmLength[0]
            # Retrieves information about z coords of target knuckles as well as distance in between.
        _, thumbOnLeft, _, _ = self.findOrientation()
        frame = self.drawMarkers(self.tipIds[1] - 3, self.tipIds[4] - 3, "red", frame)

        bufferAndScalingFactor = 0.1
            #A value from 0-1 which determines how much the hand needs to be rotated from the starting position to activate rotation tracking and also how senstive the rotation is past this buffer point.
        
        if not hasattr(HandTrackingDynamic, 'maxPalmLength'):
            HandTrackingDynamic.maxPalmLength = palmLengthXY
            # Initialize class variable.
            # We use the HandTrackingDynamic. class reference to make sure the variable "sticks".
        if palmLengthXY > HandTrackingDynamic.maxPalmLength: 
            HandTrackingDynamic.maxPalmLength = palmLengthXY
                # If absolute distance ever grows larger than any point in the past during the current instance, maxPalmLength is updated.

        print(HandTrackingDynamic.maxPalmLength)
        unbufferedRotation = 1 - (palmLengthXY/HandTrackingDynamic.maxPalmLength)
            # This statement inherently makes it so that the neutral position is when the palm faces the camera. 
            # As the hand turns to the side in either direction, the distance will get smaller, increasing the unbuffered rotation value. 
            # By dividing by maxPalmLength, unsighnedRotion will never be over 1 

        if unbufferedRotation > bufferAndScalingFactor:
            unsignedRotation = ((unbufferedRotation - bufferAndScalingFactor)/(1 - bufferAndScalingFactor))
                #This kicks in when rotation actually starts to get counted and scales it such that its still in the 0-1 range. 
                # If the handrotation is higher than the buffer value, then the temporary value of unsignedRotation kicks in and is scaled by the buffer factor to make up for not kicking in until reaching buffer point.  
        else: 
            unsignedRotation= 0

        #print(round(pointerBaseKnuckleZ,4) , round(pinkieBaseKnuckleZ,4) , round(WristZ,4))

        if ((pointerBaseKnuckleZ > pinkieBaseKnuckleZ) and thumbOnLeft) or \
           ((pointerBaseKnuckleZ < pinkieBaseKnuckleZ) and not(thumbOnLeft)):
            rotation =  unsignedRotation * 1.5
        else: 
            rotation = unsignedRotation * -1 * 2.1

                    #If the hand rotates to the left, the rotation value will cause counterclockwise (positive) rotaiton. If not, counter-clockwise (negative) rotation will occur. 
                    #Statements have been made such that this behavior occurs regardless of hand being used. 
                    #Gave the negative hand movement an extra kick given how hard it is to move in that direction. 

        rotation = max(min(1, rotation), -1)
            #Clamps rotation to be in the 0 - 1 range. 
        rotation = round(rotation, 2)
            #Rounds off two 2 decimal points. 

        if rotation == 0:
            HandTrackingDynamic.maxPalmLength = palmLengthXY
            #Resets maxPalmLength to the current palm length if rotation is small enough. Prevents the maxPalmLength from getting too large. 
            #In other words, if the hand is not rotating, reset the maxPalmLength to the current palm length.

        return rotation, HandTrackingDynamic.maxPalmLength
    

    def findTilt(self, frame):
        wristZ, middleFingerSecondKnuckleZ = self.lmsList[0][3], self.lmsList[self.tipIds[2] - 2][3]
        wristToMiddleFingerSecondKnuckleDist, wristToMiddleFingerSecondKnuckleHortizontalness, _, _ = self.defineDistanceAndOrientation(0, 9)
            # Measures horizontalness from the wrist to the first landmark along the pointer finger.
            # Very similar approach to calculating rotation. 
            #SEE findRotation METHOD COMMENTS TO SEE EXPLANATION ON METHODS USED BELOW. 
        
        wTMFBKDist = wristToMiddleFingerSecondKnuckleDist
        wTMFBKHorizontalness = wristToMiddleFingerSecondKnuckleHortizontalness
        #Abbreviations cause normal names are massive
        
        wTMFBKDist_XY = wTMFBKDist[0]
        handIsUpright, _, _, _ = self.findOrientation()

        forwardBufferAndScalingFactor = 0.1
        
        if not hasattr(HandTrackingDynamic, 'max_wTMFBKDist'):
            HandTrackingDynamic.max_wTMFBKDist = wTMFBKDist_XY
            # Initialize class variable.
            # We use the HandTrackingDynamic. class reference to make sure the variable "sticks".
        if wTMFBKDist_XY > HandTrackingDynamic.max_wTMFBKDist: 
            HandTrackingDynamic.max_wTMFBKDist = wTMFBKDist_XY
                # If absolute distance ever grows larger than any point in the past during the current instance, maxPalmLength is updated.

        if handIsUpright:
            unbufferedForwardTilt =  1- (wTMFBKDist_XY/HandTrackingDynamic.max_wTMFBKDist)
                #If the hand is upright, forward tilt increases as the distance from wrist to middle finger base knuckle gets smaller.
        else:
            unbufferedForwardTilt = (wTMFBKDist_XY/HandTrackingDynamic.max_wTMFBKDist)
                #If its downwards, forward tilt continues getting higher the more and more the middle finger base knuckle passes the wrist. 

        if (unbufferedForwardTilt > forwardBufferAndScalingFactor) and handIsUpright:
            unsignedForwardTilt = ((unbufferedForwardTilt- forwardBufferAndScalingFactor)/(1 - forwardBufferAndScalingFactor))
                # This kicks in when tilt actually starts to get counted from the starting position only and scales it such that its still in the 0-1 range. 
        elif not(handIsUpright):
            unsignedForwardTilt = unbufferedForwardTilt * 1.75
                #Condition for when the hand is downward, when buffering isn't needed. 
        else: 
            unsignedForwardTilt= 0

        if middleFingerSecondKnuckleZ < wristZ:
            forwardTilt = unsignedForwardTilt
        else:
            forwardTilt = unsignedForwardTilt * -1 * 1.5
                #forwardTils is postive (forward) when middle finger base knuckle is in front of wrist. Otherwise, negative. 
                #Given a little kick to make up for the lack of backwards mobility in the hand from starting position. 

        forwardTilt = max(min(1, forwardTilt), -1)
            #Clamps forwardTilt to be in the 0 - 1 range. 
        forwardTilt = round(forwardTilt, 2)
            #Rounds off two 2 decimal points. 
        
        if forwardTilt == 0:
            HandTrackingDynamic.max_wTMFBKDist = wTMFBKDist_XY
            #Resets maxPalmLength to the current palm length if rotation is small enough. Prevents the maxPalmLength from getting too large. 
            #In other words, if the hand is not rotating, reset the maxPalmLength to the current palm length.

        unbufferedSidewaysTilt = wTMFBKHorizontalness[0]
            #Assigns rounded XY uprightness to sideways tilt. 
            #Unfortunately, this method doesn't work for the foward tilt, z coords are really shitty for whatever reason. Hence why we use the method similar to rotation above. 
        sidewaysTiltSign = unbufferedSidewaysTilt/abs(unbufferedSidewaysTilt)
            # The horizontalness attribute has sign built in, so we take it out and save it for later use. 

        sidewaysBufferAndScalingFactor= 0.45

        if (abs(unbufferedSidewaysTilt) > sidewaysBufferAndScalingFactor):
            sidewaysTilt = ((abs(unbufferedSidewaysTilt)- sidewaysBufferAndScalingFactor)/(1 - sidewaysBufferAndScalingFactor)) * sidewaysTiltSign
        else: 
            sidewaysTilt = 0

        if sidewaysTilt > 0:
            sidewaysTilt = sidewaysTilt * 2
            #Add an extra kick to compensate for it being harder to sideways tilt to the right. 
        
        sidewaysTilt = max(min(1, sidewaysTilt), -1)
            #Clamps sidewaysTilt to be in the 0 - 1 range. 
        sidewaysTilt = round(sidewaysTilt, 4)

        frame = self.drawMarkers(0, self.tipIds[2] - 2, "blue", frame)

        return forwardTilt, sidewaysTilt


    def findAndMarkCenterOfMass(self, frame):
        
        def avgDimension(targetList,targetDimension):
            targetValues = []
            sum = 0
            average = 0
            
            for i, _ in enumerate(targetList):
                if targetDimension == "x":
                    targetValues.append(targetList[i][1])
                elif targetDimension == "y":
                    targetValues.append(targetList[i][2])
                elif targetDimension == "z":
                    targetValues.append(targetList[i][3])

            for i, _ in enumerate(targetValues):
                sum += targetValues[i]
            
            average = int(sum/len(targetValues))
                
            return average
    
        lastIDofLmsList = len(self.lmsList) - 1
        nextLmsListIDAvailable = lastIDofLmsList + 1
        #Identifies the next id available in the lmsList.

        h, _, _ = frame.shape
        
        centerOfMassWithFingersX = avgDimension(self.lmsList, "x")
            #Average of x components for ALL landmarks.
        centerOfMassWithFingersY = avgDimension(self.lmsList, "y")
            #Average of Y components for ALL landmarks.
        centerOfMassWithFingersZ = avgDimension(self.lmsList, "z")
            #Average of z components for ALL landmarks.
        centerOfMassWithFingersYDraw = (h - centerOfMassWithFingersY)
        centerOfMassWithFingers = [nextLmsListIDAvailable, centerOfMassWithFingersX, centerOfMassWithFingersY, centerOfMassWithFingersZ, centerOfMassWithFingersYDraw]

        baseKnuckleList = self.lmsList[1:17:4]
        baseKnuckleList.append(self.lmsList[0])
        completeNoFingersList = baseKnuckleList

        centerOfMassNoFingersX = avgDimension(completeNoFingersList, "x")
            #Average of X components for all base knuckle landmarks and wrist.
        centerOfMassNoFingersY = avgDimension(completeNoFingersList, "y")
            #Average of Y components for all base knuckle landmarks and wrist.
        centerOfMassNoFingersZ = avgDimension(completeNoFingersList, "z")
            #Average of Z components for all base knuckle landmarks and wrist.
        centerOfMassNoFingersYDraw = (h - centerOfMassNoFingersY)
        centerOfMassNoFingers = [nextLmsListIDAvailable + 1, centerOfMassNoFingersX, centerOfMassNoFingersY, centerOfMassNoFingersZ, centerOfMassNoFingersYDraw]
        
        self.lmsList.append(centerOfMassWithFingers)
        self.lmsList.append(centerOfMassNoFingers)
        #Adds the centers of mass to the lmsList as landmarks with ID 21 and 22.

        frame = cv2.circle(frame, (centerOfMassWithFingersX, centerOfMassWithFingersYDraw), 5, (255,255,0), cv2.FILLED)
        frame = cv2.circle(frame, (centerOfMassNoFingersX, centerOfMassNoFingersYDraw), 5, (0,255,255), cv2.FILLED)
                #cy = int((1 - lm.y) * h) 
                #cyDraw = cy
                #Defines seperate y coordinates specifically for drawing because the flip applied below flips the orientation of drawings if used as is.

                
                #Flips the y axis to go from bottom to top as described earlier, causing (0,0) to be at the bottom right.
                #self.lmsList.append([id, cx, cy, cz, cyDraw])

        return centerOfMassWithFingers, centerOfMassNoFingers


    def findFingersOpen(self):
        fingers=[]
        handIsUpright, _, handVerticalOrientationXY, _ = self.findOrientation()
        wristZ, middleFingerBaseKnuckleZ = self.lmsList[0][3], self.lmsList[self.tipIds[2] - 3][3]
       

       #Thumb orientation. 
        if handIsUpright:
            centerOfMassttoThumbTipDistance, _ , _ , _ = self.defineDistanceAndOrientation(22, self.tipIds[0])
            centerOfMassttoThumbComparisonKnuckleDistance, _ , _ , _ = self.defineDistanceAndOrientation(22, self.tipIds[0] - 1)
                #When hand is upright, use center of mass with fingers as the point of comparison. 

        else: 
            centerOfMassttoThumbTipDistance, _ , _ , _ = self.defineDistanceAndOrientation(21, self.tipIds[0] - 1)
            centerOfMassttoThumbComparisonKnuckleDistance, _ , _ , _ = self.defineDistanceAndOrientation(21, self.tipIds[0] - 3)   
                #When the hand isn't upright, the detection works better when the point of comparison is center of mass without fingers. 

        centerOfMassttoThumbTipDistanceXY = abs(centerOfMassttoThumbTipDistance[0])
                #measures XY distance from center of mass to finger tip. 
        centerOfMassttoThumbComparisonKnuckleDistanceXY = abs(centerOfMassttoThumbComparisonKnuckleDistance[0])
                #measures XY distance from center of mass to landmark 1 step below thumb tip.
            
        if centerOfMassttoThumbComparisonKnuckleDistanceXY > centerOfMassttoThumbTipDistanceXY:
                #The moment the finger tip to center of mass distance is smaller than the center of mass to second knuckle distance, finger is closed. 
            fingers.append(0)
        else:
            fingers.append(1)


        #Finger orientation.
        for id, _ in enumerate(self.tipIds[1:5]):
            id += 1
                #accounts for count starting at 0 instead of 1. 
            _ , _ , FingerComparisonKnuckletoTipUprightness , _ = self.defineDistanceAndOrientation(self.tipIds[id] -2, self.tipIds[id])

            if (handIsUpright and middleFingerBaseKnuckleZ > wristZ) or not(handIsUpright):
                centerOfMassttoFingerTipDistance, _ , _ , _ = self.defineDistanceAndOrientation(22, self.tipIds[id])
                centerOfMassttoFingerComparisonKnuckleDistance, _ , _ , _ = self.defineDistanceAndOrientation(22, self.tipIds[id] - 2)
                    #When hand is upright and not forwared tilted OR downright, use center of mass (without fingers) VS. second knuckles as the point of comparison. 
            
                    #Determines extension based on uprightness. 
            else: 
                centerOfMassttoFingerTipDistance, _ , _ , _ = self.defineDistanceAndOrientation(21, self.tipIds[id])
                centerOfMassttoFingerComparisonKnuckleDistance, _ , _ , _ = self.defineDistanceAndOrientation(21, self.tipIds[id]  - 3)    
                    #When the hand is upright and tilted forward, the detection works better when the point of comparison is center of mass (with fingers) VS. base kunckles.
                
            if handIsUpright:
                PositiveOrientation = FingerComparisonKnuckletoTipUprightness[0]
            else:
                PositiveOrientation = -1 * FingerComparisonKnuckletoTipUprightness[0]
                    #Determines extension based on uprightness, but applies negative to account for downward direction. 

            centerOfMassttoFingerTipDistanceXY = abs(centerOfMassttoFingerTipDistance[0])
                    #measures XY distance from center of mass to finger tip. 
            centerOfMassttoFingerComparisonKnuckleDistanceXY = abs(centerOfMassttoFingerComparisonKnuckleDistance[0])
                    #measures XY distance from center of mass to landmark 1 step below thumb tip.
                    
            if abs(handVerticalOrientationXY) > 0.8: 
                #If the hand is not in the transitional area, incorporate use of the PositiveOrientation value.
                #0.8 is about the value where it enters the transitional area. 
                if (centerOfMassttoFingerTipDistanceXY < centerOfMassttoFingerComparisonKnuckleDistanceXY) or (PositiveOrientation < 0):
                        #The moment the finger tip to center of mass distance is smaller than the center of mass to comparison knuckle distance, finger is closed. 
                    fingers.append(0)
                else:
                    fingers.append(1)
            else: 
                #If the hand is in the transitional area, don't use the PositiveOrientation value. 
                if (centerOfMassttoFingerTipDistanceXY < centerOfMassttoFingerComparisonKnuckleDistanceXY):
                        #The moment the finger tip to center of mass distance is smaller than the center of mass to comparison knuckle distance, finger is closed. 
                    fingers.append(0)
                else:
                    fingers.append(1)
    
        if sum(fingers[1:5]) == 0:
            handisClosed = True
            handMsg = "closed"
                #Regardless of thumb, if the four fingers of a hand are down, hand is closed. 
                #Remember, count starts from 0. 
                #Also, remember that index ranges are exclusive on the end index. 
        else: 
            if 0 < sum(fingers[0:5]) < 5:
                handMsg = "partially open"
            else: 
                handMsg = "open"
            
            handisClosed = False
                #regardless of how many fingers are up, if all four aren't down, the hand will be considered not closed. 
        
        return fingers, handMsg, handisClosed

    def completeInfo(self,frame):
        landmarkCoordinates = self.lmsList
        #List of 23 landmark coordinates, each item being a list of 4 elements: [id, x, y, z]
        #Landmarks21 and 22 are the center of mass with and without fingers respectively. 
        centerOfMassWithFingers, centerOfMassNoFingers = self.findAndMarkCenterOfMass()
        #Both of these are lists of 4 elements: [id, x, y, z]
        handIsUpright, thumbOnLeft, handVerticalOrientationXY, handHorizontalOrientationXY = self.findOrientation()
        #The first two of these are booleans. The other two are values from 0 - 1 which represent verticality and horizontality respectively. 
        #Add the last two if you need them to the return list below. I don't want to mess with your retrieval. 
        rotation, _ = self.findRotation()
        #A float between -1 and 1. 
        forwardTilt, sidewaysTilt = self.findTilt()
        #Both of these are floats between -1 and 1. 
        fingers, handMsg, handisClosed = self.findFingersOpen()
        #fingers is a list of 5 integers between 0 and 1. 
        #handMsg string that is either "closed", "partially open", or "open". 
        #handisClosed is a boolean.
        webcamDimensions = [h,w,_] = frame.shape
        #A list which contains the height and width of webcam. 

        return landmarkCoordinates, centerOfMassWithFingers, centerOfMassNoFingers, handIsUpright, thumbOnLeft, handVerticalOrientationXY, handHorizontalOrientationXY, rotation, forwardTilt, sidewaysTilt, fingers, handMsg, handisClosed, webcamDimensions
    

def main():
        
        ctime = 0
        ptime = 0
        #Setting values for important variables for later. 

        cap = cv2.VideoCapture(0)
        #Takes video input from the first deteted camera. 
        
        detector = HandTrackingDynamic()
            # This declares detector to be an object of the HandTrackingDyanmic class, which gives it access to all the functions (methods) above.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        while True:
            ret, frame = cap.read()
                #take camera input

            frame = detector.processAndCorrectView(frame)
                #process camera input and flip view
            frame = detector.drawHandLandmarks(frame)
                #draw intial landmark drawings
            lmsList = detector.findAndMark_Positions(frame)
                #Determine pixel positions and do secondary landmark drawing.

            ctime = time.time()
            fps = 1/(ctime-ptime)
            ptime = ctime
                #ctime is the time at which the loop was last ran. ptime stores the previous ctime. 
                #Hence, the FPS actually refers to how often landmark (knuckle) locations are calculated per second. 

            fontSize = 1.2
            fontThickness = 2

            cv2.putText(frame, ("FPS: " + str(int(fps))), (5,30), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)
                #On screen FPS counter. Second argument is text to be displayed, third is location and the rest is font/color/formatting.

            if len(lmsList[0]) != 0:
                # This if statement is necessary because without it, the program kills itself when your hand isn't on screen lol. 
                frame = detector.markOrientation(frame)
                rotation, _ = detector.findRotation(frame)
                forwardTilt, sidewaysTilt = detector.findTilt(frame)
                centerOfMassWithFingers, centerOfMassNoFingers = detector.findAndMarkCenterOfMass(frame)
                fingers, handMsg, _ = detector.findFingersOpen()

                #print(detector.lmsList[0][3]*5e5)
                #print("Fingers Open: ", fingers, (sum(fingers[0:5])), "  ", verticalDistance, handHorizontalOrientation, handVerticalOrientation, horizontalDistance, maxDistance, rotation)
                    # Output finger states and other info to console
                
                cv2.putText(frame, ("Fingers Open: " + str(fingers) + " " + str(sum(fingers[0:5])) + "  Hand is " + handMsg), (5,60), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)
                    #On-screen hand status. 
                cv2.putText(frame, ("Rotation: " + str(rotation)), (5,90), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)
                cv2.putText(frame, ("Forward Tilt: " + str(forwardTilt) + "  Sideways Tilt:" + str(sidewaysTilt)), (5,120), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)
                cv2.putText(frame, ("Center of Mass:"), (5,160), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)
                cv2.putText(frame, ("  With Fingers: " + str(centerOfMassWithFingers[1:4])), (5,190), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)
                cv2.putText(frame, ("  Without Fingers: " + str(centerOfMassNoFingers[1:4])), (5,220), cv2.FONT_HERSHEY_PLAIN, fontSize, (0,255,0), fontThickness)

            else: 
                cv2.putText(frame, ("Awaiting Hand..."), (5,70), cv2.FONT_HERSHEY_PLAIN, 2, (74,26,255), 2)
                    #On screen FPS counter. Second argument is text to be displayed, third is location and the rest is font/color/formatting.
            
            #if len(lmsList)!=0:
                #print(lmsList[0])
                    #This is a print used for debugging, commenented out for now.  

            if cv2.waitKey(1) == ord('x'):
                break
                    #break condition: if x is pressed, stops loop
 
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #Creates a greyscale version of the camera output. It doesn't get used. 
            
           
            cv2.imshow('Hand Movement Interpreter', frame)
                #Opens a window with the name Hand Movement Interpreter and displays the result of running the above code on the camera input. 

if __name__ == "__main__":
            main()

        #These two lines just make sure that main() doesnt run unless this script is run directly. Prevents it from running unintentionally if this script is imported into another program. 