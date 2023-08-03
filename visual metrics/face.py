import cv2 as cv
import mediapipe as mp
import numpy as np
import math




def landmarksDetection(img, results):
    img_height, img_width = img.shape[:2]
    mesh_coord = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(
        int) for p in results.multi_face_landmarks[0].landmark])
    return mesh_coord



def euclaideanDistance(point1, point2):

    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance


LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
            249, 263, 466, 388, 387, 386, 385, 384, 398]

RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]


def eye_ratio(img, landmarks, right_indices, left_indices):
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)
    if (rvDistance and lvDistance):
        reRatio = rhDistance/rvDistance
        leRatio = lhDistance/lvDistance

        ratio = (reRatio+leRatio)/2
        return ratio
    return 1.0


HOR_LIP = [61, 291]
VER_LIP = [0, 17]

def surpriseRatio(frame, landmarks, horlip, verlip):
    # right eyebrow
    lip_left = landmarks[horlip[0]]
    lip_right = landmarks[horlip[1]]

    # left eyebrow
    lip_up = landmarks[verlip[0]]
    lip_down = landmarks[verlip[1]]
    horDistance = euclaideanDistance(lip_left, lip_right)
    verDistance = euclaideanDistance(lip_up, lip_down)

    surprise_ratio = verDistance/horDistance
    return surprise_ratio



CHEEK_ENDS= [454,234]
LIPS_ENDS =[61,291]


def happyRatio(img,landmarks,cheek_ends,lips_ends):

    # two cheek end points
    ce_right = landmarks[cheek_ends[0]]
    ce_left= landmarks[cheek_ends[1]]

    #lips end points

    le_right = landmarks[lips_ends[0]]
    le_left = landmarks[lips_ends[1]]

    #distance between cheek end points 
    ce_dist = euclaideanDistance(ce_right,ce_left)

    #distance between mouth end points
    le_dist = euclaideanDistance(le_left,le_right)

    ratio = le_dist/ce_dist
    return ratio
    
CR1=[108,285]
CR2=[337,55]

def angerRatio(frame, landmarks,cr1,cr2):
    # cross1
    cr1_left = landmarks[cr1[0]]
    cr1_right = landmarks[cr1[1]]

    # cross2
    cr2_left = landmarks[cr2[0]]
    cr2_right= landmarks[cr2[1]]

    horDistance = euclaideanDistance(cr1_left,cr1_right)
    verDistance = euclaideanDistance(cr2_left,cr2_right)

    anger_ratio =verDistance/horDistance
    return anger_ratio

RIGHT_EYE_LINE=[285,258,295,286]
LEFT_EYE_LINE=[55,28,65,56]


def lineRatio(frame,landmarks,line1,line2):
    # cross1
    cr1_right = landmarks[line1[0]]
    cr2_right = landmarks[line1[1]]
    cr3_right = landmarks[line1[2]]
    cr4_right = landmarks[line1[3]]

    # cross2
    cr1_left=landmarks[line2[0]]
    cr2_left=landmarks[line2[1]]
    cr3_left=landmarks[line2[2]]
    cr4_left=landmarks[line2[3]]

    rhDistance=euclaideanDistance(cr1_right,cr2_right)
    rvDistance=euclaideanDistance(cr3_right,cr4_right)

    lhDistance=euclaideanDistance(cr1_left,cr2_left)
    lvDistance=euclaideanDistance(cr3_left,cr4_left)
    if (rvDistance and lvDistance):
        reRatio = rhDistance/rvDistance
        leRatio = lhDistance/lvDistance

        ratio = (reRatio+leRatio)/2
        return ratio
    return 1


def findMax(json_data):
    vals = json_data.values()

    for i in json_data:
        if(json_data[i]==max(vals)):
            return i


emotion_data = []

emotion_data_for_fifty = []

FRAME_COUNT = 0
# camera = cv.VideoCapture('emotions.mp4')
camera = cv.VideoCapture("theory.mp4")
map_face_mesh = mp.solutions.face_mesh
with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True) as face_mesh:

    while True:
        ret, frame = camera.read()
        FRAME_COUNT+=1
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:

            mesh_coords = landmarksDetection(frame, results)

            # surprise polylines 


            # cv.polylines(frame, [mesh_coords[HOR_LIP]],
            #              True, (0, 255, 0), 1, cv.LINE_AA)
            # cv.polylines(frame, [mesh_coords[VER_LIP]],
            #              True, (0, 255, 0), 1, cv.LINE_AA)


            #eye ratio calculations
            eyeRatio = eye_ratio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

            #surprise calculations 
            surRatio = surpriseRatio(frame, mesh_coords, HOR_LIP, VER_LIP)

            
                
            cv.polylines(frame, [mesh_coords[HOR_LIP]],
                         True, (0, 255, 0), 1, cv.LINE_AA)
            cv.polylines(frame, [mesh_coords[VER_LIP]],
                         True, (0, 255, 0), 1, cv.LINE_AA)

            
            #happy emotion 
            happy_ratio = happyRatio(frame,mesh_coords,CHEEK_ENDS,LIPS_ENDS)

            
            

            # cv.putText(frame, f'happy ratio  {happy_ratio}', (0, 300),
            #            cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), thickness=2)


            # cv.polylines(frame, [mesh_coords[CHEEK_ENDS]],
            #              True, (0, 255, 0), 1, cv.LINE_AA)
            # cv.polylines(frame, [mesh_coords[LIPS_ENDS]],
            #              True, (0, 255, 0), 1, cv.LINE_AA)


            #angry emotion
            anger_ratio=angerRatio(frame,mesh_coords,CR1,CR2)
            line_ratio=lineRatio(frame,mesh_coords,RIGHT_EYE_LINE,LEFT_EYE_LINE)

            if(FRAME_COUNT<50):
                emotion = ""
                # conditions for emotions
                if eyeRatio<3.0 and line_ratio>1.2:
                    cv.putText(frame, f'emotion is anger', (50, 50), cv.FONT_HERSHEY_TRIPLEX,1.0,(255,0,0),thickness=2)
                    emotion="angry"
                elif (surRatio > 0.7 and eyeRatio < 2.5):
                    cv.putText(frame, f'emotion is surprise', (50, 50),
                            cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 0, 0), thickness=2)
                    emotion="surprise"
                elif(happy_ratio>0.44):
                    cv.putText(frame, f'emotion is happy', (50,50),
                        cv.FONT_HERSHEY_TRIPLEX, 1.0, (255, 0, 0), thickness=2)
                    emotion="happy"
                else:
                    cv.putText(frame, f'emotion is neutral', (50, 50), cv.FONT_HERSHEY_TRIPLEX,1.0,(255,0,0),thickness=2)
                    emotion="neutral"


                json_data = {
                    "surprise":0,
                    "happy":0,
                    "angry":0,
                    "neutral":0
                }
                json_data[emotion]=1

                emotion_data_for_fifty.append(json_data)
            else:
                main_json_data = {
                    "surprise":0,
                    "happy":0,
                    "angry":0,
                    "neutral":0
                }
                for i in emotion_data_for_fifty:
                    main_json_data["surprise"]+=i['surprise']
                    main_json_data["happy"]+=i['happy']
                    main_json_data["neutral"]+=i['neutral']
                    main_json_data["angry"]+=i['angry']
                
                main_json_data["main-lead-emotion"]=findMax(main_json_data)
                emotion_data.append(main_json_data)
                

                emotion_data_for_fifty=[]
                FRAME_COUNT=0

                    
            
            

            

            cv.imshow('Video', frame)
            if cv.waitKey(20) & 0xFF == ord('q'):
                break


camera.release()
print(emotion_data)
print("length ",len(emotion_data))
cv.destroyAllWindows()


