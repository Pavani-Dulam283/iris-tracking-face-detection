import cv2 as cv
import numpy as np
import mediapipe as mp
import math

TOTAL_BLINKS = 0
FRAME_COUNT = 0
res_data = []
frameCheck = True
L_H_LEFT = [33]  # RIGHT EYE RIGHT MOST LANDMARK
L_H_RIGHT = [133]  # RIGHT EYE LEFT MOST LANDMARK
R_H_LEFT = [362]  # LEFT EYE RIGHT MOST LANDMARK
R_H_RIGHT = [263]  # LEFT EYE LEFT MOST LANDMARK

RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390,
            249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154,
             155, 133, 173, 157, 158, 159, 160, 161, 246]

              


def euclaideanDistance(point1, point2):

    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance


# def euclidean_distance(point1, point2):
#     x1, y1 = point1.ravel()
#     x2, y2 = point2.ravel()
#     distance = math.sqrt((x2-x1)**2+(y2-y1)**2)
#     return distance


def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclaideanDistance(iris_center, right_point)
    total_distance = euclaideanDistance(right_point, left_point)
    ratio1 = center_to_right_dist/total_distance
    iris_position = ""
    if ratio1 < 0.42:
        iris_position = "right"
    elif ratio1 > 0.42 and ratio1 < 0.571:
        iris_position = "center"
    else:
        iris_position = "left"
    return iris_position


def blink_ratio(img, landmarks, right_indices, left_indices):
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
    return 1

mp_face_mesh = mp.solutions.face_mesh


# FACE_PO = [x for x in range(478)]
cap = cv.VideoCapture("theory2.mp4")

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, frame = cap.read()
        FRAME_COUNT += 1
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            # print(results.multi_face_landmarks)
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(
                int) for p in results.multi_face_landmarks[0].landmark])
            
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(
                mesh_points[RIGHT_IRIS])

            # center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            cv.polylines(frame, [mesh_points[LEFT_EYE]],
                         True, (0, 255, 0), 1, cv.LINE_AA)
            cv.polylines(frame, [mesh_points[RIGHT_EYE]],
                         True, (0, 255, 0), 1, cv.LINE_AA)
            cv.polylines(frame, [mesh_points[RIGHT_EYE]],
                         True, (0, 255, 0), 1, cv.LINE_AA)
            cv.polylines(frame, [mesh_points[LEFT_EYE]],
                         True, (0, 255, 0), 1, cv.LINE_AA)
            ratio = blink_ratio(frame, mesh_points, RIGHT_EYE, LEFT_EYE)
            # cv.putText(frame, f'ratio {ratio}', (100, 100), cv.FONT_HERSHEY_TRIPLEX,1.0,(0,0,255),thickness=4)
            iris_pos = iris_position(
                center_right, mesh_points[R_H_RIGHT][0], mesh_points[R_H_LEFT][0]
            )  # print(mesh_points)

            if (FRAME_COUNT < 50 ):
                 
                if (float(ratio) > 4.0 and frameCheck==True):
                    TOTAL_BLINKS += 1
                    frameCheck = False
                FRAME_COUNT+=1
                    
            else:
                FRAME_COUNT = 0
                frameCheck = True
                json_data={
                            "iris_position": iris_pos,
                            "Total_blinks": TOTAL_BLINKS                    
                        }
                res_data.append(json_data)


            cv.putText(frame, f'TOTAL_BLINKS {TOTAL_BLINKS}', (150, 150),
                       cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 255), thickness=4)
            cv.putText(frame, f'FRAME COUNT {FRAME_COUNT}', (200, 200), cv.FONT_HERSHEY_TRIPLEX,1.0,(0,0,255),thickness=4)

            

            

            # if(dist>60)
            cv.putText(
                frame,
                f"Iris pos:{iris_pos}",
                (100, 100),
                cv.FONT_HERSHEY_TRIPLEX,
                1.2,
                (0, 255, 0),
                1,
                cv.LINE_AA
            )

        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key == ord('q'):
            # print(res_data)
            # print(len(res_data))
            break

cap.release()
print(res_data)
# print(len(res_data))
cv.destroyAllWindows()
