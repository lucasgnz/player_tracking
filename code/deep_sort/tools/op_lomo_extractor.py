""""//     {0,  "Nose"},
//     {1,  "Neck"},
//     {2,  "RShoulder"},
//     {3,  "RElbow"},
//     {4,  "RWrist"},
//     {5,  "LShoulder"},
//     {6,  "LElbow"},
//     {7,  "LWrist"},
//     {8,  "MidHip"},
//     {9,  "RHip"},
//    {10, "RKnee"},
//     {11, "RAnkle"},
//     {12, "LHip"},
//     {13, "LKnee"},
//     {14, "LAnkle"},
//     {15, "REye"},
//     {16, "LEye"},
//     {17, "REar"},
//     {18, "LEar"},
//     {19, "LBigToe"},
//     {20, "LSmallToe"},
//     {21, "LHeel"},
//     {22, "RBigToe"},
//     {23, "RSmallToe"},
//     {24, "RHeel"},




Regarder quels points parmi 0-24 sont le plus souvent detectés

Tous les couples de points qui forment un segment sur le schema (https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/media/keypoints_pose_25.png) sont des longueurs fixes (l'estimation des poses en 2D seulement est une approximation susceptible de poser problème).
17 - 18, les deux oreilles, ne sont pas reliées sur le schéma mais on peut considérer leur distance relative comme constante
"""


import numpy as np
import json
import math
import lomo


body_parts = {
    'head' : (17, 18, 1.1),
    'left_arm': (5, 6, 0.3),
    'right_arm': (2, 3, 0.3),
    'left_forearm': (6, 7, 0.3),
    'right_forearm': (3 ,4 ,0.3),
    'left_upper_leg': (12, 13, 0.3),
    'right_upper_leg': (9, 10, 0.3),
    'left_lower_leg': (13, 14, 0.3),
    'right_lower_leg': (10, 11, 0.3),
    'torso': (1,8, 0.6)
}

def op_lomo_extractor(keypoints, config, bgr_image):
    #Returns:
    # features : vect(n_features)
    #       a LOMO descriptor of different parts of the body from a set of openpose keypoints associated to a player
    # conf : float
    #        confidence / strength of the signal

    #We could add other features, but they need to reflect a constant property (i.e. invariant of the position on the field and current body pose) of the player
    #e.g. arms length / legs length ratio
    #the camera has a single perspective on the field so 2D estimation is a problem for estimating lengths of body parts (or ratios)

    conf = 1
    features = []

    for k, (a, b, aspect_ratio) in body_parts.items():
        Ax = keypoints[a][0]
        Bx = keypoints[b][0]
        Ay = keypoints[a][1]
        By = keypoints[b][1]
        Aconf = keypoints[a][2]
        Bconf = keypoints[b][2]
        if Aconf>0.1 and Bconf>0.1:
            center = ((Ax + Bx)/2, (Ay + By)/2)
            width = 101#int(math.sqrt((Ax - Bx) ** 2 + (Ay - By) ** 2))
            height = width#int(width * aspect_ratio)

            #width += width % config['lomo']['block_step']
            #height += height % config['lomo']['block_step']

            #theta=0 correspond au cas où Ay=By
            theta = math.atan((Ay - By) / (Ax - Bx))
            if (Ax - Bx) < 0:
                theta += 180

            patch = subimage(bgr_image, center, theta, width, height)

            #cv2.imshow("",patch)
            #cv2.waitKey()

            l = lomo.LOMO(patch, config)

            print(k, l.shape)
            features.append(l)
    return np.array(features).reshape(-1), conf



import cv2
import numpy as np

def subimage(image, center, theta, width, height):

   '''
   Rotates OpenCV image around center with angle theta (in deg)
   then crops the image according to width and height.
   '''

   # Uncomment for theta in radians
   #theta *= 180/np.pi

   shape = ( image.shape[1], image.shape[0] ) # cv2.warpAffine expects shape in (length, height)

   matrix = cv2.getRotationMatrix2D( center=center, angle=theta, scale=1 )
   image = cv2.warpAffine( src=image, M=matrix, dsize=shape )

   x = int( center[0] - width/2  )
   y = int( center[1] - height/2 )

   image = image[ y:y+height, x:x+width ]

   return image

if __name__ == "__main__":

    print("")