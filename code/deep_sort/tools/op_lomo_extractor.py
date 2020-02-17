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

#Chaque partie du corps est encodée sous la forme d'un quadruplet (Point A du segment, Point B du segment, Ratio largeur / longeur)
body_parts = {
    'head' : (1, 0, 1.0),
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

def op_lomo_extractor(keypoints, config, bgr_image, show_patch=False):
    #Returns:
    # features : vect(n_features)
    #       a LOMO descriptor of different parts of the body from a set of openpose keypoints associated to a player
    # conf : float
    #        confidence / strength of the signal
    #LOMO [4] is a descriptor for person Re-ID that divides each image into horizontal
    # bands and finds the maximum bins of color and texture histograms in each stripe. We
    # modified this code to use it on body parts.
    # https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liao_Person_Re-Identification_by_2015_CVPR_paper.pdf
    #
    #We could add other features, but they need to reflect a constant property (i.e. invariant of the position on the field and current body pose) of the player
    #e.g. arms length / legs length ratio
    #the camera has a single perspective on the field so 2D estimation is a problem for estimating lengths of body parts (or ratios)

    block_size_col = config['lomo']['block_size_col']
    block_step_col = config['lomo']['block_step_col']

    block_size_row = config['lomo']['block_size_row']
    block_step_row = config['lomo']['block_step_row']
    patch_width = config['lomo']['patch_width']
    conf = 0
    features = []

    """(8 ∗ 8 ∗ 8 color
                bins + 3^4 ∗ 2 SILTP bins ) ∗ (24 + 11 + 5 horizontal groups
                ) = 26, 960 dimensions"""
    n_lomo_features = ( config['lomo']['hsv_bin_size'] ** 3 + 3 ** 4 * len(config['lomo']['R_list']) ) * (
                        int((patch_width - (block_size_row - block_step_row)) / block_step_row) +
                        int(int(patch_width / 2 - (block_size_row - block_step_row)) / block_step_row) +
                        int(int(patch_width / 4 - (block_size_row - block_step_row)) / block_step_row)
                           )

    for k, (a, b, aspect_ratio) in body_parts.items():
        Ax = keypoints[a][0]
        Bx = keypoints[b][0]
        Ay = keypoints[a][1]
        By = keypoints[b][1]

        Aconf = keypoints[a][2]
        Bconf = keypoints[b][2]

        center = ((Ax + Bx)/2, (Ay + By)/2)
        width = int(math.sqrt((Ax - Bx) ** 2 + (Ay - By) ** 2))
        height = int(width * aspect_ratio)



        l = np.zeros(n_lomo_features)

        if Aconf > 0.1 and Bconf > 0.1 and abs(Ax - Bx) > 1 and abs(Ay - By) > 1:

            # theta=0 correspond au cas où Ay=By
            theta = 90 + math.atan((Ay - By) / (Ax - Bx))
            if (Ax - Bx) < 0:
                theta += 180

            patch = subimage(bgr_image, center, theta, width, height)
            #print("patch shape ", patch.shape)

            if patch.shape[0]>0 and patch.shape[1]>0:
                patch_reshaped = cv2.resize(patch, (patch_width, int(patch_width * aspect_ratio)))
                patch_reshaped = np.transpose(patch_reshaped, [1, 0, 2])

                #print("reshaped patch shape ", patch_reshaped.shape)

                if(show_patch):
                    cv2.imshow(k+str(np.random.rand()), patch)
                    cv2.waitKey()
                    cv2.imshow("reshaped_"+k + str(np.random.rand()), patch_reshaped)
                    cv2.waitKey()

                l = lomo.LOMO(patch_reshaped, config)
                l *= Aconf * Bconf
                conf += (Aconf * Bconf) / len(body_parts)


        features += list(l.reshape(-1))

    return np.array(features), conf



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