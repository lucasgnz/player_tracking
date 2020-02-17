import numpy as np
import cv2

import retinex
import siltp
import channel_histogram

def averagePooling(img):

    if img.shape[0] % 2 != 0:
        img = img[:-1]    
    if img.shape[1] % 2 != 0:
        img = img[:, :-1]
        
    img_pool = img[0::2] + img[1::2]
    img_pool = img_pool[:, 0::2] + img_pool[:, 1::2]

    img_pool = img_pool / 4

    return img_pool    

def LOMO(img, config):

    sigma_list   = config['retinex']['sigma_list']
    G            = config['retinex']['G']
    b            = config['retinex']['b']
    alpha        = config['retinex']['alpha']
    beta         = config['retinex']['beta']
    low_clip     = config['retinex']['low_clip']
    high_clip    = config['retinex']['high_clip']
    R_list       = config['lomo']['R_list']
    tau          = config['lomo']['tau']
    hsv_bin_size = config['lomo']['hsv_bin_size']
    block_size_col = config['lomo']['block_size_col']
    block_step_col = config['lomo']['block_step_col']
    block_size_row = config['lomo']['block_size_row']
    block_step_row = config['lomo']['block_step_row']

    img_retinex = retinex.MSRCP(img, sigma_list, low_clip, high_clip)

    siltp_feat = np.array([])
    hsv_feat = np.array([])
    for pool in range(3):
        row_num = int((img.shape[0] - (block_size_row - block_step_row)) / block_step_row)
        col_num = int((img.shape[1] - (block_size_col - block_step_col)) / block_step_col)
        for row in range(row_num):
            for col in range(col_num):
                img_block = img[
                    row*block_step_row:row*block_step_row+block_size_row,
                    col*block_step_col:col*block_step_col+block_size_col
                ]

                siltp_hist = np.array([])
                for R in R_list:                                    
                    siltp4 = siltp.SILTP4(img_block.astype(np.float32), R, tau)
                    unique, count = np.unique(siltp4, return_counts=True)
                    siltp_hist_r = np.zeros([3**4])
                    for u, c in zip(unique, count):
                        siltp_hist_r[u] = c
                    siltp_hist = np.concatenate([siltp_hist, siltp_hist_r], 0)
               
                img_block = img_retinex[
                    row*block_step_row:row*block_step_row+block_size_row,
                    col*block_step_col:col*block_step_col+block_size_col
                ]                
                    
                img_hsv = cv2.cvtColor(img_block.astype(np.float32), cv2.COLOR_BGR2HSV)
                hsv_hist = channel_histogram.jointHistogram(
                    img_hsv,
                    [0, 255],
                    hsv_bin_size
                )

                if col == 0:
                    siltp_feat_col = siltp_hist
                    hsv_feat_col = hsv_hist
                else:
                    siltp_feat_col = np.maximum(siltp_feat_col, siltp_hist)
                    hsv_feat_col = np.maximum(hsv_feat_col, hsv_hist)

            siltp_feat = np.concatenate([siltp_feat, siltp_feat_col], 0)
            hsv_feat = np.concatenate([hsv_feat, hsv_feat_col], 0)

        img = averagePooling(img)
        img_retinex = averagePooling(img_retinex)

    siltp_feat = np.log(siltp_feat + 1.0)
    if int(siltp_feat.shape[0])%2 == 1:
        print("ATTENTION")
    siltp_feat[:int(siltp_feat.shape[0]/2)] /= np.linalg.norm(siltp_feat[:int(siltp_feat.shape[0]/2)])
    siltp_feat[int(siltp_feat.shape[0]/2):] /= np.linalg.norm(siltp_feat[int(siltp_feat.shape[0]/2):])
    
    hsv_feat = np.log(hsv_feat + 1.0)
    hsv_feat /= np.linalg.norm(hsv_feat)

    lomo = np.concatenate([siltp_feat, hsv_feat], 0)
    
    return lomo
