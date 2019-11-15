import os
import sys
import cv2
import numpy as np

def Mouse_Handler(event, x, y, flags, data) :
    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(data['im'], (x,y),3, (0,255,255), 5, 16);
        cv2.imshow("Image", data['im']);
        if len(data['points']) < 4 :
            data['points'].append([x,y])

def Get_Four_Corner(im):
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['points'] = []
    #Set the callback function for any mouse event
    cv2.imshow("Image",im)
    cv2.setMouseCallback("Image", Mouse_Handler, data)
    cv2.waitKey(0)
    # Convert array to np.array
    points = np.vstack(data['points']).astype(float)
    return points

def Compute_Projection_Matrix(pts_src, pts_dst):#src::3D, ds::2D
    r, c = pts_src.shape
    A = np.zeros((r*2, 8))
    for i in range(r):
        A[i*2, 0:2] = pts_src[i, :]
        A[i*2, 2] = 1
        A[i*2, 6:] = -pts_dst[i][0] * pts_src[i]
#        A[i*2, -1]= -pts_dst[i][0]
        A[i*2+1, 3:5] = pts_src[i,:]
        A[i*2+1, 5] = 1
        A[i*2+1, 6:] = -pts_dst[i][1] * pts_src[i]
#        A[i*2 + 1, -1] = -pts_dst[i][1]
    A_pt = np.linalg.pinv(A)
    B = pts_dst.reshape((8,1))
    Projection_Matrix = A_pt.dot(B)
    Projection_Matrix = np.append(Projection_Matrix, [1])
    Projection_Matrix = Projection_Matrix.reshape((3,3))
#    print("Projection_Matrix");print(Projection_Matrix)
    return Projection_Matrix

def to_mtx(img):
    H,V,C = img.shape
    mtr = np.zeros((V,H,C), dtype='int')
    for i in range(img.shape[0]):
        mtr[:,i] = img[i]
    return mtr

def to_img(mtr):
    V,H,C = mtr.shape
    img = np.zeros((H,V,C), dtype='int')
    for i in range(mtr.shape[0]):
        img[:,i] = mtr[i]
    return img.astype(np.uint8)

def Forward_Warpping(img, M, dsize):
    mtr = to_mtx(img)
    R,C = dsize
    dst = np.zeros((R,C,mtr.shape[2]))
    for i in range(mtr.shape[0]):
        for j in range(mtr.shape[1]):
            res = np.dot(M, [i,j,1])
            i2,j2,_ = (res / res[2] + 0.5).astype(int)
            if i2 >= 0 and i2 < R:
                if j2 >= 0 and j2 < C:
                    dst[i2,j2] = mtr[i,j]
    return to_img(dst)
    
def Backward_Warpping(img, M, dsize):
    mtr = (img)
    C,R = dsize
    dst = np.zeros((R,C,mtr.shape[2]))
#    print("M");print(M)
    invM = np.linalg.inv(M)
#    print("invM");print(invM)
    for i in range(R):
        for j in range(C):
            res = np.dot(invM, [i,j,1])
            i2,j2,_ = (res / res[2] + 0.5).astype(int)
            if i2 >= 0 and i2 < mtr.shape[0]:
                if j2 >= 0 and j2 < mtr.shape[1]:
                    dst[i,j] = mtr[i2,j2]
    return dst.astype(np.uint8)

def Switch_Picture(pts_src, pts_dst, im_src, im_dst, index):
#    h, status = cv2.findHomography(pts_src, pts_dst);
    h0 = Compute_Projection_Matrix(pts_src, pts_dst)
    h1 = Compute_Projection_Matrix(pts_dst, pts_src)

#    im_temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))
    im_temp_F0 = Forward_Warpping(im_src, h0, (im_dst.shape[1],im_dst.shape[0]))
#    cv2.imshow("Forward Warpping_"+str(index), im_temp_F0);
    im_cut_F0 = im_temp_F0.copy()
    cv2.fillConvexPoly(im_cut_F0, pts_dst.astype(int), 0, 16);
    im_remain_F0 = im_temp_F0 - im_cut_F0
    
    im_temp_F1 = Forward_Warpping(im_dst, h1, (im_dst.shape[1],im_dst.shape[0]))
    im_cut_F1 = im_temp_F1.copy()
    cv2.fillConvexPoly(im_cut_F1, pts_src.astype(int), 0, 16);
    im_remain_F1 = im_temp_F1 - im_cut_F1
    
    
    im_result_F = im_dst.copy()
    cv2.fillConvexPoly(im_result_F, pts_dst.astype(int), 0, 16);
    cv2.fillConvexPoly(im_result_F, pts_src.astype(int), 0, 16);
    im_result_F = im_result_F + im_remain_F0 + im_remain_F1;
    cv2.imshow("Result_A_ForwardWarping_", im_result_F);
    cv2.imwrite("./hw2-2/Result_A_ForwardWarping"+".jpg", im_result_F);
    cv2.waitKey(0);
    
    im_temp_B0 = Backward_Warpping(im_src, h0, (im_dst.shape[1],im_dst.shape[0]))
#    cv2.imshow("Backward Warpping_"+str(index), im_temp_B);
    im_cut_B0 = im_temp_B0.copy()
    cv2.fillConvexPoly(im_cut_B0, pts_dst.astype(int), 0, 16);
    im_remain_B0 = im_temp_B0 - im_cut_B0
    
    im_temp_B1 = Forward_Warpping(im_dst, h1, (im_dst.shape[1],im_dst.shape[0]))
    im_cut_B1 = im_temp_B1.copy()
    cv2.fillConvexPoly(im_cut_B1, pts_src.astype(int), 0, 16);
    im_remain_B1 = im_temp_B1 - im_cut_B1
    
    im_result_B = im_dst.copy()
    cv2.fillConvexPoly(im_result_B, pts_dst.astype(int), 0, 16);
    cv2.fillConvexPoly(im_result_B, pts_src.astype(int), 0, 16);
    im_result_B = im_result_B + im_remain_B0 + im_remain_B1;
    
    cv2.imshow("Result_A_BackwardWarping", im_result_B);
    cv2.imwrite("./hw2-2/Result_A_BackwardWarping"+".jpg", im_result_B);
    cv2.waitKey(0);
    
if __name__ == '__main__' :
    # Read source image.
    im_src = cv2.imread('./hw2-2/data_2-2_A.jpg');
    # Read destination image
    im_dst = cv2.imread('./hw2-2/data_2-2_A.jpg');
    if os.path.isfile('./hw2-2/pts_src_A.npy'):
        pts_src = np.load('./hw2-2/pts_src_A.npy')
    else:
        pts_src = Get_Four_Corner(im_src)
        np.save('hw2-2/pts_src_A', np.asarray(pts_src))
    if os.path.isfile('./hw2-2/pts_dst_A.npy'):
        pts_dst = np.load('./hw2-2/pts_dst_A.npy')
    else:
        # Get four corners of the billboard
        print ('Click on four corners of a billboard and then press ENTER')
        pts_dst = Get_Four_Corner(im_dst)
        np.save('hw2-2/pts_dst_A', np.asarray(pts_dst))
    Switch_Picture(pts_src, pts_dst, im_src, im_dst, 0)



