import os
import cv2
import numpy as np
import scipy.linalg
import visualize as vl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

Points_2D = []
rows = 9
cols = 4
Data_index = 1
Data_number = 2
index = 0

num_points = rows * cols
Points_3D = np.loadtxt("./data/Point3D.txt", dtype='i', delimiter=' ')

def Get_Points_2D(event, x, y, flags, param):
    global Points_2D, index
    if event == cv2.EVENT_LBUTTONDOWN:
        Points_2D += [[x, y]]
        np.save('hw2-1/Points_2D_'+str(Data_index), np.asarray(Points_2D))
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img, (x, y), 5, (0, 255, 255), -1)
        cv2.imshow('image', img)
        index += 1
        if index >= num_points:
            cv2.destroyAllWindows()
    return Points_2D

def Compute_Projection_Matrix():
    global Points_2D, index
    global A,Eigen_Values, Eigen_Vectors
    r, c = Points_3D.shape
    A = np.zeros((r*2, 12))
    for i in range(r):
        A[i*2, 0:3] = Points_3D[i, :]
        A[i*2, 3] = 1
        A[i*2, 8:-1] = -Points_2D[i][0] * Points_3D[i]
        A[i*2, -1]= -Points_2D[i][0]
        A[i*2+1, 4:7] = Points_3D[i,:]
        A[i*2+1, 7] = 1
        A[i*2+1, 8:-1] = -Points_2D[i][1] * Points_3D[i]
        A[i*2 + 1, -1] = -Points_2D[i][1]

    Eigen_Values, Eigen_Vectors = np.linalg.eig(A.transpose().dot(A))
    Projection_Matrix = Eigen_Vectors[:, np.argmin(Eigen_Values)]  
    Projection_Matrix = Projection_Matrix.reshape((3,4))
    print("Projection_Matrix");print(Projection_Matrix)
    return Projection_Matrix

def Decompose_into_KRT(Projection_Matrix):
    if np.linalg.det(Projection_Matrix[:,:-1])<0:
        Projection_Matrix *= -1
    K, R = scipy.linalg.rq(Projection_Matrix[:,:-1])
    D = np.zeros((3, 3))
    for i in range(3):
        D[i, i] = np.sign(K[i][i])
#    print("D");print(D)
    K = K.dot(D)
#    print("K");print(K)
    T = np.linalg.inv(K).dot(Projection_Matrix[:, -1])
    T=T[..., None]
    R = D.dot(R)
    K /= K[-1, -1]
#    print("K") ;print(K)
#    print("R") ;print(R)
#    print("T") ;print(T)
    return K,R,T

def Reproject_into_Points_2D(K, R, T, Points_3D):
    global Points_2D,img
    RT = np.concatenate((R.transpose(), T.transpose()))
    RT = RT.transpose()
    New_Projection_Matrix = K.dot(RT)
    r, c = Points_3D.shape
    Points_3D_4N = np.c_[Points_3D, np.ones(r)]
    Points_2D_3N = New_Projection_Matrix.dot(Points_3D_4N.transpose())
    Points_2D_3N = Points_2D_3N.transpose()
    Points_2D_3N[:, 0] = np.divide(Points_2D_3N[:, 0], Points_2D_3N[:, 2])
    Points_2D_3N[:, 1] = np.divide(Points_2D_3N[:, 1], Points_2D_3N[:, 2])
    print("Points_2D_3N") ;print(Points_2D_3N)
    RMS=0
    for i in range(r):
        cv2.circle(img, (int(Points_2D[i][0]), int(Points_2D[i][1])), 5, (0, 255, 255), -1)
        cv2.circle(img, (int(Points_2D_3N[i][0]), int(Points_2D_3N[i][1])), 3, (255, 0, 255), -1)
        RMS+=(Points_2D[i][0]-Points_2D_3N[i][0])**2+(Points_2D[i][1]-Points_2D_3N[i][1])**2
    RMS /= r
    RMS=np.sqrt(RMS)
    print("RMS") ;print(RMS)
    cv2.imshow('image', img)
    cv2.imwrite('./hw2-1/Result' + str(Data_index) + '.jpg', img)
    return Points_2D_3N[:, 0:2], RMS
def Compute_Camera_Position(R, T):
    return R.transpose().dot(T)
def Compute_Camera_Angle():
    return
    
if __name__ == '__main__':
    for i in range(1, Data_number+1):
        index = 0
        img_path = 'data/data_' + str(Data_index) +'.jpg'
#        print("Image Path") ;print(img_path)
        img = cv2.imread(img_path)
        h, w, c = img.shape
        if os.path.isfile('./hw2-1/Points_2D_'+str(Data_index)+'.npy'):
            print("Showing Image...")
            Points_2D = np.load('./hw2-1/Points_2D_'+str(Data_index)+'.npy')
            print("Points_2D") ;print(Points_2D)
            print("Points_3D") ;print(Points_3D)
        else:
            print("You need to labelize it...")
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', w, h)
            cv2.setMouseCallback('image', Get_Points_2D)
            cv2.imshow('image', img)
            cv2.waitKey(0)
            np.save('hw2-1/Points_2D'+str(Data_index), np.asarray(Points_2D))
        Projection_Matrix = Compute_Projection_Matrix()
        np.save('./hw2-1/Projection_matrix_'+str(Data_index), np.asarray(Points_2D))
        K, R, T = Decompose_into_KRT(Projection_Matrix)
        np.save('./hw2-1/K_matrix_'+str(Data_index), np.asarray(K))
        np.save('./hw2-1/R_matrix_'+str(Data_index), np.asarray(R))
        np.save('./hw2-1/T_matrix_'+str(Data_index), np.asarray(T))
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', w, h)
        cv2.imshow('image', img)
        Points_2D_3N, RMS = Reproject_into_Points_2D(K, R, T, Points_3D)
        np.save('./hw2-1/RMS_'+str(Data_index), np.asarray(RMS))
        Camera_Position = Compute_Camera_Position(R, T)
        print("Camera Position") ;print(Camera_Position)
        Data_index += 1
        cv2.waitKey(0)
        
    R1=np.load('./hw2-1/R_matrix_1.npy')
    T1=np.load('./hw2-1/T_matrix_1.npy')
    R2=np.load('./hw2-1/R_matrix_2.npy')
    T2=np.load('./hw2-1/T_matrix_2.npy')
    vl.visualize(Points_3D, R1, T1, R2, T2)

    

