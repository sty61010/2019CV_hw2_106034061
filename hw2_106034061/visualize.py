import numpy as np
import matplotlib.pyplot as plt
import math

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
   
def draw_camera(ax, R, pos, color):
    leftTopCorner = [ -0.2, 0.2, -1 ] @ R
    leftTopCorner = leftTopCorner / np.linalg.norm(leftTopCorner, 2)*4 + pos

    leftBotCorner = [ -0.2, -0.2, -1 ] @ R
    leftBotCorner = leftBotCorner / np.linalg.norm(leftBotCorner, 2)*4 + pos

    rightTopCorner = [ 0.2, 0.2, -1 ] @ R
    rightTopCorner = rightTopCorner / np.linalg.norm(rightTopCorner, 2)*4 + pos

    rightBotCorner = [ 0.2, -0.2, -1 ] @ R
    rightBotCorner = rightBotCorner / np.linalg.norm(rightBotCorner, 2)*4 + pos

    leftTriangle = np.concatenate([pos, leftTopCorner, leftBotCorner], axis=0)
    topTriangle = np.concatenate([pos, leftTopCorner, rightTopCorner], axis=0)
    rightTriangle = np.concatenate([pos, rightTopCorner, rightBotCorner], axis=0)
    botTriangle = np.concatenate([pos, leftBotCorner, rightBotCorner], axis=0)

    verts = [list(leftTriangle)]
    ax.add_collection3d(Poly3DCollection(verts, facecolors=color))
    
    verts = [list(topTriangle)]
    ax.add_collection3d(Poly3DCollection(verts, facecolors=color))
    
    verts = [list(rightTriangle)]
    ax.add_collection3d(Poly3DCollection(verts, facecolors=color))
    
    verts = [list(botTriangle)]
    ax.add_collection3d(Poly3DCollection(verts, facecolors=color))

    
def visualize(pts, R1, T1, R2, T2):
    '''
    Input:
        pts: 36x3 3D points
        R1: 3x3 rotation matrix of image 1
        T1: 3x1 translation vector of image 1
        R2: 3x3 roatation matrix of image 2
        T2: 3x1 translation vector of image 2
    
    This function will display a chessboard and two cameras.
    '''
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(-10, 5)
    ax.set_ylim3d(10, -5)
    ax.set_zlim3d(-5, 10)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # pose vector of camera
    cameraPoseVector1 = R1.T @ [0, 0, 1]
    cameraPoseVector1 /= np.linalg.norm(cameraPoseVector1)
    cameraPoseVector2 = R2.T @ [0, 0, 1]
    cameraPoseVector2 /= np.linalg.norm(cameraPoseVector2)
    


    angle = math.degrees(math.acos(np.clip(np.dot(cameraPoseVector1, cameraPoseVector2), -1, 1)))
    print('Angle between two cameras: ', angle)    

    # position of camera
    cameraPos1 = -T1.T @ R1
    cameraPos2 = -T2.T @ R2
    
    # draw
    for r in range(1, 9):
        for c in range(1, 4):
            fourCorner = np.concatenate((pts[r*4+c-5, :], pts[r*4+c-4, :], pts[r*4+c, :], pts[r*4+c-1, :]))
            fourCorner.resize((4,3))
            
            if r%2 == c%2:
                color = 'black'
            else:
                color = 'white'
                
            verts = [list(fourCorner)]
            ax.add_collection3d(Poly3DCollection(verts, facecolors=color))
    
    # draw cameras
    draw_camera(ax, R1, cameraPos1, 'blue')
    draw_camera(ax, R2, cameraPos2, 'red')
     
    plt.show()
