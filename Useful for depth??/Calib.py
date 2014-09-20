"""
These are some functions to help work with kinect camera calibration and projective
geometry.

Tasks:
- Convert the kinect depth image to a metric 3D point cloud
- Convert the 3D point cloud to texture coordinates in the RGB image

Notes about the coordinate systems:
There are three coordinate systems to worry about.

IN
1. Kinect depth image:
u,v,depth
u and v are image coordinates, (0,0) is the top left corner of the image
(640,480) is the bottom right corner of the image
depth is the raw 11-bit image from the kinect, where 0 is infinitely far away
and larger numbers are closer to the camera
(2047 indicates an error pixel)

OUT
2. Kinect rgb image:
u,v
u and v are image coordinates (0,0) is the top left corner
(640,480) is the bottom right corner
3. XYZ world coordinates:
x,y,z
The 3D world coordinates, in meters, relative to the depth camera.
(0,0,0) is the camera center.
Negative Z values are in front of the camera, and the positive Z direction points
towards the camera.
The X axis points to the right, and the Y axis points up. This is the standard
right-handed coordinate system used by OpenGL.

"""
import numpy as np



def depth2xyzuv(depth, u=None, v=None):
  """
Return a point cloud, an Nx3 array, made by projecting the kinect depth map
through intrinsic / extrinsic calibration matrices
Parameters:
depth - comes directly from the kinect
u,v - are image coordinates, same size as depth (default is the original image)
Returns:
xyz - 3D world coordinates in meters (Nx3)
uv - image coordinates for the RGB image (Nx3)
You can provide only a portion of the depth image, or a downsampled version of
the depth image if you want; just make sure to provide the correct coordinates
in the u,v arguments.
Example:
# This downsamples the depth image by 2 and then projects to metric point cloud
u,v = mgrid[:480:2,:640:2]
xyz,uv = depth2xyzuv(freenect.sync_get_depth()[::2,::2], u, v)
# This projects only a small region of interest in the upper corner of the depth image
u,v = mgrid[10:120,50:80]
xyz,uv = depth2xyzuv(freenect.sync_get_depth()[v,u], u, v)
"""
  if u is None or v is None:
    u,v = np.mgrid[:480,:640]
  
  # Build a 3xN matrix of the d,u,v data
  C = np.vstack((u.flatten(), v.flatten(), depth.flatten(), 0*u.flatten()+1))

  # Project the duv matrix into xyz using xyz_matrix()
  X,Y,Z,W = np.dot(xyz_matrix(),C)
  X,Y,Z = X/W, Y/W, Z/W
  xyz = np.vstack((X,Y,Z)).transpose()
  xyz = xyz[Z<0,:]

  # Project the duv matrix into U,V rgb coordinates using rgb_matrix() and xyz_matrix()
  #U,V,_,W = np.dot(np.dot(uv_matrix(), xyz_matrix()),C)
  U,V,_,W = np.dot(np.dot(uv_matrix(), xyz_matrix()),C)
  U,V = U/W, V/W
  uv = np.vstack((U,V)).transpose()
  uv = uv[Z<0,:]

  # Return both the XYZ coordinates and the UV coordinates
  return xyz, uv



def uv_matrix():
  """
Returns a matrix you can use to project XYZ coordinates (in meters) into
U,V coordinates in the kinect RGB image
"""
  rot = np.array([[ 9.99846e-01, -1.26353e-03, 1.74872e-02],
                  [-1.4779096e-03, -9.999238e-01, 1.225138e-02],
                  [1.747042e-02, -1.227534e-02, -9.99772e-01]])
  trans = np.array([[1.9985e-02, -7.44237e-04,-1.0916736e-02]])
  m = np.hstack((rot, -trans.transpose()))
  m = np.vstack((m, np.array([[0,0,0,1]])))
  KK = np.array([[529.2, 0, 329, 0],
                 [0, 525.6, 267.5, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])
  m = np.dot(KK, (m))
  return m

def xyz_matrix():
  fx = 594.21
  fy = 591.04
  a = -0.0030711
  b = 3.3309495
  #cx = 339.5
  cx = 325.5
  #cy = 242.7
  cy = 252.0
  mat = np.array([[1/fx, 0, 0, -cx/fx],
                  [0, -1/fy, 0, cy/fy],
                  [0, 0, 0, -1],
                  [0, 0, a, b]])
  return mat

"""
fx_rgb 5.2921508098293293e+02
fy_rgb 5.2556393630057437e+02
cx_rgb 3.2894272028759258e+02
cy_rgb 2.6748068171871557e+02
k1_rgb 2.6451622333009589e-01
k2_rgb -8.3990749424620825e-01
p1_rgb -1.9922302173693159e-03
p2_rgb 1.4371995932897616e-03
k3_rgb 9.1192465078713847e-01

    Depth 

fx_d 5.9421434211923247e+02
fy_d 5.9104053696870778e+02
cx_d 3.3930780975300314e+02
cy_d 2.4273913761751615e+02
k1_d -2.6386489753128833e-01
k2_d 9.9966832163729757e-01
p1_d -7.6275862143610667e-04
p2_d 5.0350940090814270e-03
k3_d -1.3053628089976321e+00

Relative transform between the sensors (in meters)

R
[ 9.9984628826577793e-01, 1.2635359098409581e-03,
-1.7487233004436643e-02, -1.4779096108364480e-03,
9.9992385683542895e-01, -1.2251380107679535e-02,
1.7470421412464927e-02, 1.2275341476520762e-02,
9.9977202419716948e-01 ]

T
[ 1.9985242312092553e-02, -7.4423738761617583e-04,
-1.0916736334336222e-02 ]
"""
  
def uv_matrix2():
  """
Returns a matrix you can use to project XYZ coordinates (in meters) into
U,V coordinates in the kinect RGB image
"""
  rot = np.array([[ 9.9984628826577793e-01, -1.2635359098409581e-03, -1.7487233004436643e-02],
                  [-1.4779096108364480e-03, -9.9992385683542895e-01,  1.2251380107679535e-02],
                  [ 1.7470421412464927e-02, -1.2275341476520762e-02, -9.9977202419716948e-01]])
  trans = np.array([[1.9985242312092553e-02, -7.4423738761617583e-0404,-1.0916736334336222e-02]])
  m = np.hstack((rot, -trans.transpose()))
  m = np.vstack((m, np.array([[0,0,0,1]])))
  KK = np.array([[5.2921508098293293e+02, 0, 3.2894272028759258e+02, 0],
                 [0, 5.2556393630057437e+02, 2.6748068171871557e+02, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

  m = np.dot(KK, (m))
  return m

def xyz_matrix2():
  fx = 5.9421434211923247e+02
  fy = 5.9104053696870778e+02
  a = -0.0030711
  b = 3.3309495
  #cx = 339.5
  cx = 3.3930780975300314e+02
  #cy = 242.7
  cy = 2.4273913761751615e+02
 
  mat = np.array([[1/fx, 0, 0, -cx/fx],
                  [0, -1/fy, 0, cy/fy],
                  [0, 0, 0, -1],
                  [0, 0, a, b]])
  return mat
