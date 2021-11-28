# faceswap_gl
Face swap using Python + OpenGL

![Demo gif](images/demo.gif)

This implementation finds facial landmarks on both the source and destination images, and  generates a texture to be used on a 3D model of the face. This Model is then positioned over the destination image using the landmarks.

| Source | Destination | Texture | 3D face | Result |
| ------ | ------- | ----------- | ------------------ | ------ |
| ![Source image](images/src.jpg) | ![Destination image](images/dst.png) | ![Texture generated from source image](images/texture.png) | ![Texture applied to 3D model positioned according to the face on the destination image](images/3dface.png) | ![Face swap result](images/result.png) |
