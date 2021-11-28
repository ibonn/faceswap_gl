# faceswap_gl
Face swap using Python + OpenGL

![Demo gif](images/demo.gif)

This implementation finds facial landmarks on both the source and destination images, and  generates a texture to be used on a 3D model of the face. This Model is then positioned over the destination image using the landmarks.

| Source | Destination | Texture | 3D face | Result |
| ------ | ------- | ----------- | ------------------ | ------ |
| ![Source image](images/src.jpg) | ![Destination image](images/dst.png) | ![Texture generated from source image](images/texture.png) | ![Texture applied to 3D model positioned according to the face on the destination image](images/3dface.png) | ![Face swap result](images/result.png) |

## Installation

Clone the repo
```
git clone https://github.com/ibonn/faceswap_gl.git
```

Install required modules
```
pip install -r cd faceswap_gl/requirements.txt
```

## Usage

Basic usage
```
python swap.py -s <SRC_PATH> -d <DST_PATH> -o <OUT_PATH>
```

More options:
* **-s/--src**: Path to the source image  
* **-d/--dst**: Path to the destination image/video
* **-o/--output**: Output path
* **-t/--texture**: Size of the texture. The lower the value the faster the program starts (but the source face resolution is lower). Defaults to 256px
* **-b/--border**: Padding size. Not implemented yet
* **-m/--mask**: Output a video with the mask

Example:
```
python swap.py -s tomhanks.jpg -d dicaprio.mp4 -o out.mp4 -t 1024
```
