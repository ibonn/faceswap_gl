"""
TODO
    * Render headless: https://stackoverflow.com/questions/51627603/opengl-render-view-without-a-visible-window-in-python
    * Add padding
    * Enable lights
    * Add more filters
    * Raise custom exception instead of RuntimeError
"""
import cv2
import mediapipe as mp
import sys, pygame
from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from PIL import Image, ImageOps
import progressbar

from obj_parser import OBJ

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

pygame.init()

def add_border(image, size=100):
    height, width, channels = image.shape

    new_width = 2 * size + width
    new_height = 2 * size + height

    result = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    result[size:size + height, size:size + width, :] = image

    return result

def remove_border(image, size=100):
    height, width, _ = image.shape

    original_width = width - 2 * size
    original_height = height - 2 * size

    return image[size:size + original_height, size:size + original_width, ...]

def get_landmarks(img, flip=False, detection_confidence=0.9, tracking_confidence=0.9):
    with mp_face_mesh.FaceMesh(min_detection_confidence=detection_confidence, min_tracking_confidence=tracking_confidence) as face_mesh:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img.flags.writeable = False
        results = face_mesh.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            if flip:
                return np.array([(landmark.x, landmark.z, landmark.y) for landmark in results.multi_face_landmarks[0].landmark])
            return np.array([(landmark.x, landmark.y, landmark.z) for landmark in results.multi_face_landmarks[0].landmark])
    return None

def map_image(src, src_landmarks, dst_landmarks, dst_width, dst_height, triangles):
    
    src_height, src_width, _ = src.shape

    mapped = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)
    added_triangles = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)

    for n, trng in enumerate(triangles):

        geo_trng = trng[:, 0]
        txt_trng = trng[:, 1]

        src_trng_points = src_landmarks[geo_trng, :2]
        src_trng_points = (src_trng_points * (src_width, src_height)).astype(np.float32)

        dst_trng_points = dst_landmarks[txt_trng, :2]
        dst_trng_points = (dst_trng_points * (dst_width, dst_height)).astype(np.float32)

        ret = cv2.getAffineTransform(src_trng_points, dst_trng_points)

        warped = cv2.warpAffine(src, ret, (dst_width, dst_height)).astype(np.uint8)

        mask = cv2.fillConvexPoly(np.zeros((dst_height, dst_width, 3), dtype=np.uint8), dst_trng_points.astype(int), (255, 255, 255))

        if n != 0:
            overlap = cv2.bitwise_not(cv2.bitwise_and(added_triangles, mask))
            mask = cv2.bitwise_and(mask, overlap)

        added_triangles = cv2.bitwise_or(added_triangles, mask)

        masked = cv2.bitwise_and(warped, mask)

        mapped = cv2.bitwise_or(mapped, masked)

    return mapped

def swap_face(src, dst, output, texture_size=256, border_size=100):
    dst_video = cv2.VideoCapture(dst)

    # Get video size/fps
    width = int(dst_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(dst_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(dst_video.get(cv2.CAP_PROP_FPS))
    num_frames = int(dst_video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Find face
    src_img = cv2.imread(src)
    src_landmarks = get_landmarks(src_img)

    if src_landmarks is None:
        raise RuntimeError("The source image does not contain any face")
    
    # Load face obj
    obj = OBJ("data/canonical_face_model.obj", swap=True)

    # Generate material
    mapped = map_image(src_img, src_landmarks, obj.vt, texture_size, texture_size, obj.f)
    mapped = cv2.rotate(mapped, cv2.ROTATE_180)

    cv2.imwrite("data/face_texture.png", mapped)

    # Create render surface
    pygame.display.set_mode((width, height), OPENGL | DOUBLEBUF)

    # Set lights
    # TODO make lights available through args
    # glLightfv(GL_LIGHT0, GL_POSITION, (0.5, 0.5, 0.5, 0))
    # glLightfv(GL_LIGHT0, GL_AMBIENT, (1.0, 1.0, 1.0, 1.0))
    # glLightfv(GL_LIGHT0, GL_DIFFUSE, (1.0, 1.0, 1.0, 1.0))
    # glEnable(GL_LIGHT0)
    # glEnable(GL_LIGHTING)
    # glEnable(GL_COLOR_MATERIAL)
    # glEnable(GL_DEPTH_TEST)
    # glShadeModel(GL_SMOOTH)

    # Load material and generate OpenGL object
    obj.load_material("data/face.mtl")
    obj.generate()

    # Create output video
    # TODO make codec available through args
    out_video = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    try:
        with progressbar.ProgressBar(max_value=num_frames) as pbar:
            while True:
                success, frame = dst_video.read()
                if not success:
                    break

                # Find face
                landmarks = get_landmarks(frame, flip=True)
                if landmarks is None:
                    out_video.write(frame)

                else:
                    obj.v = landmarks
                    scaled_landmarks = np.array([(x * width, y * height) for x, y in np.delete(landmarks, 1, axis=1)]).astype(np.int32)

                    obj.generate()

                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                    glLoadIdentity()

                    # Render
                    glScale(2, 2, 2)
                    glRotate(90, 1, 0, 0)
                    glTranslatef(-0.5, 0.0, -0.5)

                    glPushMatrix()
                    obj.render()
                    glPopMatrix()

                    # Get image from 3D model
                    glPixelStorei(GL_PACK_ALIGNMENT, 1)
                    data = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
                    # I'm pretty sure all of this can be done with numpy avoiding PIL
                    image = Image.frombytes("RGB", (width, height), data)
                    image = ImageOps.flip(image)
                    face = np.array(image, dtype=np.uint8)
                    face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                    
                    # Get mask
                    hull = cv2.convexHull(scaled_landmarks)
                    rect = cv2.boundingRect(hull)
                    mask = np.zeros_like(frame, dtype=np.uint8)
                    cv2.fillConvexPoly(mask, hull, (255, 255, 255))

                    # Combine images
                    merged = cv2.seamlessClone(face, frame, mask, (rect[0] + rect[2] // 2, rect[1] + rect[3] // 2), cv2.NORMAL_CLONE)
                    out_video.write(merged)

                pygame.display.flip()
                pbar.update(pbar.value + 1)
                
    except KeyboardInterrupt:
        print("[!] Interrupted. Quitting...")
    
    except Exception as e:
        print(f"[!] An exception has occured: {e}")

    # Close videos
    dst_video.release()
    out_video.release()


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--src", type=str, action="store", required=True, help="Path to source image. Must contain a face")
    parser.add_argument("-d", "--dst", type=str, action="store", required=True, help="Path to the destination video. If no face is found the output video will be a copy of this video")
    parser.add_argument("-o", "--output", type=str, action="store", required=True, help="Path where the resulting video will be saved")
    parser.add_argument("-t", "--texture", type=int, action="store", required=False, default=256, help="Texture resolution")
    parser.add_argument("-b", "--border", type=int, action="store", required=False, default=100, help="Padding size. Currently does nothing as this feature is yet not implemented")

    parsed_args = parser.parse_args(sys.argv[1:])

    try:
        swap_face(
            src=parsed_args.src, 
            dst=parsed_args.dst, 
            output=parsed_args.output, 
            texture_size=parsed_args.texture, 
            border_size=parsed_args.border,
        )
    except RuntimeError as e:
        print(f"[!] ERROR: {e}")