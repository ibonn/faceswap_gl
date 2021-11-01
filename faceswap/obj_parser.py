import os
import numpy as np
import pygame
from OpenGL.GL import *

class OBJ:
    def __init__(self, path, swap=False):
        self._path = path

        self.v = []
        self.vt = []
        self.f = []
        self.mtl = None

        with open(path, "r") as f:
            for line in f:
                parts = line.split(" ")
                if parts[0] == "v":
                    x = float(parts[1])
                    y = float(parts[3 if swap else 2])
                    z = float(parts[2 if swap else 3])

                    self.v.append((x, y, z))

                elif parts[0] == "vt":
                    x = float(parts[1])
                    y = float(parts[2])

                    self.vt.append((x, y))

                elif parts[0] == "f":
                    face_points = []
                    for face in parts[1:]:
                        face_parts = face.split("/")
                        v_idx = int(face_parts[0]) - 1
                        vt_idx = int(face_parts[1]) - 1
                        face_points.append((v_idx, vt_idx))

                    self.f.append(face_points)

        self.v = np.array(self.v)
        self.vt = np.array(self.vt)
        self.f = np.array(self.f)
    
    def generate(self):
        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)

        for face in self.f:
            if self.mtl is not None:
                if 'texture_Kd' in self.mtl:
                    # use diffuse texmap
                    glBindTexture(GL_TEXTURE_2D, self.mtl['texture_Kd'])
                else:
                    # just use diffuse colour
                    glColor(*self.mtl['Kd'])

            glBegin(GL_POLYGON)
            for point_idx in face:
                v_idx, vt_idx = point_idx
                glTexCoord2fv(self.vt[vt_idx])
                glVertex3fv(self.v[v_idx])
            glEnd()
        glDisable(GL_TEXTURE_2D)
        glEndList()

    def load_material(self, path):

        contents = {}
        mtl = None

        with open(path, "r") as f:
            for line in f:
                if line.startswith('#'): continue

                values = line.split()
                if not values: continue

                if values[0] == 'newmtl':
                    mtl = contents[values[1]] = {}

                elif mtl is None:
                    raise ValueError("mtl file doesn't start with newmtl stmt")

                elif values[0] == 'map_Kd':
                    # load the texture referred to by this declaration
                    mtl[values[0]] = values[1]
                    imagefile = os.path.join(os.path.dirname(path), mtl['map_Kd'])
                    mtl['texture_Kd'] = self.load_texture(imagefile)

                else:
                    mtl[values[0]] = list(map(float, values[1:]))

        self.mtl = mtl

    @classmethod
    def load_texture(cls, image):
        surf = pygame.image.load(image)
        image = pygame.image.tostring(surf, 'RGBA', 1)
        ix, iy = surf.get_rect().size
        texid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texid)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
        return texid

    def render(self):
        glCallList(self.gl_list)

    def free(self):
        glDeleteLists([self.gl_list])