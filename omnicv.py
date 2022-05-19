#!/usr/bin/env/python
import cv2
import numpy as np
import cupy as cp


def rmat(alpha,
         beta,
         gamma):

    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(alpha * np.pi / 180), -np.sin(alpha * np.pi / 180)],
            [0, np.sin(alpha * np.pi / 180), np.cos(alpha * np.pi / 180)],
        ]
    )
    ry = np.array(
        [
            [np.cos(beta * np.pi / 180), 0, np.sin(beta * np.pi / 180)],
            [0, 1, 0],
            [-np.sin(beta * np.pi / 180), 0, np.cos(beta * np.pi / 180)],
        ]
    )
    rz = np.array(
        [
            [np.cos(gamma * np.pi / 180), -np.sin(gamma * np.pi / 180), 0],
            [np.sin(gamma * np.pi / 180), np.cos(gamma * np.pi / 180), 0],
            [0, 0, 1],
        ]
    )

    return np.matmul(rz, np.matmul(ry, rx))


class fisheyeImgConv:
    def __init__(self,
                 param_file_path=None
                 ):
        # print("debugging1")

        self.Hd = None
        self.Wd = None
        self.map_x = None
        self.map_y = None
        self.singleLens = False
        self.filePath = param_file_path

    def fisheye2equirect(self,
                         srcFrame,
                         outShape,
                         aperture=0,
                         delx=0,
                         dely=0,
                         radius=0,
                         edit_mode=False):

        inShape = srcFrame.shape[:2]
        self.Hs = inShape[0]
        self.Ws = inShape[1]
        self.Hd = outShape[0]
        self.Wd = outShape[1]
        self.map_x = np.zeros((self.Hd, self.Wd), np.float32)
        self.map_y = np.zeros((self.Hd, self.Wd), np.float32)
        # print(self.map_x, self.map_y)

        if not radius:
            self.radius = min(inShape)
        else:
            self.radius = radius

        if not aperture:
            self.aperture = 385  # This value is determined using the GUI
        else:
            self.aperture = aperture

        if not edit_mode:
            f = open(self.filePath, "r")
            self.radius = int(f.readline())
            self.aperture = int(f.readline())
            delx = int(f.readline())
            dely = int(f.readline())
            f.close

        self.Cx = (
            self.Ws // 2 - delx
        )  # This value needs to be tuned using the GUI for every new camera
        self.Cy = (
            self.Hs // 2 - dely
        )  # This value needs to be tuned using the GUI for every new camera
        # print(self.Cx, self.Cy)


        i, j = np.meshgrid(np.arange(0, int(self.Hd)),
                           np.arange(0, int(self.Wd)))
        xyz = np.zeros((self.Hd, self.Wd, 3))
        x, y, z = np.split(xyz, 3, axis=-1)

        x = (
            self.radius
            * np.cos((i * 1.0 / self.Hd - 0.5) * np.pi)
            * np.cos((j * 1.0 / self.Hd - 0.5) * np.pi)
        )
        y = (
            self.radius
            * np.cos((i * 1.0 / self.Hd - 0.5) * np.pi)
            * np.sin((j * 1.0 / self.Hd - 0.5) * np.pi)
        )
        z = self.radius * np.sin((i * 1.0 / self.Hd - 0.5) * np.pi)
        # print(x,y,z)

        r = (
            2
            * np.arctan2(np.sqrt(x ** 2 + z ** 2), y)
            / np.pi
            * 180
            / self.aperture
            * self.radius
        )
        theta = np.arctan2(z, x)

        self.map_x = np.multiply(
            r, np.cos(theta)).T.astype(np.float32) + self.Cx
        self.map_y = np.multiply(
            r, np.sin(theta)).T.astype(np.float32) + self.Cy
        # print(self.map_x, self.map_y)

        return cv2.remap(
            srcFrame,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    # def fisheye2equirect(self, srcFrame, outShape, aperture=0, delx=0, dely=0, radius=0, edit_mode=False):
    #     inShape = srcFrame.shape[:2]
    #     self.Hs = inShape[0]
    #     self.Ws = inShape[1]
    #     self.Hd = outShape[0]
    #     self.Wd = outShape[1]
    #     self.map_x = np.zeros((self.Hd, self.Wd), np.float32)
    #     self.map_y = np.zeros((self.Hd, self.Wd), np.float32)
    #
    #     # This value needs to be tuned using the GUI for every new camera
    #     self.Cx = self.Ws//2 - delx
    #     # This value needs to be tuned using the GUI for every new camera
    #     self.Cy = self.Hs//2 - dely
    #     # print(self.Cx, self.Cy)
    #
    #     if not radius:
    #         self.radius = min(inShape)
    #     else:
    #         self.radius = radius
    #
    #     if not aperture:
    #         self.aperture = 385  # This value is determined using the GUI
    #     else:
    #         self.aperture = aperture
    #
    #     if not edit_mode:
    #         f = open(self.filePath, "r")
    #         self.radius = int(f.readline())
    #         self.aperture = int(f.readline())
    #         delx = int(f.readline())
    #         dely = int(f.readline())
    #         f.close
    #
    #     i, j = np.meshgrid(np.arange(0, int(self.Hd)),
    #                        np.arange(0, int(self.Wd)))
    #     xyz = np.zeros((self.Hd, self.Wd, 3))
    #     x, y, z = np.split(xyz, 3, axis=-1)
    #
    #     x = self.radius*np.cos((i*1.0/self.Hd - 0.5)*np.pi) * \
    #         np.cos((j*1.0/self.Hd - 0.5)*np.pi)
    #     y = self.radius*np.cos((i*1.0/self.Hd - 0.5)*np.pi) * \
    #         np.sin((j*1.0/self.Hd - 0.5)*np.pi)
    #     z = self.radius * np.sin((i*1.0/self.Hd-0.5)*np.pi)
    #     print(x,y,z)
    #
    #     # indx = np.logical_not(np.arccos(y/np.sqrt(x**2+y**2+z**2))/np.pi*180 > self.aperture/2)
    #     r = 2*np.arctan2(np.sqrt(x**2+z**2), y)/np.pi * \
    #         180/self.aperture*self.radius
    #     theta = np.arctan2(z, x)
    #     # print(theta)
    #
    #     self.map_x = np.multiply(
    #         r, np.cos(theta)).T.astype(np.float32) + self.Cx
    #     self.map_y = np.multiply(
    #         r, np.sin(theta)).T.astype(np.float32) + self.Cy
    #
    #     return cv2.remap(srcFrame, self.map_x, self.map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def equirect2cubemap(self,
                         srcFrame,
                         side=256,
                         modif=False,
                         dice=False
                         ):

        self.dice = dice
        self.side = side

        inShape = srcFrame.shape[:2]
        mesh = np.stack(
            np.meshgrid(
                np.linspace(-0.5, 0.5, num=side, dtype=np.float32),
                -np.linspace(-0.5, 0.5, num=side, dtype=np.float32),
            ),
            -1,
        )

        # Creating a matrix that contains x,y,z values of all 6 faces
        facesXYZ = np.zeros((side, side * 6, 3), np.float32)

        if modif:
            # Front face (z = 0.5)
            facesXYZ[:, 0 * side: 1 * side, [0, 2]] = mesh
            facesXYZ[:, 0 * side: 1 * side, 1] = -0.5

            # Right face (x = 0.5)
            facesXYZ[:, 1 * side: 2 * side, [1, 2]] = np.flip(mesh, axis=1)
            facesXYZ[:, 1 * side: 2 * side, 0] = 0.5

            # Back face (z = -0.5)
            facesXYZ[:, 2 * side: 3 * side, [0, 2]] = mesh
            facesXYZ[:, 2 * side: 3 * side, 1] = 0.5

            # Left face (x = -0.5)
            facesXYZ[:, 3 * side: 4 * side, [1, 2]] = np.flip(mesh, axis=1)
            facesXYZ[:, 3 * side: 4 * side, 0] = -0.5

            # Up face (y = 0.5)
            facesXYZ[:, 4 * side: 5 * side, [0, 1]] = mesh[::-1]
            facesXYZ[:, 4 * side: 5 * side, 2] = 0.5

            # Down face (y = -0.5)
            facesXYZ[:, 5 * side: 6 * side, [0, 1]] = mesh
            facesXYZ[:, 5 * side: 6 * side, 2] = -0.5

        else:
            # Front face (z = 0.5)
            facesXYZ[:, 0 * side: 1 * side, [0, 1]] = mesh
            facesXYZ[:, 0 * side: 1 * side, 2] = 0.5

            # Right face (x = 0.5)
            facesXYZ[:, 1 * side: 2 * side, [2, 1]] = mesh
            facesXYZ[:, 1 * side: 2 * side, 0] = 0.5

            # Back face (z = -0.5)
            facesXYZ[:, 2 * side: 3 * side, [0, 1]] = mesh
            facesXYZ[:, 2 * side: 3 * side, 2] = -0.5

            # Left face (x = -0.5)
            facesXYZ[:, 3 * side: 4 * side, [2, 1]] = mesh
            facesXYZ[:, 3 * side: 4 * side, 0] = -0.5

            # Up face (y = 0.5)
            facesXYZ[:, 4 * side: 5 * side, [0, 2]] = mesh
            facesXYZ[:, 4 * side: 5 * side, 1] = 0.5

            # Down face (y = -0.5)
            facesXYZ[:, 5 * side: 6 * side, [0, 2]] = mesh
            facesXYZ[:, 5 * side: 6 * side, 1] = -0.5

        # Calculating the spherical coordinates phi and theta for given XYZ
        # coordinate of a cube face
        x, y, z = np.split(facesXYZ, 3, axis=-1)
        # phi = tan^-1(x/z)
        phi = np.arctan2(x, z)
        # theta = tan^-1(y/||(x,y)||)
        theta = np.arctan2(y, np.sqrt(x ** 2 + z ** 2))

        h, w = inShape
        # Calculating corresponding coordinate points in
        # the equirectangular image
        eqrec_x = (phi / (2 * np.pi) + 0.5) * w
        eqrec_y = (-theta / np.pi + 0.5) * h
        # Note: we have considered equirectangular image to
        # be mapped to a normalised form and then to the scale of (pi,2pi)

        self.map_x = eqrec_x
        self.map_y = eqrec_y

        dstFrame = cv2.remap(srcFrame,
                             self.map_x,
                             self.map_y,
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)

        if self.dice:
            line1 = np.hstack(
                (
                    dstFrame[:, 4 * side: 5 * side, :] * 0,
                    cv2.flip(dstFrame[:, 4 * side: 5 * side, :], 0),
                    dstFrame[:, 4 * side: 5 * side, :] * 0,
                    dstFrame[:, 4 * side: 5 * side, :] * 0,
                )
            )
            line2 = np.hstack(
                (
                    dstFrame[:, 3 * side: 4 * side, :],
                    dstFrame[:, 0 * side: 1 * side, :],
                    cv2.flip(dstFrame[:, 1 * side: 2 * side, :], 1),
                    cv2.flip(dstFrame[:, 2 * side: 3 * side, :], 1),
                )
            )
            line3 = np.hstack(
                (
                    dstFrame[:, 5 * side: 6 * side, :] * 0,
                    dstFrame[:, 5 * side: 6 * side, :],
                    dstFrame[:, 5 * side: 6 * side, :] * 0,
                    dstFrame[:, 5 * side: 6 * side, :] * 0,
                )
            )
            dstFrame = np.vstack((line1, line2, line3))

        return dstFrame

    def cubemap2equirect(self,
                         srcFrame,
                         outShape
                         ):

        h, w = srcFrame.shape[:2]

        if h / w == 3 / 4:
            l1, l2, l3 = np.split(srcFrame, 3, axis=0)
            _, pY, _, _ = np.split(l1, 4, axis=1)
            nX, pZ, pX, nZ = np.split(l2, 4, axis=1)
            _, nY, _, _ = np.split(l3, 4, axis=1)

            srcFrame = np.hstack(
                (pZ, cv2.flip(pX, 1), cv2.flip(nZ, 1), nX, cv2.flip(pY, 0), nY)
            )

        inShape = srcFrame.shape[:2]
        self.Hd = outShape[0]
        self.Wd = outShape[1]
        h = self.Hd
        w = self.Wd
        face_w = inShape[0]

        phi = np.linspace(-np.pi, np.pi, num=self.Wd, dtype=np.float32)
        theta = np.linspace(np.pi, -np.pi, num=self.Hd, dtype=np.float32) / 2

        phi, theta = np.meshgrid(phi, theta)

        tp = np.zeros((h, w), dtype=np.int32)
        tp[:, : w // 8] = 2
        tp[:, w // 8: 3 * w // 8] = 3
        tp[:, 3 * w // 8: 5 * w // 8] = 0
        tp[:, 5 * w // 8: 7 * w // 8] = 1
        tp[:, 7 * w // 8:] = 2

        # Prepare ceil mask
        mask = np.zeros((h, w // 4), np.bool)
        idx = np.linspace(-np.pi, np.pi, w // 4) / 4
        idx = h // 2 - np.round(np.arctan(np.cos(idx)) * h / np.pi).astype(int)
        for i, j in enumerate(idx):
            mask[:j, i] = 1

        mask = np.roll(mask, w // 8, 1)

        mask = np.concatenate([mask] * 4, 1)

        tp[mask] = 4
        tp[np.flip(mask, 0)] = 5

        tp = tp.astype(np.int32)

        coor_x = np.zeros((h, w))
        coor_y = np.zeros((h, w))

        for i in range(4):
            mask = tp == i
            coor_x[mask] = 0.5 * np.tan(phi[mask] - np.pi * i / 2)
            coor_y[mask] = (
                -0.5 * np.tan(theta[mask]) / np.cos(phi[mask] - np.pi * i / 2)
            )

        mask = tp == 4
        c = 0.5 * np.tan(np.pi / 2 - theta[mask])
        coor_x[mask] = c * np.sin(phi[mask])
        coor_y[mask] = c * np.cos(phi[mask])

        mask = tp == 5
        c = 0.5 * np.tan(np.pi / 2 - np.abs(theta[mask]))
        coor_x[mask] = c * np.sin(phi[mask])
        coor_y[mask] = -c * np.cos(phi[mask])

        # Final renormalize
        coor_x = (np.clip(coor_x, -0.5, 0.5) + 0.5) * face_w
        coor_y = (np.clip(coor_y, -0.5, 0.5) + 0.5) * face_w

        self.map_x = coor_x.astype(np.float32)
        self.map_y = coor_y.astype(np.float32)

        dstFrame = 0
        cube_faces = np.stack(np.split(srcFrame, 6, 1), 0)
        cube_faces[1] = np.flip(cube_faces[1], 1)
        cube_faces[2] = np.flip(cube_faces[2], 1)
        cube_faces[4] = np.flip(cube_faces[4], 0)
        self.tp = tp
        for i in range(6):
            mask = self.tp == i
            dstFrame1 = cv2.remap(
                cube_faces[i],
                self.map_x,
                self.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            # We use this border mode to avoid small black lines

            dstFrame += cv2.bitwise_and(
                dstFrame1, dstFrame1, mask=mask.astype(np.uint8)
            )

        return dstFrame

    def equirect2persp(self,
                       img,
                       FOV,
                       THETA,
                       PHI,
                       Hd,
                       Wd
                       ):

        # THETA is left/right angle, PHI is up/down angle, both in degree
        equ_h, equ_w = img.shape[:2]

        equ_cx = (equ_w) / 2.0
        equ_cy = (equ_h) / 2.0

        wFOV = FOV
        hFOV = float(Hd) / Wd * wFOV

        c_x = (Wd) / 2.0
        c_y = (Hd) / 2.0

        w_len = 2 * np.tan(np.radians(wFOV / 2.0))
        w_interval = w_len / (Wd)

        h_len = 2 * np.tan(np.radians(hFOV / 2.0))
        h_interval = h_len / (Hd)

        x_map = np.zeros([Hd, Wd], np.float32) + 1
        y_map = np.tile((np.arange(0, Wd) - c_x) * w_interval, [Hd, 1])
        z_map = -np.tile((np.arange(0, Hd) - c_y) * h_interval, [Wd, 1]).T
        D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)

        xyz = np.zeros([Hd, Wd, 3], np.float)
        xyz[:, :, 0] = (x_map / D)[:, :]
        xyz[:, :, 1] = (y_map / D)[:, :]
        xyz[:, :, 2] = (z_map / D)[:, :]

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([Hd * Wd, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2] / 1)
        lon = np.zeros([Hd * Wd], np.float)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(np.bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)

        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([Hd, Wd]) / np.pi * 180
        lat = -lat.reshape([Hd, Wd]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        self.map_x = lon.astype(np.float32)
        self.map_y = lat.astype(np.float32)

        persp = cv2.remap(img,
                          lon.astype(np.float32),
                          lat.astype(np.float32),
                          cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_WRAP)

        return persp

    def cubemap2persp(self,
                      img,
                      FOV,
                      THETA,
                      PHI,
                      Hd,
                      Wd
                      ):

        # THETA is left/right angle, PHI is up/down angle, both in degree

        img = self.cubemap2equirect(img, [2 * Hd, 4 * Hd])

        equ_h, equ_w = img.shape[:2]

        equ_cx = (equ_w) / 2.0
        equ_cy = (equ_h) / 2.0

        wFOV = FOV
        hFOV = float(Hd) / Wd * wFOV

        c_x = (Wd) / 2.0
        c_y = (Hd) / 2.0

        w_len = 2 * 1 * np.sin(
            np.radians(wFOV / 2.0)) / np.cos(np.radians(wFOV / 2.0))
        w_interval = w_len / (Wd)

        h_len = 2 * 1 * np.sin(
            np.radians(hFOV / 2.0)) / np.cos(np.radians(hFOV / 2.0))
        h_interval = h_len / (Hd)

        x_map = np.zeros([Hd, Wd], np.float32) + 1
        y_map = np.tile((np.arange(0, Wd) - c_x) * w_interval, [Hd, 1])
        z_map = -np.tile((np.arange(0, Hd) - c_y) * h_interval, [Wd, 1]).T
        D = np.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = np.zeros([Hd, Wd, 3], np.float)
        xyz[:, :, 0] = (1 / D * x_map)[:, :]
        xyz[:, :, 1] = (1 / D * y_map)[:, :]
        xyz[:, :, 2] = (1 / D * z_map)[:, :]

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

        xyz = xyz.reshape([Hd * Wd, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2] / 1)
        lon = np.zeros([Hd * Wd], np.float)
        theta = np.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(np.bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(np.bool)

        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([Hd, Wd]) / np.pi * 180
        lat = -lat.reshape([Hd, Wd]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        self.map_x = lon.astype(np.float32)
        self.map_y = lat.astype(np.float32)

        persp = cv2.remap(
            img,
            lon.astype(np.float32),
            lat.astype(np.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

        return persp

    def equirect2Fisheye(self,
                         img,
                         outShape,
                         f=50,
                         xi=1.2,
                         angles=[0, 0, 0]
                         ):

        self.Hd = outShape[0]
        self.Wd = outShape[1]
        self.f = f
        self.xi = xi

        Hs, Ws = img.shape[:2]

        self.Cx = self.Wd / 2.0
        self.Cy = self.Hd / 2.0

        x = np.linspace(0, self.Wd - 1, num=self.Wd, dtype=np.float32)
        y = np.linspace(0, self.Hd - 1, num=self.Hd, dtype=np.float32)

        x, y = np.meshgrid(range(self.Wd), range(self.Hd))
        xref = 1
        yref = 1

        self.fmin = (
            np.sqrt(
                -(1 - self.xi ** 2) * ((xref - self.Cx) ** 2 +
                                       (yref - self.Cy) ** 2)
            )
            * 1.0001
        )

        x_hat = (x - self.Cx) / self.f
        y_hat = (y - self.Cy) / self.f

        x2_y2_hat = x_hat ** 2 + y_hat ** 2

        omega = np.real(
            self.xi + np.lib.scimath.sqrt(1 + (1 - self.xi ** 2) * x2_y2_hat)
        ) / (x2_y2_hat + 1)
        # print(np.max(x2_y2_hat))

        Ps_x = omega * x_hat
        Ps_y = omega * y_hat
        Ps_z = omega - self.xi

        self.alpha = angles[0]
        self.beta = angles[1]
        self.gamma = angles[2]

        R = np.matmul(
            rmat(self.alpha, self.beta, self.gamma),
            np.matmul(rmat(0, -90, 45), rmat(0, 90, 90)),
        )

        Ps = np.stack((Ps_x, Ps_y, Ps_z), -1)
        Ps = np.matmul(Ps, R.T)

        Ps_x, Ps_y, Ps_z = np.split(Ps, 3, axis=-1)
        Ps_x = Ps_x[:, :, 0]
        Ps_y = Ps_y[:, :, 0]
        Ps_z = Ps_z[:, :, 0]

        theta = np.arctan2(Ps_y, Ps_x)
        phi = np.arctan2(Ps_z, np.sqrt(Ps_x ** 2 + Ps_y ** 2))

        a = 2 * np.pi / (Ws - 1)
        b = np.pi - a * (Ws - 1)
        self.map_x = (1.0 / a) * (theta - b)

        a = -np.pi / (Hs - 1)
        b = np.pi / 2
        self.map_y = (1.0 / a) * (phi - b)

        output = cv2.remap(
            img,
            self.map_x.astype(np.float32),
            self.map_y.astype(np.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

        if self.f < self.fmin:
            r = np.sqrt(-(self.f ** 2) / (1 - self.xi ** 2))
            mask = np.zeros_like(output[:, :, 0])
            mask = cv2.circle(
                mask, (int(self.Cx), int(self.Cy)), int(r), (255, 255, 255), -1
            )
            output = cv2.bitwise_and(output, output, mask=mask)

        return output

    def equirect2Fisheye_UCM(self,
                             img,
                             outShape,
                             f=50,
                             xi=1.2,
                             angles=[0, 0, 0]
                             ):

        self.Hd = outShape[0]
        self.Wd = outShape[1]
        self.f = f
        self.xi = xi

        Hs, Ws = img.shape[:2]

        self.Cx = self.Wd / 2.0
        self.Cy = self.Hd / 2.0

        x = np.linspace(0, self.Wd - 1, num=self.Wd, dtype=np.float32)
        y = np.linspace(0, self.Hd - 1, num=self.Hd, dtype=np.float32)

        x, y = np.meshgrid(range(self.Wd), range(self.Hd))
        xref = 1
        yref = 1

        self.fmin = (
            np.lib.scimath.sqrt(
                -(1 - self.xi ** 2) *
                 ((xref - self.Cx) ** 2 + (yref - self.Cy) ** 2)
            )
            * 1.0001
        )

        if self.xi ** 2 >= 1:
            self.fmin = np.real(self.fmin)
        else:
            self.fmin = np.imag(self.fmin)

        x_hat = (x - self.Cx) / self.f
        y_hat = (y - self.Cy) / self.f

        x2_y2_hat = x_hat ** 2 + y_hat ** 2

        omega = np.real(
            self.xi + np.lib.scimath.sqrt(1 + (1 - self.xi ** 2) * x2_y2_hat)
        ) / (x2_y2_hat + 1)

        Ps_x = omega * x_hat
        Ps_y = omega * y_hat
        Ps_z = omega - self.xi

        self.alpha = angles[0]
        self.beta = angles[1]
        self.gamma = angles[2]

        R = np.matmul(
            rmat(self.alpha, self.beta, self.gamma),
            np.matmul(rmat(0, -90, 45), rmat(0, 90, 90)),
        )

        Ps = np.stack((Ps_x, Ps_y, Ps_z), -1)
        Ps = np.matmul(Ps, R.T)

        Ps_x, Ps_y, Ps_z = np.split(Ps, 3, axis=-1)
        Ps_x = Ps_x[:, :, 0]
        Ps_y = Ps_y[:, :, 0]
        Ps_z = Ps_z[:, :, 0]

        theta = np.arctan2(Ps_y, Ps_x)
        phi = np.arctan2(Ps_z, np.sqrt(Ps_x ** 2 + Ps_y ** 2))

        a = 2 * np.pi / (Ws - 1)
        b = np.pi - a * (Ws - 1)
        self.map_x = (1.0 / a) * (theta - b)

        a = -np.pi / (Hs - 1)
        b = np.pi / 2
        self.map_y = (1.0 / a) * (phi - b)

        output = cv2.remap(
            img,
            self.map_x.astype(np.float32),
            self.map_y.astype(np.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

        if self.f < self.fmin:
            r = np.sqrt(np.abs(-(self.f ** 2) / (1 - self.xi ** 2)))
            mask = np.zeros_like(output[:, :, 0])
            mask = cv2.circle(
                mask, (int(self.Cx), int(self.Cy)), int(r), (255, 255, 255), -1
            )
            output = cv2.bitwise_and(output, output, mask=mask)

        return output

    def equirect2Fisheye_EUCM(self,
                              img,
                              outShape,
                              f=50,
                              a_=0.5,
                              b_=0.5,
                              angles=[0, 0, 0]
                              ):

        self.Hd = outShape[0]
        self.Wd = outShape[1]
        self.f = f
        self.a_ = a_
        self.b_ = b_

        Hs, Ws = img.shape[:2]

        self.Cx = self.Wd / 2.0
        self.Cy = self.Hd / 2.0

        x = np.linspace(0, self.Wd - 1, num=self.Wd, dtype=np.float32)
        y = np.linspace(0, self.Hd - 1, num=self.Hd, dtype=np.float32)

        x, y = np.meshgrid(range(self.Wd), range(self.Hd))
        xref = 1
        yref = 1

        self.fmin = (
            np.lib.scimath.sqrt(
                self.b_
                * (2 * self.a_ - 1)
                * ((xref - self.Cx) ** 2 + (yref - self.Cy) ** 2)
            )
            * 1.0001
        )
        # print(self.fmin)
        if np.real(self.fmin) <= 0:
            self.fmin = np.imag(self.fmin)

        # print(self.f)
        # print(self.fmin)

        mx = (x - self.Cx) / self.f
        my = (y - self.Cy) / self.f

        r_2 = mx ** 2 + my ** 2

        mz = np.real(
            (1 - self.b_ * self.a_ * self.a_ * r_2)
            / (
                self.a_ * np.lib.scimath.sqrt(1 - (2 * self.a_ - 1) *
                                              self.b_ * r_2)
                + (1 - self.a_)
            )
        )

        coef = 1 / np.sqrt(mx ** 2 + my ** 2 + mz ** 2)

        Ps_x = mx * coef
        Ps_y = my * coef
        Ps_z = mz * coef

        self.alpha = angles[0]
        self.beta = angles[1]
        self.gamma = angles[2]

        R = np.matmul(
            rmat(self.alpha, self.beta, self.gamma),
            np.matmul(rmat(0, -90, 45), rmat(0, 90, 90)),
        )

        Ps = np.stack((Ps_x, Ps_y, Ps_z), -1)
        Ps = np.matmul(Ps, R.T)

        Ps_x, Ps_y, Ps_z = np.split(Ps, 3, axis=-1)
        Ps_x = Ps_x[:, :, 0]
        Ps_y = Ps_y[:, :, 0]
        Ps_z = Ps_z[:, :, 0]

        theta = np.arctan2(Ps_y, Ps_x)
        phi = np.arctan2(Ps_z, np.sqrt(Ps_x ** 2 + Ps_y ** 2))

        a = 2 * np.pi / (Ws - 1)
        b = np.pi - a * (Ws - 1)
        self.map_x = (1.0 / a) * (theta - b)

        a = -np.pi / (Hs - 1)
        b = np.pi / 2
        self.map_y = (1.0 / a) * (phi - b)

        output = cv2.remap(
            img,
            self.map_x.astype(np.float32),
            self.map_y.astype(np.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

        if self.f < self.fmin:
            r = np.sqrt(np.abs((self.f ** 2) / (self.b_ * (2 * self.a_ - 1))))
            mask = np.zeros_like(output[:, :, 0])
            mask = cv2.circle(
                mask, (int(self.Cx), int(self.Cy)), int(r), (255, 255, 255), -1
            )
            output = cv2.bitwise_and(output, output, mask=mask)

        return output

    def equirect2Fisheye_FOV(self,
                             img,
                             outShape,
                             f=50,
                             w_=0.5,
                             angles=[0, 0, 0]
                             ):

        self.Hd = outShape[0]
        self.Wd = outShape[1]
        self.f = f
        self.w_ = w_

        Hs, Ws = img.shape[:2]

        self.Cx = self.Wd / 2.0
        self.Cy = self.Hd / 2.0

        x = np.linspace(0, self.Wd - 1, num=self.Wd, dtype=np.float32)
        y = np.linspace(0, self.Hd - 1, num=self.Hd, dtype=np.float32)

        x, y = np.meshgrid(range(self.Wd), range(self.Hd))

        mx = (x - self.Cx) / self.f
        my = (y - self.Cy) / self.f

        rd = np.sqrt(mx ** 2 + my ** 2)

        Ps_x = mx * np.sin(rd * self.w_) / (2 * rd * np.tan(self.w_ / 2))
        Ps_y = my * np.sin(rd * self.w_) / (2 * rd * np.tan(self.w_ / 2))
        Ps_z = np.cos(rd * self.w_)

        self.alpha = angles[0]
        self.beta = angles[1]
        self.gamma = angles[2]

        R = np.matmul(
            rmat(self.alpha, self.beta, self.gamma),
            np.matmul(rmat(0, -90, 45), rmat(0, 90, 90)),
        )

        Ps = np.stack((Ps_x, Ps_y, Ps_z), -1)
        Ps = np.matmul(Ps, R.T)

        Ps_x, Ps_y, Ps_z = np.split(Ps, 3, axis=-1)
        Ps_x = Ps_x[:, :, 0]
        Ps_y = Ps_y[:, :, 0]
        Ps_z = Ps_z[:, :, 0]

        theta = np.arctan2(Ps_y, Ps_x)
        phi = np.arctan2(Ps_z, np.sqrt(Ps_x ** 2 + Ps_y ** 2))

        a = 2 * np.pi / (Ws - 1)
        b = np.pi - a * (Ws - 1)
        self.map_x = (1.0 / a) * (theta - b)

        a = -np.pi / (Hs - 1)
        b = np.pi / 2
        self.map_y = (1.0 / a) * (phi - b)

        output = cv2.remap(
            img,
            self.map_x.astype(np.float32),
            self.map_y.astype(np.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

        return output

    def equirect2Fisheye_DS(self,
                            img,
                            outShape,
                            f=50,
                            a_=0.5,
                            xi_=0.5,
                            angles=[0, 0, 0]
                            ):

        self.Hd = outShape[0]
        self.Wd = outShape[1]
        self.f = f
        self.a_ = a_
        self.xi_ = xi_

        Hs, Ws = img.shape[:2]

        self.Cx = self.Wd / 2.0
        self.Cy = self.Hd / 2.0

        x = np.linspace(0, self.Wd - 1, num=self.Wd, dtype=np.float32)
        y = np.linspace(0, self.Hd - 1, num=self.Hd, dtype=np.float32)

        x, y = np.meshgrid(range(self.Wd), range(self.Hd))
        xref = 1
        yref = 1

        self.fmin = np.sqrt(np.abs((2 * self.a_ - 1) *
                            ((xref - self.Cx) ** 2 + (yref - self.Cy) ** 2))
                            )

        mx = (x - self.Cx) / self.f
        my = (y - self.Cy) / self.f

        r_2 = mx ** 2 + my ** 2

        mz = np.real(
            (1 - self.a_ * self.a_ * r_2)
            / (self.a_ * np.lib.scimath.sqrt(1 - (2 * self.a_ - 1) * r_2) +
               1 - self.a_)
        )

        omega = np.real(
            (mz * self.xi_ + np.lib.scimath.sqrt(mz ** 2 +
                                                 (1 - self.xi_ ** 2) * r_2))
            / (mz ** 2 + r_2)
        )

        Ps_x = omega * mx
        Ps_y = omega * my
        Ps_z = omega * mz - self.xi_

        self.alpha = angles[0]
        self.beta = angles[1]
        self.gamma = angles[2]

        R = np.matmul(
            rmat(self.alpha, self.beta, self.gamma),
            np.matmul(rmat(0, -90, 45), rmat(0, 90, 90)),
        )

        Ps = np.stack((Ps_x, Ps_y, Ps_z), -1)
        Ps = np.matmul(Ps, R.T)

        Ps_x, Ps_y, Ps_z = np.split(Ps, 3, axis=-1)
        Ps_x = Ps_x[:, :, 0]
        Ps_y = Ps_y[:, :, 0]
        Ps_z = Ps_z[:, :, 0]

        theta = np.arctan2(Ps_y, Ps_x)
        phi = np.arctan2(Ps_z, np.sqrt(Ps_x ** 2 + Ps_y ** 2))

        a = 2 * np.pi / (Ws - 1)
        b = np.pi - a * (Ws - 1)
        self.map_x = (1.0 / a) * (theta - b)

        a = -np.pi / (Hs - 1)
        b = np.pi / 2
        self.map_y = (1.0 / a) * (phi - b)

        output = cv2.remap(
            img,
            self.map_x.astype(np.float32),
            self.map_y.astype(np.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

        if self.f < self.fmin:
            r = np.sqrt(np.abs((self.f ** 2) / (2 * self.a_ - 1)))
            mask = np.zeros_like(output[:, :, 0])
            mask = cv2.circle(
                mask, (int(self.Cx), int(self.Cy)), int(r), (255, 255, 255), -1
            )
            output = cv2.bitwise_and(output, output, mask=mask)

        return output

    def applyMap(self,
                 map,
                 srcFrame
                 ):

        if map == 0:
            return cv2.remap(
                srcFrame,
                self.map_x,
                self.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        if map == 1:
            dstFrame = cv2.remap(
                srcFrame,
                self.map_x,
                self.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

            if self.dice:
                line1 = np.hstack(
                    (
                        dstFrame[:, 4 * self.side: 5 * self.side, :] * 0,
                        cv2.flip(dstFrame[:, 4 *
                                          self.side: 5 * self.side, :], 0),
                        dstFrame[:, 4 * self.side: 5 * self.side, :] * 0,
                        dstFrame[:, 4 * self.side: 5 * self.side, :] * 0,
                    )
                )
                line2 = np.hstack(
                    (
                        dstFrame[:, 3 * self.side: 4 * self.side, :],
                        dstFrame[:, 0 * self.side: 1 * self.side, :],
                        cv2.flip(dstFrame[:, 1 *
                                          self.side: 2 * self.side, :], 1),
                        cv2.flip(dstFrame[:, 2 *
                                          self.side: 3 * self.side, :], 1),
                    )
                )
                line3 = np.hstack(
                    (
                        dstFrame[:, 5 * self.side: 6 * self.side, :] * 0,
                        dstFrame[:, 5 * self.side: 6 * self.side, :],
                        dstFrame[:, 5 * self.side: 6 * self.side, :] * 0,
                        dstFrame[:, 5 * self.side: 6 * self.side, :] * 0,
                    )
                )
                dstFrame = np.vstack((line1, line2, line3))
            return dstFrame

        if map == 2:
            h, w = srcFrame.shape[:2]
            if h / w == 3 / 4:
                l1, l2, l3 = np.split(srcFrame, 3, axis=0)
                _, pY, _, _ = np.split(l1, 4, axis=1)
                nX, pZ, pX, nZ = np.split(l2, 4, axis=1)
                _, nY, _, _ = np.split(l3, 4, axis=1)
                srcFrame = np.hstack(
                    (pZ, cv2.flip(pX, 1), cv2.flip(nZ, 1),
                     nX, cv2.flip(pY, 0), nY)
                )

            dstFrame = 0
            cube_faces = np.stack(np.split(srcFrame, 6, 1), 0)
            cube_faces[1] = np.flip(cube_faces[1], 1)
            cube_faces[2] = np.flip(cube_faces[2], 1)
            cube_faces[4] = np.flip(cube_faces[4], 0)

            for i in range(6):
                mask = self.tp == i
                dstFrame1 = cv2.remap(
                    cube_faces[i],
                    self.map_x,
                    self.map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                # We use this border mode to avoid small black lines

                dstFrame += cv2.bitwise_and(
                    dstFrame1, dstFrame1, mask=mask.astype(np.uint8)
                )

            return dstFrame

        if map == 3:

            dstFrame = cv2.remap(
                srcFrame,
                self.map_x.astype(np.float32),
                self.map_y.astype(np.float32),
                cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_WRAP,
            )

            if self.f < self.fmin:
                r = np.sqrt(-(self.f ** 2) / (1 - self.xi ** 2))
                mask = np.zeros_like(dstFrame[:, :, 0])
                mask = cv2.circle(
                    mask, (int(self.Cx), int(self.Cy)),
                    int(r), (255, 255, 255), -1
                )
                dstFrame = cv2.bitwise_and(dstFrame, dstFrame, mask=mask)

            return dstFrame

        if map == 4:
            return cv2.remap(
                srcFrame,
                self.map_x,
                self.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )

        else:
            return print("WRONG MAP ENTERED")

    # CUDA

    def rmat_cuda(alpha,
                  beta,
                  gamma):

        rx = cp.array(
            [
                [1, 0, 0],
                [0, cp.cos(alpha * cp.pi / 180), -cp.sin(alpha * cp.pi / 180)],
                [0, cp.sin(alpha * cp.pi / 180), cp.cos(alpha * cp.pi / 180)],
            ]
        )
        ry = cp.array(
            [
                [cp.cos(beta * cp.pi / 180), 0, cp.sin(beta * cp.pi / 180)],
                [0, 1, 0],
                [-cp.sin(beta * cp.pi / 180), 0, cp.cos(beta * cp.pi / 180)],
            ]
        )
        rz = cp.array(
            [
                [cp.cos(gamma * cp.pi / 180), -cp.sin(gamma * cp.pi / 180), 0],
                [cp.sin(gamma * cp.pi / 180), cp.cos(gamma * cp.pi / 180), 0],
                [0, 0, 1],
            ]
        )

        return cp.asnumpy(cp.matmul(rz, cp.matmul(ry, rx)))

    def cubemap2equirect_cuda(self,
                              srcFrame,
                              outShape
                              ):

        srcFrame_cp = cp.asarray(srcFrame)
        h, w = srcFrame_cp.shape[:2]

        if h / w == 3 / 4:
            l1, l2, l3 = cp.split(srcFrame_cp, 3, axis=0)
            _, pY, _, _ = cp.split(l1, 4, axis=1)
            nX, pZ, pX, nZ = cp.split(l2, 4, axis=1)
            _, nY, _, _ = cp.split(l3, 4, axis=1)

            srcFrame_cp = cp.hstack(
                (pZ, cv2.flip(pX, 1), cv2.flip(nZ, 1), nX, cv2.flip(pY, 0), nY)
            )

        inShape = srcFrame_cp.shape[:2]
        self.Hd = outShape[0]
        self.Wd = outShape[1]
        h = self.Hd
        w = self.Wd
        face_w = inShape[0]

        phi = cp.linspace(-cp.pi, cp.pi, num=self.Wd, dtype=cp.float32)
        theta = cp.linspace(cp.pi, -cp.pi, num=self.Hd, dtype=cp.float32) / 2

        phi, theta = cp.meshgrid(phi, theta)

        tp = cp.zeros((h, w), dtype=cp.int32)
        tp[:, : w // 8] = 2
        tp[:, w // 8: 3 * w // 8] = 3
        tp[:, 3 * w // 8: 5 * w // 8] = 0
        tp[:, 5 * w // 8: 7 * w // 8] = 1
        tp[:, 7 * w // 8:] = 2

        # Prepare ceil mask
        mask = cp.zeros((h, w // 4), cp.bool)
        idx = cp.linspace(-cp.pi, cp.pi, w // 4) / 4
        idx = h // 2 - cp.round(cp.arctan(cp.cos(idx)) * h / cp.pi).astype(int)
        for i, j in enumerate(idx):
            mask[:j, i] = 1

        mask = cp.roll(mask, w // 8, 1)

        mask = cp.concatenate([mask] * 4, 1)

        tp[mask] = 4
        tp[cp.flip(mask, 0)] = 5

        tp = tp.astype(cp.int32)

        coor_x = cp.zeros((h, w))
        coor_y = cp.zeros((h, w))

        for i in range(4):
            mask = tp == i
            coor_x[mask] = 0.5 * cp.tan(phi[mask] - cp.pi * i / 2)
            coor_y[mask] = (
                -0.5 * cp.tan(theta[mask]) / cp.cos(phi[mask] - cp.pi * i / 2)
            )

        mask = tp == 4
        c = 0.5 * cp.tan(cp.pi / 2 - theta[mask])
        coor_x[mask] = c * cp.sin(phi[mask])
        coor_y[mask] = c * cp.cos(phi[mask])

        mask = tp == 5
        c = 0.5 * cp.tan(cp.pi / 2 - cp.abs(theta[mask]))
        coor_x[mask] = c * cp.sin(phi[mask])
        coor_y[mask] = -c * cp.cos(phi[mask])

        # Final renormalize
        coor_x = (cp.clip(coor_x, -0.5, 0.5) + 0.5) * face_w
        coor_y = (cp.clip(coor_y, -0.5, 0.5) + 0.5) * face_w

        map_x_np = cp.asnumpy(coor_x.astype(cp.float32))
        map_y_np = cp.asnumpy(coor_y.astype(cp.float32))
        # self.map_x = cv2.cuda_GpuMat(map_x_np)
        # self.map_y = cv2.cuda_GpuMat(map_y_np)

        dstFrame = 0
        cube_faces_cp = cp.stack(cp.split(srcFrame_cp, 6, 1), 0)
        cube_faces_cp[1] = cp.flip(cube_faces_cp[1], 1)
        cube_faces_cp[2] = cp.flip(cube_faces_cp[2], 1)
        cube_faces_cp[4] = cp.flip(cube_faces_cp[4], 0)
        self.tp = cp.asnumpy(tp)
        for i in range(6):
            mask = self.tp == i
            mask = mask.astype(np.uint8)
            cube_faces = cp.asnumpy(cube_faces_cp[i])
            dstFrame1 = cv2.remap(
                cube_faces,
                map_x_np,
                map_y_np,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            # We use this border mode to avoid small black lines

            dstFrame += cv2.bitwise_and(dstFrame1, dstFrame1, mask=mask)

        return dstFrame

    def equirect2cubemap_cuda(self,
                              srcFrame,
                              side=256,
                              modif=False,
                              dice=False
                              ):

        self.dice = dice
        self.side = side

        srcFrame_cuda = cv2.cuda_GpuMat(srcFrame)
        inShape = srcFrame.shape[:2]
        mesh = cp.stack(
            cp.meshgrid(
                cp.linspace(-0.5, 0.5, num=side, dtype=cp.float32),
                -cp.linspace(-0.5, 0.5, num=side, dtype=cp.float32),
            ),
            -1,
        )

        # Creating a matrix that contains x,y,z values of all 6 faces
        facesXYZ = cp.zeros((side, side * 6, 3), cp.float32)

        if modif:
            # Front face (z = 0.5)
            facesXYZ[:, 0 * side: 1 * side, [0, 2]] = mesh
            facesXYZ[:, 0 * side: 1 * side, 1] = -0.5

            # Right face (x = 0.5)
            facesXYZ[:, 1 * side: 2 * side, [1, 2]] = cp.flip(mesh, axis=1)
            facesXYZ[:, 1 * side: 2 * side, 0] = 0.5

            # Back face (z = -0.5)
            facesXYZ[:, 2 * side: 3 * side, [0, 2]] = mesh
            facesXYZ[:, 2 * side: 3 * side, 1] = 0.5

            # Left face (x = -0.5)
            facesXYZ[:, 3 * side: 4 * side, [1, 2]] = cp.flip(mesh, axis=1)
            facesXYZ[:, 3 * side: 4 * side, 0] = -0.5

            # Up face (y = 0.5)
            facesXYZ[:, 4 * side: 5 * side, [0, 1]] = mesh[::-1]
            facesXYZ[:, 4 * side: 5 * side, 2] = 0.5

            # Down face (y = -0.5)
            facesXYZ[:, 5 * side: 6 * side, [0, 1]] = mesh
            facesXYZ[:, 5 * side: 6 * side, 2] = -0.5

        else:
            # Front face (z = 0.5)
            facesXYZ[:, 0 * side: 1 * side, [0, 1]] = mesh
            facesXYZ[:, 0 * side: 1 * side, 2] = 0.5

            # Right face (x = 0.5)
            facesXYZ[:, 1 * side: 2 * side, [2, 1]] = mesh
            facesXYZ[:, 1 * side: 2 * side, 0] = 0.5

            # Back face (z = -0.5)
            facesXYZ[:, 2 * side: 3 * side, [0, 1]] = mesh
            facesXYZ[:, 2 * side: 3 * side, 2] = -0.5

            # Left face (x = -0.5)
            facesXYZ[:, 3 * side: 4 * side, [2, 1]] = mesh
            facesXYZ[:, 3 * side: 4 * side, 0] = -0.5

            # Up face (y = 0.5)
            facesXYZ[:, 4 * side: 5 * side, [0, 2]] = mesh
            facesXYZ[:, 4 * side: 5 * side, 1] = 0.5

            # Down face (y = -0.5)
            facesXYZ[:, 5 * side: 6 * side, [0, 2]] = mesh
            facesXYZ[:, 5 * side: 6 * side, 1] = -0.5

        # Calculating the spherical coordinates phi and theta for given XYZ
        # coordinate of a cube face
        x, y, z = cp.split(facesXYZ, 3, axis=-1)
        # phi = tan^-1(x/z)
        phi = cp.arctan2(x, z)
        # theta = tan^-1(y/||(x,y)||)
        theta = cp.arctan2(y, cp.sqrt(x ** 2 + z ** 2))

        h, w = inShape
        # Calculating corresponding coordinate points in
        # the equirectangular image
        eqrec_x = (phi / (2 * cp.pi) + 0.5) * w
        eqrec_y = (-theta / cp.pi + 0.5) * h
        # Note: we have considered equirectangular image to
        # be mapped to a normalised form and then to the scale of (pi,2pi)

        self.map_x = cv2.cuda_GpuMat(cp.asnumpy(eqrec_x))
        self.map_y = cv2.cuda_GpuMat(cp.asnumpy(eqrec_y))

        dstFrame_cuda = cv2.cuda.remap(srcFrame_cuda,
                                       self.map_x,
                                       self.map_y,
                                       interpolation=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT)

        dstFrame_np = dstFrame_cuda.download()

        dstFrame_cp = cp.asarray(dstFrame_np)

        if self.dice:
            line1 = cp.hstack(
                (
                    dstFrame_cp[:, 4 * side: 5 * side, :] * 0,
                    cv2.flip(dstFrame_cp[:, 4 * side: 5 * side, :], 0),
                    dstFrame_cp[:, 4 * side: 5 * side, :] * 0,
                    dstFrame_cp[:, 4 * side: 5 * side, :] * 0,
                )
            )
            line2 = cp.hstack(
                (
                    dstFrame_cp[:, 3 * side: 4 * side, :],
                    dstFrame_cp[:, 0 * side: 1 * side, :],
                    cv2.flip(dstFrame_cp[:, 1 * side: 2 * side, :], 1),
                    cv2.flip(dstFrame_cp[:, 2 * side: 3 * side, :], 1),
                )
            )
            line3 = cp.hstack(
                (
                    dstFrame_cp[:, 5 * side: 6 * side, :] * 0,
                    dstFrame_cp[:, 5 * side: 6 * side, :],
                    dstFrame_cp[:, 5 * side: 6 * side, :] * 0,
                    dstFrame_cp[:, 5 * side: 6 * side, :] * 0,
                )
            )
            dstFrame_cp = cp.vstack((line1, line2, line3))

        return cp.asnumpy(dstFrame_cp)

    def fisheye2equirect_cuda(self,
                              srcFrame,
                              outShape,
                              aperture=0,
                              delx=0,
                              dely=0,
                              radius=0,
                              edit_mode=False):

        cu_frame = cv2.cuda_GpuMat()
        cu_frame.upload(srcFrame)

        inShape = srcFrame.shape[:2]

        self.Hs = inShape[0]
        self.Ws = inShape[1]
        self.Hd = outShape[0]
        self.Wd = outShape[1]

        map_x_cp = cp.zeros((self.Hd, self.Wd), cp.float32)
        map_y_cp = cp.zeros((self.Hd, self.Wd), cp.float32)
        # print(self.map_x, self.map_y)

        self.Cx = (
            self.Ws // 2 - delx
        )  # This value needs to be tuned using the GUI for every new camera
        self.Cy = (
            self.Hs // 2 - dely
        )  # This value needs to be tuned using the GUI for every new camera
        # print(self.Cx, self.Cy)

        if not radius:
            self.radius = min(inShape)
        else:
            self.radius = radius

        if not aperture:
            self.aperture = 385  # This value is determined using the GUI
        else:
            self.aperture = aperture

        if not edit_mode:
            f = open(self.filePath, "r")
            self.radius = int(f.readline())
            self.aperture = int(f.readline())
            delx = int(f.readline())
            dely = int(f.readline())
            f.close

        i, j = cp.meshgrid(cp.arange(0, int(self.Hd)),
                           cp.arange(0, int(self.Wd)))
        xyz = cp.zeros((self.Hd, self.Wd, 3))
        x, y, z = cp.split(xyz, 3, axis=-1)

        x = (
            self.radius
            * cp.cos((i * 1.0 / self.Hd - 0.5) * cp.pi)
            * cp.cos((j * 1.0 / self.Hd - 0.5) * cp.pi)
        )
        y = (
            self.radius
            * cp.cos((i * 1.0 / self.Hd - 0.5) * cp.pi)
            * cp.sin((j * 1.0 / self.Hd - 0.5) * cp.pi)
        )
        z = self.radius * cp.sin((i * 1.0 / self.Hd - 0.5) * cp.pi)
        # print(x,y,z)

        r = (
            2
            * cp.arctan2(cp.sqrt(x ** 2 + z ** 2), y)
            / cp.pi
            * 180
            / self.aperture
            * self.radius
        )
        theta = cp.arctan2(z, x)

        map_x_cp = cp.multiply(
            r, cp.cos(theta)).T.astype(cp.float32) + self.Cx
        map_y_cp = cp.multiply(
            r, cp.sin(theta)).T.astype(cp.float32) + self.Cy
        self.map_x = cp.asnumpy(map_x_cp)
        self.map_y = cp.asnumpy(map_y_cp)
        # print(type(self.map_x), type(self.map_y))
        cu_map_x = cv2.cuda_GpuMat(self.map_x)
        cu_map_y = cv2.cuda_GpuMat(self.map_y)

        cu_rect_img = cv2.cuda.remap(
            cu_frame,
            cu_map_x,
            cu_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        return cu_rect_img.download()
