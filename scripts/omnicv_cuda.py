#!/usr/bin/env/python
import cv2
import numpy as np
import cupy as cp


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


class fisheyeImgConvGPU:
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

    # CUDA

    def fisheye2equirect(self,
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

    def equirect2cubemap(self, srcFrame, side=256, modif=False, dice=False):

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

    def cubemap2equirect(self, srcFrame, outShape):

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
        self.map_x = cv2.cuda_GpuMat(map_x_np)
        self.map_y = cv2.cuda_GpuMat(map_y_np)

        dstFrame = 0
        # outShape1 = outShape.append(3)
        # dstFrame = np.zeros(outShape).astype(np.uint8)
        # dstFrame = cv2.cuda_GpuMat(dstFrame)
        cube_faces_cp = cp.stack(cp.split(srcFrame_cp, 6, 1), 0)
        cube_faces_cp[1] = cp.flip(cube_faces_cp[1], 1)
        cube_faces_cp[2] = cp.flip(cube_faces_cp[2], 1)
        cube_faces_cp[4] = cp.flip(cube_faces_cp[4], 0)
        self.tp = tp
        for i in range(6):
            mask = self.tp == i
            # mask = cp.stack([mask, mask, mask], axis=2)
            mask = mask.astype(cp.uint8)
            mask = cp.asnumpy(mask)
            # print(mask.shape)
            mask = cv2.cuda_GpuMat(mask)
            cube_faces = cv2.cuda_GpuMat(cp.asnumpy(cube_faces_cp[i]))
            dstFrame1 = cv2.cuda.remap(
                cube_faces,
                self.map_x,
                self.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            ).download()
            # print(dstFrame1.download().shape)
            # # We use this border mode to avoid small black lines
            #
            dstFrame += cv2.bitwise_and(dstFrame1, dstFrame1, mask=mask.download())
            # dstFrame2 = cv2.bitwise_not(dstFrame2, mask=mask.download())
            # dstFrame = cv2.add(dstFrame, dstFrame2)
            # dstFrame2 = cv2.cuda.bitwise_not(dstFrame1, mask=mask)
            # dstFrame2 = cv2.cuda.bitwise_not(dstFrame2, mask=mask)
            # print(dstFrame.download().shape)
            # print(dstFrame2.download().shape)
            # dstFrame = cv2.add(dstFrame, dstFrame2.download())

        return dstFrame

    def cubemap2persp(self, img, FOV, THETA, PHI, Hd, Wd):

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

        x_map = cp.zeros([Hd, Wd], cp.float32) + 1
        y_map = cp.tile((cp.arange(0, Wd) - c_x) * w_interval, [Hd, 1])
        z_map = -cp.tile((cp.arange(0, Hd) - c_y) * h_interval, [Wd, 1]).T
        D = cp.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)
        xyz = cp.zeros([Hd, Wd, 3], cp.float)
        xyz[:, :, 0] = (1 / D * x_map)[:, :]
        xyz[:, :, 1] = (1 / D * y_map)[:, :]
        xyz[:, :, 2] = (1 / D * z_map)[:, :]

        y_axis = cp.array([0.0, 1.0, 0.0], cp.float32)
        z_axis = cp.array([0.0, 0.0, 1.0], cp.float32)
        [R1, _] = cv2.Rodrigues(cp.asnumpy(z_axis * cp.radians(THETA)))
        R1 = cp.asarray(R1)
        [R2, _] = cv2.Rodrigues(cp.asnumpy(cp.dot(R1, y_axis) * cp.radians(-PHI)))
        R2 = cp.asarray(R2)

        xyz = xyz.reshape([Hd * Wd, 3]).T
        xyz = cp.dot(R1, xyz)
        xyz = cp.dot(R2, xyz).T
        lat = cp.arcsin(xyz[:, 2] / 1)
        lon = cp.zeros([Hd * Wd], np.float)
        theta = cp.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(cp.bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(cp.bool)

        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + cp.pi
        lon[idx4] = theta[idx4] - cp.pi

        lon = cp.asnumpy(lon.reshape([Hd, Wd]) / cp.pi * 180)
        lat = cp.asnumpy(-lat.reshape([Hd, Wd]) / cp.pi * 180)
        lon = cp.asnumpy(lon / 180 * equ_cx + equ_cx)
        lat = cp.asnumpy(lat / 90 * equ_cy + equ_cy)

        self.map_x = lon.astype(cp.float32)
        self.map_y = lat.astype(cp.float32)

        persp = cv2.cuda.remap(
            cv2.cuda_GpuMat(img),
            cv2.cuda_GpuMat(lon.astype(np.float32)),
            cv2.cuda_GpuMat(lat.astype(np.float32)),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

        return persp.download()

    def equirect2persp(self, img, FOV, THETA, PHI, Hd, Wd):

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

        x_map = cp.zeros([Hd, Wd], cp.float32) + 1
        y_map = cp.tile((cp.arange(0, Wd) - c_x) * w_interval, [Hd, 1])
        z_map = -cp.tile((cp.arange(0, Hd) - c_y) * h_interval, [Wd, 1]).T
        D = cp.sqrt(x_map ** 2 + y_map ** 2 + z_map ** 2)

        xyz = cp.zeros([Hd, Wd, 3], cp.float)
        xyz[:, :, 0] = (x_map / D)[:, :]
        xyz[:, :, 1] = (y_map / D)[:, :]
        xyz[:, :, 2] = (z_map / D)[:, :]

        y_axis = cp.array([0.0, 1.0, 0.0], cp.float32)
        z_axis = cp.array([0.0, 0.0, 1.0], cp.float32)
        [R1, _] = cv2.Rodrigues(cp.asnumpy(z_axis * cp.radians(THETA)))
        R1 = cp.array(R1)
        [R2, _] = cv2.Rodrigues(cp.asnumpy(cp.dot(R1, y_axis) * cp.radians(-PHI)))
        R2 = cp.array(R2)

        xyz = xyz.reshape([Hd * Wd, 3]).T
        xyz = cp.dot(R1, xyz)
        xyz = cp.dot(R2, xyz).T
        lat = cp.arcsin(xyz[:, 2] / 1)
        lon = cp.zeros([Hd * Wd], cp.float)
        theta = cp.arctan(xyz[:, 1] / xyz[:, 0])
        idx1 = xyz[:, 0] > 0
        idx2 = xyz[:, 1] > 0

        idx3 = ((1 - idx1) * idx2).astype(cp.bool)
        idx4 = ((1 - idx1) * (1 - idx2)).astype(cp.bool)

        lon[idx1] = theta[idx1]
        lon[idx3] = theta[idx3] + np.pi
        lon[idx4] = theta[idx4] - np.pi

        lon = lon.reshape([Hd, Wd]) / np.pi * 180
        lat = -lat.reshape([Hd, Wd]) / np.pi * 180
        lon = lon / 180 * equ_cx + equ_cx
        lat = lat / 90 * equ_cy + equ_cy

        self.map_x = cv2.cuda_GpuMat(cp.asnumpy(lon.astype(np.float32)))
        self.map_y = cv2.cuda_GpuMat(cp.asnumpy(lat.astype(np.float32)))

        persp = cv2.cuda.remap(cv2.cuda_GpuMat(img),
                          self.map_x,
                          self.map_y,
                          cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_WRAP)

        return persp.download()

