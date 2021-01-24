from time import time
from copy import deepcopy
import numpy as np
import torch
from torch.nn import functional as F
import cv2


class AugmentationParams1:
    """ Augmentation params v1: probabilities and ranges """

    def __init__(self, **kwargs):

        self.p_fliplr = 0.0

        self.p_scale = 1.0
        self.scale_range = (0.66, 1.5)

        self.p_rotate = 1.0
        self.rotate_range = (-60, 60)
        self.p_skew = 0.0
        self.skew_range = (-0.1, 0.1)

        self.p_translate = 1.0
        self.translate_range = (0.2, 0.8)

        self.p_blur = 0.2
        self.blur_size_range = (0.5, 5)
        self.blur_angle_range = (0.0, 360.0)

        for key, val in kwargs.items():
            setattr(self, key, val)

    def items(self):
        return vars(self).items()


class AugmentationParams2:
    """ Augmentation params v2: selections """

    def __init__(self, **kwargs):

        self.num_aug = 20
        self.location = [(0.5, 0.5)]  # Center location - fraction of image (width, height)
        self.rotation = [5, -5, 10, -10, 20, -20, 30, -30, 45, -45, 60, -60]
        self.fliplr = [False, False, True]
        self.scale = [0.7, 1.0, 1.5, 2.0, '0.25', '0.5', '1.0']  # Strings: fraction of frame height
        self.skew = [(0.0, 0.0), (0.0, 0.0), (0.1, 0.1)]
        self.blur_size = [0.0, 0.0, 0.0, 2.0, 5.0]
        self.blur_angle = [0, 45, 90, 135]

        for key, val in kwargs.items():
            setattr(self, key, val)

    def items(self):
        return vars(self).items()

    def __repr__(self):
        return str(vars(self))


class AugmentationSpec:

    def __init__(self, **kwargs):
        """
        :param location:   New target center (x,y)  [fractions of image width and height]
        :param rotation:   Rotation angle [degrees]
        :param fliplr:     Whether to mirror on the x-axis
        :param scale:      number: scale change, string: absolute size (fraction of image height)
        :param skew:       Skew (x,y) [Transform matrix skew coefficients]
        :param blur_size:  Blur size [pixels]
        :param blur_angle: Blur rotation angle [degrees]
        :param min_size:   Minimum size (with and height) of the augmented object [pixels]
        """

        self.location = None
        self.rotation = 0.0
        self.fliplr = False
        self.scale = 1.0
        self.skew = (0, 0)
        self.blur_size = 0
        self.blur_angle = 0
        self.min_size = 10

        for key, val in kwargs.items():
            setattr(self, key, val)

        assert self.location is not None

    def __repr__(self):
        return str(vars(self))


class ImageAugmenter:

    def __init__(self, parameters: dict):

        self.params = parameters
        self.T_generate = 0
        self.max_retries = 100

    @staticmethod
    def _scale(sx, sy):
        return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

    @staticmethod
    def _rotate(a):
        ca, sa = np.cos(a), np.sin(a)
        return np.array([[ca, sa, 0], [-sa, ca, 0], [0, 0, 1]])

    @staticmethod
    def _translate(dx, dy):
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])

    @staticmethod
    def _skew(kx, ky):
        return np.array([[1, kx, 0], [ky, 1, 0], [0, 0, 1]])

    @staticmethod
    def _blur_kernel(sx, sy, R):
        """
        :param sx:  x axis sigma
        :param sy:  y axis sigma
        :param R:   rotation matrix
        :return: rotated Gauss blur filter
        """
        cov = R @ np.diag((sx, sy)) @ R.T

        s = int(np.max((sx, sy)) / 2 + 0.5)
        s = (s + (s + 1) % 2)  # odd size
        r = np.arange(-s, s + 1)
        X = np.stack(np.meshgrid(r, r))

        X = (X * np.tensordot(np.linalg.inv(cov), X, axes=[1, 0])).sum(0)
        G = np.exp(-0.5 * X)
        G = G / G.sum() * 1.0
        return G.astype(np.float32)

    @staticmethod
    def warp_affine(src, H, dsize, interpolation='bicubic'):
        """ Warp a single image or a batch of images with an affine transform.
        :param src:    Input to warp, shape = (..., H, W)
        :param H:      Forward warp affine transform
        :param dsize:  Output (height, width) in pixels
        :param interpolation:  Interpolation mode. Default 'bicubic'
        :return:  Warped output
        """
        dsize = int(dsize[1]), int(dsize[0])
        H = H[:2].astype(np.float)
        flags = dict(bicubic=cv2.INTER_CUBIC, bilinear=cv2.INTER_LINEAR, nearest=cv2.INTER_NEAREST)[interpolation]

        im = src.permute(1, 2, 0).squeeze().cpu().numpy()
        dst = cv2.warpAffine(im, H, dsize, flags=flags)
        dst = torch.from_numpy(np.atleast_3d(dst).transpose(2, 0, 1)).to(src.device)

        return dst

    def generate_specs1(self, N, aparams: AugmentationParams1, tg_bbox, im_size, force_unity_scale=False):

        t0 = time()

        ap = aparams
        h, w = im_size
        tg_x, tg_y = tg_bbox[:2]
        choose = self._choose

        aspecs = []

        for i in range(N):
            t = choose(ap.p_translate, range=ap.translate_range, default=(tg_x / w, tg_y / h), size=(2,))
            r = choose(ap.p_rotate, range=ap.rotate_range, default=0.0)
            m = choose(ap.p_fliplr, choices=[True, False])
            s = choose(ap.p_scale, range=ap.scale_range, default=1.0) if not force_unity_scale else 1.0
            k = choose(ap.p_skew, range=ap.skew_range, default=(0., 0.), size=(2,))
            use_blur = np.random.uniform(0, 1) < ap.p_blur
            if use_blur:
                bs = np.random.uniform(*ap.blur_size_range)
                ba = np.random.uniform(*ap.blur_angle_range)
            else:
                bs, ba = (0, 0), 0

            aspec = AugmentationSpec(tcenter=t, rotation=r, fliplr=m, scale=s, skew=k, blur_size=bs, blur_angle=ba)
            aspecs.append(aspec)

        self.T_generate += time() - t0

        return aspecs

    def generate_target_locations(self, N, im_size):
        """ Lay out a grid of new target object centers, shuffle the order and pick the first N """
        h, w = im_size
        aspect = w / h
        nrows = int(np.ceil(np.sqrt(N / aspect)))
        ncols = int(np.ceil(aspect * nrows))

        tcenters = []
        co_max = 0.5 / ncols  # Max column offset (fraction of image size)
        ro_max = 0.5 / nrows  # Max row ...
        for r in range(nrows):
            for c in range(ncols):
                x = (c + 0.5) / ncols
                y = (r + 0.5) / nrows
                # Randomize a little
                x += np.random.normal(0, co_max / 4)
                y += np.random.normal(0, ro_max / 4)
                x = np.round(x, 3)  # Rounding makes printing of the parameters prettier,
                y = np.round(y, 3)  # but should not really affect the target_models.
                tcenters.append((x, y))

        np.random.shuffle(tcenters)
        tcenters = tcenters[:N]
        return tcenters

    def generate_specs2(self, aparams: AugmentationParams2):
        """ Generate augmentation specs - method 2 """

        t0 = time()

        N = aparams.num_aug - 1  # The original image is always included and does not need a spec.
        aug1 = dict()

        # Independently shuffle and combine the parameter lists

        for k, a in aparams.items():

            if k in ('num_aug',):  # Ignore these
                continue
            a = a * ((N + len(a) - 1) // len(a))  # Ensure enough values to choose from
            np.random.shuffle(a)  # Select N parameters without replacement
            aug1[k] = a[:N]       #

        # Dict of lists -> list of dicts

        aug2 = [dict() for i in range(N)]
        for i in range(N):
            for k in aug1.keys():
                aug2[i][k] = aug1[k][i]

        # Generate specs

        aspecs = [AugmentationSpec(**a) for a in aug2]
        self.T_generate += time() - t0
        return aspecs

    def get_transform(self, aspec: AugmentationSpec, tg_bbox, im_size, limit_scale=True, use_blur=True):
        """ Get the affine transform and blur kernel from an AugmentationSpec object
        :param tg_bbox:  Target bounding box (center_x, center_y, width, height)
        :param im_size:  Image (height, width)
        :param limit_scale:  Whether to restrict the scale so that the target will not be larger than im_size.
        :return: geometric transform [3x3 matrix], blur kernel
        """

        tg_x, tg_y, tg_w, tg_h = tg_bbox
        assert tg_w > 0 and tg_h > 0
        im_h, im_w = im_size

        t, a, s, k = aspec.location, aspec.rotation, aspec.scale, aspec.skew

        # Set up the scaling

        if isinstance(s, str):
            new_tg_h = float(s) * im_h  # s = target fraction of image height
            s = new_tg_h / tg_h

        if limit_scale:
            if s * tg_w > im_w or s * tg_h > im_h:  # Upper bound
                s = min(im_w / tg_w, im_h / tg_h)

            if s * tg_w < aspec.min_size or s * tg_h < aspec.min_size:  # Lower bound
                s = max(aspec.min_size / tg_w, aspec.min_size / tg_h)

        m = -1 if aspec.fliplr else 1  # mirror
        s = (m*s, s)

        # Create the affine transform

        d2r = np.pi / 180
        T = self._translate(t[0] * im_w, t[1] * im_h) @ self._skew(*k) @ self._rotate(a * d2r) @ \
            self._scale(*s) @ self._translate(-tg_x, -tg_y)  # The rightmost _translate moves the target to the origin

        # Create the blur filter

        if use_blur and aspec.blur_size > 0:
            R = self._rotate(aspec.blur_angle * d2r)[:2, :2]
            G = self._blur_kernel(aspec.blur_size, 0.1, R)
        else:
            G = np.array([[1.0]], dtype=np.float32)

        return T, G

    @staticmethod
    def _choose(p, choices=None, range=None, default=None, **kwargs):
        """ Choose a parameter randomly (uniformly)
        :param p:        Probability of *not* choosing the default value. Ignored if default=None
        :param choices:  List of choices. If None, use a range instead
        :param range:    Value range (low, high). If None, use the list of choices instead
        :param default:  Default parameter value
        :param kwargs:   Additional arguments to np.random.choice (if choices is not None) or np.random.uniform
        :return: the chosen parameter
        """

        assert choices is not None or range is not None
        assert default is not None

        make_choice = (p > 0.0 and np.random.uniform(0.0, 1.0) < p) or (default is None)
        if make_choice and choices is not None:
            return np.random.choice(choices, **kwargs)
        elif make_choice and range is not None:
            return np.random.uniform(*range, **kwargs)
        else:
            raise ValueError

    @staticmethod
    def cut_and_inpaint(im, mask: torch.Tensor, d=9, f=3, fast=False):
        """ Cut out the target object and inpaint the hole
        :param im:      Source image (torch Tensor, shape (C, H, W))
        :param mask:    Inpainting mask (torch tensor, shape (H, W) or (1, H, W)). Nonzero pixels will be inpainted
        :param d:       Target mask dilation (default: 3 pixel diameter)
        :param f:       Target alpha feather width (default: 3 pixels)
        :return: target cutout (RGBA) , inpainted image (RGB)
        """
        image = im.detach().cpu().numpy().transpose((1, 2, 0))
        mask = (mask.squeeze() > 0).byte().detach().cpu().numpy()
        mask = mask[..., None]

        # Extract the target

        target = mask * image
        # Feathered mask -> alpha channel
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (f, f))
        alpha = cv2.blur(cv2.erode(mask, k) * 255, (f, f))
        alpha = alpha[..., None]
        target = np.concatenate((target, alpha), axis=-1)

        # Inpaint the target area

        mask0 = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
        mask1 = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d*2, d*2)))

        if not fast:
            image = cv2.inpaint(image, mask1, inpaintRadius=d, flags=cv2.INPAINT_TELEA)
        else:
            mask2 = cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR)
            image = cv2.bitwise_and(image, (1 - mask2) * 255) + mask2 * 128

        # Blur the inpainted border

        m = (1 - mask0) * mask1
        m = cv2.blur(m * 255, ksize=(d, d))
        m = m[..., None] / 255
        bim = cv2.blur(image, ksize=(d, d))
        image = (bim * m + (1 - m) * image).astype(np.uint8)

        target = torch.from_numpy(target.transpose((2, 0, 1))).to(im.device)
        image = torch.from_numpy(image.transpose((2, 0, 1))).to(im.device)

        return target, image

    @staticmethod
    def filter_image(image, kernel):

        kernel = np.array(kernel, dtype=np.float32)
        if kernel is not None and kernel.shape != (1, 1):
            fh, fw = kernel.shape
            fpad = fh // 2, fw // 2
            kernel = torch.as_tensor(kernel).float().view(1, 1, fh, fw).to(image.device)
            image = F.conv2d(image.unsqueeze(1), kernel, padding=fpad).squeeze(1)  # per-channel filtering

        return image

    @classmethod
    def warp_and_filter_image(cls, image, H, kernel):
        """ Warp and filter a whole image
        :param image:   Image to transform (RGB, 0-255 value, byte tensor, shape C,H,W)
        :param H:       Linear transform (3x3 matrix)
        :param kernel:  2D filter kernel to apply to the target before warping and pasting
        """
        image = image.float()
        sz = image.shape[-2:]

        H = np.array(H, dtype=np.float32)
        image = cls.warp_affine(image, H, sz).clamp(0, 255)
        image = cls.filter_image(image, kernel)

        return image

    @classmethod
    def warp_distractors(cls, distractors, H):
        sz = distractors.shape[-2:]
        distractors = cls.warp_affine(distractors, H, sz, 'nearest')
        return distractors

    @classmethod
    def warp_filter_and_paste(cls, image, target, labels, H, kernel):
        """ Warp the target and its mask according the the H transform and paste the target into the image.
        :param image:   Background image to paint into (RGB, 0-255 value, byte tensor)
        :param target:  Target image (RGBA, 0-255 value, byte tensor)
        :param labels:  Target labels
        :param H:       Linear transform (3x3 matrix)
        :param kernel:  2D filter kernel to apply to the target before warping and pasting
        :return: (image with warped target pasted into it, warped mask)
        """

        image = image.float()
        target = target.float()

        # Warp

        sz = image.shape[-2:]
        H = np.array(H, dtype=np.float32)
        target = cls.warp_affine(target, H, sz).clamp(0, 255)
        labels = cls.warp_affine(labels, H, sz, 'nearest')

        # Filter

        target = cls.filter_image(target, kernel)
        # Not blurring the labels

        # Paste

        alpha = target[3].unsqueeze(0) / 255
        target = target[:3]
        image = target * alpha + image * (1 - alpha)
        image = image.byte()

        return image, labels

    def augment_from_specs(self, image, target, target_mask, tg_aspec: AugmentationSpec, tg_bbox,
                           bg_aspec: AugmentationSpec=None, distractors=None):
        """  Create one augmented image from specifications.
        :param image:        Inpainted Source image (tensor, shape (3,H,W)
        :param target:       Target source image (tensor, shape (3,H,W)
        :param target_mask:  Target mask. Nonzero pixels will be cut out
        :param tg_aspec:     Target AugmentationSpec
        :param tg_bbox:      Target bounding box (center_x, center_y, width, height)
        :param bg_aspec:     [Optional] Background AugmentationSpec

        :return: Warped (image, labels)
        """

        if bg_aspec is not None:
            h, w = image.shape[-2:]
            bg_bbox = (w / 2, h / 2, w, h)
            T, G = self.get_transform(bg_aspec, bg_bbox, image.shape[-2:], limit_scale=False)
            wimage = self.warp_and_filter_image(image, T, G)
            if distractors is not None:
                wdistractors = self.warp_distractors(distractors, T)
        else:
            wimage = image

        T, G = self.get_transform(tg_aspec, tg_bbox, wimage.shape[-2:])
        wimage, wlabels = self.warp_filter_and_paste(wimage, target, target_mask, T, G)
        if distractors is not None:
            wlabels = wlabels + wdistractors
            wlabels[wlabels == 3] = 1  # Target wins over distractor

        return wimage, wlabels

    @staticmethod
    def center_bbox_from_mask(mask):
        """ Find the axis-aligned bounding box covering all non-zero pixels in the input mask.
        :param mask:  mask image, tensor shaped (1, H, W) or (H, W)
        :return: list of N bounding box-tuples (center_x, center_y, w, h). w == 0 and h == 0 if the mask is empty
        """
        # Find the indices of all nonzero rows and columns in the mask
        ys = mask.squeeze().sum(dim=-1).nonzero(as_tuple=False).view(-1).cpu().numpy()
        xs = mask.squeeze().sum(dim=-2).nonzero(as_tuple=False).view(-1).cpu().numpy()

        if len(ys) > 0 and len(xs) > 0:
            x, y = xs[0], ys[0]
            w = xs[-1] - xs[0] + 1
            h = ys[-1] - ys[0] + 1
        else:
            x, y, w, h = 0, 0, 0, 0

        x += w/2
        y += h/2

        return x, y, w, h

    def verify_frame(self, obj_ids, wlabels, have_no_background):
        """ Verify all target objects are visible in the new frame
        :param obj_ids: list of object ids (excluding the background) to test for
        :param wlabels:  Labels image
        :param have_no_background:  Whether the background is completely hidden by objects
        :return:  True on success
        """
        min_px_count = self.params.min_px_count
        max_px_count = wlabels.shape[-1] * wlabels.shape[-2] - min_px_count
        good_frame = True

        for oid in obj_ids:
            px_count = (wlabels == oid).sum().item()
            good_frame = (px_count >= min_px_count) and (px_count < max_px_count or have_no_background)
            if not good_frame:
                break

        return good_frame

    def augment_first_frame(self, im, lb):

        p = self.params
        im_sz = im.shape[-2:]

        target_mask = (lb == 1).byte()
        distractors = None

        # Verify target object

        tg_bbox = self.center_bbox_from_mask(target_mask)
        if tg_bbox[-2:] == (0, 0):
            raise ValueError("Augmentation failed: No object to augment.")

        obj_ids, obj_pix_counts = np.unique(lb.cpu().numpy(), return_counts=True)
        no_background = (obj_ids[0] != 0)
        if not no_background:  # Have background -> remove the background class
            obj_ids = obj_ids[1:]
            obj_pix_counts = obj_pix_counts[1:]

        if np.any(obj_pix_counts < self.params.min_px_count):
            raise ValueError("Augmentation failed: Target object is too small.")

        # Cut and inpaint

        target, inpainted_image = self.cut_and_inpaint(im, target_mask, d=1, f=1, fast=True)

        # Finalize parameters

        fg_params = deepcopy(p.fg_aug_params)
        fg_params.location = self.generate_target_locations(p.num_aug, im_sz)
        bg_params = deepcopy(p.bg_aug_params) if 'bg_aug_params' in p else None

        N = p.num_aug - 1

        aug_images = []
        aug_labels = []
        retries = -1

        while len(aug_images) < N:

            retries += 1
            if retries > self.max_retries:
                RuntimeError("Augmentation failed: Not enough samples after %d retries." % self.max_retries)

            # Generate N augmentation specs

            fg_aspecs = self.generate_specs2(AugmentationParams2(**fg_params))
            if bg_params is not None:
                bg_aspecs = self.generate_specs2(AugmentationParams2(**bg_params))
            else:
                bg_aspecs = [None] * N

            # Warp, blur and paste

            for fg_aspec, bg_aspec in zip(fg_aspecs, bg_aspecs):
                wimage, wlabels = self.augment_from_specs(inpainted_image, target, target_mask, fg_aspec, tg_bbox, bg_aspec, distractors=distractors)
                good_frame = self.verify_frame([1], wlabels, no_background)
                if good_frame:
                    aug_images.append(wimage)
                    aug_labels.append(wlabels)

        if len(aug_images) > N:
            # Needed to retry ... shuffle again and crop to N
            iis = list(range(len(aug_images)))
            np.random.shuffle(iis)
            iis = iis[:N]
            aug_images = [aug_images[i] for i in iis]
            aug_labels = [aug_labels[i] for i in iis]

        # Insert original frame first

        aug_images.insert(0, im)
        aug_labels.insert(0, lb)

        aug_images = torch.stack(aug_images)
        aug_labels = torch.stack(aug_labels)

        return aug_images, aug_labels
