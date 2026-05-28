"""Resampling, intensity normalization, and NIfTI export helpers."""

import os

import numpy as np
import SimpleITK as sitk
from skimage.transform import resize


def resample_volume(outspacing, vol, mask=False):
    """Resample a SimpleITK volume to the target spacing."""
    outsize = [0, 0, 0]
    inputsize = vol.GetSize()
    inputspacing = vol.GetSpacing()

    outsize[0] = round(inputsize[0] * inputspacing[0] / outspacing[0])
    outsize[1] = round(inputsize[1] * inputspacing[1] / outspacing[1])
    outsize[2] = round(inputsize[2] * inputspacing[2] / outspacing[2])
    outsize = [outsize[2], outsize[1], outsize[0]]

    image_data = sitk.GetArrayFromImage(vol)
    if mask:
        im = resize(image_data, output_shape=outsize, order=0, anti_aliasing=False)
    else:
        im = resize(
            image_data,
            output_shape=outsize,
            order=3,
            mode="edge",
            anti_aliasing=False,
        )
    image = sitk.GetImageFromArray(im)
    image.SetSpacing(outspacing)
    image.SetDirection(vol.GetDirection())
    image.SetOrigin(vol.GetOrigin())
    return image


def normalize_data(data, ifzhao=True):
    """Intensity normalization for FLARE-style CT volumes."""
    data = np.clip(data, -22.0, 325.0)
    data = np.array(data, dtype=np.float32)
    if ifzhao:
        data -= 214.68231
        data /= 100.240135
    else:
        data -= -100
        data /= 100
    return data


def cure_image(new, old):
    """Copy spacing, origin, and direction from ``old`` onto ``new``."""
    new.SetSpacing(old.GetSpacing())
    new.SetOrigin(old.GetOrigin())
    new.SetDirection(old.GetDirection())
    return new


def save_nii_file(arr, path, direction=False, ifzhao=True, mask=False):
    """Resample, optionally normalize, and write a NIfTI file."""
    saveimg = arr
    vol_resampled = resample_volume(
        [1.254798173904419, 1.254798173904419, 2.5], saveimg, mask=mask
    )

    if mask:
        savemask = vol_resampled
        savemask.SetSpacing(vol_resampled.GetSpacing())
        if direction:
            savemask.SetDirection(direction)
        else:
            savemask.SetDirection(vol_resampled.GetDirection())
        savemask.SetOrigin(vol_resampled.GetOrigin())
        sitk.WriteImage(savemask, path)
        return

    resize_imgarr = sitk.GetArrayFromImage(vol_resampled)
    nor_resize_imgarr = normalize_data(resize_imgarr, ifzhao=ifzhao)
    nor_resize_img = sitk.GetImageFromArray(nor_resize_imgarr)
    nor_resize_img.SetSpacing(vol_resampled.GetSpacing())
    nor_resize_img.SetOrigin(vol_resampled.GetOrigin())
    if direction:
        nor_resize_img.SetDirection(direction)
    else:
        nor_resize_img.SetDirection(vol_resampled.GetDirection())
    sitk.WriteImage(nor_resize_img, path)


# Legacy camelCase alias
resampleVolume = resample_volume
