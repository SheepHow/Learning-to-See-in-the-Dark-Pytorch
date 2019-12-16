from __future__ import division
import os, scipy.io, scipy.misc
import torch
import numpy as np
import rawpy
import glob

from unet import UNetFuji

input_dir = './dataset/Fuji/short/'
gt_dir = './dataset/Fuji/long/'
checkpoint_dir = './checkpoint/Fuji/'
result_dir = './result_Fuji/'
ckpt = checkpoint_dir + 'model.pth'

# get test IDs
test_fns = glob.glob(gt_dir + '1*.RAF')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

def pack_raw(raw):
    # pack X-Trans image to 9 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 1024, 0) / (16383 - 1024)  # subtract the black level

    img_shape = im.shape
    H = (img_shape[0] // 6) * 6
    W = (img_shape[1] // 6) * 6

    out = np.zeros((H // 3, W // 3, 9))

    # 0 R
    out[0::2, 0::2, 0] = im[0:H:6, 0:W:6]
    out[0::2, 1::2, 0] = im[0:H:6, 4:W:6]
    out[1::2, 0::2, 0] = im[3:H:6, 1:W:6]
    out[1::2, 1::2, 0] = im[3:H:6, 3:W:6]

    # 1 G
    out[0::2, 0::2, 1] = im[0:H:6, 2:W:6]
    out[0::2, 1::2, 1] = im[0:H:6, 5:W:6]
    out[1::2, 0::2, 1] = im[3:H:6, 2:W:6]
    out[1::2, 1::2, 1] = im[3:H:6, 5:W:6]

    # 1 B
    out[0::2, 0::2, 2] = im[0:H:6, 1:W:6]
    out[0::2, 1::2, 2] = im[0:H:6, 3:W:6]
    out[1::2, 0::2, 2] = im[3:H:6, 0:W:6]
    out[1::2, 1::2, 2] = im[3:H:6, 4:W:6]

    # 4 R
    out[0::2, 0::2, 3] = im[1:H:6, 2:W:6]
    out[0::2, 1::2, 3] = im[2:H:6, 5:W:6]
    out[1::2, 0::2, 3] = im[5:H:6, 2:W:6]
    out[1::2, 1::2, 3] = im[4:H:6, 5:W:6]

    # 5 B
    out[0::2, 0::2, 4] = im[2:H:6, 2:W:6]
    out[0::2, 1::2, 4] = im[1:H:6, 5:W:6]
    out[1::2, 0::2, 4] = im[4:H:6, 2:W:6]
    out[1::2, 1::2, 4] = im[5:H:6, 5:W:6]

    out[:, :, 5] = im[1:H:3, 0:W:3]
    out[:, :, 6] = im[1:H:3, 1:W:3]
    out[:, :, 7] = im[2:H:3, 0:W:3]
    out[:, :, 8] = im[2:H:3, 1:W:3]
    return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
unet = UNetFuji()
unet.load_state_dict(torch.load(ckpt))
unet.to(device)

if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')

with torch.no_grad():
    unet.eval()
    for test_id in test_ids:
        # test the first image in each sequence
        in_files = glob.glob(input_dir + '%05d_00*.RAF' % test_id)
        for k in range(len(in_files)):
            in_path = in_files[k]
            in_fn = os.path.basename(in_path)
            print(in_fn)
            gt_files = glob.glob(gt_dir + '%05d_00*.RAF' % test_id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)

            raw = rawpy.imread(in_path)
            input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
            im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio #scale the low-light image using the same ratio
            scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

            input_full = np.minimum(input_full, 1.0)
            in_img = torch.from_numpy(input_full).permute(0,3,1,2).to(device)
            out_img = unet(in_img)
            output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
            output = np.minimum(np.maximum(output, 0), 1)

            _, H, W, _ = output.shape

            output = output[0, :, :, :]
            gt_full = gt_full[0, 0:H, 0:W, :]
            scale_full = scale_full[0, 0:H, 0:W, :]
            scale_full = scale_full * np.mean(gt_full) / np.mean(
                scale_full)  # scale the low-light image to the same mean of the groundtruth

            scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + 'final/%5d_00_%d_out.png' % (test_id, ratio))
            scipy.misc.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + 'final/%5d_00_%d_scale.png' % (test_id, ratio))
            scipy.misc.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + 'final/%5d_00_%d_gt.png' % (test_id, ratio))
