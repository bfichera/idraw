from pathlib import Path
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt

data_cmap = 'Greys'
cmap = 'viridis'
ttcf_cmap = 'magma'

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=lambda s: Path(s))
parser.add_argument('--show', action='store_true')
parser.add_argument('--output-dir', type=lambda s: Path(s))
cfg = parser.parse_args()
do_show = cfg.show
output_dir = cfg.output_dir
input_path = cfg.input_path

params = {}

if do_show:

    def show():
        plt.show()
        plt.close()

else:

    def show():
        plt.close()


with open(input_path, 'rb') as fh:
    results = pickle.load(fh)

# Plot a few frames of the input data; phase and amplitude
num_frames = results['img_stk.shape'][0]
frames_of_interest = np.linspace(0, num_frames, 3, endpoint=False, dtype=int)
for frame in frames_of_interest:
    mappable_0 = plt.imshow(results['img_stk'][frame, 0, :, :], cmap=data_cmap)
    plt.colorbar()
    plt.title(f'Frame {frame}: Phase')
    plt.savefig(output_dir / f'img_stk_{frame:03d}_0.pdf')
    show()
    plt.imshow(results['img_stk'][frame, 1, :, :], cmap=data_cmap)
    plt.colorbar()
    plt.title(f'Frame {frame}: Amplitude')
    plt.savefig(output_dir / f'img_stk_{frame:03d}_1.pdf')
    show()

# Plot the RGB reduction
plt.imshow(results['mapper_low_res_rgb'][0, :, :, :], cmap=data_cmap)
plt.colorbar()
plt.title('RGB reduction')
plt.savefig(output_dir / 'low_res_rgb.pdf')
show()

# Plot one of the ttcfs
ttcf_i_idx = 0
ttcf_j_idx = 0
ttcf = results['window_ttcf'][0, ttcf_i_idx, ttcf_j_idx, :, :]
plt.imshow(ttcf, origin='lower', cmap=ttcf_cmap)
plt.colorbar()
plt.title('TFCF of the ({ttcf_i_idx}, {ttcf_j_idx})\'th window')
plt.savefig(output_dir / f'ttcf_{ttcf_i_idx}_{ttcf_j_idx}.pdf')
show()

# Plot sums of the input data
im = np.sum(results['img_stk'][:, 0, :, :], axis=0)
plt.imshow(im, cmap=cmap)
plt.colorbar()
plt.title('Sum of the input stack: Phase')
plt.savefig(output_dir / 'sum_img_stk0.pdf')
show()
im = np.sum(results['img_stk'][:, 1, :, :], axis=0)
plt.imshow(im, cmap=cmap)
plt.colorbar()
plt.title('Sum of the input stack: Amplitude')
plt.savefig(output_dir / 'sum_img_stk1.pdf')
show()

# Plot the data points in RGB space
fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
u0, u1, u2, u3 = results['mapper_low_res_rgb'].shape
collapsed_rgb = results['mapper_low_res_rgb'].reshape(u1 * u2, 3)
ax.scatter(*(collapsed_rgb[:, i] for i in range(3)))
ax.set_xlabel('r')
ax.set_ylabel('g')
ax.set_zlabel('b')
plt.savefig(output_dir / 'rgb_scatter.pdf')
show()
