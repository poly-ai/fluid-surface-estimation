import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import glob
import cv2

INPUT_DATA = '../data/1D-waves/output/wave-sin-1.npy'
OUTPUT_DIR = "./temp-frames"
OUTPUT_FRAME_PREFIX = 'frame'
VIDEO_FILENAME = 'output.mp4'
FRAME_RATE = 24

# Create output dir
if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)
  print("Created dir './temp-frames'")

# Delete old video
if os.path.exists(VIDEO_FILENAME):
  os.remove(VIDEO_FILENAME)

# Delete old frames
files = glob.glob(f'{OUTPUT_DIR}/*.png', recursive=True)
for f in files:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))


def get_point_height(x, y, data):
    return data[y, x]

# Generate a single frame
def generate_frame(single_frame_data, output_path, maxz, minz):
    height = single_frame_data.shape[1];
    width = single_frame_data.shape[0];
    X = np.arange(width)
    Y = np.arange(height)
    X, Y = np.meshgrid(X, Y)
    zs = np.array(get_point_height(np.ravel(X), np.ravel(Y), single_frame_data))
    Z = zs.reshape(X.shape)
    

    # Create figure
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap=cm.plasma, linewidth=0, antialiased=False)

    ax1.set_xlabel('x axis')
    ax1.set_ylabel('y axis')
    ax1.set_zlabel('z axis')
    ax1.set_zlim(minz,maxz)
    plt.savefig(output_path)
    plt.close()


def main():
    # Load data
    data = np.load(INPUT_DATA)
    num_frames = data.shape[0];

    max_h = np.max(data);
    min_h = np.min(data);
        
    # Generate frame sequence for single example
    for i in range(num_frames):
        single_frame = data[i,:,:]
        filename = f"{OUTPUT_FRAME_PREFIX}-{i:04}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        generate_frame(single_frame, output_path, max_h, min_h)

    # Generate video file
    img_array = []
    frame_filenames = sorted(glob.glob(f'{OUTPUT_DIR}/*.png'))
    for filename in frame_filenames:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(VIDEO_FILENAME, fourcc, FRAME_RATE, size)

    # Write frames to video
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

    print(f"Saved video to {VIDEO_FILENAME}")


if __name__ == "__main__":
    main()