import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import cv2

OUTPUT_DIR = "./temp-frames"
OUTPUT_FRAME_PREFIX = 'frame'

# Create output dir
if not os.path.exists(OUTPUT_DIR):
  os.makedirs(OUTPUT_DIR)
  print("Created dir './temp-frames'")

files = glob.glob(f'{OUTPUT_DIR}/*.png', recursive=True)
for f in files:
    try:
        os.remove(f)
    except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))


# Generate a single frame
def generate_frame(single_frame_data, output_path):
    height, width = single_frame_data.shape
    x3 = []
    y3 = []
    z3 = []
    dx = []
    dy = []
    dz = []
    for i in range(height):
        for j in range(width):
            x3.append(i)
            y3.append(j)
            z3.append(0)
            dx.append(1)
            dy.append(1)
            dz.append(single_frame_data[i,j])

    # Create figure
    fig = plt.figure(figsize=(5,5))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.bar3d(x3, y3, z3, dx, dy, dz)

    ax1.set_xlabel('x axis')
    ax1.set_ylabel('y axis')
    ax1.set_zlabel('z axis')
    plt.savefig(output_path)


def main():
    # Load data
    x_train = np.load('../data/1D-waves/output/x-train-adv.npy')
    single_example = x_train[0,:,:,:] # 100,100, NUM_FRAMES
    num_frames = single_example.shape[2]
    print(single_example.shape)

    # Generate frame sequence for single example
    for i in range(num_frames):
        single_frame = single_example[:,:,i]
        filename = f"{OUTPUT_FRAME_PREFIX}-{i}.png"
        output_path = os.path.join(OUTPUT_DIR, filename)
        generate_frame(single_frame, output_path)

    # Generate video file
    img_array = []
    for filename in glob.glob(f'{OUTPUT_DIR}/*.png'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    fps = 3
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, size)

    
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == "__main__":
    main()
