from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

FPS = 30  # Framerate of outputted video file
INTERVAL = 50  # Interval between frames during preview animation
CMAP = "plasma"  # Colormap to use


# ------------------------------------------------------------------------------
# Create 2D video
# ------------------------------------------------------------------------------
def animate_2D(frame_number, image_ref: AxesImage, data):
    image_ref.set_array(data[frame_number, :, :])
    image_ref.set_cmap(CMAP)
    return frame_number


def create_2D_animation(data, output_filepath):
    fig = plt.figure()
    plt.axes()
    im = plt.imshow(data[0, :, :], cmap=CMAP)
    num_frames = data.shape[0]
    anim = animation.FuncAnimation(
        fig, animate_2D, interval=INTERVAL, fargs=(im, data), frames=num_frames
    )

    anim.save(output_filepath, fps=FPS, extra_args=["-vcodec", "libx264"])
    plt.show()


# ------------------------------------------------------------------------------
# Create 3D video
# ------------------------------------------------------------------------------
def animate_3D(frame_number, plot, ax, X, Y, data):
    plot[0].remove()
    plot[0] = ax.plot_surface(X, Y, data[frame_number, :, :], cmap=CMAP)


def create_3D_animation(data, output_filepath):
    num_frames = data.shape[0]
    height = data.shape[1]
    width = data.shape[2]
    min_z = np.min(data)
    max_z = np.max(data)

    X = np.arange(width)
    Y = np.arange(height)
    X, Y = np.meshgrid(X, Y)

    # Attaching 3D axis to the figure
    # Create figure
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("x axis")
    ax.set_ylabel("y axis")
    ax.set_zlabel("z axis")
    ax.set_zlim(min_z, max_z)
    plot = [
        ax.plot_surface(
            X, Y, data[0, :, :], cmap=CMAP, linewidth=0, rstride=1, cstride=1
        )
    ]

    anim = animation.FuncAnimation(
        fig,
        animate_3D,
        interval=INTERVAL,
        fargs=(plot, ax, X, Y, data),
        frames=num_frames,
    )

    anim.save(output_filepath, fps=FPS, extra_args=["-vcodec", "libx264"])
    plt.show()
