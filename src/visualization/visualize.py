import click
import logging
import numpy as np
import src.visualization.create_animation as animation


@click.command()
@click.argument("data_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path())
@click.option("-t", "--three-dimensional", default=False)
def main(data_filepath, output_filepath, three_dimensional):
    """Create visualization from a sequence of frames"""
    logger = logging.getLogger(__name__)

    # Load data
    data = np.load(data_filepath, allow_pickle=True)

    # Create animation
    if three_dimensional:
        logger.info(f"Saving 3D animation to {output_filepath}")
        animation.create_3D_animation(data, output_filepath)
    else:
        logger.info(f"Saving 2D animation to {output_filepath}")
        animation.create_2D_animation(data, output_filepath)


if __name__ == "__main__":
    log_fmt = "%(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
