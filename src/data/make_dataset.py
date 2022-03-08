import click
import logging
from cv2 import log
import numpy as np
from pathlib import Path
import src.data.wave_advection_diffusion as advection_diffusion


@click.command()
@click.argument('output_filepath', type=click.Path())
@click.argument('image_dimension', type=click.INT)
@click.argument('num_frames', type=click.INT)
def main(output_filepath, image_dimension, num_frames):
    """
    Create dataset from parameters
    Example:               OUTPUT_FILEPATH                  IMAGE_DIMENSION
    python make_dataset.py "../../data/processed/waves.npy" 64
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Number of frames  {num_frames}')
    logger.info(f'Image dimensions  {image_dimension} x {image_dimension}')

    # Create wave data
    data = advection_diffusion.create_wave(image_dimension=image_dimension,
                                              num_frames=num_frames)
    # Save output
    np.save(output_filepath, data)
    logger.info(f'Saving dataset to {output_filepath}')


if __name__ == '__main__':
    log_fmt = '%(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    main()
