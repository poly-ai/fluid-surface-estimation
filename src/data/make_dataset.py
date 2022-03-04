# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
from pathlib import Path
import src.data.wave_adv_diff as wave_adv_diff


# python src/data/make_dataset.py "./data/processed/1d_wave.npy" 64 1000 5

@click.command()
@click.argument('output_filepath', type=click.Path())
@click.argument('image_dimension', type=click.INT)
@click.argument('num_frames', type=click.INT)
@click.argument('duration', type=click.INT)
def main(output_filepath, image_dimension, num_frames, duration):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    '''
    Create dataset from parameters
    Example:               OUTPUT_FILEPATH                  IMAGE_DIMENSION
    python make_dataset.py "../../data/processed/waves.npy" 64
    '''
    logger = logging.getLogger(__name__)
    logger.info(f'Image dimensions: {image_dimension}x{image_dimension}')

    # Create wave data
    data = wave_adv_diff.create_adv_diff_wave(image_dimension=image_dimension,
                                              num_frames=num_frames,
                                              duration=duration)
    np.save(output_filepath, data)
    logger.info(f'Saving dataset to {output_filepath}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    main()
