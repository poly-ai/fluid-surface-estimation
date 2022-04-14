import numpy as np
from .wave_cir import create_cir_wave

def make_cir_wave_dataset(output_filepath, image_dimension, num_frames):

    wave_freq_list = [1,10]
    wave_number_list = [1,3]
    x_center_list = [1,5]

    cir_data = []

    for wave_freq in wave_freq_list:
        for wave_number in wave_number_list:
            for x_center in x_center_list:
                print(f"create cir wave wave freq:{wave_freq}, wave number {wave_number}, x center {x_center}")
                data = create_cir_wave(image_dimension=image_dimension, 
                                        num_frames=num_frames, 
                                        wave_freq=wave_freq,
                                        wave_number=wave_number,
                                        x_center=x_center)
                cir_data.append(data)
    
    print(f"Created cir wave data")
    cir_data = np.stack(cir_data)
    print("Dataset shape:", cir_data.shape)

    # Save output
    np.save(output_filepath, cir_data)
    print(f'Saving dataset to {output_filepath}')

def main():
    OUTPUT_PATH = 'wave-cir.npy'
    make_cir_wave_dataset(OUTPUT_PATH, image_dimension = 64, num_frames = 200)

if __name__ == "__main__":
    main()


