# 1D Advection and Diffusion Wave in 2D domain 
The code generates data for different wave behaviors in 2D domain.

NOTE: the data processing of train_X and train_Y will be handled inside the Conv-LSTM Code. 

## Run Code to generate 1D Wave (expanding in 2D domain) 
```
python3 wave-1d.py
```

## Run Code to Visualize Wave Ouput Data (Take Inputs as the .npy files)
```
python3 anim-wave.py
```

## Output File Format
wave-[label].npy

Example:

wave-sin-1.npy	represents a sin wave travelling rightward.

wave-dif-1.npy	represents a diffusion wave spreading from the center.

wave-1.npy	represents a combination of traveling sin wave and spreading diffusion wave.

## Output Wave Data Structure
3-dimensional tensor

[FRAME,[Height values in 2D Array]];

Example:
```
data[0,:,:]
```
return the Height values in 2D Array at time t = 0 (of at frame 0)

## Output Data Visuallization
![sin wave animation](animations/sin-1.gif)

![diffusion wave animation](animations/dif-1.gif)

## Objectives
1: Machine Learn Model that can predict a single sin wave traveling (wave-sin-1.npy)

2: Machine Learn Model that can predict different kinds of wave behaviors (wave-sin-1.npy, wave-dif-1.npy, wave-1.npy ... etc)

Ultimate: Machine Learn Model can predict the fluid data from Shallow.py code (Real 2D Wave)
