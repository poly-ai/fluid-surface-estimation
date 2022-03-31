import numpy as np
import torch
import matplotlib.pyplot as plt
import gc

PRED_FRAMES = 50      # Choose the number of frames to be predicted
VISUAL_FREQ = 10       # Visaulize every VISUAL_FRAMES

def predict_model(model, test_data, device):
   
    test_wave = np.float32(((test_data*0.5)+0.5)) 

    predict_wave = torch.zeros(10,64,64)
    predict_wave[0:10,:,:] = torch.from_numpy(test_wave[0:10,:,:])

    out_wave = np.zeros((10+PRED_FRAMES,64,64))
    out_wave[0:10,:,:] = test_wave[0:10,:,:]
    err_wave = np.zeros((10+PRED_FRAMES,64,64))

    # Make prediction and Visualize the prediction result
    for i in range(0,PRED_FRAMES):

        # Prepare the input data to consistent size, and load it to cuda
        test_input = predict_wave[0:10,:,:].clone().detach()
        test_input = test_input.unsqueeze(0).unsqueeze(0)            
        test_input = test_input.to(device)

        with torch.no_grad():                   # Reduce the cuda memory used
            out = model(test_input).squeeze(0)  # Predict the output

        # Store the output to numpy array  
        old = predict_wave[1:10,:,:].to(device)
        predict_wave = torch.cat((old,out),0)
        out_wave[i+10,:,:] = out.cpu().detach().numpy()

        # visualize every VISUAL_FREQ frames
        if i%VISUAL_FREQ == 0:
            print("frame:\t",i)
            fig = plt.figure()
            ax = plt.axes()
            im = plt.imshow(out_wave[i+10,:,:], cmap = "gray")
            plt.show()

        # Clean cuda memory (just in case)
        del test_input
        del out
        del old
        gc.collect()

    del predict_wave
    gc.collect()
    err_wave = out_wave - test_wave[:10+PRED_FRAMES,:,:]