import numpy as np
import torch
import matplotlib.pyplot as plt
import gc
import config


def predict_model(model, test_data, device):

    test_wave = np.float32(((test_data * 0.5) + 0.5))

    predict_wave = torch.zeros(10, 64, 64)
    predict_wave[0:10, :, :] = torch.from_numpy(test_wave[0:10, :, :])

    out_wave = np.zeros((10 + config.PRED_FRAMES, 64, 64))
    out_wave[0:10, :, :] = test_wave[0:10, :, :]
    # err_wave = np.zeros((10 + PRED_FRAMES, 64, 64))

    # Make prediction and Visualize the prediction result
    for i in range(0, config.PRED_FRAMES):

        # Prepare the input data to consistent size, and load it to cuda
        test_input = predict_wave[0:10, :, :].clone().detach()
        test_input = test_input.unsqueeze(0).unsqueeze(0)
        test_input = test_input.to(device)

        with torch.no_grad():  # Reduce the cuda memory used
            out = model(test_input).squeeze(0)  # Predict the output
            # print(out)

        # Store the output to numpy array
        old = predict_wave[1:10, :, :].to(device)
        predict_wave = torch.cat((old, out), 0)
        out_wave[i + 10, :, :] = out.cpu().detach().numpy()

        # visualize every VISUAL_FREQ frames
        if i % config.VISUAL_FREQ == 0:
            print("frame:\t", i)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle("Prediction vs CFD")
            _ = ax1.imshow(out_wave[i + 10, :, :], cmap="gray")
            _ = ax2.imshow(test_wave[i + 10, :, :], cmap="gray")
            ax1.title.set_text("Prediction")
            ax2.title.set_text("CFD")
            plt.show()

        # Clean cuda memory (just in case)
        del test_input
        del out
        del old
        gc.collect()

    del predict_wave
    gc.collect()
    # err_wave = out_wave - test_wave[: 10 + PRED_FRAMES, :, :]
