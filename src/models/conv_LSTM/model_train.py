import os
import numpy as np
import torch
from definitions import PRE_TRAINED_MODEL_DIR
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, optim, 
                criterion, num_examples, device, num_epochs, save_path, best_loss=100000000):

    SAVED_MODEL_PATH = os.path.join(PRE_TRAINED_MODEL_DIR, save_path)

    best_val_loss = best_loss 

    improved = False
    error_factor = 255 # Jordan: I found this will improve the training
    # List to store the training history
    train_loss_list = []
    val_loss_list = []

    # Data Structure
    # Model Input  (batch_size, channel=1, frames=10, 64, 64)
    # Model Output (batch_size, channel=1, 64, 64)
    # Target       (batch_size, channel=1, 64, 64)

    # Training
    for epoch in range(1, num_epochs+1):
        train_loss = 0                                                 
        model.train()

        # Select a batch from train_loader
        for batch_num, (input, target) in enumerate(train_loader, 1):

            # Predict the 11th frame from the 10 inputted frames
            output = model(input) 
            
            # Catch nan Conv-LSTM output error
            if (torch.isnan(output[0,0,0,0])):
                print("ERROR nan OUTPUT!!!!!! Try to Restart Colab (This nan eror happens to me several times, I solve it by restarting Colab. Maybe it is becasue of the poor initialization of the model parameters")        
                break

            # Optimize model
            # compute the error between the predicted frame and the target frame
            loss = criterion(error_factor*output.flatten(), error_factor*target.flatten())
            loss.backward()     # backward propogation of the loss
            optim.step()        # update the model
            optim.zero_grad()
            train_loss += loss.item()

        train_loss = train_loss / (num_examples*0.8)   # (N*0.8) size of train data                                   

        # Check Validation Loss
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for input, target in val_loader:
                output = model(input)
                loss = criterion(error_factor*output.flatten(), error_factor*target.flatten())
                val_loss += loss.item() 
            
            # val_loss and train_loss is not in the same unit. because train_loss has way more samples and here I just add it up the loss                              
            val_loss = val_loss / (num_examples*0.1)   # (N*0.1): size of val data

        print("Epoch:{} Training Loss:{:4.4f} Validation Loss:{:4.4f}\n".format(epoch, train_loss, val_loss))
        
        # Store training history (train_loss, val_loss)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        # Save model with best validation loss
        if val_loss < best_val_loss:
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'val_loss_history': np.array(val_loss_list),
                'train_loss_history': np.array(train_loss_list)
            }
            improved = True
            torch.save(state, SAVED_MODEL_PATH)
            print("Best Model Saved to: ",SAVED_MODEL_PATH)
        
        # Auto-Save Training History (every 20 (or whatever number) epochs)
        #if epoch % 20 == 0:
        #    train_loss_list.append(np.array(train_loss))
        #    val_loss_list.append(np.array(val_loss))
        #    # TODO: Save the train_list and val_list

    # After the Training, load the best model if it exists 
    # (For model retraining, both the model and the optimizer need to be saved)
    if improved:
        checkpoint = torch.load(SAVED_MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['val_loss']
        print("Show training history")
        train_loss_history = checkpoint['train_loss_history']
        val_loss_history = checkpoint['val_loss_history']
        e = np.arange(train_loss_history.shape[0]) + 1
        plt.figure(1);
        plt.plot(e,train_loss_history)
        plt.plot(e,val_loss_history)
        plt.title('Train History')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['train','val'])
        plt.show() 
        del checkpoint
