import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def train_RLCNN(policy, criterion, optim, input_data, load_path, save_path, last_path, hist_path, stop_crit, r_train, n_train_vid, t_frame, r_freq, r_epoch, n_epoch, dev, w_play, save_flag):
  # Init
  epoch_loss_history = []
  best_loss = np.inf
  Improved = False
  schedule_idx = 0
  stop_criteria = stop_crit
  ReTrain = r_train
  num_train_videos = n_train_vid
  target_frame = t_frame
  render_freq = r_freq
  render_epoch = r_epoch
  num_epoch = n_epoch
  device = dev
  weight_play = w_play
  isSave = save_flag

  # Shallow
  x_dim = input_data.shape[2]
  y_dim = input_data.shape[3]

  # Configure Path
  MODEL_LOAD_PATH = load_path
  MODEL_SAVE_PATH = save_path
  MODEL_LAST_PATH = last_path
  TRAIN_HIST      = hist_path

  # Load pre-trained model if exists
  if ReTrain:
    print("Loading pre-trained model")
    checkpoint = torch.load(MODEL_LOAD_PATH,map_location=device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.train()
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    best_loss = checkpoint['loss']
    play_frame = checkpoint['game_step']
    print("load model from: ",MODEL_LOAD_PATH)
    print("best Play frame before training: ",play_frame)
    print("best loss before training: ",best_loss)
    del checkpoint

  print("START TRAINING")
  ### Start For Loop
  for e in range(1,1+num_epoch):
    # Set render Flag
    if e % render_epoch == 0:
      torender = True
    else:
      torender = False


    # Train the model
    state = train_epoch(x_dim, y_dim, policy, criterion, optim, e, input_data, num_train_videos, stop_criteria, weight_play, device, target_frame, render_freq , render=torender, start_index=-1)
  
    # Log Info
    if torender:
      print("===== Epoch =====",e)
      print("stop criteria: {:7d}".format(stop_criteria))
      print("f-frame error: {:7.0f}".format(state['final_frame_error']))
      print("RL Best Loss:  {:7.2f}".format(best_loss))
      print("RL Epoch Loss: {:7.2f}".format(state['loss']))
      print("Traget Frame:  {:3d}".format(state['target_frame']))
      print("Play Frame:    {:3d}".format(state['game_step']))
      print("\n")

    # Append the training history
    epoch_loss_history.append(state['loss'])

    # Save Training History every 1000 epochs
    if e % render_epoch == 0 and isSave:
      np.save(TRAIN_HIST,np.array(epoch_loss_history))
      print("history saved")
      print("Get Better Model?\t",Improved)
  
    # Save Best Model
    if state['loss'] < best_loss and isSave:
      # Save
      torch.save(state, MODEL_SAVE_PATH)
      best_loss = state['loss']
      Improved = True
      print("***** Save Best Policy. Epoch: {:5d} ".format(state['epoch']),
            "Final Frame Error: {:5.2f} ".format(state['final_frame_error']),
            "Play Frame: {:4d} ".format(state['game_step']),
            "RL Loss: {:5.2f} *****".format(state['loss']))
      

    ### End For Loop ###
  
  # After Training, load the best model if it exists
  if Improved:
    print("Model Improved. Load the Best Model")
    checkpoint = torch.load(MODEL_SAVE_PATH,map_location=device)
    policy.load_state_dict(checkpoint['model_state_dict'])
    best_loss = checkpoint['loss']
    final_frame_error = checkpoint['final_frame_error']
    del checkpoint
    print("Best Model Info")
    print("RL Loss after training:\t",best_loss)
    print("Final Frame Error (accum)",final_frame_error)

  # Save Last Model
  torch.save(state,MODEL_LAST_PATH)
  print("last epoch save to: ",MODEL_LAST_PATH)


def train_epoch(x_dim, y_dim, policy, criterion, optim, epoch, input, num_videos, stop_criteria, weight_play, device, target_frame = 200, renderfreq = 20, render = False, start_index = -1):
  
  ### Setup ###
  WEIGHT_PLAY = weight_play

  # IsStop: flag to indicate when the cumulative error exceeds stop_criteria
  # IsDone: flag to indicate the epoch is done and the policy is updated

  # Choose the starting index (choose which "level" to start the game)
  if start_index == -1:
    #start_index = np.random.randint(0,1000-target_frame)
    #Rand Start
    start_index = np.random.randint(0,1000-target_frame-20, size=num_videos)
  if render:
    print("\ncheck epoch: ", epoch)
    #print("start index: ", start_index)
  
  ### Init ###
  epoch_done = False
  IsStop = False
  step_error = 0
  i = 0	# i: the step of the training (the clock of the game)

  # Obtain the first ten frames (Detect the first situaitons of the game)
  obs = torch.zeros(num_videos,10,x_dim,y_dim)
  #obs[:,0:10,:,:] = torch.clone(input[:,0+start_index:10+start_index,:,:])
  #Rand Start
  for k in range(num_videos):
    obs[k,0:10:,:] = torch.clone(input[k,0+start_index[k]:10+start_index[k],:,:])
  obs = obs.to(device)

  # Start the game 
  while epoch_done == False:
    
    ##### 1. Playing Loop #####
    if IsStop == False:
      # Prediction (make an action)
      with torch.no_grad():
        act = policy(torch.clone(obs[:,0:10,:,:])).view(num_videos,1,x_dim,y_dim) # act shape (num_videos,1,64,64)
        out = act + obs[:,9,:,:].unsqueeze(1)
  
      # Update Obs (change the game situation based on the action)
      #obs[:,0:9,:,:] = torch.clone(input[:,i+1+start_index:i+10+start_index,:,:])
      #Rand Start
      for k in range(num_videos):
        obs[k,0:9,:,:] = torch.clone(input[k,i+1+start_index[k]:i+10+start_index[k],:,:])

      obs[:,9,:,:] = out.view(num_videos,x_dim,y_dim) 
      
      # compute the current error and sum it up (compute the current score of the game, and sum it up with the past scores)
      #step_error += criterion(255*out.flatten(),255*input[:,i+10+start_index,:,:].to(device).flatten()) /num_videos
      #Rand Start
      for k in range(num_videos):
        step_error += (criterion(255*out[k].flatten(),255*input[k,i+10+start_index[k],:,:].to(device).flatten()))/num_videos
      #step_error = step_error / num_videos

      # logging the error 
      if i%renderfreq == 0 and render == True:
        print("step: ",i,"\tstep error {0:4.5f}".format(step_error.item()))
        #render_predict(num_videos,i,step_error,epoch,out,input,start_index) 

      # Each Prediction Step
      i = i+1
      
      # Conditions when the game is terminated
      # 1. cumulative error exceeds stop_criteria
      if (step_error > stop_criteria):
        IsStop = True
      if (i) >= target_frame:
        IsStop = True
    
    ##### 2. Calculating Result of Playing #####
    # Cumulattive error exceeds stop_criteria, get the last action, compute the loss(reward), and update the policy
    # (the game stops, count the score and optimize based on the score)
    else:
      act = policy(torch.clone(obs[:,0:10,:,:])).view(num_videos,1,x_dim,y_dim)
      out = act + obs[:,9,:,:].unsqueeze(1)

      # Update Obs
      #obs[:,0:9,:,:] = torch.clone(input[:,i+1+start_index:i+10+start_index,:,:])
      #Rand Start
      for k in range(num_videos):
        obs[k,0:9,:,:] = torch.clone(input[k,i+1+start_index[k]:i+10+start_index[k],:,:])

      obs[:,9,:,:] = out.view(num_videos,x_dim,y_dim)

      ### 2.1 RL Part ###
      #step_error += criterion(255*out.flatten(),255*input[:,i+10+start_index,:,:].to(device).flatten()) /num_videos # normalize error
      #Rand Start
      for k in range(num_videos):
        step_error += (criterion(255*out[k].flatten(),
                                255*input[k,i+10+start_index[k],:,:].to(device).flatten()))/ num_videos
      #step_error = step_error / num_videos

      # RL Reward Function: default target frame = 200
      #WEIGHT_PLAY = 10  # Jordan 04.04 (Testing This Effect..., default: 1. Idea: encouraging playing game with more steps)
      # loss = step_error + WEIGHT_PLAY*(target_frame-i)
      loss = step_error/stop_criteria + WEIGHT_PLAY*(target_frame-i)/target_frame


      ### 2.2 Update the policy ###
      loss.backward()
      optim.step()
      optim.zero_grad()

      # Finish Epoch Check (for nan error, rare)
      if render:
        print("%%% CHECK Action %%%\tmax: ",torch.max(act).item(),"min: ", torch.min(act).item() )

      # Epoch Done (Game over flag)
      epoch_done = True

      # checkpoint of this epoch result
      state = {'epoch': epoch,
               'model_state_dict': policy.state_dict(),
               'optimizer_state_dict': optim.state_dict(),
               'game_step':i,
               'target_frame': target_frame,
               'final_frame_error':step_error.item(),
               'loss': loss.item(),
               }

      return state
  
  ### End While loop ###

def render_predict(num_videos,step,step_error,epoch,out,input,start_index):
    error_txt = " Step: {:3d}".format(step) + '\nError: {:4.2f}'.format(step_error.item())
    fig, axs = plt.subplots(num_videos, 2, figsize=(5,5))
    fig.suptitle('Epoch: {:4d}'.format(epoch) + error_txt + '\nPrediction v.s. CFD', fontdict = {'fontsize' : 7})
    for video_index in range(0, num_videos):
       im = axs[video_index,0].imshow(out[video_index,0,:,:].detach().cpu().numpy(), cmap = "gray")
       im = axs[video_index,1].imshow(input[video_index,step+10+start_index[video_index],:,:].detach().cpu().numpy(), cmap = "gray")
       axs[video_index,0].axis('off')
       axs[video_index,1].axis('off')
       axs[video_index,0].set_title(str(step+10+start_index[video_index]),fontdict = {'fontsize' : 7})

    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.1, 
                        hspace=0.4)
    plt.show()
