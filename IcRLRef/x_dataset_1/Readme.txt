This folder contains two datsets. (x_dataset 1) and (x_dataset 2).

Both implement the same architecture and same process. 

For (x_dataset 1), in that folder

Total process: 3 steps. 

1. Process the data: Run Final_submission.ipynb file until cell 10 to get the processed file for further steps.



 We have 4 reward function. We have chosen reward function 3 (Dataset 1 reward function 3 folder). Lets see that. Go to reward function 3 folder

2. DDPG network:
   Contains two files : DDPG.py has the architecture, env.py has the enviornmnet. 
   Paste the csv file generated using step 1 in this folder. 
   To trian the ddpg network, in DDPG.py,  Uncomment sess.run(tf.global_variables_initializer()), line 771 and train(), line 773. 
   Once it starts running it will first show you alidation acccuracy and then test accuracy. 
   It generates results.csv which includes the results for each participant and gives the prediction of train data "predicted_data_train.csv
   Folder "hard" will contain the trained weights, copay these files and paste into a folder called 'Base_model'.   
	

DDPG.py has these functions train() - to train (sess.run(tf.global_variables_initializer()) needed to intialise weights)
            Need saver.restore(sess, tf.train.latest_checkpoint(path)) to load the weights for validation /testing
	    eval(i,j)- to validate
            testp(i,j)- test accuracy of each participants
            calculate_se()- Standard error calculation 
            test(i, j)- test accuracy and MRDE of each groups
            eval1(i,j)- this give the folder "predicted_data_train.csv


3. Personalisation: To personalise first we have to generate our dataset which includes tarin data where MRDE <1 and add participants data.
   To achive this run personalisation.ipynb, it generates files for each participants. For example for Group 1, particpant 17: Group1_17_662_1143_data.csv 
   Now paste Group1_17_662_1143_data.csv into G1_P17 folder. 
	G1_P17 folder has ddpg.py and env.py. 
	Run ddpg.py. It will give you the personalised acccuracy. 

	
