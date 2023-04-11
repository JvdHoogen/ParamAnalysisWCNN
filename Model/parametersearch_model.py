import itertools
import yaml
import datetime
import sys
import os
from tensorflow.keras import regularizers
import warnings

with open('parametersearch_combinations.yaml', 'r') as stream:
    try:
        inputdict = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

def print_time():
    parser = datetime.datetime.now() 
    return parser.strftime("%d-%m-%Y %H:%M:%S")


from tensorflow.keras.callbacks import Callback

class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """
    def __init__(self, monitor='accuracy', baseline=0.99, early=0):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline
        self.early = early


    def on_epoch_end(self, epoch, logs=None):
        
        logs = logs or {}
        acc = logs.get(self.monitor)
        if acc is not None:
            if acc >= self.baseline:
                self.early += 1
                print(self.early)
            if self.early == 5:                
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True
                self.early = 0


dataset = #add your own dataset to initialise the search, expected to receive a numpy array


total_list = [inputdict[key] for key in inputdict]
combinations = list(itertools.product(*total_list))
print(len(combinations))
print(combinations[0][0])


# sys.exit()
for i in combinations:
    print(i[0])
    came_here = False
    try:

        exec(open('imports_copy.py').read())
        keras.backend.clear_session()
        tf.keras.backend.clear_session()
        #%%
        # Read imports and utility functions
        seed = 1
        import os
        from tensorflow.python.eager import context
        from tensorflow.python.framework import ops
        import random
        import tensorflow
        os.environ['PYTHONHASHSEED']=str(seed)
        random.seed(seed)
        np.random.seed(seed)
        np.random.permutation(seed)
        tensorflow.random.set_seed(seed)
        os.environ['PYTHONHASHSEED']=str(seed)# 2. Set python built-in pseudo-random generator at a fixed value
        random.seed(seed)# 3. Set numpy pseudo-random generator at a fixed value
        tf.random.set_seed(seed)
        np.random.RandomState(seed)
        np.random.seed(seed)
        context.set_global_seed(seed)
        ops.get_default_graph().seed = seed

        #pip install tensorflow-determinism needed
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        np.random.seed(seed)
        
        import sys
        data = np.load(dataset)
        y = np.load(dataset)

        num_classes = len(np.unique(y))
        data = np.squeeze(data)
        
        from sklearn.model_selection import train_test_split
        y = y.flatten()
        num_classes = len(np.unique(y))
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.80,random_state = 1)


        

        scaler = StandardScaler()
        x_train2 = scaler.fit_transform(x_train)
        x_test2 = scaler.transform(x_test)

        start = time.time()
        #%%
        def wd_layers(nr_blocks,first_layer,seq_len,sens,wda,first_kernel_size,first_stride):
            model_list = []
            input_shape2 = []
            for i in range(nr_blocks):
                input_shape = Input((seq_len,sens))
                xx = Conv1D(filters=first_layer,kernel_size=first_kernel_size,strides=first_stride,padding='same')(input_shape)
                xx = BatchNormalization()(xx)
                xx = Activation('relu')(xx)
                xx = AveragePooling1D(strides=2)(xx)
                xx = Dropout(0.15)(xx)
                model_list.append(xx)
                input_shape2.append(input_shape)
            return(model_list,input_shape2)


        

        #%%
        # Full model function
        def full_model(x_train2,x_test2,y_train2,y_test2):

            #Create dummies for the labels
            y_train2 = pd.DataFrame(y_train2, columns=['label'])
            dummies = pd.get_dummies(y_train2['label']) # Classification
            products = dummies.columns
            y = dummies.values
            
            # Sequence length
            seq_len = x_test2.shape[1]
            sens = 1

            # Give values for the parameter search
            first_kernel_size = int(i[0])
            first_stride = int(i[1])
            first_layer = int(i[2])

            # Initialize k-folds
            kf = StratifiedKFold(3, shuffle=True,random_state=2) # Use for StratifiedKFold classification
            fold = 0

            # Build empty lists for results
            oos_y = []
            oos_pred = []
            oos_test_pred = []
            oos_test_y = []
            oos_test_prob = []
            oos_test_activations = []
            
            # Earlystopping callback
            earlystop_loss = EarlyStopping(monitor= 'val_loss', min_delta=0 , patience=10, verbose=0, mode='auto')
            earlystop_acc = TerminateOnBaseline(monitor='val_accuracy', baseline=1.0)
            
            
            # Initialize loop for every kth fold
            for train, test in kf.split(x_train2, y_train2['label']): 
                keras.backend.clear_session()
                tf.keras.backend.clear_session()
                model_checkpoint = ModelCheckpoint('model.hdf5', monitor='val_loss',verbose=0, save_best_only=True)
                fold+=1
                print(f"Fold #{fold}")
                x_train = x_train2[train]
                y_train = y[train]
                x_test = x_train2[test]
                y_test = y[test]


                xx,input_shape = wd_layers(1,first_layer,seq_len,sens,wda,first_kernel_size,first_stride)
                if len(xx) > 1:
                    xx = concatenate([k for k in xx])
                else:
                    xx = xx[0]
                
                xx = Conv1D(filters=i[3],kernel_size=i[4],strides=i[5],padding='same')(xx)
                xx = BatchNormalization()(xx)
                xx = Activation('relu')(xx)
                xx = AveragePooling1D(strides=2)(xx)
                xx = Dropout(0.15)(xx)
                
                ls = ['same','same','valid']
                for j in ls:
                    xx = Conv1D(filters=i[6],kernel_size=i[7],strides=i[8],padding=j)(xx)
                    xx = BatchNormalization()(xx)
                    xx = Activation('relu')(xx)
                    xx = AveragePooling1D(strides=2)(xx)
                    xx = Dropout(0.15)(xx)

                xx = Flatten()(xx)
                xx = Dense(100, activation = 'sigmoid')(xx)
                xx = Dropout(0.5)(xx)
                output = Dense(num_classes, activation = "sigmoid")(xx)

                # Create model
                wcnn = Model(inputs=input_shape,outputs=output)
                print(wcnn.summary())


                
            
                nr_params = wcnn.count_params()
                # initialize optimizer and random generator within one fold
                keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=156324)
                opt = tf.optimizers.Adam(learning_rate=0.01)
                wcnn.compile(optimizer=opt,
                            loss='mean_squared_error',
                            metrics=['accuracy'])

                # # Fit the model
                wcnn.fit(x_train, y_train,validation_data = (x_test,y_test), epochs = 100, batch_size = 64, verbose=1, 
                            callbacks =[earlystop_loss,earlystop_acc,model_checkpoint], shuffle = True)


                n_epochs = len(wdcnn_multi.history.history['loss'])


                
                wcnn = load_model('./model.hdf5')

                # Predictions on the validation set
                predictions = wcnn.predict(x_test)
                #predictions = wdcnn_multi.predict([x_test10,x_test20,x_test30,x_test40])




                # Append actual labels of the validation set to empty list
                oos_y.append(y_test)
                # Raw probabilities to chosen class (highest probability)
                predictions = np.argmax(predictions,axis=1)
                # Append predictions of the validation set to empty list
                oos_pred.append(predictions)  
                
                # Measure this fold's accuracy on validation set compared to actual labels
                y_compare = np.argmax(y_test,axis=1) 
                score = metrics.accuracy_score(y_compare, predictions)
                print(f"Validation fold score(accuracy): {score}")
                
                # Predictions on the test set
                test_predictions_loop = wdcnn_multi.predict(x_test2)
                #test_predictions_loop = wdcnn_multi.predict([datax_test1,datax_test2,datax_test3,datax_test4])

                # Append actual labels of the test set to empty list
                oos_test_y.append(y_test2)
                # Append raw probabilities of the test set to empty list
                oos_test_prob.append(test_predictions_loop)
                # Raw probabilities to chosen class (highest probability)
                test_predictions_loop = np.argmax(test_predictions_loop, axis=1)
                # Append predictions of the test set to empty list
                oos_test_pred.append(test_predictions_loop)
                
                # Measure this fold's accuracy on test set compared to actual labels
                test_score = metrics.accuracy_score(y_test2, test_predictions_loop)
                print(f"Test fold score (accuracy): {test_score}")
                                
                keras.backend.clear_session()
                tf.keras.backend.clear_session()
                

            # Build the prediction list across all folds
            oos_y = np.concatenate(oos_y)
            oos_pred = np.concatenate(oos_pred)
            oos_y_compare = np.argmax(oos_y,axis=1) 

            # Measure aggregated accuracy across all folds on the validation set
            aggregated_score = metrics.accuracy_score(oos_y_compare, oos_pred)
            print(f"Aggregated validation score (accuracy): {aggregated_score}")    
            
            # Build the prediction list across all folds
            oos_test_y = np.concatenate(oos_test_y)
            oos_test_pred = np.concatenate(oos_test_pred)
            oos_test_prob = np.concatenate(oos_test_prob)
            
            # Measure aggregated accuracy across all folds on the test set
            aggregated_test_score = metrics.accuracy_score(oos_test_y, oos_test_pred)
            print(f"Aggregated test score (accuracy): {aggregated_test_score}")

            end = time.time()
            runtime = exec_time(start,end)
            time_and_date = print_time()
            return(oos_test_prob, oos_test_y, aggregated_score, aggregated_test_score, runtime, 
                oos_test_activations, oos_test_y,nr_params,first_layer,first_stride,first_kernel_size,time_and_date,n_epochs)


        # Initialize the full_model_WDMTCNN function
        oos_test_y = []
        oos_test_prob = []
        aggregated_score = 0
        aggregated_test_score = 0
        earlystop = 0
        runtime = 0
        oos_test_activations = []
        oos_test_y = []
        

        oos_test_prob, oos_test_y, aggregated_score, aggregated_test_score, runtime, oos_test_activations, oos_test_y,nr_params,first_layer,first_stride,first_kernel_size,time_and_date,n_epochs= full_model(x_train2,x_test2,y_train,y_test)

        # %%
        ls = []
        ls.append([time_and_date,i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7],i[8],aggregated_score, aggregated_test_score, nr_params, earlystop,runtime, 
                num_classes,n_epochs])

        with open('results.txt', 'a') as f:
            for item in ls:
                f.write("%s\n" % item)
    
