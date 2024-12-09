# https://github.com/p-koo/evoaug_analysis/blob/main/evoaug_analysis/model_zoo.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, average_precision_score
#import tqdm
import numpy as np
import random

""" 1: TF ATF2 binds, 0: does not bind (h5: from ENCODE dataset)"""

"""
class ChipCNN1(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(ChipCNN1, self).__init__()
        initializer = torch.nn.init.kaiming_uniform_
        
        # First conv layer
        #self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=64, kernel_size=7, padding=3)
        self.conv1 = nn.Conv1d(in_channels=input_shape[0], out_channels=64, kernel_size=7, padding=3)
        self.batchnorm1 = nn.BatchNorm1d(64)
        
        # Second conv layer
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=5, padding=2)
        self.batchnorm2 = nn.BatchNorm1d(96)
        
        # Third conv layer
        self.conv3 = nn.Conv1d(in_channels=96, out_channels=128, kernel_size=5, padding=2)
        self.batchnorm3 = nn.BatchNorm1d(128)
        
        # Dense layers
        self.fc1 = nn.Linear(128 * (input_shape[0] // 32), 256)
        self.batchnorm4 = nn.BatchNorm1d(256)
        
        # Output layer
        self.fc2 = nn.Linear(256, output_shape)

    def forward(self, x):
        # First conv layer
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = F.dropout(x, p=0.2, training=self.training)

        # Second conv layer
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = F.dropout(x, p=0.2, training=self.training)

        # Third conv layer
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2)
        x = F.dropout(x, p=0.2, training=self.training)

        # Dense layers
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.batchnorm4(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Output layer
        x = self.fc2(x)
        output = torch.sigmoid(x)
        return output
"""

"""
class ChipCNN(nn.Module):
    def __init__(self):
        super(ChipCNN, self).__init__()
        ##self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=7, padding=3)
        #self.conv1 = nn.Conv1d(in_channels=200, out_channels=64, kernel_size=7, padding=3)
        self.conv1 = nn.Conv1d(in_channels=200, out_channels=64, kernel_size=7, padding="same") #AC orig
        #self.conv1 = nn.Conv1d(in_channels=200, out_channels=64, kernel_size=7, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(kernel_size=4)
        self.dropout1 = nn.Dropout(0.2)

        #self.conv2 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=5, padding=2)
        #self.conv2 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=5, padding="same") #AC orig
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=5, padding=4)
        self.bn2 = nn.BatchNorm1d(96)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(kernel_size=4)
        ##self.maxpool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.2)

        #self.conv3 = nn.Conv1d(in_channels=96, out_channels=128, kernel_size=5, padding=2)
        #self.conv3 = nn.Conv1d(in_channels=96, out_channels=128, kernel_size=5, padding="same") #AC orig
        self.conv3 = nn.Conv1d(in_channels=96, out_channels=128, kernel_size=5, padding=3)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool1d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.2)

        self.flatten = nn.Flatten()
        #self.fc1 = nn.Linear(128 * 25, 256) #AC orig
        self.fc1 = nn.Linear(128, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x = self.dropout1(self.maxpool1(self.relu1(self.bn1(self.conv1(x)))))
        #x = self.dropout2(self.maxpool2(self.relu2(self.bn2(self.conv2(x)))))
        #x = self.dropout3(self.maxpool3(self.relu3(self.bn3(self.conv3(x)))))
        #x = self.dropout4(self.relu4(self.bn4(self.fc1(self.flatten(x)))))
        out=x
        #
        out=self.conv1(out)
        #print(f"{out.shape=}")
        out=self.bn1(out)
        #print(f"{out.shape=}")
        out=self.relu1(out)
        #print(f"{out.shape=}")
        out=self.maxpool1(out)
        #print(f"{out.shape=}")
        out=self.dropout1(out)
        #print(f"{out.shape=}")
        #
        out=self.conv2(out)
        #print(f"{out.shape=}")
        out=self.bn2(out)
        #print(f"{out.shape=}")
        out=self.relu2(out)
        #print(f"{out.shape=}")
        #print(f"relu2 {out.shape=}")
        out=self.maxpool2(out)
        #print(f"{out.shape=}")
        #print(f"maxpool2 {out.shape=}")
        out=self.dropout2(out)
        #print(f"{out.shape=}")
        #
        out=self.conv3(out)
        out=self.bn3(out)
        out=self.relu3(out)
        out=self.maxpool3(out)
        out=self.dropout3(out)
        #
        #print(f"dropout3; {out.shape=}")
        out=self.flatten(out)
        #print(f"flatten: {out.shape=}")
        out=self.fc1(out)
        out=self.bn4(out)
        out=self.relu4(out)
        out=self.dropout4(out)
        #
        out = self.sigmoid(self.fc2(out))
        return out
"""

class In_House_CNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        
        self.activation1 = nn.ReLU()
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.2)
        self.dropout4 = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.output_activation = nn.Sigmoid()

        # Layer 1 (convolutional), constituent parts
        self.conv1_filters = torch.nn.Parameter(torch.zeros(64, 4, 7))
        torch.nn.init.kaiming_uniform_(self.conv1_filters)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.maxpool1 = nn.MaxPool1d(4)
        
        # Layer 3 (convolutional), constituent parts
        self.conv2_filters = torch.nn.Parameter(torch.zeros(96, 64, 5))
        torch.nn.init.kaiming_uniform_(self.conv2_filters)
        self.batchnorm2 = nn.BatchNorm1d(96)
        self.maxpool2 = nn.MaxPool1d(4)
        
        # Layer 4 (convolutional), constituent parts
        self.conv3_filters = torch.nn.Parameter(torch.zeros(128, 96, 5))
        torch.nn.init.kaiming_uniform_(self.conv3_filters)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.maxpool3 = nn.MaxPool1d(2)
        
        # Layer 5 (fully connected), constituent parts
        self.fc4 = nn.LazyLinear(256, bias=True)
        self.batchnorm4 = nn.BatchNorm1d(256)
        
        # Output layer (fully connected), constituent parts
        self.fc5 = nn.LazyLinear(output_dim, bias=True)
    
    def forward(self, x):
        # Layer 1
        cnn = torch.conv1d(x, self.conv1_filters, stride=1, padding="same")
        cnn = self.batchnorm1(cnn)
        cnn = self.activation1(cnn)
        cnn = self.maxpool1(cnn)
        cnn = self.dropout1(cnn)
        
        # Layer 2
        cnn = torch.conv1d(cnn, self.conv2_filters, stride=1, padding="same")
        cnn = self.batchnorm2(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool2(cnn)
        cnn = self.dropout2(cnn)
        
        # Layer 3
        cnn = torch.conv1d(cnn, self.conv3_filters, stride=1, padding="same")
        cnn = self.batchnorm3(cnn)
        cnn = self.activation(cnn)
        cnn = self.maxpool3(cnn)
        cnn = self.dropout3(cnn)
        
        # Layer 4
        cnn = self.flatten(cnn)
        cnn = self.fc4(cnn)
        cnn = self.batchnorm4(cnn)
        cnn = self.activation(cnn)
        cnn = self.dropout4(cnn)
        
        # Output layer
        logits = self.fc5(cnn) 
        y_pred = self.output_activation(logits)
        
        return y_pred

def mcdropout_prediction(model,x,device,seed=41):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'): #QUIQUINONURG endswith ropout???
            m.train()
    out=model(x.to(device))
    x.detach().cpu() #.to('cpu')
    return out

def eval_InHouseCNN(model, dataloader, loss_fn, device):
    model.eval()

    with torch.no_grad():
        model.eval()
        eval_loss = 0.0

        accuracy = 0.0
        roc_auc = 0.0
        aupr = 0.0
        for inputs, labels in dataloader:
            #print(f"{len(inputs)=}")
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs)
            eval_loss += loss_fn(output,labels).item() * inputs.size(0) # QUIQUI

            predicted = output.round().squeeze(1) #.max(1)
            accuracy += predicted.eq(labels.squeeze(1)).sum().item() / len(labels)

            outp=output.cpu()
            labs=labels.cpu()

            """
            roc_auc += roc_auc_score(labs, outp)
            aupr += average_precision_score(labs, outp)
            """
            if len(np.unique(predicted.detach().cpu().numpy()))!=1: #may not be necessary for torch 2.0.1 #QUIQUIURG this is to prevent the error of ROC_AUC not calculable if only one class, but is this correct to implement? Or should I prevent the error at all costs?
                roc_auc += roc_auc_score(labs, outp)
                aupr += average_precision_score(labs, outp)
            else:
                roc_auc += 0.
                aupr += 0.

        eval_loss /= len(dataloader)
        accuracy /= len(dataloader) # /labels.size(0)
        roc_auc /= len(dataloader)
        aupr /= len(dataloader)

    return roc_auc,aupr,accuracy,eval_loss

#def incremental_train_InHouseCNN(model, indexes_dataloader, dataloader, loss_fn, optimizer, device, valid_dataloader, scheduler):
def incremental_train_InHouseCNN(model, 
                               new_X, new_y, 
                               X_train, y_train, #train_dataloader, 
                               loss_fn, optimizer, device, valid_dataloader, scheduler, 
                               incremental_method='random'):
    if incremental_method=='random':
        new_dataloader=torch.utils.data.DataLoader(list(zip(new_X,new_y)), batch_size=batch_size, shuffle=True)
        model, train_loss, valid_loss, train_acc, valid_accr = train_InHouseCNN(model, new_dataloader, loss_fn, optimizer, device, valid_dataloader, scheduler)
        indexes=np.arange(len(X_train))[:len(new_X)*10] #QUIQUIURG
        prev_dataloader=torch.utils.data.DataLoader(list(zip(X_train,y_train))[indexes], batch_size=batch_size, shuffle=True)
        model, train_loss_prev, valid_loss_prev, train_acc_prev, valid_accr_prev = train_InHouseCNN(model, prev_dataloader, loss_fn, optimizer, device, valid_dataloader, scheduler)
    elif incremental_method=='RL':
        print("Selecting what old batches to reuse as RL is not implemented yet (and will raise the doubt on how can it generalize to games not-played-yet)")
        exit()
    return model, train_loss, valid_loss, train_acc, valid_accr, train_loss_prev, valid_loss_prev, train_acc_prev, valid_accr_prev

def train_InHouseCNN(model, dataloader, loss_fn, optimizer, device, valid_dataloader, scheduler):
    #model.train()
    total_train_loss = 0.0
    total_valid_loss = 0.0
    num_samples = 0

    train_acc=0.0
    valid_acc=0.0
    for inputs, labels in dataloader:
    #for inputs, labels in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc='Training CNNFrAmb', colour='white'):
        model.train()
        inputs, labels = inputs.to(device), labels.to(device)
        #print(f"{inputs.shape}")
        #print(f"{inputs.shape=}")
        #print(f"{labels.shape=}")

        optimizer.zero_grad()

        outputs = model(inputs)

        """
        print(f"{outputs[:5]=} {labels[:5]=}")
        print(f"{outputs[:5].round()=}")
        print(f"{outputs[:5].round().squeeze(1)=}")
        print(f"{outputs[:5].round().squeeze(1).eq(labels[:5].squeeze(1))=}")
        print(f"{(outputs.round().squeeze(1)).eq(labels.squeeze(1)).sum().item()=} {len(labels)=}")
        """

        on_the_fly_acc=(outputs.round().squeeze(1)).eq(labels.squeeze(1)).sum().item()/len(labels) 
        #print(f"{on_the_fly_acc=}")
        train_acc+=on_the_fly_acc/len(dataloader)

        #print(f"{outputs.shape=}")
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        #roc_auc,aupr,accuracy,valid_loss = eval_InHouseCNN(model, test_dataloader, loss_fn, device)
        valid_roc_auc,valid_aupr,valid_accuracy,valid_loss = eval_InHouseCNN(model, valid_dataloader, loss_fn, device)
        valid_acc+=valid_accuracy
        #print(f"{valid_accuracy=}")

        total_train_loss += loss.item() * inputs.size(0) #QUIQUI
        total_valid_loss += valid_loss # NO * inputs.size(0) a prescindere qui: lo metto nelle linee con .item()
        num_samples += inputs.size(0)

    #print(f"{inputs.size(0)=} {len(dataloader)=}") # inputs.size(0)=187 len(dataloader)=106

    average_train_loss = total_train_loss / len(dataloader) # / num_samples
    average_valid_loss = total_valid_loss / len(dataloader) # / num_samples
    valid_acc/=len(dataloader) #num_samples
    #print(f"{valid_acc=} {valid_accuracy=}")
    #print(f"{otf_acc=} {average_valid_loss=}")
    #print("LETS EVALUATE SO FAR: SHOULD BE: \n106/106 - 4s - loss: 0.5395 - accuracy: 0.7458 - auroc: 0.8228 - aupr: 0.8352 - val_loss: 1.7032 - val_accuracy: 0.5358 - val_auroc: 0.8435 - val_aupr: 0.7787 - lr: 0.0010 - 4s/epoch - 39ms/step")
    #scheduler.step(valid_loss) 
    scheduler.step(valid_acc) # to be done at the end of an epoch: https://pytorch.org/docs/stable/optim.html

    return model, average_train_loss, average_valid_loss, train_acc, valid_acc

class EarlyStopper:
    # https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        #elif validation_loss > (self.min_validation_loss + self.min_delta):
        elif (validation_loss-self.min_validation_loss) < self.min_delta:
            self.counter += 1
            #print(f"{validation_loss=} {self.min_validation_loss=} {validation_loss-self.min_validation_loss=}  {self.min_delta=}")
            if self.counter >= self.patience:
                return True
        return False



###########################################################################



if __name__=='__main__':
    print("WARNING The model has been altered: vedi AC orig")
    import h5py
    #import torch.multiprocessing as mp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h5file='../inputs/ATF2_200.h5'
    #h5file='../inputs/ATF2_200_remade.h5'
    data = h5py.File(h5file, 'r')
    #print(h5f.keys())
    #print(h5f['x_train'].shape)
    #print(h5f['y_train']) #.shape)

    #X=h5f['x_train']
    #y=h5f['y_train']

    #X_train = torch.tensor(np.transpose(data['x_train'][:, :4, :], (0, 2, 1)), dtype=torch.float32)
    X_train = torch.tensor(np.transpose(data['x_train'][:, :4, :], (0, 1, 2)), dtype=torch.float32) #CNNAMBER
    y_train = torch.tensor(data['y_train'][()], dtype=torch.float32)
    #X_valid = torch.tensor(np.transpose(data['x_valid'][:, :4, :], (0, 2, 1)), dtype=torch.float32)
    X_valid = torch.tensor(np.transpose(data['x_valid'][:, :4, :], (0, 1, 2)), dtype=torch.float32) #CNNAMBER
    y_valid = torch.tensor(data['y_valid'][()], dtype=torch.float32)
    #X_test = torch.tensor(np.transpose(data['x_test'][:, :4, :], (0, 2, 1)), dtype=torch.float32)
    X_test = torch.tensor(np.transpose(data['x_test'][:, :4, :], (0, 1, 2)), dtype=torch.float32) #CNNAMBER
    y_test = torch.tensor(data['y_test'][()], dtype=torch.float32)
    
    print(f"{X_train.shape=}")
    print(f"{X_test.shape=}")
    print(f"{X_valid.shape=}")

    #def make_Xy(X,y):
    #    Xy=torch.tensor(np.array([X,y]))
    #    print(Xy.shape)
    #    return Xy
    #Xy_train=make_Xy(X_train,y_train)


    batch_size=256 #25 #GOODOLD: 100 #256 #10 #256 #100 ORIG: 256
    loss_fn = nn.BCELoss().to(device)
    learning_rate = 0.001 #0.01 # 1e-6 #1e-6 #0.001 ORIG: 1e-3
    num_epochs = 40 #100 # GOODOLD: 40 #100 #100 #20 #5 #100 ORIG:100
    n_deep_ensembles=5 #50 #5

    torch.manual_seed(41)
    train_dataloader=torch.utils.data.DataLoader(list(zip(X_train,y_train)), batch_size=batch_size, shuffle=True)
    test_dataloader=torch.utils.data.DataLoader(list(zip(X_test,y_test)), batch_size=batch_size, shuffle=True)
    valid_dataloader=torch.utils.data.DataLoader(list(zip(X_valid,y_valid)), batch_size=batch_size, shuffle=True)

    # Assuming input_shape and output_shape are known
    input_shape = X_train[0].shape #(input_length, input_channels)  # Update input_length and input_channels
    output_shape = 1 #y_train[0].shape
    print(f"{input_shape=} {output_shape=}")



    def single_model(i_de):
        #global device, learning_rate, loss_fn, num_epochs #QUIQUIURG
        np.random.seed(41+i_de)
        torch.manual_seed(41+i_de)
        
        #model = ChipCNN1(input_shape=input_shape, output_shape=output_shape).to(device)
        #model = ChipCNN().to(device)
        model = In_House_CNN(output_dim=1).to(device) #CNNAMBER

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, min_lr=1e-6, factor=0.2)

        # Loading
        #pthfile='../../outputs_DALdna/rank_random_6/DAL_Model_j-0_ChosenModel-CNNFrAmb_pN-2707_gU-2560_mxe-40_ALc-90_itr-0_bao-1_seqmeth-Xy-from-ds_unc-Deep_Ensemble_pristmeth-random_rank-random_seedadd-46.pth' #BAD ONE
        #pthfile='../../outputs_DALdna/rank_mcdropout5_shuffletest/DAL_Model_j-0_ChosenModel-CNNFrAmb_pN-2707_gU-2560_mxe-40_ALc-96_itr-retrain_bao-1_seqmeth-Xy-from-ds_pristmeth-random_rank-mc_dropout_5_seedadd-46.pth' #BAD ONE
        #pthfile='../outputs/jobseed5_mcdr5_corrupted.pth'
        #model.load_state_dict(torch.load(pthfile))

        # Training loop
        #early_stopper = EarlyStopper(patience=10, min_delta=1e-3)
        early_stopper = EarlyStopper(patience=5, min_delta=0.5)
        for epoch in range(num_epochs):
            model, train_loss, valid_loss, train_acc, valid_accr = train_InHouseCNN(model, train_dataloader, loss_fn, optimizer, device, valid_dataloader, scheduler)
            #print(f"Model {i_de}, Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")
            freq=5 #500 #5 
            #if epoch%(int(num_epochs/freq))==0: print(f"Model {i_de}, Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")
            #if epoch%100==0: print(f"Model {i_de}, Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}")
            if epoch%5==0: print(f"Model {i_de}, Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Valid Accuracy: {valid_accr:.4f}")

            #if early_stopping(train_loss, valid_loss, tolerance=0.001): #min_delta=10, tolerance = 20):
            """
            if early_stopper.early_stop(valid_loss):             
                print(f"Early Stopping at epoch: {epoch}")
                break
            """

        #print("Saving model.")
        ##torch.save(model.state_dict(),'../inputs/chipcnn_aft2.pth') 
        ##torch.save(model.state_dict(),'../inputs/cnnamb_aft2.pth')   #CNNAMBER
        #torch.save(model.state_dict(),'../inputs/cnnamb_aft2_new.pth')   #CNNAMBER #GOODOLD

        #print("=== Test:")
        # Test loop
        #test_loss = test_InHouseCNN(model, test_dataloader, loss_fn, device)
        roc_auc,aupr,accuracy,test_loss = eval_InHouseCNN(model, test_dataloader, loss_fn, device)
        print(f"{test_loss=} {valid_loss=}")
        return roc_auc,aupr,accuracy


    accuracies=[]
    roc_aucs=[]
    auprs=[]
    #for i_de in range(n_deep_ensembles):
    for i_de in range(1,n_deep_ensembles+1): #in this way, seed 46 will be included ### If I Still Want To Investigate If It Depends On The Initialization: Save All Initial Weights And Make DimRed With An AutoEncoder
    #for i_de in [5]: #[405]: #test seed 46
        print()
        """
        mp.set_start_method('spawn')  # 'spawn' is usually preferred on Windows
        process_pool=mp.Pool(processes=self.N_for_uncertainty) 
        #multiproc_results = [process_pool.apply(self.jModel) for j in range(self.N_for_uncertainty)] #args=(arg1, arg2, ...)) for _ in range(num_processes)]
        # Close the process pool and wait for all processes to finish
        process_pool.close()
        process_pool.join()
        """
        roc_auc,aupr,accuracy=single_model(i_de)

        #print(f"Test Loss: {test_loss:.4f}")
        print(f"{accuracy=} {roc_auc=} {aupr=} ")
        roc_aucs.append(roc_auc)
        accuracies.append(accuracy)
        auprs.append(aupr)

    print(f"\naccuracy: {np.mean(accuracies)} +- {np.std(accuracies)} \nroc_auc: {np.mean(roc_aucs)} +- {np.std(roc_aucs)} \naupr: {np.mean(auprs)} +- {np.std(auprs)} ")

    print("SCRIPT END.")

    print("TODO: Ray Tune?")
    print("Apply EIG to DAL?")
    print("cum per unc in DAL_Pipeline for Classif")
    print("wandb")