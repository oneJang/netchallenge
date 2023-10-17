#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : client_isic.py
# Modified   : 08.03.2022
# By         : Sandra Carrasco <sandra.carrasco@ai.se>
import os
from collections import OrderedDict
import numpy as np  
from typing import List, Tuple, Dict
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader 
from argparse import ArgumentParser 
import flwr as fl 
import utils
from utils import Net, seed_everything  , training_transforms, testing_transforms
import time
import wandb 
from env_config import Settings
import warnings

warnings.filterwarnings("ignore")
seed = 2022
seed_everything(seed)

EXCLUDE_LIST = [
    #"running",
    #"num_batches_tracked",
    #"bn",
]
settings = Settings()

class Client(fl.client.NumPyClient):
    """Flower client implementing melanoma classification using PyTorch."""

    def __init__(
        self,
        model: Net,
        trainloader: torch.utils.data.DataLoader,
        valloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        num_examples: Dict, 
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_properties(self, config):
        return {} 
    
    def get_parameters(self, config={}) -> List[np.ndarray]:
        print("get_parameter호출 -- 로컬 모델에서 파라미터 뽑기")
        parameters = []
        for i, (name, tensor) in enumerate(self.model.state_dict().items()):
            #print(f"  [layer {i}] {name}, {type(tensor)}, {tensor.shape}, {tensor.dtype}")

            # Check if this tensor should be included or not
            exclude = False
            for forbidden_ending in EXCLUDE_LIST:
                if forbidden_ending in name:
                    exclude = True
            if exclude:
                continue

            # Convert torch.Tensor to NumPy.ndarray
            parameters.append(tensor.cpu().numpy())

        return parameters


    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        print("set_parameters호출 -- 로컬 모델에 가져온 파라미터 적용하기")
        keys = []
        for name in self.model.state_dict().keys():
            # Check if this tensor should be included or not
            exclude = False
            for forbidden_ending in EXCLUDE_LIST:
                if forbidden_ending in name:
                    exclude = True
            if exclude:
                continue

            # Add to list of included keys
            keys.append(name)

        params_dict = zip(keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=False)
        #torch.save(self.model, "./aggregation_straggler_parameters/edge1/late_model.pth")
        #joblib.dump(self.model.state_dict(), './model/model-1.pkl')
        # print("set_parameters를 통해 적용한 모델 저장!")
    

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        print("fit시작")
        self.set_parameters(parameters)
        torch.save(self.model, './model/initial_model_edge%s.pth'%settings.EDGE_NUM) #이게 total_round - 1에서의 글로벌 모델이 될 수도..
        start = time.time()
        self.model = utils.train(self.model, self.trainloader, self.valloader, self.num_examples, args.partition, 
                                args.nowandb, device, args.log_interval, epochs=args.epochs, es_patience=3)
        end = time.time()
        if end-start > 1800.0:
            if not os.path.exists("./aggregation_straggler_parameters/edge%s"%settings.EDGE_NUM):
                env = "edge"+settings.EDGE_NUM
                dir_path = "./aggregation_straggler_parameters/"+env
                os.mkdir(dir_path)
                os.chmod(dir_path, 0o777)
                
            self.set_parameters(self.get_parameters())
            torch.save(self.model, "./aggregation_straggler_parameters/edge%s/late_model.pth"%settings.EDGE_NUM)
            print("edge%s의 straggler 모델 저장!"%settings.EDGE_NUM)

        return self.get_parameters(), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]: 
        # Set model parameters, evaluate model on local test dataset, return result
        print("evalute() 호출")
        self.set_parameters(parameters)
        loss, auc, accuracy, f1 = utils.val(self.model, self.testloader, nn.BCEWithLogitsLoss(), f"_test",args.nowandb, device)
        print("accuracy:",accuracy, "auc:", auc, "f1:", f1)

        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy), "auc": float(auc), "f1": float(f1)}

if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument("--model", type=str, default="efficientnet-b2") 
    parser.add_argument("--log_interval", type=int, default=100)  
    parser.add_argument("--epochs", type=int, default=2)  
    parser.add_argument("--batch_train", type=int, default=32) 
    parser.add_argument("--num_partitions", type=int, default=5) 
    parser.add_argument("--partition", type=int, default=0)   
    parser.add_argument("--gpu", type=int, default=0)   
    parser.add_argument("--tags", type=str, default="Exp 5. FedBN") 
    parser.add_argument("--nowandb", action="store_true") 
    parser.add_argument("--path", type=str, default="./workspace/melanoma_isic_dataset")
    parser.add_argument("--host", type=str, default=settings.SERVER)
    parser.add_argument("--port", type=str, default=settings.PORT)
    args = parser.parse_args()

    # Setting up GPU for processing or CPU if GPU isn't available
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Load model
    model = utils.load_model(args.model, device)
    
    if not args.nowandb:
        wandb.init(project="dai-healthcare" , entity="eyeforai", group="FL", tags=[args.tags], config={"model": args.model})
        wandb.config.update(args) 
        # wandb.watch(model, log="all")
    
    # Load data
    # Normal partition
    trainset, valset, num_examples = utils.load_isic_data(args.path)
    trainset, valset, num_examples = utils.load_partition(trainset, valset, num_examples, idx=args.partition, num_partitions=args.num_partitions)
    # Exp 1
    # trainset, testset, num_examples = utils.load_exp1_partition(trainset, testset, num_examples, idx=args.partition)
    # Exp 2-6
    # train_df, validation_df, num_examples = utils.load_isic_by_patient(args.partition, args.path)
    # trainset = utils.CustomDataset(df = train_df, train = True, transforms = training_transforms) 
    # valset = utils.CustomDataset(df = validation_df, train = True, transforms = testing_transforms ) 
    testset =  utils.load_isic_by_patient(-1, args.path) #동일하게 해놔서 모든 클라이언트들 결과가 똑같을듯
    
    print(f"Train dataset: {len(trainset)}, Val dataset: {len(valset)}, Test dataset: {len(testset)}")

    train_loader = DataLoader(trainset, batch_size=args.batch_train, num_workers=0, worker_init_fn=utils.seed_worker, shuffle=True) 
    val_loader = DataLoader(valset, batch_size=16, num_workers=0, worker_init_fn=utils.seed_worker, shuffle = False)   
    test_loader = DataLoader(testset, batch_size=16, num_workers=0, worker_init_fn=utils.seed_worker, shuffle = False)   

    # Start client 
    client = Client(model, train_loader, val_loader, test_loader, num_examples)
    fl.client.start_numpy_client(server_address=args.host + ":" + args.port, client=client)
    
