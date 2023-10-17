import flwr as fl
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import ( #common: 서버와 클라이언트 사이에 공유되는 공통 컴포넌트
    FitIns, # client에 대한 지시사항 수행
    FitRes, # client로부터 온 응답 수행
    MetricsAggregationFn,
    NDArrays, # N차원 넘파이
    Parameters,
    Scalar,
    ndarrays_to_parameters, # N차원 넘파이를 파라미터 객체로 변환
    parameters_to_ndarrays, # 파라미터 객체를 N차원 넘파이로 변환
)
import timeit
from flwr.server.history import History
import numpy as np
from logging import DEBUG, INFO
import os
from collections import OrderedDict
import torch
import random
from flwr.common.logger import log
import concurrent.futures
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from flwr.server.criterion import Criterion
from env_config import Settings

FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[Union[Tuple[ClientProxy, FitRes], BaseException]],
]

# def handle_exception(future):
#     try:
#         # Future에서 예외 확인
#         future.result()
#     except Exception as e:        
#         # 예외 처리 작업 수행
#         print(f"Exception occurred in Future: {e}")
settings = Settings()

def get_parameters(model) -> List[np.ndarray]:
        print("get_parameter호출 -- 로컬 모델에서 파라미터 뽑기")
        parameters = []
        for i, (name, tensor) in enumerate(model.state_dict().items()):
            print(f"  [layer {i}] {name}, {type(tensor)}, {tensor.shape}, {tensor.dtype}")

            # Check if this tensor should be included or not
            # exclude = False
            # for forbidden_ending in EXCLUDE_LIST:
            #     if forbidden_ending in name:
            #         exclude = True
            # if exclude:
            #     continue

            # Convert torch.Tensor to NumPy.ndarray
            parameters.append(tensor.cpu().numpy())

        return parameters
    
def set_parameters(model, parameters: List[np.ndarray]): # -> None
    print("set_parameters호출 -- 모델에 파라미터 적용하기")
    keys = []
    for name in model.state_dict().keys():
        # Check if this tensor should be included or not
        # exclude = False
        # for forbidden_ending in EXCLUDE_LIST:
        #     if forbidden_ending in name:
        #         exclude = True
        # if exclude:
        #     continue

        # # Add to list of included keys
        keys.append(name)

    params_dict = zip(keys, parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)
    return model
    #torch.save(model, "./model/global_model.pth")
    
def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float]
) -> Tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    print("server.py의 fit_client() 호출")
    fit_res = client.fit(ins, timeout=timeout)
    return client, fit_res

def fit_clients( #서버에서 선택된 클라이언트들에게 모델 파라미터를 학습시키고 결과를 받음
    client_instructions: List[Tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> FitResultsAndFailures:
    """Refine parameters concurrently on all selected clients."""
    print("myserver.py의 fit_clients() 호출, timeout = ", timeout)
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []
    current_clients: List[Tuple[ClientProxy, FitIns]] = []
    
    for client in client_instructions:
        current_clients.append(client)
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        print("max_workers:", max_workers)
        submitted_fs = {
            executor.submit(fit_client, client_proxy, ins, timeout) #선택된 client에서 fit_client를 비동기적으로 실행하는 Future객체 리턴
            for client_proxy, ins in client_instructions
        }
        # wait하면서 unregister가 실행됨..오류뜨는 애들은 unregister되나봄 => 이걸 막아야할듯한디 -> unregister는 상관없고 예외처리를 해야할거같은디 -> ㄴㄴ unregister되면 안됨
        finished_fs, unfinished_fs = concurrent.futures.wait( # wait에서 쓰레드가 종료될 때까지 기다림, finished_fs: 완료된 작업, 완료되지 않은 작업: unfinished_fs
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )
        for future in finished_fs:
            try:
                # Future의 결과를 가져오고 첫 번째 요소 (ClientProxy)를 추출
                result: Tuple[ClientProxy, FitRes] = future.result() 
                results.append(result) # 정상적인 애들만 모일걸로 예상
            except Exception as e:
                print(f"Exception occurred in Future: {e}")
                # 예외 처리 (예외 발생 시 failures에는 Tuple(None, None) 추가
                failure: Tuple[ClientProxy, FitRes] = (None, None)
                failures.append(failure)
            
    print("len(finished):", len(finished_fs))
    print("len(unfinished):", len(unfinished_fs))
        
    # Gather results
    # for future in finished_fs: #finished_fs에 있는 내용만큼 _handle함수 호출
    #     _handle_finished_future_after_fit( # 여기서 성공과 실패 여부 나눔
    #         future=future, results=results, failures=failures
    #     )
        
    print("len(results):", len(results))
    print("len(failures):", len(failures))
    result_clientProxy = []
    for result in results:
        result_clientProxy.append(result[0])
    
    for clientProxy in result_clientProxy:
        for current_client in current_clients:
            if current_client[0] != clientProxy:
                continue
            else:
                current_clients.remove(current_client) 
    ## failure에 clientproxy, fitres 나 clientproxy, None이 들어가야함. current_clients는 clientproxy, fitIns가 들어가있어서 fitIns를 어찌해야할듯
    # 예외처리된 녀석의 fitRes도 얻기 쉽지 않음..일단 None으로 하는수밖에
    failures.clear()
    for current_client in current_clients:
        failures.append((current_client[0], None))
        # myClientManager.register(current_client[0])
        #여기에 failures에 있는 애들을 다시 register하는거를 하면 어떻게 될까 -> register()하는 방법을 모르겠음..그냥 register하면 안되는거같어 lock충돌이 일어나는듯
    
    return results, failures    

class myServer(fl.server.server.Server):

    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
            """Run federated averaging for a number of rounds."""
            print("server.py의 fit() 호출, timeout = ",timeout) #timeout이 얼마인지 보기 위해 추가해봄
            history = History()

            # Initialize parameters
            log(INFO, "Initializing global parameters")
            self.parameters = self._get_initial_parameters(timeout=timeout) #전략 또는 랜덤 클라이언트에게 초기 파라미터 가져옴
            log(INFO, "Evaluating initial parameters")
            res = self.strategy.evaluate(0, parameters=self.parameters)
            if res is not None:
                log(
                    INFO,
                    "initial parameters (loss, other metrics): %s, %s",
                    res[0],
                    res[1],
                )
                history.add_loss_centralized(server_round=0, loss=res[0])
                history.add_metrics_centralized(server_round=0, metrics=res[1])

            # Run federated learning for num_rounds
            log(INFO, "FL starting")
            start_time = timeit.default_timer()

            for current_round in range(1, num_rounds + 1):
                # Train model and replace previous global model
                res_fit = self.fit_round(
                    server_round=current_round,
                    timeout=timeout,
                )
                if res_fit is not None:
                    print("res_fit is not None..")
                    parameters_prime, fit_metrics, _ = res_fit  # fit_metrics_aggregated
                    #if current_round == num_rounds and parameters_prime:
                    if parameters_prime:
                        print("(last round... and )parameters_prime is not None..")
                        self.parameters = parameters_prime
                        
                        #parameter: List = []
                        # for byte in parameters_prime.tensors:
                        #     parameter.append(bytes_to_ndarray(byte))
                        parameter = parameters_to_ndarrays(parameters_prime)
                        for i in range(5):
                            if os.path.exists("./model/initial_model_edge%d.pth"%(i+1)):
                                model = torch.load("./model/initial_model_edge%d.pth"%(i+1))
                                break #일단 client-2꺼가 먼저 실행되어야할듯
                        global_model = set_parameters(model, parameter)
                        torch.save(global_model, "./model/global_model.pth")
                        print("\n==========global model을 저장함!!==========\n")
                        
                        global_parameters = get_parameters(global_model)
                        self.parameters = ndarrays_to_parameters(global_parameters)
                        
                        
                    history.add_metrics_distributed_fit(
                        server_round=current_round, metrics=fit_metrics
                    )
                # Evaluate model using strategy implementation
                res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
                if res_cen is not None:
                    loss_cen, metrics_cen = res_cen
                    log(
                        INFO,
                        "fit progress: (%s, %s, %s, %s)",
                        current_round,
                        loss_cen,
                        metrics_cen,
                        timeit.default_timer() - start_time,
                    )
                    history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                    history.add_metrics_centralized(
                        server_round=current_round, metrics=metrics_cen
                    )

                # Evaluate model on a sample of available clients
                res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
                if res_fed is not None:
                    loss_fed, evaluate_metrics_fed, _ = res_fed
                    if loss_fed is not None:
                        history.add_loss_distributed(
                            server_round=current_round, loss=loss_fed
                        )
                        history.add_metrics_distributed(
                            server_round=current_round, metrics=evaluate_metrics_fed
                        )
                
            
            # Bookkeeping
            end_time = timeit.default_timer()
            elapsed = end_time - start_time
            log(INFO, "FL finished in %s", elapsed)
            return history
        
    def fit_round(
            self,
            server_round: int,
            timeout: Optional[float],
        ) -> Optional[
            Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
        ]:
            """Perform a single round of federated averaging."""
            # Get clients and their respective instructions from strategy
            print("server.py의 fit_round() 호출")
            print("configure_fit을 통해 선택된 clientproxy와 fit_ins가져오기")
            

            client_instructions = self.strategy.configure_fit( #configure_fit을 통해 선택된 clientproxy와 fit_ins가져오기
                server_round=server_round,
                parameters=self.parameters,
                client_manager=self._client_manager,
            )
            if not client_instructions:
                log(INFO, "fit_round %s: no clients selected, cancel", server_round)
                return None
            log(
                DEBUG,
                "fit_round %s: strategy sampled %s clients (out of %s)",
                server_round,
                len(client_instructions),
                self._client_manager.num_available(),
            )
            #현재 참여중인 클라이언트 저장
            
            # Collect `fit` results from all clients participating in this round
            results, failures = fit_clients(
                client_instructions=client_instructions,
                max_workers=self.max_workers,
                timeout=timeout,
            )
            log(
                DEBUG,
                "fit_round %s received %s results and %s failures",
                server_round,
                len(results),
                len(failures),
            )
            file_path = "./aggregation_straggler_parameters"
            #여기서 엣지들의 디렉토리랑 마운트된 서버의 디렉토리에서 파라미터? 가져와서 results에 포함시키기
            for i in range(5):
                # 파라미터 가져와서 results에 넣기
                if os.path.exists(file_path + "/edge%d/late_model.pth" %(i+1)):
                    print("straggler 모델 가져오기")
                    # straggler_weights_array = np.load(file_path + "/edge%d/late_parameters.npz" %(i+1))
                    model = torch.load(file_path + "/edge%d/late_model.pth" %(i+1))
                    print("straggler model(edge%d)"%i)
                    straggler_weights_ndarray = get_parameters(model)
                    straggler_weights = ndarrays_to_parameters(straggler_weights_ndarray)
                    # straggler_weights = ndarrays_to_parameters(straggler_weights_array)
                    
                    straggler_fit_res = FitRes(
                    status=None,
                    parameters=straggler_weights,
                    num_examples=6023,
                    metrics=None,
                    )
                    print("num_examples:",straggler_fit_res.num_examples)
                    print("results에 straggler parameters추가")
                    results.append((None, straggler_fit_res))
                    print("추가 했으니 straggler model 삭제")
                    os.remove(file_path + "/edge%d/late_model.pth" %(i+1))
                else:
                    continue
            for _, fit_res in results:
                print(type(fit_res.parameters))
                # if 디렉토리에 해당 파라미터가 없다면 continue
            # Aggregate training results
            aggregated_result: Tuple[
                Optional[Parameters],
                Dict[str, Scalar],
            ] = self.strategy.aggregate_fit(server_round, results, failures)

            parameters_aggregated, metrics_aggregated = aggregated_result
            return parameters_aggregated, metrics_aggregated, (results, failures)

class myClientManager(fl.server.client_manager.SimpleClientManager):
    
    def num_available(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        print("MyClientManager의 num_available()!! num_available =", len(self), "client..:", self)
        return len(self)
    
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        straggler: Dict[str, Tuple[int, int]] = {},
        criterion: Optional[Criterion] = None, # criterion: 다음 라운드에 참여할 자격이 있는지 필터링할지 말지
    ) -> List[ClientProxy]:
        
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        print("sample() 호출")
        if min_num_clients is None:
            min_num_clients = num_clients
        
        print("wait for clients until connecting server.....")
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients) # self.clients: Dict[str, ClientProxy] = {}, list로 감싸면 key만 list에 포함
        print("available_cids:", available_cids)
        
        # straggler를 제외하고 나머지를 우선적으로 num_clients만큼 랜덤 샘플링
        # straggler를 제외한 나머지가 num_clients보다 적으면 나머지를 전체 선택 후 (num_clients-나머지)를 straggler에서 랜덤 샘플링
        tier1_available_cids = list()
        for cid in available_cids:
            tier1_available_cids.append(cid)
        print("tier1_available_cids=", tier1_available_cids)
        
        straggler_cids = list()
        if len(straggler):
            print("straggler가 있으면 이거 실행")
            straggler_cids = list(straggler)    
            for cid in straggler_cids:
                if not cid in tier1_available_cids:
                    print("tier1_available_cids에 cid가 없으므로 패스")
                    continue
                else:
                    print("tier1_available_cids에 cid가 있으므로 그 cid 삭제하기") 
                    tier1_available_cids.remove(cid)           
        print("straggler_cids =", straggler_cids)
        print("tier1_available_cids =", tier1_available_cids)
            
        if criterion is not None:
            print("if criterion on not None:")
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]
            print("criterion.. available_cids=",available_cids)
            
        # if num_clients > len(available_cids):
        #     log(
        #         INFO,
        #         "Sampling failed: number of available clients"
        #         " (%s) is less than number of requested clients (%s).",
        #         len(available_cids),
        #         num_clients,
        #     )
        #     return []
        
        sampled_cids = list()
        print("num_clients =", num_clients)
        # num_clients가 더 많으면 tier1_available_cids는 걍 다 sampled_cids에 추가해버리고 나머지는 straggler_cide에서 랜덤 뽑기
        if num_clients >= len(tier1_available_cids):
            print("if num_clients >= tier1_available_cids..")
            if len(tier1_available_cids) >= 1:
                print("if len(tier1_available_cids)>=1..")
                for cid in tier1_available_cids:
                    sampled_cids.append(cid)
                    num_clients -= 1
                if num_clients > 0:
                    print("straggler random sample!!")
                    straggler_sampled_cid = random.sample(straggler_cids, num_clients)
                    for scid in straggler_sampled_cid:
                        sampled_cids.append(scid)
                    print("sampled_cids=",sampled_cids)
        else:    
            print("num_clients < len(tier1_available_cids)...")
            sampled_cids = random.sample(tier1_available_cids, num_clients)
            print("sample_cids=", sampled_cids)
        print("sample_cids:", sampled_cids, "전체 client에서 sampled_cids에 있는 cid에 해당하는 clientproxy 리턴하기")
        return [self.clients[cid] for cid in sampled_cids]
    
    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy
        """
        print("unregister() 호출, client:", client)
        # if client.cid in self.clients:
        #     del self.clients[client.cid]

        #     with self._cv:
        #         self._cv.notify_all()

class testStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = settings.FRACTION_FIT,
        fraction_evaluate: float = 0.1,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 4,
        min_available_clients: int = settings.AVAILABLE_CLIENTS,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        straggler: Dict[str, Tuple[int, int]] = {},
    ) -> None:
        super().__init__()
        self.straggler = straggler
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        
        
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: myClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        print("configure_fit() 호출!")
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)
        
        print("self.fraction_fit = ", self.fraction_fit)
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        print("sample_size: %d, min_num_clients: %d" %(sample_size, min_num_clients))
        straggler = self.straggler
        
        print("client sampling start!!")
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients, straggler=straggler)
        print("clients:", clients)
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        print("aggregation_fit() 호출")
        
        if len(self.straggler):
            key_list = []
            for key, val in self.straggler.items():
                if val[1] > 0:
                    self.straggler[key] = (val[0], val[1] - 1)
                    if self.straggler[key][1] == 0:
                        key_list.append(key)
            for key in key_list:
                del self.straggler[key]        
        if failures: #not self.accept_failures and failures
            print("# if failures are existed!!")
            for item in failures:
                print("# for item in failures")
                if isinstance(item, tuple) and len(item) == 2:
                    client_proxy, _ = item
                    #print("# if isinstance(item, tuple) and len(item) == 2:")
                    #if isinstance(client_proxy, ClientProxy):
                        #print("# if isinstance(client_proxy, ClientProxy):")
                    if not client_proxy.cid in self.straggler:
                        self.straggler[client_proxy.cid] = (server_round, 1)
                    else:
                        _, penalty = self.straggler[client_proxy.cid]
                        self.straggler[client_proxy.cid] = (server_round, penalty*2)
                else:
                    print("####failures are not tuple.. BaseException!!")
            print("straggler:", self.straggler)                 
            #return None, {}
        
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        
        # straggler의 패널티를 1줄이고 줄인 패널티가 0이 되면 straggler에서 제외하기
                
        print("straggler:", self.straggler)
        
        # Convert results
        print("# Convert results")
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        print("weights_results개수:", len(weights_results))
        for _, num in weights_results:
            print("num_examples:", num)
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        
        # Aggregate custom metrics if aggregation fn was provided
        print("# Aggregate custom metrics if aggregation fn was provided")
        
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
    
    # def aggregate_evaluate(
    #     self,
    #     rnd: int,
    #     results: List[Tuple[ClientProxy, EvaluateRes]],
    #     failures: List[BaseException],
    # ) -> Tuple[Optional[float], Dict[str, Scalar]]:
    #     """Aggregate evaluation losses using weighted average."""
    #     print("aggregate_evaluate()..")
    #     if not results:
    #         return None, {}
    #     # Do not aggregate if there are failures and failures are not accepted
    #     if not self.accept_failures and failures:
    #         return None, {}
    #     #print(rnd, results, failures)
    #     loss_aggregated = weighted_loss_avg(
    #         [
    #             (evaluate_res.num_examples, evaluate_res.loss,)
    #             for _, evaluate_res in results
    #         ]
    #     )
    #     accuracy_aggregated = weighted_loss_avg(
    #         [
    #             (
    #                 evaluate_res.num_examples,
    #                 #evaluate_res.metrics.get("accuracy", 0.0),
    #                 evaluate_res.loss,
    #             )
    #             for _, evaluate_res in results
    #         ]
    #     )
    #     return loss_aggregated, {"accuracy": accuracy_aggregated}

    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy
        """
        print("unregister() 호출, client:", client)
        # if client.cid in self.clients:
        #     del self.clients[client.cid]

        #     with self._cv:
        #         self._cv.notify_all()

if __name__ == "__main__":
    #fl.server.start_server("0.0.0.0:8080", config={"num_rounds": 3}, strategy=fl.server.strategy.FedAvg())
        print("start_server() starts")
        # EXCLUDE_LIST = [
        #     #"num_batches_tracked",
        #     #"running",
        #     #"bn", #FedBN
        # ]
        # parser = ArgumentParser()
        # parser.add_argument("--model", type=str, default='efficientnet-b2')
        # parser.add_argument("--path_data", type=str, default='./workspace/melanoma_isic_dataset') 
        # parser.add_argument("--nowandb", action="store_true")
        # args = parser.parse_args()
        # device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
        # model = utils.load_model(args.model, device).eval() 
        
        client_manager = myClientManager()
        strategy = testStrategy()
        server = myServer(client_manager=client_manager, strategy=strategy)
        fl.server.server.fit_clients = fit_clients
        fl.server.server.fit_client = fit_client
       
        for i in range(5):
            dir_path = "./aggregation_straggler_parameters/edge%d"%(i+1)
            os.chmod(dir_path, 0o777)
        os.chmod("./model", 0o777)

        fl.server.start_server(
        server_address="0.0.0.0:"+settings.PORT, config=fl.server.ServerConfig(num_rounds=settings.ROUND_NUM, round_timeout=settings.TIMEOUT), server=server, client_manager=client_manager, strategy=strategy)
        print("start_server() finished")
