import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from comet_ml import Experiment
import os
from loguru import logger
import time
import numpy as np

class Trainer:
    def __init__(self, train_loader, val_loader, test_loader, model, criterion, optimizer, device, output_path,scheduler=None,predict_path=None, model_params=None, experiment=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.criterion = criterion
        self.scheduler = scheduler  
        self.optimizer = optimizer
        self.device = device
        self.experiment = experiment  # 增加 experiment 参数
        self.best_val_metric = float('inf')  # 初始化最佳指标值
        self.best_model_path = output_path  # 定义模型保存路径
        self.predict_path = predict_path
        self.current_epoch = 0
        self.model_params = model_params  # 模型参数


    def train_one_epoch(self):
        self.model.train()
        self.current_epoch += 1
        epoch_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(tqdm(self.train_loader, desc="Training")):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)  # Reconstruction task
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            epoch_loss += loss.item()

            # 实时记录每个批次的损失
            logger.info(f"Batch [{batch_idx+1}/{len(self.train_loader)}] Train Batch Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(self.train_loader)
        if self.experiment:
            self.experiment.log_metric("train_loss", avg_loss)  # 记录训练损失
        logger.info(f"Train Loss: {avg_loss:.4f}")
        return avg_loss
    
    def train_epoch_grid(self):
        self.model.train()
        self.current_epoch += 1
        epoch_loss = 0.0
        for batch_idx, data in enumerate(tqdm(self.train_loader, desc="Training")):
            inputs, targets, _ = data
            # print("inputs: ", inputs.shape)
            # print("targets: ", targets.shape)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)  # Reconstruction task
            loss.backward()
            
            self.optimizer.step()
            self.scheduler.step()
            epoch_loss += loss.item()
            # 实时记录每个批次的损失
            logger.info(f"Batch [{batch_idx+1}/{len(self.train_loader)}] Train Batch Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(self.train_loader)
        if self.experiment:
            self.experiment.log_metric("train_loss", avg_loss)  # 记录训练损失
        logger.info(f"Train Loss: {avg_loss:.4f}")
        return avg_loss
    
    def train_epoch_graph(self):
        self.model.train()
        self.current_epoch += 1
        epoch_loss = 0.0
        for batch_idx, data in enumerate(tqdm(self.train_loader, desc="Training")):
            inputs, targets, adj, _, _ = data
            adj = adj[0]
            inputs, targets, adj = inputs.to(self.device), targets.to(self.device), adj.to(self.device)
            self.optimizer.zero_grad()
            # print("inputs: ", inputs.shape)
            # print("targets: ", targets.shape)
            # print("adj: ", adj.shape)
            outputs = self.model(inputs, adj)
            loss = self.criterion(outputs, targets)  # Reconstruction task
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            epoch_loss += loss.item()
            # 实时记录每个批次的损失
            logger.info(f"Batch [{batch_idx+1}/{len(self.train_loader)}] Train Batch Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(self.train_loader)
        if self.experiment:
            self.experiment.log_metric("train_loss", avg_loss)  # 记录训练损失
        logger.info(f"Train Loss: {avg_loss:.4f}")
        return avg_loss

    def train_epoch_grid_graph(self):
        self.model.train()
        self.current_epoch += 1
        epoch_loss = 0.0
        for batch_idx, data in enumerate(tqdm(self.train_loader, desc="Training")):
            grid_inputs, grid_targets, _ = data[0]
            graph_inputs, graph_targets, graph_adj, _, _ = data[1]
            graph_adj = graph_adj[0]
            grid_inputs, grid_targets = grid_inputs.to(self.device), grid_targets.to(self.device)
            graph_inputs, graph_targets, graph_adj = graph_inputs.to(self.device), graph_targets.to(self.device), graph_adj.to(self.device)

            self.optimizer.zero_grad()
            grid_outputs, graph_outputs = self.model(grid_inputs,(graph_inputs, graph_adj))

            loss_grid = self.criterion(grid_outputs, grid_targets)  # Reconstruction task
            loss_graph = self.criterion(graph_outputs, graph_targets)  # Reconstruction task

            alpha = 100
            loss = loss_grid * alpha + loss_graph 

            logger.info(f"Batch [{batch_idx+1}/{len(self.train_loader)}] Train Batch Loss: {loss.item():.6f} | loss_grid: {loss_grid.item():.6f} | loss_graph: {loss_graph.item():.6f} with grid alpha: {alpha}")

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            epoch_loss += loss.item()
            # 实时记录每个批次的损失
        
        avg_loss = epoch_loss / len(self.train_loader)
        if self.experiment:
            self.experiment.log_metric("train_loss", avg_loss)  # 记录训练损失
        logger.info(f"Train Loss: {avg_loss:.4f}")

    def validate(self):
        self.model.eval()
        epoch_loss = 0.0
        metrics = {"MSE": 0.0, "RMSE": 0.0, "MSE_grid":0.0,"RMSE_grid":0.0,"MSE_graph":0.0,"RMSE_graph":0.0,}
        count = 0

        with torch.no_grad():
            for batch_idx, (data_grid, data_graph) in enumerate(tqdm(self.val_loader, desc="Validation")):
                input_tensor_grid, output_tensor_grid, time_grid = data_grid
                input_tensor_graph, output_tensor_graph, time_graph, lan_lon = data_graph

                input_tensor_grid = input_tensor_grid.to(self.device)
                input_tensor_graph = input_tensor_graph.to(self.device)
                output_tensor_grid = output_tensor_grid.to(self.device)
                output_tensor_graph = output_tensor_graph.to(self.device)
                lan_lon = lan_lon.to(self.device)
                time_graph = time_graph.to(self.device)
                time_grid = time_grid.to(self.device)

                predicted_grid, predicted_graph = self.model(input_tensor_graph, input_tensor_grid, lan_lon, time_graph, time_grid)
                            
                loss1 = self.criterion(predicted_grid, output_tensor_grid)
                # print("output_tensor_graph ", output_tensor_graph.shape)
                loss2 = self.criterion(predicted_graph, output_tensor_graph)

                # print("loss1: ", loss1)
                # print("loss2: ", loss2)

                # loss平衡：大概差1个数量级
                alpha = 0.1
                loss = loss1 + loss2 * alpha
                epoch_loss += loss.item()

                epoch_loss += loss.item()
                # 实时记录每个批次的损失
                logger.info(f"Batch [{batch_idx+1}/{len(self.train_loader)}] Train Batch Loss: {loss.item():.4f} | loss_grid: {loss1.item():.4f} | loss_graph: {loss2.item():.4f} with alpha: {alpha}")


                # Metrics calculation
                mse_grid = torch.mean((predicted_grid - output_tensor_grid) ** 2).item()
                rmse_grid = torch.sqrt(torch.mean((predicted_grid - output_tensor_grid) ** 2)).item()
                mse_graph = torch.mean((predicted_graph - output_tensor_graph) ** 2).item()
                rmse_graph = torch.sqrt(torch.mean((predicted_graph - output_tensor_graph) ** 2)).item()
                metrics["MSE_grid"] += mse_grid
                metrics["RMSE_grid"] += rmse_grid
                metrics["MSE_graph"] += mse_graph
                metrics["RMSE_graph"] += rmse_graph
                metrics["MSE"] =  mse_grid + mse_graph
                metrics["RMSE"] =  rmse_grid + rmse_graph
                count += 1

        for key in metrics:
            metrics[key] /= count

        avg_loss = epoch_loss / len(self.val_loader)
        if self.experiment:
            self.experiment.log_metric("val_loss", avg_loss)
            for key, value in metrics.items():
                self.experiment.log_metric(f"val_{key}", value)

        # 更新最佳模型
        if metrics["RMSE"] < self.best_val_metric:
            self.best_val_metric = metrics["RMSE"]
            best_model_name = f"model_epoch={self.current_epoch}_RMSE={metrics['RMSE']:.4f}.pth"
            best_model_path = os.path.join(self.best_model_path, best_model_name)
            torch.save(self.model.state_dict(), best_model_path)
            logger.info(f"New best model saved: {best_model_name} with RMSE: {self.best_val_metric:.4f}")

        logger.info(f"Validation Loss: {avg_loss:.4f}, Validation MSE: {metrics['MSE']:.4f}, Validation RMSE: {metrics['RMSE']:.4f}")
        return avg_loss, metrics

    def validate_grid(self):
        self.model.eval()
        metrics = {"MSE": 0.0, "RMSE": 0.0}

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.val_loader, desc="Validation")):
                inputs, targets, _ = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)
              
                outputs = self.model(inputs)

                # Metrics calculation
                mse_grid = torch.mean((outputs - targets) ** 2).item()
                rmse_grid = torch.sqrt(torch.mean((outputs - targets) ** 2)).item()
                metrics["MSE"] += mse_grid
                metrics["RMSE"] += rmse_grid

        if self.experiment:
            for key, value in metrics.items():
                self.experiment.log_metric(f"val_{key}", value)

        # 更新最佳模型
        if metrics["RMSE"] < self.best_val_metric:
            self.best_val_metric = metrics["RMSE"]
            best_model_name = f"model_epoch={self.current_epoch}_RMSE={metrics['RMSE']:.4f}.pth"
            best_model_path = os.path.join(self.best_model_path, best_model_name)
            torch.save(self.model.state_dict(), best_model_path)
            logger.info(f"New best model saved: {best_model_name} with RMSE: {self.best_val_metric:.4f}")

        logger.info(f"Validation MSE: {metrics['MSE']:.4f}, Validation RMSE: {metrics['RMSE']:.4f}")
        return metrics

    def validate_graph(self):
        self.model.eval()
        metrics = {"MSE": 0.0, "RMSE": 0.0}

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.val_loader, desc="Validation")):
                inputs, targets, adj, _, _ = data
                adj = adj[0]
                inputs, targets, adj = inputs.to(self.device), targets.to(self.device), adj.to(self.device)

                outputs = self.model(inputs, adj)
                # Metrics calculation
                mse_graph = torch.mean((outputs - targets) ** 2).item()
                rmse_graph = torch.sqrt(torch.mean((outputs - targets) ** 2)).item()
                metrics["MSE"] += mse_graph
                metrics["RMSE"] += rmse_graph

        if self.experiment:
            for key, value in metrics.items():
                self.experiment.log_metric(f"val_{key}", value)

        # 更新最佳模型
        if metrics["RMSE"] < self.best_val_metric:
            self.best_val_metric = metrics["RMSE"]
            best_model_name = f"model_epoch={self.current_epoch}_RMSE={metrics['RMSE']:.4f}.pth"
            best_model_path = os.path.join(self.best_model_path, best_model_name)
            torch.save(self.model.state_dict(), best_model_path)
            logger.info(f"New best model saved: {best_model_name} with RMSE: {self.best_val_metric:.4f}")

        return metrics


    def validate_epoch_grid_graph(self):
        self.model.eval()
        metrics = {"MSE": 0.0,
                   "RMSE": 0.0,
                   "MSE_grid": 0.0,
                   "RMSE_grid": 0.0,
                   "MSE_graph": 0.0,
                   "RMSE_graph": 0.0,
                   }
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.val_loader, desc="Validation")):
                grid_inputs, grid_targets, _ = data[0]
                graph_inputs, graph_targets, graph_adj, _, _ = data[1]
                graph_adj = graph_adj[0]
                grid_inputs, grid_targets = grid_inputs.to(self.device), grid_targets.to(self.device)
                graph_inputs, graph_targets, graph_adj = graph_inputs.to(self.device), graph_targets.to(self.device), graph_adj.to(self.device)

                self.optimizer.zero_grad()
                grid_outputs, graph_outputs = self.model(grid_inputs,(graph_inputs, graph_adj))

                loss_grid = self.criterion(grid_outputs, grid_targets)  # Reconstruction task
                loss_graph = self.criterion(graph_outputs, graph_targets)  # Reconstruction task

                alpha = 1
                loss = loss_grid + loss_graph * alpha

                logger.info(f"Batch [{batch_idx+1}/{len(self.train_loader)}] Train Batch Loss: {loss.item():.6f} | loss_grid: {loss_grid.item():.6f} | loss_graph: {loss_graph.item():.6f} with alpha: {alpha}")
               
                # Metrics calculation
                mse_grid = torch.mean((grid_outputs - grid_targets) ** 2).item()
                rmse_grid = torch.sqrt(torch.mean((grid_outputs - grid_outputs) ** 2)).item()
                mse_graph = torch.mean((graph_outputs - graph_targets) ** 2).item()
                rmse_graph = torch.sqrt(torch.mean((graph_outputs - graph_targets) ** 2)).item()

                metrics["MSE_grid"] += mse_grid
                metrics["RMSE_grid"] += rmse_grid
                metrics["MSE_graph"] += mse_graph
                metrics["RMSE_graph"] += rmse_graph
                mse = mse_grid + mse_graph
                rmse = rmse_grid + rmse_graph
                metrics['MSE'] += mse
                metrics['RMSE'] += rmse

        # 更新最佳模型
        if metrics["RMSE"] < self.best_val_metric:
            self.best_val_metric = metrics["RMSE"]
            best_model_name = f"model_epoch={self.current_epoch}_RMSE={metrics['RMSE']:.6f}.pth"
            best_model_path = os.path.join(self.best_model_path, best_model_name)
            torch.save(self.model.state_dict(), best_model_path)
            logger.info(f"New best model saved: {best_model_name} with RMSE: {self.best_val_metric:.6f}")

        logger.info(f"Validation MSE: {metrics['MSE']:.6f}, Validation RMSE: {metrics['RMSE']:.6f}")
        return metrics


    def predict_grid(self):
        if self.model_params is not None:
            self.model.load_state_dict(torch.load(self.model_params))
            print("load model params from: ", self.model_params)
        self.model.eval()
        metrics = {"MSE": 0.0, "RMSE": 0.0}

        # Define directories for saving predictions and targets
        predictions_dir = os.path.join(self.predict_path, "predictions")
        targets_dir = os.path.join(self.predict_path, "targets")
        os.makedirs(predictions_dir, exist_ok=True)
        os.makedirs(targets_dir, exist_ok=True)

        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.test_loader, desc="Predicting")):
                inputs, targets, _ = data
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                # Save predictions and targets as .npy files
                prediction_file = os.path.join(predictions_dir, f"batch_{batch_idx}.npy")
                target_file = os.path.join(targets_dir, f"batch_{batch_idx}.npy")

                if outputs.shape[0] == 1:
                    outputs = outputs.squeeze(0)
                    targets = targets.squeeze(0)
                np.save(prediction_file, outputs.cpu().numpy())
                np.save(target_file, targets.cpu().numpy())

                # Metrics calculation
                mse_grid = torch.mean((outputs - targets) ** 2).item()
                rmse_grid = torch.sqrt(torch.mean((outputs - targets) ** 2)).item()
                metrics["MSE"] += mse_grid
                metrics["RMSE"] += rmse_grid

        logger.info(f"Validation MSE: {metrics['MSE']:.4f}, Validation RMSE: {metrics['RMSE']:.4f}")
        return metrics
    
    def predict_graph(self):
        if self.model_params is not None:
            self.model.load_state_dict(torch.load(self.model_params))
            print("load model params from: ", self.model_params)
        self.model.eval()
        metrics = {"MSE": 0.0, "RMSE": 0.0}

        # Define directories for saving predictions and targets
        predictions_dir = os.path.join(self.predict_path, "predictions")
        targets_dir = os.path.join(self.predict_path, "targets")
        os.makedirs(predictions_dir, exist_ok=True)
        os.makedirs(targets_dir, exist_ok=True)
        
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.test_loader, desc="Predicting")):
                inputs, targets, adj, _, _ = data
                adj = adj[0]
                inputs, targets, adj = inputs.to(self.device), targets.to(self.device), adj.to(self.device)
                # print("inputs: ", inputs.dtype)
                # print("targets: ", targets.dtype)
                outputs = self.model(inputs, adj)

                # Save predictions and targets as .npy files
                prediction_file = os.path.join(predictions_dir, f"batch_{batch_idx}.npy")
                target_file = os.path.join(targets_dir, f"batch_{batch_idx}.npy")

                if outputs.shape[0] == 1:
                    outputs = outputs.squeeze(0)
                    targets = targets.squeeze(0)
                np.save(prediction_file, outputs.cpu().numpy())
                np.save(target_file, targets.cpu().numpy())

                # Metrics calculation
                mse_grid = torch.mean((outputs - targets) ** 2).item()
                rmse_grid = torch.sqrt(torch.mean((outputs - targets) ** 2)).item()
                metrics["MSE"] += mse_grid
                metrics["RMSE"] += rmse_grid

        logger.info(f"Validation MSE: {metrics['MSE']:.4f}, Validation RMSE: {metrics['RMSE']:.4f}")
        return metrics
    
    
    def precict_grid_graph(self):
        if self.model_params is not None:
            self.model.load_state_dict(torch.load(self.model_params))
            print("load model params from: ", self.model_params)

        # Define directories for saving predictions and targets
        grid_predictions_dir = os.path.join(self.predict_path, "predictions_grid")
        grid_targets_dir = os.path.join(self.predict_path, "targets_grid")
        graph_predictions_dir = os.path.join(self.predict_path, "predictions_graph")
        graph_targets_dir = os.path.join(self.predict_path, "targets_graph")
        os.makedirs(grid_predictions_dir, exist_ok=True)
        os.makedirs(grid_targets_dir, exist_ok=True)
        os.makedirs(graph_predictions_dir, exist_ok=True)
        os.makedirs(graph_targets_dir, exist_ok=True)

        self.model.eval()
        metrics = {"MSE": 0.0,
                   "RMSE": 0.0,
                   "MSE_grid": 0.0,
                   "RMSE_grid": 0.0,
                   "MSE_graph": 0.0,
                   "RMSE_graph": 0.0,
                   }
        with torch.no_grad():
            for batch_idx, data in enumerate(tqdm(self.test_loader, desc="Predicting")):
                grid_inputs, grid_targets, _ = data[0]
                graph_inputs, graph_targets, graph_adj, _, _ = data[1]
                graph_adj = graph_adj[0]
                grid_inputs, grid_targets = grid_inputs.to(self.device), grid_targets.to(self.device)
                graph_inputs, graph_targets, graph_adj = graph_inputs.to(self.device), graph_targets.to(self.device), graph_adj.to(self.device)

                self.optimizer.zero_grad()
                grid_outputs, graph_outputs = self.model(grid_inputs,(graph_inputs, graph_adj))

                # Save predictions and targets as .npy files
                grid_prediction_file = os.path.join(grid_predictions_dir, f"batch_{batch_idx}.npy")
                grid_target_file = os.path.join(grid_targets_dir, f"batch_{batch_idx}.npy")
                graph_prediction_file = os.path.join(graph_predictions_dir, f"batch_{batch_idx}.npy")
                graph_target_file = os.path.join(graph_targets_dir, f"batch_{batch_idx}.npy")

                if grid_outputs.shape[0] == 1:
                    grid_outputs = grid_outputs.squeeze(0)
                    grid_targets = grid_targets.squeeze(0)

                if graph_outputs.shape[0] == 1:
                    graph_outputs = graph_outputs.squeeze(0)
                    graph_targets = graph_targets.squeeze(0)

                np.save(grid_prediction_file, grid_outputs.cpu().numpy())
                np.save(grid_target_file, grid_targets.cpu().numpy())
                np.save(graph_prediction_file, graph_outputs.cpu().numpy())
                np.save(graph_target_file, graph_targets.cpu().numpy())

               
                # Metrics calculation
                mse_grid = torch.mean((grid_outputs - grid_targets) ** 2).item()
                rmse_grid = torch.sqrt(torch.mean((grid_outputs - grid_outputs) ** 2)).item()
                mse_graph = torch.mean((graph_outputs - graph_targets) ** 2).item()
                rmse_graph = torch.sqrt(torch.mean((graph_outputs - graph_targets) ** 2)).item()

                metrics["MSE_grid"] += mse_grid
                metrics["RMSE_grid"] += rmse_grid
                metrics["MSE_graph"] += mse_graph
                metrics["RMSE_graph"] += rmse_graph
                mse = mse_grid + mse_graph
                rmse = rmse_grid + rmse_graph
                metrics['MSE'] += mse
                metrics['RMSE'] += rmse

        # 更新最佳模型
        if metrics["RMSE"] < self.best_val_metric:
            self.best_val_metric = metrics["RMSE"]
            best_model_name = f"model_epoch={self.current_epoch}_RMSE={metrics['RMSE']:.6f}.pth"
            best_model_path = os.path.join(self.best_model_path, best_model_name)
            torch.save(self.model.state_dict(), best_model_path)
            logger.info(f"New best model saved: {best_model_name} with RMSE: {self.best_val_metric:.6f}")

        logger.info(f"Validation MSE: {metrics['MSE']:.6f}, Validation RMSE: {metrics['RMSE']:.6f}")
        return metrics