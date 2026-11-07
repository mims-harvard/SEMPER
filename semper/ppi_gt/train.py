# General
import numpy as np
import random
import argparse
import os
import copy

# Pytorch
import torch
import torch.nn as nn
from torch_geometric.utils.convert import to_networkx, to_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling

# Center loss
from center_loss import CenterLoss

# W&B
import wandb

# Own code
from generate_input import read_data, get_metapaths, get_centerloss_labels
import model as mdl
import utils
import minibatch_utils as mb_utils
from parse_args import get_args, get_hparams

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import socket
import io
from torch.nn import SyncBatchNorm

# Seed
seed = 3
print("SEED:", seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
np.random.seed(seed)
random.seed(seed)
# torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# Setup
args = get_args()
hparams_raw = get_hparams(args)

save_log = args.save_prefix + "_gnn_train.log"
save_graph = args.save_prefix + "_graph.pkl"
save_model = args.save_prefix + "_model_save.pth"
save_plots = args.save_prefix + "_train_embed_plots.pdf"
save_ppi_embed = args.save_prefix + "_protein_embed.pth"
save_mg_embed = args.save_prefix + "_mg_embed.pth"
save_labels_dict = args.save_prefix + "_labels_dict.txt"

log_f = open(save_log, "w")
log_f.write("Number of epochs: %s \n" % args.epochs)
log_f.write("Save model directory: %s \n" % save_model)
log_f.write("Save embeddings directory: %s, %s \n" % (save_ppi_embed, save_mg_embed))
log_f.write("Save graph: %s \n" % save_graph)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

best_val_acc = -1
best_model = None
eps = 10e-4

# Initialize W&B tracking. The entity is read from the WANDB_ENTITY env var so
# no personal account is baked into the repo. To run with NO tracking and no
# login, set WANDB_MODE=disabled in your shell — wandb.log/watch become no-ops
# and wandb.config is still populated from hparams_raw.
wandb.init(
    config=hparams_raw,
    project="ppigt",
    entity=os.environ.get("WANDB_ENTITY"),  # None -> your default W&B entity
)

hparams = wandb.config

def train(epoch, model, optimizer, center_loss, rank, local_rank, train_mask, val_mask, test_mask, ppi_metapaths, mg_metapaths, ppi_data, mg_data, edge_attr_dict, celltype_map, tissue_neighbors, ppi_layers, metagraph):

    #global args, ppi_data, mg_data, best_model, best_val_acc, hparams
    global args, best_model, best_val_acc, hparams

    print(f"[GPU {local_rank}] Training started on global rank {rank}...")
    torch.cuda.set_device(local_rank)
    model.to(device)

    # Generate PPI batches
    ppi_train_loader_dict, _, ppi_metapaths_train, ppi_x_ori = mb_utils.generate_batch(ppi_data, ppi_metapaths, edge_attr_dict, "train", args.batch_size, device, ppi=True, loader_type=args.loader)
    ppi_val_loader_dict, _, ppi_metapaths_val, _ = mb_utils.generate_batch(ppi_data, ppi_metapaths, edge_attr_dict, "val", args.batch_size, device, ppi=True, loader_type=args.loader)

    ppi_train_loader_dict = mb_utils.split_loader_for_ddp(ppi_train_loader_dict, rank)
    ppi_val_loader_dict = mb_utils.split_loader_for_ddp(ppi_val_loader_dict, rank)
    

    # Generate metagraph batches
    _, mg_data_train, mg_metapaths_train, mg_x_ori = mb_utils.generate_batch({0: mg_data}, mg_metapaths, edge_attr_dict, "train", args.batch_size, device, ppi=False, loader_type=args.loader)
    _, mg_data_val, mg_metapaths_val, _ = mb_utils.generate_batch({0: mg_data}, mg_metapaths, edge_attr_dict, "val", args.batch_size, device, ppi=False, loader_type=args.loader)


    mg_x_ori = mg_x_ori[0]
    mg_data_train = mg_data_train[0]
    mg_data_val = mg_data_val[0]
    mg_metapaths_train = mg_metapaths_train[0]
    mg_metapaths_val = mg_metapaths_val[0]
    for i, val in enumerate(mg_metapaths_train):
        mg_metapaths_train[i] = val.to(device)
    for key, val in ppi_metapaths_train.items():
        ppi_metapaths_train[key] = [val[0].to(device)]

    
    model.train()

    # Run batch training
    _, _, mg_pred, ppi_preds_all, ppi_data_train_y, loss = mb_utils.iterate_train_batch(ppi_train_loader_dict, ppi_x_ori, ppi_metapaths, mg_x_ori, mg_metapaths_train, mg_data_train, tissue_neighbors, model, hparams, device, wandb, center_loss, optimizer, train_mask)


    # Training metrics
    roc_score, ap_score, train_acc, train_f1 = utils.calc_metrics(mg_pred, mg_data_train, ppi_preds_all, ppi_data_train_y)
    #print("Training Metrics:", "ROC", roc_score, "AP", ap_score, "ACC", train_acc, "F1", train_f1)
    wandb.log({"train_roc": roc_score, "train_ap": ap_score, "train_acc": train_acc, "train_f1": train_f1})

    utils.metrics_per_rel(mg_pred, mg_data_train, ppi_preds_all, ppi_data_train_y, edge_attr_dict, celltype_map, log_f, wandb, "train")

    # Validation set predictions
    ppi_x, _, mg_pred, ppi_preds_all, ppi_data_val_y = mb_utils.iterate_predict_batch(ppi_val_loader_dict, ppi_x_ori, ppi_metapaths_train, mg_x_ori, mg_metapaths_train, mg_data_val, tissue_neighbors, model, hparams, device)  # Using train metapaths.
    
    # Validation metrics
    roc_score, ap_score, val_acc, val_f1 = utils.calc_metrics(mg_pred, mg_data_val, ppi_preds_all, ppi_data_val_y)
    #print("Validation Metrics:", "ROC", roc_score, "AP", ap_score, "ACC", val_acc, "F1", val_f1)
    utils.metrics_per_rel(mg_pred, mg_data_val, ppi_preds_all, ppi_data_val_y, edge_attr_dict, celltype_map, log_f, wandb, "val")

    calinski_harabasz, davies_bouldin = utils.calc_cluster_metrics(ppi_x)
    
    # Save metrics
    res = "\t".join(["Epoch: %04d" % (epoch + 1), 
                     "train_loss = {:.5f}".format(loss), 
                     "val_roc = {:.5f}".format(roc_score), 
                     "val_ap = {:.5f}".format(ap_score), 
                     "val_f1 = {:.5f}".format(val_acc), 
                     "val_acc = {:.5f}".format(val_f1)])
    #print(res)
    log_f.write(res + "\n")
    wandb.log({"total_loss": loss, "total_val_roc": roc_score, "total_val_ap": ap_score, "total_val_acc": val_acc, "total_val_f1": val_f1, "total_val_calinski_harabasz_score": calinski_harabasz, "total_val_davies_bouldin_score": davies_bouldin})

    # Save best model and parameters
    if best_val_acc <= np.mean(val_acc) + eps:
        best_val_acc = np.mean(val_acc)
        if rank == 0:  # only one DDP process writes the checkpoint
            with open(save_model, 'wb') as f:
                torch.save({"epoch": epoch, "model": model, "optimizer": optimizer}, f)
        best_model = copy.deepcopy(model)
    
    for i, val in enumerate(mg_metapaths_train):
        mg_metapaths_train[i] = val.detach().cpu()
    for key, val in ppi_metapaths_train.items():
        ppi_metapaths_train[key] = [val[0].detach().cpu()]
    
    return ppi_metapaths_train, mg_metapaths_train, ppi_metapaths_val, mg_metapaths_val


@torch.no_grad()
def test(model, ppi_metapaths_test, mg_metapaths_test,train_mask, val_mask, test_mask, ppi_metapaths, mg_metapaths, ppi_data, mg_data, edge_attr_dict, celltype_map, tissue_neighbors, ppi_layers, metagraph):

    #model.to(device)
    torch.cuda.set_device(0)
    model.to(device)
    model.eval()
    
    # Generate PPI batches
    ppi_test_loader_dict, _, _, ppi_x = mb_utils.generate_batch(ppi_data, ppi_metapaths, edge_attr_dict, "test", args.batch_size, device, ppi=True, loader_type=args.loader)
    
    # Generate metagraph batches
    _, mg_data_test, _, mg_x = mb_utils.generate_batch({0: mg_data}, mg_metapaths, edge_attr_dict, "test", args.batch_size, device, ppi=False, loader_type=args.loader)
    mg_data_test = mg_data_test[0]
    mg_x = mg_x[0]

    _, _, mg_pred, ppi_preds_all, ppi_data_test_y = mb_utils.iterate_predict_batch(ppi_test_loader_dict, ppi_x, ppi_metapaths_test, mg_x, mg_metapaths_test, mg_data_test, tissue_neighbors, model, hparams, device)

    roc_score, ap_score, test_acc, test_f1 = utils.calc_metrics(mg_pred, mg_data_test, ppi_preds_all, ppi_data_test_y)
    
    #print('Test ROC score: {:.5f}'.format(roc_score))
    #print('Test AP score: {:.5f}'.format(ap_score))
    #print('Test Accuracy: {:.5f}'.format(test_acc))
    #print('Test F1 score: {:.5f}'.format(test_f1))
    log_f.write('Test ROC score: {:.5f}\n'.format(roc_score))
    log_f.write('Test AP score: {:.5f}\n'.format(ap_score))
    log_f.write('Test Accuracy: {:.5f}\n'.format(test_acc))
    log_f.write('Test F1 score: {:.5f}\n'.format(test_f1))

    wandb.log({"test_roc": roc_score, "test_ap": ap_score, "test_acc": test_acc, "test_f1": test_f1})
    utils.metrics_per_rel(mg_pred, mg_data_test, ppi_preds_all, ppi_data_test_y, edge_attr_dict, celltype_map, log_f, wandb, "test")


def setup_ddp():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    #initialize process group
    dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)

    return rank, local_rank, world_size

def broadcast_tensor(tensor, src=0):
    """ Broadcast a tensor to all ranks. """
    if tensor is None:
        tensor = torch.zeros(1, dtype=torch.float32, device="cuda")
    dist.broadcast(tensor, src=src)
    return tensor

def broadcast_list_of_tensors(tensor_list, src=0):
    """ Broadcast a list of tensors to all ranks. """
    sizes = torch.tensor([t.numel() if t is not None else 0 for t in tensor_list], dtype=torch.long, device="cuda")
    dist.broadcast(sizes, src=src)

    max_size = sizes.max().item()
    broadcasted_tensors = []

    for i, size in enumerate(sizes):
        temp_tensor = torch.zeros(max_size, dtype=torch.float32, device="cuda") if size > 0 else None
        if tensor_list[i] is not None:
            temp_tensor[:size] = tensor_list[i].view(-1)[:size]
        dist.broadcast(temp_tensor, src=src)
        broadcasted_tensors.append(temp_tensor[:size] if size > 0 else None)

    return broadcasted_tensors

def main():
    global args, best_model, hparams, device
    
    rank, local_rank, world_size = setup_ddp()

    if rank == 0:
        ppi_data, mg_data, edge_attr_dict, celltype_map, tissue_neighbors, ppi_layers, metagraph = read_data(args.G_f, args.ppi_dir, args.mg_f, hparams['feat_mat'], hparams['n_heads'])
        ppi_metapaths, mg_metapaths = get_metapaths()
        center_loss_labels, train_mask, val_mask, test_mask = get_centerloss_labels(args, celltype_map, ppi_layers)

    else:
        ppi_data, mg_data, edge_attr_dict, celltype_map, tissue_neighbors, ppi_layers, metagraph = None, None, None, None, None, None, None
        ppi_metapaths, mg_metapaths = None, None
        train_mask, val_mask, test_mask, center_loss_labels = None, None, None, None

    dist.barrier()

    broadcast_list = [center_loss_labels, train_mask, val_mask, test_mask]
    dist.broadcast_object_list(broadcast_list, src=0)

    center_loss_labels, train_mask, val_mask, test_mask = broadcast_list

    if rank == 0:
        buffer = io.BytesIO()
        torch.save([ppi_metapaths, mg_metapaths, ppi_data, mg_data, edge_attr_dict, celltype_map, tissue_neighbors, ppi_layers, metagraph], buffer)
        serialized_size = torch.tensor([buffer.getbuffer().nbytes], dtype=torch.long, device="cuda")
    else:
        serialized_size = torch.tensor([0], dtype=torch.long, device="cuda")

    dist.broadcast(serialized_size, src=0)

    if rank == 0:
        serialized_tensor = torch.tensor(list(buffer.getbuffer()), dtype=torch.uint8, device="cuda")
    else:
        serialized_tensor = torch.empty(serialized_size.item(), dtype=torch.uint8, device="cuda")

    dist.broadcast(serialized_tensor, src=0)

    if rank != 0:
        buffer = io.BytesIO(serialized_tensor.cpu().numpy().tobytes())
        ppi_metapaths, mg_metapaths, ppi_data, mg_data, edge_attr_dict, celltype_map, tissue_neighbors, ppi_layers, metagraph = torch.load(buffer)

    print(f"[Rank {rank}] Broadcast complete!")
 
    # Set up
    if args.resume_run != "":
        save_model = "%s_model_save.pth" % args.resume_run
        #print("Resuming", save_model)
        checkpoint = torch.load(save_model)
        model = checkpoint["model"]
        optimizer = checkpoint["optimizer"]
        params = list(model.parameters())
    else:
        model = mdl.PPIGT(mg_data.x.shape[1], hparams['hidden'], hparams['output'], len(ppi_metapaths), len(mg_metapaths), ppi_data, hparams['n_heads'], hparams['pc_att_channels'], hparams['dropout']).to(local_rank)


        model = SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
        params = list(model.parameters())
        optimizer = torch.optim.Adam(params, lr = hparams['lr'], weight_decay = hparams['wd'])

    center_loss = CenterLoss(num_classes=len(set(center_loss_labels)), feat_dim=hparams['output'] * hparams['n_heads'], use_gpu=torch.cuda.is_available())
    params += list(center_loss.parameters())
    wandb.watch(model)

    def count_trainable_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    print("Trainable params (full model):", f"{count_trainable_params(model):,}")

    # Train model
    for epoch in range(args.epochs):
        ppi_metapaths_train, mg_metapaths_train, ppi_metapaths_val, mg_metapaths_val = train(epoch, model, optimizer, center_loss, rank, local_rank, train_mask, val_mask, test_mask, ppi_metapaths, mg_metapaths, ppi_data, mg_data, edge_attr_dict, celltype_map, tissue_neighbors, ppi_layers, metagraph)

    if rank == 0:
        # Generate test metapaths
        ppi_metapaths_test = {}
        mg_metapaths_test = []
        for key in ppi_metapaths_train.keys():
            ppi_metapaths_test[key] = [torch.cat(ppi_metapaths_val[key] + ppi_metapaths_train[key], dim=1)]
        for mg_mt_t, mg_mt_v in zip(mg_metapaths_train, mg_metapaths_val):
            mg_metapaths_test.append(torch.cat([mg_mt_t, mg_mt_v], dim=1))

        # Test (w/train metapaths trained node embeddings and test links)
        test(best_model, ppi_metapaths_test, mg_metapaths_test,train_mask, val_mask, test_mask, ppi_metapaths, mg_metapaths, ppi_data, mg_data, edge_attr_dict, celltype_map, tissue_neighbors, ppi_layers, metagraph)

    del ppi_metapaths_train, ppi_metapaths_val, mg_metapaths_train, mg_metapaths_val
    torch.cuda.empty_cache()
    
    if rank == 0:
        del ppi_metapaths_test, mg_metapaths_test
        # Generate and save best embeddings (w/complete metapaths trained node embeddings)
        _, ppi_data_all, ppi_metapaths_adjs, ppi_x = mb_utils.generate_batch(ppi_data, ppi_metapaths, edge_attr_dict, "all", args.batch_size, device, ppi = False, loader_type=args.loader)
        _, mg_data_all, mg_metapaths_adjs, mg_x = mb_utils.generate_batch({0: mg_data}, mg_metapaths, edge_attr_dict, "all", args.batch_size, device, ppi = False, loader_type=args.loader)
    
        # Generate final embeddings
        best_ppi_x, best_mg_x = utils.get_embeddings(best_model, ppi_x, mg_x[0], ppi_metapaths_adjs, mg_metapaths_adjs[0], ppi_data_all, mg_data_all[0]["total_edge_index"], tissue_neighbors)

        # Save outputs
        for celltype, x in best_ppi_x.items():
            best_ppi_x[celltype] = x.to(device)
        best_mg_x = best_mg_x.to(device)
        torch.save(best_ppi_x, save_ppi_embed)
        torch.save(best_mg_x, save_mg_embed)

        # Generate plots
        labels_dict = utils.plot_emb(best_ppi_x, best_mg_x, celltype_map, ppi_layers, metagraph, wandb, center_loss_labels, hparams['plot'])
    
        # Save labels
        labels_fout = open(save_labels_dict, "w")
        labels_fout.write(str(labels_dict))
        labels_fout.close()

        # Save only protein/metagraph labels
        protein_celltype_labels = []
        protein_name_labels = []
        metagraph_celltype_labels = []
        metagraph_name_labels = []
        for c, p in zip(labels_dict["Cell Type"], labels_dict["Name"]):
            if c.startswith("Sanity"): continue
            if c.startswith("BTO") or c.startswith("CCI"):
                metagraph_celltype_labels.append(c)
                metagraph_name_labels.append(p)
            else:
                protein_celltype_labels.append(c)
                protein_name_labels.append(p)

        protein_labels_dict = {"Cell Type": protein_celltype_labels, "Name": protein_name_labels}
        labels_fout = open(args.save_prefix + "_protein_labels_dict.txt", "w")
        labels_fout.write(str(protein_labels_dict))
        labels_fout.close()

        mg_labels_dict = {"Cell Type": metagraph_celltype_labels, "Name": metagraph_name_labels}
        labels_fout = open(args.save_prefix + "_mg_labels_dict.txt", "w")
        labels_fout.write(str(mg_labels_dict))
        labels_fout.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
