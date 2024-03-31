import torch
from model import BoltNet 
import torch.optim as optim
from loss import BoltLoss
import sys
import argparse
import numpy as np
from model import BoltNet
import open3d as o3d
import utils
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance
import sys
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--point_cloud_source", type=str, help="Path to the point cloud")
    parser.add_argument("-n", "--n_final_sample", type=int, help="Number of final samples", default=4096)
    parser.add_argument("-e", "--n_epochs", type=int, help="Number of epochs", default=100)
    args = parser.parse_args()

    dataset = utils.PointCloudData(args.point_cloud_source, pc_class='02691156', preloaded=False, n_final_sample=args.n_final_sample)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    num_classes = dataset.num_classes() 
    num_shapes = dataset.num_classes()

    model = BoltNet(num_classes, num_shapes, out_points=args.n_final_sample)
    model.to('cuda')
    #if os.path.exists('model.pth'):
    #    model.load_state_dict(torch.load('model.pth'))

    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    progress_bar = tqdm(range(args.n_epochs), postfix={"Loss": 0})

    losses = []

    for epoch in progress_bar:
        model.train()
        in_epoch_progress_bar = tqdm(enumerate(data_loader), leave=False, total=len(data_loader), postfix={"Loss": 0})
        for i, data in in_epoch_progress_bar:
            optimizer.zero_grad()
            inputs, true_pc = data
            outputs = model(inputs)
            loss, _ = chamfer_distance(outputs, true_pc)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            in_epoch_progress_bar.set_postfix({"Loss": loss.item()})
        sys.stdout.flush()
        sys.stderr.flush()

        progress_bar.set_postfix({"Loss": loss.item()})
        torch.save(model.state_dict(), f"model.pth")

    json.dump({'losses': losses}, open('losses.json', 'w'))

    print("Finished training")
