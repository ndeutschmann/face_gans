"""Training functions for dcgan32 inverter"""

import os
import datetime as dt
import torch

import src.data.dcgan32.create_inversion_dataset as ds_create
import src.data.dcgan32.load_inversion_dataset as ds_load
from src.data.util import cycle

from src.models.dcgan32 import DCGAN32Inverter


def prepare_dataset(dataset_root="data/processed/dcgan32_inversion",
                    dataset_size=100000,
                    val_size=10000,
                    test_size=10000,
                    batch_size=128,
                    torch_seed=42,
                    val_torch_seed=None,
                    test_torch_seed=None,
                    generator_checkpoint_path="models/dcgan32v1/model_weights/checkpointG.2020_04_26",
                    ):

    print("Creating the training dataset at "+os.path.join(dataset_root,"train"))

    ds_create.generate_dcgan32_inversion_dataset_two_h5_tables(
        os.path.join(dataset_root, "train"),
        dataset_size=dataset_size,
        batch_size=batch_size,
        torch_seed=torch_seed,
        generator_checkpoint_path=generator_checkpoint_path
    )

    print("Creating the validation dataset at "+os.path.join(dataset_root, "val"))

    ds_create.generate_dcgan32_inversion_dataset_two_h5_tables(
        os.path.join(dataset_root, "val"),
        dataset_size=val_size,
        batch_size=batch_size,
        torch_seed=val_torch_seed if val_torch_seed is not None else torch_seed+1,
        generator_checkpoint_path=generator_checkpoint_path
    )

    print("Creating the test dataset at "+os.path.join(dataset_root, "test"))

    ds_create.generate_dcgan32_inversion_dataset_two_h5_tables(
        os.path.join(dataset_root, "test"),
        dataset_size=test_size,
        batch_size=batch_size,
        torch_seed=test_torch_seed if test_torch_seed is not None else torch_seed+2,
        generator_checkpoint_path=generator_checkpoint_path
    )


def create_dataloaders(data_path="data/processed/dcgan32_inversion/train",
                       val_path="data/processed/dcgan32_inversion/val",
                       batch_size=128,
                       val_batch_size=32):

    dataloader = ds_load.create_dcgan32_inversion_dataloader_hdf5_tables(
        root=data_path,
        batch_size=batch_size)

    val_loader = ds_load.create_dcgan32_inversion_dataloader_hdf5_tables(
        root=val_path,
        batch_size=val_batch_size)

    return dataloader, val_loader


def train_one_step(imgs, zs, model, optim, loss):
    optim.zero_grad()
    predicted_zs = model(imgs)
    L = loss(predicted_zs, zs)
    L.backward()
    optim.step()
    return L.detach().cpu().item()


def compute_vloss(imgs, zs, model, loss):
    with torch.no_grad():
        predicted_zs = model(imgs)
        return loss(predicted_zs, zs).cpu().item()


def train_inversion_model(n_channels=350,
                          learning_rate=5.e-4,
                          optim_class=torch.optim.Adam,
                          n_epochs=70,
                          data_root="data/processed/dcgan32_inversion",
                          device=None,
                          exp_root="tmp/inversion_experiment",
                          exp_id="1",
                          loss_report_period=150):

    os.makedirs(exp_root, exist_ok=True)
    exp_dir = os.path.join(exp_root,exp_id)
    try:
        os.makedirs(exp_dir,exist_ok=False)
    except FileExistsError as e:
        print("Experiments must run in a unique folder given by exp_root/exp_id")
        raise

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    dataloader, val_loader = create_dataloaders(data_path=os.path.join(data_root,"train"),
                       val_path=os.path.join(data_root,"val"),
                       batch_size=128,
                       val_batch_size=32)

    infinite_val_loader = cycle(val_loader)

    invgan = DCGAN32Inverter(channels=n_channels).to(device)

    optim = optim_class(invgan.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()

    Losses = []
    vLosses = []

    best_checkpoint = {
        "validation_loss" : float("inf"),
        "state_dict": None,
        "epoch": -1
    }

    for epoch in range(n_epochs):
        print("Starting epoch {}/{}".format(epoch+1,n_epochs))

        for i, data in enumerate(dataloader):
            imgs = data[0].to(device)
            zs = data[1].to(device)
            Loss = train_one_step(imgs, zs, invgan, optim, loss_function)

            vdata = next(infinite_val_loader)
            imgs = vdata[0].to(device)
            zs = vdata[1].to(device)

            vLoss = compute_vloss(imgs,zs,invgan,loss_function)

            if (i % loss_report_period) == 0:
                print("step {0}| Loss = {1:.3e}| vLoss = {2:.3e}".format(i, Loss, vLoss))

            Losses.append(Loss)
            vLosses.append(vLoss)

        ts = dt.datetime.timestamp(dt.datetime.now())
        torch.save({
            "timestamp": ts,
            "epoch": epoch,
            "Losses": Losses,
            "vLosses": vLosses
        }, os.path.join(exp_dir, "run_summary.pkl"))

        if vLosses[-1] < best_checkpoint["validation_loss"]:
            print("overwriting the checkpoint with current state")
            best_checkpoint = {
                "validation_loss": vLosses[-1],
                "state_dict": invgan.state_dict(),
                "epoch": epoch
            }
            torch.save(
                best_checkpoint,
                os.path.join(exp_dir, "best_checkpoint.pkl")
            )






