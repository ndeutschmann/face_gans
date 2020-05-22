"""CLI to train dcgan32-related models"""

import click
from .dcgan32_inversion import prepare_dataset, train_inversion_model


@click.group()
def dcgan32_inversion():
    """Commands related to the inversion model"""


@dcgan32_inversion.command("prepare-dataset",
                           help="Load a checkpoint for the DCGAN32 generator and generate an hdf5 inversion dataset")
@click.option("--dataset_root", default="data/processed/dcgan32_inversion", type=click.Path())
@click.option("--dataset_size", default=100000, type=click.INT)
@click.option("--val_size", default=10000, type=click.INT)
@click.option("--test_size", default=10000, type=click.INT)
@click.option("--batch_size", default=128, type=click.INT)
@click.option("--torch_seed", default=42, type=click.INT)
@click.option("--val_torch_seed", default=None, type=click.INT)
@click.option("--test_torch_seed", default=None, type=click.INT)
@click.option("--generator_checkpoint_path",
              default="models/dcgan32v1/model_weights/checkpointG.2020_04_26",
              type=click.Path(exists=True, file_okay=True, dir_okay=False))
def prepare_dataset_cli(dataset_root,
                        dataset_size,
                        val_size,
                        test_size,
                        batch_size,
                        torch_seed,
                        val_torch_seed,
                        test_torch_seed,
                        generator_checkpoint_path):
    prepare_dataset(dataset_root=dataset_root,
                    dataset_size=dataset_size,
                    val_size=val_size,
                    test_size=test_size,
                    batch_size=batch_size,
                    torch_seed=torch_seed,
                    val_torch_seed=val_torch_seed,
                    test_torch_seed=test_torch_seed,
                    generator_checkpoint_path=generator_checkpoint_path)


@dcgan32_inversion.command("train", help="Train a regression model to invert the generator of DCGAN32")
@click.option("--n_channels", default=350, type=click.INT)
@click.option("--learning_rate", default=5e-4, type=click.FLOAT)
@click.option("--n_epochs", default=70, type=click.INT)
@click.option("--data_root", default="data/processed/dcgan32_inversion", type=click.Path())
@click.option("--exp_root", default="tmp/inversion_experiment", type=click.Path())
@click.option("--exp_id", default="1", type=click.STRING)
@click.option("--loss_report_period", default=150, type=click.INT)
def train(n_channels,
          learning_rate,
          n_epochs,
          data_root,
          exp_root,
          exp_id,
          loss_report_period):
    train_inversion_model(n_channels=n_channels,
                          learning_rate=learning_rate,
                          n_epochs=n_epochs,
                          data_root=data_root,
                          exp_root=exp_root,
                          exp_id=exp_id,
                          loss_report_period=loss_report_period)
