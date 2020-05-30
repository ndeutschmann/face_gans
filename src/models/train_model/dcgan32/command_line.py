"""CLI to train dcgan32-related models"""

import click
from .dcgan32_inversion import prepare_dataset, train_inversion_model
from .train_dcgan32 import train_dcgan32


@click.group()
def dcgan32():
    """Commands related to DCGAN32"""


train_dcgan32_options = [
    click.Option(["--n_epochs"], default=100, type=click.INT),
    click.Option(["--exp_root"], default="tmp/dcgan32_training", type=click.STRING),
    click.Option(["--exp_id"], default="1", type=click.STRING),
    click.Option(["--loss_report_period"], default=150, type=click.INT),
    click.Option(["--batch_size"], default=128, type=click.INT),
    click.Option(["--mem_batch_size"], default=16, type=click.INT),
    click.Option(["--mem_capacity"], default=128, type=click.INT),
    click.Option(["--mem_update_size"], default=16, type=click.INT),
    click.Option(["--mem_update_prob"], default=0.7, type=click.FLOAT),
    click.Option(["--learning_rate_g"], default=3.e-5, type=click.FLOAT),
    click.Option(["--learning_rate_d"], default=6.e-5, type=click.FLOAT),
    click.Option(["--noise_level"], default=0.2, type=click.FLOAT),
    click.Option(["--noise_damping"], default=1.5, type=click.FLOAT),
    click.Option(["--noise_damping_period"], default=5, type=click.INT),
    click.Option(["--data_root"], default="celebA/unlabelled_centered", type=click.STRING),
    click.Option(["--num_workers"], default=1, type=click.INT),
    click.Option(["--torch_seed"], default=None, type=click.INT),
    click.Option(["--checkpoint_period"], default=10, type=click.INT),
]

train_dcgan32_cli = click.Command(name="train",
                                  callback=train_dcgan32,
                                  help="Train DCGAN32",
                                  params=train_dcgan32_options)

dcgan32.add_command(train_dcgan32_cli)


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
    # This is a pretty bad design pattern I should avoid this extra layer which makes it hard to
    # Propagate values.
    prepare_dataset(dataset_root=dataset_root,
                    dataset_size=dataset_size,
                    val_size=val_size,
                    test_size=test_size,
                    batch_size=batch_size,
                    torch_seed=torch_seed,
                    val_torch_seed=val_torch_seed,
                    test_torch_seed=test_torch_seed,
                    generator_checkpoint_path=generator_checkpoint_path)


train_inversion_options = [
    click.Option(["--n_channels"], default=128, type=click.INT),
    click.Option(["--learning_rate"], default=5e-4, type=click.FLOAT),
    click.Option(["--dropout_rate"], default=.3, type=click.FLOAT),
    click.Option(["--weight_decay"], default=0., type=click.FLOAT),
    click.Option(["--noise_level"], default=.1, type=click.FLOAT),
    click.Option(["--batch_size"], default=32, type=click.INT),
    click.Option(["--val_batch_size"], default=None, type=click.INT),
    click.Option(["--n_epochs"], default=70, type=click.INT),
    click.Option(["--data_root"], default="data/processed/dcgan32_inversion", type=click.Path()),
    click.Option(["--exp_root"], default="tmp/inversion_experiment", type=click.Path()),
    click.Option(["--exp_id"], default="1", type=click.STRING),
    click.Option(["--loss_report_period"], default=150, type=click.INT),
    click.Option(["--model_version"], default=2, type=click.INT),
]

train_inversion_cli = click.Command(
    name="train",
    help="Train a regression model to invert the generator of DCGAN32",
    callback=train_inversion_model,
    params=train_inversion_options
)

dcgan32_inversion.add_command(train_inversion_cli)
