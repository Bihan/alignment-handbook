from alignment import H4ArgumentParser, ModelArguments, DataArguments, SFTConfig, get_datasets
import logging

logger = logging.getLogger(__name__)


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    ###############
    # Load datasets
    ###############
    # raw_datasets = get_datasets(
    #     data_args,
    #     splits=data_args.dataset_splits,
    #     configs=data_args.dataset_configs,
    #     columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label"],
    # )
    # logger.info(
    #     f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    # )


if __name__ == "__main__":
    main()
