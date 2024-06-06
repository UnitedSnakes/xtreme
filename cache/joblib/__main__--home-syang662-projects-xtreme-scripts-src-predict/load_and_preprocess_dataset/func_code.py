# first line: 239
@conditional_cache
def load_and_preprocess_dataset(
    language: str,
    split: str,
    add_index: bool = True,
) -> DatasetDict:
    dataset = load_dataset_by_name(
        language, split=split, switch_source_and_target=Config.swap_sentences
    )

    max_length = 0

    if Config.dataset_name == "multi_nli":
        tokenizer = ModelLoader.get_tokenizer(Config.model)
        all_sentences = [
            sentence
            for example in dataset
            for sentence in [example["premise"], example["hypothesis"]]
        ]

        max_length = max(
            len(tokenizer(sentence, truncation=True).input_ids)
            for sentence in all_sentences
        )
        print(f"max length:{max_length}")

    dataset = preprocess_dataset(dataset, max_length)

    if add_index and not Config.is_fine_tune:
        dataset = add_index_column(dataset)

    dataset.set_format(
        type="torch",
        columns=["input_ids", "token_type_ids", "attention_mask", "labels"],
    )

    return dataset
