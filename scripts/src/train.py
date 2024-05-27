from transformers import Trainer, TrainingArguments

def train_model(model_loader, train_dataset):
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',  # Log directory
        logging_steps=10,      # Log every 10 steps
    )

    trainer = Trainer(
        model=model_loader.load_model(inference_only=False),
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=model_loader.load_tokenizer(),
    )

    trainer.train()
    model_loader.save_model(Config.finetuned_model_dir)