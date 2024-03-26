import numpy as np
import tensorflow as tf
from tqdm import tqdm
import transformers


def tokenize(
    tokenizer: transformers.PreTrainedTokenizerFast, data: list[str], max_len: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tokenizes a list of text strings into input IDs and attention masks suitable for input into pre-trained models.

    Args:
        data (list[str]): A list of text strings to be tokenized. Each text string in this list represents one piece of input data.
        max_len (int, optional): The maximum length of the tokenized sequences.
                                If a sequence is shorter than this length, it will be padded up to `max_len`.
                                If it is longer, it will be truncated to this length.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing two elements:
            - The first element is a NumPy array of input IDs.
              Each row in the array corresponds to the tokenized version of the corresponding text string in the input data.
            - The second element is a NumPy array of attention masks.
              Each row in the array is an attention mask corresponding to the tokenized version of the corresponding text string in the input data.
              It has 1s for real tokens and 0s for padding tokens.

    Note:
        This function assumes that a pre-trained tokenizer is available and has been instantiated as `tokenizer`.
    """
    input_ids = []
    attention_masks = []

    # tokenize data
    for i in tqdm(range(len(data))):
        encoded = tokenizer.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )

        # append input_ids, attention_masks
        input_ids.append(encoded["input_ids"])
        attention_masks.append(encoded["attention_mask"])

    return np.array(input_ids), np.array(attention_masks)


def create_model(
    roberta_model: transformers.PreTrainedModel, max_len: int = 512
) -> tf.keras.Model:
    """
    Initializes and compiles a TensorFlow model with a pre-trained RoBERTa model from the Hugging Face's Transformers library.
    This model is designed for a multi-class classification task, taking sequences of input IDs and attention masks,
    processing them through the RoBERTa model, and outputting class probabilities through a final dense layer.

    Args:
        roberta_model (transformers.PreTrainedModel): The pre-trained RoBERTa model
        max_len (int, optional): The maximum sequence length for model inputs. Input sequences will be padded or truncated to this length.

    Returns:
        tf.keras.Model: A compiled Keras model ready for training.
    """
    # model metrics
    opt = tf.keras.optimizers.Adam()
    loss = "binary_crossentropy"
    accuracy = "accuracy"

    # input layers
    input_ids = tf.keras.Input(shape=(max_len,), dtype="int32")
    attention_masks = tf.keras.Input(shape=(max_len,), dtype="int32")

    # roberta model
    output = roberta_model([input_ids, attention_masks])
    output = output[1]

    # dropout layer for regulization (might overfitting because random over sampler)
    output = tf.keras.layers.Dropout(0.15)(output)

    # dense layer with sigmoid activation (binary problem)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(output)

    # create and compile model
    model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)
    model.compile(opt, loss=loss, metrics=accuracy)
    return model
