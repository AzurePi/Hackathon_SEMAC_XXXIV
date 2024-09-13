import keras_nlp
import pandas as pd
import argparse


def load_model(args):
    if args.model == 'roberta':
        model = keras_nlp.models.RobertaClassifier.from_preset(
            "roberta_large_en",
            num_classes=2,
        )
    elif args.model == 'distilbert':
        model = keras_nlp.models.DistilBertClassifier.from_preset(
            "distil_bert_base_en",
            num_classes=2
        )
    else:
        model = None

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='roberta')
    parser.add_argument('--classifier', '-c', type=str, default='none')
    args = parser.parse_args()

    reader = pd.read_csv("data/train.csv")
    train_x = reader['text'].values
    train_y = reader['feeling'].values

    reader = pd.read_csv("test.csv")
    test_x = reader['text'].values

    model = load_model(args)

    model.fit(x=train_x, y=train_y, batch_size=64, epochs=10)
    model.predict(x=test_x, batch_size=64)

    return 0


if __name__ == '__main__':
    main()
