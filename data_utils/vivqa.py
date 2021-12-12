import torch
from torch.utils import data
from torch.utils.data.dataset import random_split
from data_utils.utils import preprocess_question, preprocess_answer, get_transform
from data_utils.vocab import Vocab

import os
from PIL import Image
import json
import config

class ViVQA(data.Dataset):
    """ VQA dataset, open-ended """
    def __init__(self, json_path, image_size, vocab=None):
        super(ViVQA, self).__init__()
        with open(json_path, 'r') as fd:
            json_data = json.load(fd)

        # vocab
        self.vocab = Vocab([json_path]) if vocab is None else vocab

        # questions and answers
        self.questions, self.answers, self.image_paths = self.load_json(json_data)

        # images
        self.transform = get_transform(image_size)

    @property
    def max_question_length(self):
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.questions)) + 2
        return self._max_length + 2

    @property
    def num_tokens(self):
        return len(self.vocab.stoi)

    def load_json(self, json_data):
        questions = []
        answers = []

        for ann in json_data["annotations"]:
            questions.append(preprocess_question(ann["question"]))
            answers.append(preprocess_answer(ann["answer"]))

        image_paths = []
        for image in json_data["images"]:
            image_paths.append(os.path.join(config.qa_path, image["filepath"], image["file_name"]))

        return questions, answers, image_paths

    def _load_image(self, idx):
        """ Load an image """
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        return self.transform(image)

    def __getitem__(self, idx):
        q = self.vocab._encode_question(self.questions[idx])
        a = self.vocab._encode_answer(self.answers[idx])
        v = self._load_image(idx)

        return v, q, a

    def __len__(self):
        return len(self.questions)


def get_loader(train_dataset, test_dataset=None):
    """ Returns a data loader for the desired split """

    fold_size = int(len(train_dataset) * 0.2)

    subdatasets = random_split(train_dataset, [fold_size, fold_size, fold_size, fold_size, len(train_dataset) - fold_size*4], generator=torch.Generator().manual_seed(13))
    
    folds = []
    for subdataset in subdatasets:
        folds.append(
            torch.utils.data.DataLoader(
                subdataset,
                batch_size=config.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=config.data_workers))

    if test_dataset:
        test_fold = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=config.batch_size,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=config.data_workers)

        return folds, test_fold

    return folds