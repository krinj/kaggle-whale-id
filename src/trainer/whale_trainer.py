
# -*- coding: utf-8 -*-

"""
<ENTER DESCRIPTION HERE>
"""
import random
from typing import List

import numpy as np
from raid import Trainer, IDataset
from raid.data.functional.score_counter import ScoreCounter
from raid.logic.config import Config

import torch
import torch.nn.functional as F
from torch import optim
from torchvision import transforms

from data.siamese_sample import SiameseSample
from trainer.whale_net import WhaleNet

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        same = label
        diff = 1 - label
        loss_contrastive = torch.mean(same * torch.pow(euclidean_distance, 2) +
                                      diff * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


class WhaleTrainer(Trainer):

    def __init__(self, config: Config):

        freeze = config.get("freeze", False)
        nf = config.get("nf", False)
        size = config.get("size", 256)

        self.net = WhaleNet(freeze, nf).cuda()
        self.criterion = ContrastiveLoss()
        self.optimizer = optim.Adam(self.net.parameters())
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

        super().__init__(config)

    def step(self, data_batch: List[SiameseSample]) -> float:

        image_1_tensors = [self.transform(s.sample_1.pil_image) for s in data_batch]
        image_2_tensors = [self.transform(s.sample_2.pil_image) for s in data_batch]
        labels_tensor = torch.FloatTensor([s.label for s in data_batch]).cuda()

        input_1_tensor = torch.stack(image_1_tensors, dim=0).cuda()
        input_2_tensor = torch.stack(image_2_tensors, dim=0).cuda()

        image_1_encoding = self.net.forward(input_1_tensor)
        image_2_encoding = self.net.forward(input_2_tensor)

        self.optimizer.zero_grad()
        loss = self.criterion(image_1_encoding, image_2_encoding, labels_tensor)
        loss.backward()
        self.optimizer.step()
        print(loss.item())
        return loss.item()

    def validate(self, validation_set: IDataset) -> float:

        n_correct = 0

        # score_counter = ScoreCounter(validation_set.label_map)

        self.net.eval()

        # n_gallery_sample_max = 36
        # gallery_samples = []
        # gallery_predictions = []
        n_batch_validation = 8

        n_total = (len(validation_set.samples) // n_batch_validation) * n_batch_validation
        mini_batch = []

        all_distances = []
        all_labels = []

        print("Validating")

        for s_index, s in enumerate(validation_set.samples):

            mini_batch.append(s)

            # Continue batch until it is full, or we are at the end of the samples.
            if len(mini_batch) < n_batch_validation and s_index < len(validation_set) - 1:
                continue

            image_1_tensors = [self.transform(s.sample_1.pil_image) for s in mini_batch]
            image_2_tensors = [self.transform(s.sample_2.pil_image) for s in mini_batch]
            input_1_tensor = torch.stack(image_1_tensors, dim=0).cuda()
            input_2_tensor = torch.stack(image_2_tensors, dim=0).cuda()

            image_1_encodings = self.net.forward(input_1_tensor)
            image_2_encodings = self.net.forward(input_2_tensor)
            d = F.pairwise_distance(image_1_encodings, image_2_encodings)

            all_distances += d.cpu().detach().numpy().tolist()
            all_labels += [s.label for s in mini_batch]

            # Clear the mini-batch for the next run.
            mini_batch.clear()

        neg_dist = [all_distances[i] for i, label in enumerate(all_labels) if label == 0]
        pos_dist = [all_distances[i] for i, label in enumerate(all_labels) if label == 1]

        pos_median = np.median(pos_dist)
        neg_median = np.median(neg_dist)

        recommended_t_median = float((neg_median - pos_median) * 0.5 + pos_median)
        self.add_metric("threshold", recommended_t_median)

        for i in range(n_total):
            d = all_distances[i]
            label = all_labels[i]

            if d < recommended_t_median and label == 1:
                n_correct += 1
            elif d >= recommended_t_median and label == 0:
                n_correct += 1

        print("Score", n_correct/n_total)
        # Switch back to training mode.
        self.net.train()
        return n_correct/n_total

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)
        pass
