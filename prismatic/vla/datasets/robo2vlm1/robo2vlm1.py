from datasets import load_dataset
import tensorflow_datasets as tfds
import json
import os

from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf

_CITATION = (
    '@misc{chen2025robo2vlmvisualquestionanswering,'
      'title={Robo2VLM: Visual Question Answering from Large-Scale In-the-Wild Robot Manipulation Datasets}, '
      'author={Kaiyuan Chen and Shuangyu Xie and Zehan Ma and Pannag R Sanketi and Ken Goldberg},'
      'year={2025},'
      'eprint={2505.15517},'
      'archivePrefix={arXiv},'
      'primaryClass={cs.RO},'
      'url={https://arxiv.org/abs/2505.15517},}')

class Robo2VLM1(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Robo2VLM1 dataset."""

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description='Robo2VLM1 dataset for VQA tasks.',
            features=tfds.features.FeaturesDict({
                'image/id': np.int32,
                'image/filename': tfds.features.Text(),
                'image': tfds.features.Image(encoding_format='jpeg'),
                'question_id': np.int32,
                'question_type': tfds.features.Text(),
                'question_text': tfds.features.Text(),
                'answer_type': tfds.features.Text(),
                'answers': tfds.features.Sequence(tfds.features.Text()),
                'answer_confidences': tfds.features.Sequence(
                    tfds.features.ClassLabel(names=['no', 'maybe', 'yes'])),
                'top_answer': tfds.features.Text(),
            }),
            homepage='https://berkeleyautomation.github.io/robo2vlm/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        return {
            'train': self._generate_examples('train'),
            'val': self._generate_examples('val'),
            'test': self._generate_examples('test'),
        }

    def _generate_examples(self, split, image_folder=None):
        """Yields (key, example) tuples from the dataset."""
        image_folder = image_folder or split
        os.makedirs(image_folder, exist_ok=True)

        dataset = load_dataset('keplerccc/Robo2VLM-1', split=split)

        for item in dataset:
            qid = item['id']
            question = item['question']
            choices = item['choices']
            correct_answer = item['correct_answer']
            image = item['image']
            answer_type = 'other'
            question_type = 'unknown'

            image_filepath = os.path.join(image_folder, f'{image_id}.jpg')
            image.save(image_filepath)

            yield qid, {
                'image/id': image_id,
                'image/filename': f'{image_id}.jpg',
                'image': image_filepath,
                'question_id': qid,
                'question_type': question_type,
                'question_text': question,
                'answer_type': answer_type,
                'answers': answers,
                'answer_confidences': item.get('answer_confidences'),
                'top_answer': item.get('top_answer'),
            }
