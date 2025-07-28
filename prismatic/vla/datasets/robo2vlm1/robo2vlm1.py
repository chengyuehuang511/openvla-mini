import tensorflow_datasets as tfds
import json
import os

from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf

r"""
make sure to rename the datasets folder before running the script!

TFDS_DATA_DIR=/storage/home/hcoda1/8/mzhang445/p-smukhopadhyay6-0/datasets/tensorflow-datasets
tfds build --datasets=robo2vlm1


"""


_CITATION = (
    '@misc{chen2025robo2vlmvisualquestionanswering,'
      'title={Robo2VLM: Visual Question Answering from Large-Scale In-the-Wild Robot Manipulation Datasets}, '
      'author={Kaiyuan Chen and Shuangyu Xie and Zehan Ma and Pannag R Sanketi and Ken Goldberg},'
      'year={2025},'
      'eprint={2505.15517},'
      'archivePrefix={arXiv},'
      'primaryClass={cs.RO},'
      'url={https://arxiv.org/abs/2505.15517},}')

_IMAGE_DIR = '/nethome/chuang475/flash/robo2vlm1/images'

class Robo2vlm1(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for Robo2VLM1 dataset."""
    VERSION = tfds.core.Version('3.0.0')

    code_path = None

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description='Robo2VLM1 dataset for VQA tasks.',
            features=tfds.features.FeaturesDict({
                'image/id': np.int32,
                'image/filename': tfds.features.Text(),
                'image': tfds.features.Image(encoding_format='jpeg'),
                'question_id': tfds.features.Text(),
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
            'test': self._generate_examples('test'),
        }

    def _generate_examples(self, split):
        """Yields (key, example) tuples from the dataset."""
        from datasets import load_dataset
        image_folder = os.path.join(_IMAGE_DIR, split)
        os.makedirs(image_folder, exist_ok=True)

        dataset = load_dataset('keplerccc/Robo2VLM-1', split=split)

        for idx, item in enumerate(dataset):
            image_id = idx
            qid = item['id']
            
            image = item['image']
            choices = eval(item['choices'])
            answer_type = 'other'
            question_type = 'unknown'

            answer = str(item['correct_answer'])

            options = ", ".join([f"{i}: {choice}" for i, choice in enumerate(choices)])
            question = item['question'] + f" Please select the number corresponding to your answer from among these options: [{options}]."

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
                'answers': [answer] * 10,
                'answer_confidences': ['yes'] * 10,
                'top_answer': answer,
            }
