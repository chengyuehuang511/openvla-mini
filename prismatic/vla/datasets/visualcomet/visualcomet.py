import os
import json
import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

_DESCRIPTION = 'VisualCOMET dataset in VQA format with bbox-guided questions.'
_CITATION = """
@inproceedings{park2020visualcomet,
  title={Visualcomet: Reasoning about the dynamic context of a still image},
  author={Park, Jae Sung and Bhagavatula, Chandra and Mottaghi, Roozbeh and Farhadi, Ali and Choi, Yejin},
  booktitle={European Conference on Computer Vision},
  pages={508--524},
  year={2020},
  organization={Springer}
}
"""

_IMAGE_DIR = '/home/hice1/skim3513/scratch/VLA-VQA/datasets_src/images/visualcomet-bbox'
_TRAIN_JSON_PATH = '/home/hice1/skim3513/scratch/VLA-VQA/datasets_src/datasets_files/visualcomet/train_annots.json'
_VAL_JSON_PATH = '/home/hice1/skim3513/scratch/VLA-VQA/datasets_src/datasets_files/visualcomet/val_annots.json'

class Visualcomet(tfds.core.GeneratorBasedBuilder):
  """TFDS-style VisualCOMET dataset."""

  VERSION = tfds.core.Version('1.0.0')

  def _info(self) -> tfds.core.DatasetInfo:
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
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
        homepage='https://visualcomet.xyz',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    return {
        'train': self._generate_examples(_TRAIN_JSON_PATH),
        'val': self._generate_examples(_VAL_JSON_PATH),
    }

  def _generate_examples(self, json_path):
    """Yields (key, example) tuples."""
    with open(json_path, 'r') as f:
        items = json.load(f)

    question_templates = {
        "event": "what is happening here?",
        "before": "what happened before?",
        "after": "what will happen after?",
        "intent": "what is the intent?"
    }

    bbox_guide = (
        "We have drawn bounding boxes around each person in the image. "
        "Please refer to each person by their number: red is person 1, orange is 2, yellow is 3, green is 4, and blue is 5. "
        "When answering the question, use these numbers (e.g., '1' instead of 'the red person')."
    )

    uid = 0
    for item in items:
        img_fn = item["img_fn"]
        full_img_path = os.path.join(_IMAGE_DIR, img_fn)
        if not os.path.exists(full_img_path):
            continue

        for key in ["event", "before", "after", "intent"]:
            if key in item and item[key]:
                question_text = f"{bbox_guide} {question_templates[key]}"
                answers = item[key] if isinstance(item[key], list) else [item[key]]

                yield uid, {
                    "image/id": int(uid),
                    "image/filename": img_fn,
                    "image": full_img_path,
                    "question_id": str(uid),
                    "question_type": key,
                    "question_text": question_text,
                    "answer_type": "",  
                    "answers": answers,
                    "answer_confidences": ["yes"] * len(answers),
                    "top_answer": answers[0],
                }
                uid += 1
