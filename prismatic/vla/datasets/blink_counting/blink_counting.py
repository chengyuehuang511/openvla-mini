import os
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow_datasets as tfds
from datasets import load_dataset

_DESCRIPTION = 'BLINK Benchmark dataset for visual multiple-choice reasoning.'
_CITATION = """
@inproceedings{fu2024blink,
  title={Blink: Multimodal large language models can see but not perceive},
  author={Fu, Xingyu and Hu, Yushi and Li, Bangzheng and Feng, Yu and Wang, Haoyu and Lin, Xudong and Roth, Dan and Smith, Noah A and Ma, Wei-Chiu and Krishna, Ranjay},
  booktitle={European Conference on Computer Vision},
  pages={148--166},
  year={2024},
  organization={Springer}
}
"""

def concat_images_horizontally_with_margin(images, margin=10):
    max_height = max(image.height for image in images)
    total_width = sum(image.width for image in images) + margin * (len(images) - 1)
    new_image = Image.new('RGB', (total_width, max_height), (0, 0, 0))

    x_offset = 0
    for image in images:
        y_offset = (max_height - image.height) // 2
        new_image.paste(image, (x_offset, y_offset))
        x_offset += image.width + margin
    return new_image


class Blink(tfds.core.GeneratorBasedBuilder):
    """TFDS builder for BLINK dataset."""

    VERSION = tfds.core.Version('1.0.0')

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'image/id': tfds.features.Text(),
                'image': tfds.features.Image(encoding_format='jpeg'),
                'question_id': tfds.features.Text(),
                'question_text': tfds.features.Text(),
                'answers': tfds.features.Sequence(tfds.features.Text()),
                'answer_confidences': tfds.features.Sequence(tfds.features.ClassLabel(names=["yes"])),
                'top_answer': tfds.features.Text(),
                'multiple_choices': tfds.features.Sequence(tfds.features.Text()),
            }),
            homepage='https://huggingface.co/datasets/BLINK-Benchmark/BLINK',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return {
            'train': self._generate_examples(split='val'),
            'validation': self._generate_examples(split='val'),
            'test': self._generate_examples(split='test'),
        }

    def _generate_examples(self, split):
        dataset = load_dataset("BLINK-Benchmark/BLINK", "Counting", split=split)

        for idx, example in enumerate(dataset):
            image_list = []
            for i in range(1, 5):
                img = example.get(f"image_{i}", None)
                if img:
                    if isinstance(img, Image.Image):
                        image_list.append(img)
                    elif isinstance(img, dict) and "bytes" in img:
                        image = Image.open(BytesIO(img["bytes"])).convert("RGB")
                        image_list.append(image)

            if not image_list:
                continue

            concat_image = concat_images_horizontally_with_margin(image_list)

            question = f"{example.get('prompt', '').strip()}\n{example['question'].strip()}".strip()
            answer = example["answer"]
            answers = [answer] if answer != "hidden" else []
            top_answer = answers[0] if answers else ""
            answer_confidences = ["yes"] * len(answers)

            yield str(idx), {
                "image/id": str(idx),
                "image": concat_image,
                "question_id": str(example["idx"]),
                "question_text": question,
                "answers": answers,
                "answer_confidences": answer_confidences,
                "top_answer": top_answer,
                "multiple_choices": example["choices"],
            }
