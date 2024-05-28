import os
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
import tensorflow_datasets as tfds
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
import unittest

# Assuming MyCustomDataset is defined in my_custom_dataset.py
from my_custom_dataset import MyCustomDataset


class MyCustomDatasetTest(unittest.TestCase):
    DATASET_CLASS = MyCustomDataset
    SPLITS = {
        "train": 3,  # Number of fake train examples
        "val": 1,  # Number of fake val examples
        "test": 1,  # Number of fake test examples
    }

    DL_EXTRACT_RESULT = {
        'images_folder': 'path/to/fake_images_folder',
        'annotations_file': 'path/to/fake_instances.json',
    }

    def setUp(self):
        self.dataset = MyCustomDataset()

    @patch('my_custom_dataset.os.path.join')
    @patch('my_custom_dataset.COCO')
    @patch('my_custom_dataset.train_test_split')
    def test_split_generators(self, mock_train_test_split, mock_COCO, mock_path_join):
        mock_path_join.side_effect = lambda *args: "/".join(args)
        mock_coco_instance = MagicMock()
        mock_COCO.return_value = mock_coco_instance
        mock_train_test_split.side_effect = [
            (['img1', 'img2', 'img3'], ['img4', 'img5']),
            (['img4'], ['img5'])
        ]

        splits = self.dataset._split_generators(dl_manager=None)

        self.assertEqual(set(splits.keys()), {'train', 'val', 'test'})
        self.assertTrue(hasattr(splits['train'], '__iter__'))
        self.assertTrue(hasattr(splits['val'], '__iter__'))
        self.assertTrue(hasattr(splits['test'], '__iter__'))

    @patch('my_custom_dataset.cv2.imread')
    @patch('my_custom_dataset.cv2.resize')
    def test_generate_examples(self, mock_resize, mock_imread):
        images_folder = "/dummy/images_folder"
        mock_coco = MagicMock()
        mock_coco.loadImgs.return_value = [
            {"id": 1, "file_name": "image1.jpg"},
            {"id": 2, "file_name": "image2.jpg"}
        ]

        mock_imread.return_value = np.zeros((1024, 768, 3), dtype=np.uint8)
        mock_resize.side_effect = lambda img, size: np.zeros(
            (size[1], size[0], img.shape[2] if len(img.shape) > 2 else 1), dtype=img.dtype)

        examples = list(self.dataset._generate_examples(images_folder, mock_coco, [1, 2]))

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0][0], "image1.jpg")
        self.assertEqual(examples[1][0], "image2.jpg")
        self.assertIn("image", examples[0][1])
        self.assertIn("mask", examples[0][1]) # Check normalization


if __name__ == "__main__":
    unittest.main()