import unittest
from unittest.mock import patch, MagicMock
import tensorflow as tf
import tensorflow_datasets as tfds
import os
from model import Segmentator

class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.segmentator = Segmentator()

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('tensorflow_datasets.load')
    def test_load_dataset(self, mock_tfds_load, mock_path_exists, mock_makedirs):
        mock_path_exists.return_value = False
        dataset_mock = MagicMock()
        dataset_mock.map.return_value.batch.return_value.prefetch.return_value = 'dataset'
        mock_tfds_load.return_value = dataset_mock

        dataset = self.segmentator.load_dataset('train')

        # Check if the dataset directory was created
        mock_makedirs.assert_called_with('../dataset')
        # Check if the dataset was loaded correctly
        mock_tfds_load.assert_called_with('my_custom_dataset', split='train', data_dir='../dataset')
        # Check if the preprocessing, batching, and prefetching were applied
        self.assertEqual(dataset, 'dataset')

    @patch('tensorflow.keras.models.Model')
    def test_save_model(self, MockModel):
        mock_model = MockModel.return_value
        self.segmentator.model = mock_model
        self.segmentator.save_model('fake_model_path')
        mock_model.save.assert_called_with('fake_model_path')

    def test_set_epochs(self):
        self.segmentator.set_epochs(40)
        self.assertEqual(self.segmentator.epochs, 40)
        self.segmentator.set_epochs(-1)
        self.assertNotEqual(self.segmentator.epochs, -1)

    def test_set_batch_size(self):
        self.segmentator.set_batch_size(32)
        self.assertEqual(self.segmentator.batch_size, 32)
        self.segmentator.set_batch_size(-1)
        self.assertNotEqual(self.segmentator.batch_size, -1)

    def test_build_model(self):
        self.segmentator.build_model()
        self.assertIsNotNone(self.segmentator.model)
        self.assertIsInstance(self.segmentator.model, tf.keras.Model)
        self.assertEqual(self.segmentator.model.input_shape, (None, 320, 240, 3))
        self.assertEqual(self.segmentator.model.output_shape, (None, 320, 240, 11))

    @patch('tensorflow.keras.Model.fit')
    def test_train_model(self, mock_fit):
        self.segmentator.build_model()
        train_dataset = MagicMock()
        val_dataset = MagicMock()
        self.segmentator.train_model(train_dataset, val_dataset)
        mock_fit.assert_called_with(
            train_dataset,
            epochs=self.segmentator.epochs,
            validation_data=val_dataset,
            callbacks=self.segmentator.callbacks
        )

if __name__ == '__main__':
    unittest.main()
