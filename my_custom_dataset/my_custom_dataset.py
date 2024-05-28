import tensorflow_datasets as tfds
import tensorflow as tf
import os
from pycocotools.coco import COCO
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

class MyCustomDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for my_dataset dataset."""

    VERSION = tfds.core.Version('1.0.0')

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Tensor(shape=(320, 240, 3), dtype=np.float32),
                'mask': tfds.features.Tensor(shape=(320, 240), dtype=np.uint8),
            }),
            supervised_keys=('image', 'mask'),
        )

    def _split_generators(self, dl_manager):
        root_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
        images_folder = os.path.join(root_folder, 'images')
        annotations_file = os.path.join(root_folder, 'instances.json')

        coco = COCO(annotations_file)
        img_ids = coco.getImgIds()

        train_ids, test_ids = train_test_split(img_ids, test_size=0.3, random_state=42)
        val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=42)

        return {
            'train': self._generate_examples(images_folder, coco, train_ids),
            'val': self._generate_examples(images_folder, coco, val_ids),
            'test': self._generate_examples(images_folder, coco, test_ids),
        }

    def __create_masks(self,image_id, coco):
        mask = np.zeros((1024, 768), dtype=np.uint8)
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            category_id = ann['category_id']
            binary_mask = coco.annToMask(ann)
            mask[binary_mask == 1] = category_id
        return mask

    def _generate_examples(self, images_folder, coco, ids):
        # TODO(my_dataset): Yields (key, example) tuples from the dataset

        image_data = coco.loadImgs(ids)
        for img in image_data:
            image_name = img["file_name"]
            image_path = os.path.join(images_folder, image_name)
            image = cv2.imread(image_path)

            resized_image = cv2.resize(image, (240, 320))
            normalized_image = resized_image.astype(np.float32) / 255.0

            mask = self.__create_masks(img['id'], coco)
            resized_mask = cv2.resize(mask, (240, 320))

            yield image_name, {
                "image": normalized_image,
                "mask": resized_mask}
