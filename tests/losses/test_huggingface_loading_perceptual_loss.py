# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from monai.losses import PerceptualLoss
from monai.utils import optional_import
from tests.test_utils import skip_if_downloading_fails, skip_if_quick

_, has_torchvision = optional_import("torchvision")


@unittest.skipUnless(has_torchvision, "Requires torchvision")
@skip_if_quick
class TestHuggingFaceLoadingPerceptualLoss(unittest.TestCase):
    def test_medicalnet_resnet10_loading(self):
        """Test MedicalNet ResNet10 loading from Hugging Face."""
        with skip_if_downloading_fails():
            loss = PerceptualLoss(spatial_dims=3, network_type="medicalnet_resnet10_23datasets", is_fake_3d=False)

        input_tensor = torch.randn(1, 1, 32, 32, 32)
        target_tensor = torch.randn(1, 1, 32, 32, 32)
        result = loss(input_tensor, target_tensor)

        self.assertEqual(result.shape, torch.Size([]))
        self.assertIsInstance(result.item(), float)

    def test_medicalnet_resnet50_loading(self):
        """Test MedicalNet ResNet50 loading from Hugging Face."""
        with skip_if_downloading_fails():
            loss = PerceptualLoss(spatial_dims=3, network_type="medicalnet_resnet50_23datasets", is_fake_3d=False)

        input_tensor = torch.randn(1, 1, 32, 32, 32)
        target_tensor = torch.randn(1, 1, 32, 32, 32)
        result = loss(input_tensor, target_tensor)

        self.assertEqual(result.shape, torch.Size([]))
        self.assertIsInstance(result.item(), float)

    def test_radimagenet_loading(self):
        """Test RadImageNet ResNet50 loading from Hugging Face."""
        with skip_if_downloading_fails():
            loss = PerceptualLoss(spatial_dims=2, network_type="radimagenet_resnet50")

        input_tensor = torch.randn(1, 1, 64, 64)
        target_tensor = torch.randn(1, 1, 64, 64)
        result = loss(input_tensor, target_tensor)

        self.assertEqual(result.shape, torch.Size([]))
        self.assertIsInstance(result.item(), float)

    def test_checksum_failure(self):
        """Test that a RuntimeError is raised when checksum verification fails."""
        with patch("monai.losses.perceptual.hf_hub_download", return_value="/tmp/dummy_path"):
            with patch("monai.losses.perceptual.check_hash", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "Hash mismatch for file"):
                    PerceptualLoss(spatial_dims=3, network_type="medicalnet_resnet10_23datasets", is_fake_3d=False)


if __name__ == "__main__":
    unittest.main()
