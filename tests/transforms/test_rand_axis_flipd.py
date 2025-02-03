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

import numpy as np
import torch

from monai.data import MetaTensor, set_track_meta
from monai.transforms import RandAxisFlipd
from tests.lazy_transforms_utils import test_resampler_lazy
from tests.test_utils import TEST_NDARRAYS_ALL, NumpyImageTestCase3D, assert_allclose, test_local_inversion


class TestRandAxisFlip(NumpyImageTestCase3D):
    def test_correct_results(self):
        for p in TEST_NDARRAYS_ALL:
            flip = RandAxisFlipd(keys="img", prob=1.0)
            flip.set_random_state(seed=1234)
            im = p(self.imt[0])
            call_param = {"data": {"img": im}}
            result = flip(**call_param)

            # test lazy
            test_resampler_lazy(flip, result, call_param=call_param, output_key="img", seed=1234)
            flip.lazy = False

            test_local_inversion(flip, result, {"img": im}, "img")
            expected = [np.flip(channel, flip.flipper._axis) for channel in self.imt[0]]
            assert_allclose(result["img"], p(np.stack(expected)), type_test="tensor")

            set_track_meta(False)
            result = flip({"img": im})["img"]
            self.assertNotIsInstance(result, MetaTensor)
            self.assertIsInstance(result, torch.Tensor)
            set_track_meta(True)


if __name__ == "__main__":
    unittest.main()
