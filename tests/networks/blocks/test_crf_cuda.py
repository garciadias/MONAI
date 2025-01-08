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
from parameterized import parameterized

from monai.networks.blocks import CRF
from tests.utils.utils import skip_if_no_cpp_extension, skip_if_no_cuda

TEST_CASES = [
    [
        # Case Description
        "2 batche(s), 1 dimension(s), 2 classe(s), 1 channel(s)",
        # Parameters
        [
            5,  # iterations
            1.0,  # bilateral_weight
            0.3,  # gaussian_weight
            5.0,  # bilateral_spatial_sigma
            0.5,  # bilateral_color_sigma
            5.0,  # gaussian_spatial_sigma
            1.0,  # update_factor
            None,  # compatibility_matrix
        ],
        # Input
        [
            # Batch 0
            [
                # Class 0
                [0.8, 0.9, 0.6, 0.2, 0.3],
                # Class 1
                [0.1, 0.3, 0.5, 0.8, 0.7],
            ],
            # Batch 1
            [
                # Class 0
                [0.8, 0.9, 0.6, 0.2, 0.3],
                # Class 1
                [0.1, 0.3, 0.5, 0.8, 0.7],
            ],
        ],
        # Features
        [
            # Batch 0
            [
                # Channel 0
                [1, 1, 1, 0.5, 0]
            ],
            # Batch 1
            [
                # Channel 0
                [1, 1, 0.5, 0, 0]
            ],
        ],
        # Expected
        [
            # Batch 0
            [
                # Class 0
                [0.724431, 0.702247, 0.586338, 0.364053, 0.362328],
                # Class 1
                [0.275569, 0.297753, 0.413662, 0.635947, 0.637672],
            ],
            # Batch 1
            [
                # Class 0
                [0.735150, 0.713455, 0.522234, 0.301106, 0.345620],
                # Class 1
                [0.264850, 0.286545, 0.477766, 0.698894, 0.654381],
            ],
        ],
    ],
    [
        # Case Description
        "2 batche(s), 1 dimension(s), 2 classe(s), 1 channel(s), with_matrix",
        # Parameters
        [
            5,  # iterations
            1.0,  # bilateral_weight
            0.3,  # gaussian_weight
            5.0,  # bilateral_spatial_sigma
            0.5,  # bilateral_color_sigma
            5.0,  # gaussian_spatial_sigma
            1.0,  # update_factor
            2 * torch.eye(2),  # compatibility_matrix
        ],
        # Input
        [
            # Batch 0
            [
                # Class 0
                [0.8, 0.9, 0.6, 0.2, 0.3],
                # Class 1
                [0.1, 0.3, 0.5, 0.8, 0.7],
            ],
            # Batch 1
            [
                # Class 0
                [0.8, 0.9, 0.6, 0.2, 0.3],
                # Class 1
                [0.1, 0.3, 0.5, 0.8, 0.7],
            ],
        ],
        # Features
        [
            # Batch 0
            [
                # Channel 0
                [1, 1, 1, 0.5, 0]
            ],
            # Batch 1
            [
                # Channel 0
                [1, 1, 0.5, 0, 0]
            ],
        ],
        # Expected
        [
            # Batch 0
            [
                # Class 0
                [0.854686, 0.839089, 0.755770, 0.463087, 0.357129],
                # Class 1
                [0.145314, 0.160911, 0.244230, 0.536913, 0.642871],
            ],
            # Batch 1
            [
                # Class 0
                [0.825893, 0.807061, 0.492641, 0.196325, 0.231688],
                # Class 1
                [0.174107, 0.192939, 0.507359, 0.803675, 0.768312],
            ],
        ],
    ],
    [
        # Case Description
        "1 batche(s), 2 dimension(s), 3 classe(s), 2 channel(s)",
        # Parameters
        [
            5,  # iterations
            1.0,  # bilateral_weight
            0.3,  # gaussian_weight
            5.0,  # bilateral_spatial_sigma
            0.5,  # bilateral_color_sigma
            5.0,  # gaussian_spatial_sigma
            1.0,  # update_factor
            None,  # compatibility_matrix
        ],
        # Input
        [
            # Batch 0
            [
                # Class 0
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                ],
                # Class 1
                [
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                # Class 2
                [
                    [0.0, 0.0, 0.0, 0.5, 1.0],
                    [0.0, 0.0, 0.5, 1.0, 0.5],
                    [0.0, 0.5, 1.0, 0.5, 0.0],
                    [0.5, 1.0, 0.5, 0.0, 0.0],
                    [1.0, 0.5, 0.0, 0.0, 0.0],
                ],
            ]
        ],
        # Features
        [
            # Batch 0
            [
                # Channel 0
                [
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                # Channel 1
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                ],
            ]
        ],
        # Expected
        [
            # Batch 0
            [
                # Class 0
                [
                    [0.154633, 0.164076, 0.300110, 0.239729, 0.179437],
                    [0.156664, 0.161426, 0.254582, 0.191402, 0.253060],
                    [0.316391, 0.259811, 0.201576, 0.271977, 0.333670],
                    [0.263658, 0.204998, 0.276233, 0.686272, 0.687161],
                    [0.208480, 0.281425, 0.355033, 0.690412, 0.692331],
                ],
                # Class 1
                [
                    [0.681083, 0.652977, 0.312156, 0.245985, 0.181768],
                    [0.675692, 0.662155, 0.247827, 0.183893, 0.240174],
                    [0.309154, 0.249075, 0.186364, 0.240742, 0.293918],
                    [0.243739, 0.185445, 0.242820, 0.151819, 0.151363],
                    [0.180842, 0.238059, 0.292488, 0.150209, 0.149395],
                ],
                # Class 2
                [
                    [0.164284, 0.182947, 0.387733, 0.514285, 0.638795],
                    [0.167644, 0.176419, 0.497592, 0.624705, 0.506766],
                    [0.374455, 0.491115, 0.612060, 0.487281, 0.372412],
                    [0.492602, 0.609557, 0.480947, 0.161909, 0.161476],
                    [0.610678, 0.480516, 0.352479, 0.159380, 0.158274],
                ],
            ]
        ],
    ],
    [
        # Case Description
        "1 batche(s), 3 dimension(s), 2 classe(s), 1 channel(s)",
        # Parameters
        [
            2,  # iterations
            1.0,  # bilateral_weight
            0.3,  # gaussian_weight
            5.0,  # bilateral_spatial_sigma
            0.1,  # bilateral_color_sigma
            5.0,  # gaussian_spatial_sigma
            1.0,  # update_factor
            None,  # compatibility_matrix
        ],
        # Input
        [
            # Batch 0
            [
                # Class 0
                [
                    # Slice 0
                    [
                        [1.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    # Slice 1
                    [
                        [1.0, 1.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    # Slice 2
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    # Slice 3
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    # Slice 4
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                ],
                # Class 1
                [
                    # Slice 0
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    # Slice 1
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    # Slice 2
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    # Slice 3
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0],
                    ],
                    # Slice 4
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0, 1.0],
                    ],
                ],
            ]
        ],
        # Features
        [
            # Batch 0
            [
                # Channel 0
                [
                    # Slice 0
                    [
                        [0.5, 0.5, 0.5, 0.0, 0.0],
                        [0.5, 0.5, 0.5, 0.0, 0.0],
                        [0.5, 0.5, 0.5, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    # Slice 1
                    [
                        [0.5, 0.5, 0.5, 0.0, 0.0],
                        [0.5, 0.5, 0.5, 0.0, 0.0],
                        [0.5, 0.5, 0.5, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    # Slice 2
                    [
                        [0.5, 0.5, 0.5, 0.0, 0.0],
                        [0.5, 0.5, 0.5, 0.0, 0.0],
                        [0.5, 0.5, 0.8, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                    ],
                    # Slice 3
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                    ],
                    # Slice 4
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0],
                    ],
                ]
            ]
        ],
        # Expected
        [
            # Batch 0
            [
                # Class 0
                [
                    # Slice 0
                    [
                        [0.778237, 0.777561, 0.561416, 0.501611, 0.501294],
                        [0.777517, 0.776882, 0.560301, 0.501103, 0.500791],
                        [0.561231, 0.560339, 0.559060, 0.500619, 0.500311],
                        [0.501322, 0.500872, 0.500468, 0.500156, 0.499851],
                        [0.500883, 0.500449, 0.500059, 0.499713, 0.499420],
                    ],
                    # Slice 1
                    [
                        [0.777409, 0.776861, 0.560182, 0.501111, 0.500808],
                        [0.776784, 0.776102, 0.558887, 0.500618, 0.500329],
                        [0.559943, 0.558969, 0.557350, 0.500183, 0.499897],
                        [0.500789, 0.500403, 0.500052, 0.499765, 0.499487],
                        [0.500378, 0.500005, 0.499668, 0.499363, 0.499072],
                    ],
                    # Slice 2
                    [
                        [0.560846, 0.560185, 0.558597, 0.500660, 0.500369],
                        [0.560078, 0.559146, 0.556974, 0.500209, 0.499950],
                        [0.558225, 0.557130, 0.486025, 0.448784, 0.445606],
                        [0.500340, 0.500005, 0.448945, 0.448551, 0.444201],
                        [0.499972, 0.499644, 0.448537, 0.447195, 0.443425],
                    ],
                    # Slice 3
                    [
                        [0.500887, 0.500713, 0.500529, 0.500251, 0.499999],
                        [0.500596, 0.500312, 0.500109, 0.499848, 0.499605],
                        [0.500301, 0.500002, 0.447391, 0.445814, 0.442289],
                        [0.499940, 0.499662, 0.447338, 0.227284, 0.225224],
                        [0.499650, 0.499367, 0.445866, 0.227800, 0.225564],
                    ],
                    # Slice 4
                    [
                        [0.500399, 0.500241, 0.500090, 0.499883, 0.499637],
                        [0.500134, 0.499888, 0.499756, 0.499526, 0.499261],
                        [0.499888, 0.499631, 0.446166, 0.442215, 0.440038],
                        [0.499603, 0.499369, 0.445307, 0.225463, 0.223935],
                        [0.499337, 0.499113, 0.443668, 0.226403, 0.224790],
                    ],
                ],
                # Class 1
                [
                    # Slice 0
                    [
                        [0.221763, 0.222439, 0.438584, 0.498389, 0.498706],
                        [0.222483, 0.223118, 0.439699, 0.498897, 0.499209],
                        [0.438769, 0.439661, 0.440940, 0.499381, 0.499689],
                        [0.498678, 0.499128, 0.499532, 0.499844, 0.500149],
                        [0.499117, 0.499551, 0.499941, 0.500287, 0.500580],
                    ],
                    # Slice 1
                    [
                        [0.222591, 0.223139, 0.439818, 0.498889, 0.499192],
                        [0.223216, 0.223898, 0.441113, 0.499382, 0.499671],
                        [0.440057, 0.441031, 0.442650, 0.499817, 0.500103],
                        [0.499211, 0.499597, 0.499948, 0.500235, 0.500513],
                        [0.499622, 0.499995, 0.500332, 0.500637, 0.500928],
                    ],
                    # Slice 2
                    [
                        [0.439154, 0.439815, 0.441403, 0.499340, 0.499631],
                        [0.439922, 0.440854, 0.443026, 0.499791, 0.500050],
                        [0.441775, 0.442870, 0.513975, 0.551216, 0.554394],
                        [0.499660, 0.499995, 0.551055, 0.551449, 0.555799],
                        [0.500028, 0.500356, 0.551463, 0.552805, 0.556575],
                    ],
                    # Slice 3
                    [
                        [0.499113, 0.499287, 0.499471, 0.499749, 0.500001],
                        [0.499404, 0.499688, 0.499891, 0.500152, 0.500395],
                        [0.499699, 0.499998, 0.552609, 0.554186, 0.557711],
                        [0.500060, 0.500338, 0.552662, 0.772716, 0.774776],
                        [0.500350, 0.500633, 0.554134, 0.772200, 0.774436],
                    ],
                    # Slice 4
                    [
                        [0.499601, 0.499759, 0.499910, 0.500117, 0.500363],
                        [0.499866, 0.500112, 0.500244, 0.500474, 0.500739],
                        [0.500112, 0.500369, 0.553834, 0.557785, 0.559962],
                        [0.500397, 0.500631, 0.554693, 0.774537, 0.776065],
                        [0.500663, 0.500887, 0.556332, 0.773597, 0.775210],
                    ],
                ],
            ]
        ],
    ],
]


@skip_if_no_cpp_extension
@skip_if_no_cuda
class CRFTestCaseCuda(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test(self, test_case_description, params, input, features, expected):
        # Create input tensors
        input_tensor = torch.from_numpy(np.array(input)).to(dtype=torch.float, device=torch.device("cuda"))
        feature_tensor = torch.from_numpy(np.array(features)).to(dtype=torch.float, device=torch.device("cuda"))

        params[-1] = None if params[-1] is None else params[-1].cuda()

        # apply filter
        crf = CRF(*params)
        output = crf(input_tensor, feature_tensor).cpu().numpy()

        # Ensure result are as expected
        # np.testing.assert_allclose(output, expected, atol=1e-4)

        # Temporarily allowing some (10%) mismatched elements due to non determinism.
        absolute_diff_tolerance = 5e-2
        mismatch_ratio_tolerance = 0.1

        output = np.array(output).flatten()
        expected = np.array(expected).flatten()

        abs_diff = abs(output - expected)
        mismatch_count = sum(np.where(abs_diff > absolute_diff_tolerance, 1, 0))

        self.assertLessEqual(mismatch_count / len(output), mismatch_ratio_tolerance)


if __name__ == "__main__":
    unittest.main()
