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
from tests.util import skip_if_no_cpp_extension

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
                [0.726896, 0.704883, 0.589467, 0.376669, 0.380321],
                # Class 1
                [0.273104, 0.295117, 0.410533, 0.623331, 0.619679],
            ],
            # Batch 1
            [
                # Class 0
                [0.741916, 0.720671, 0.551116, 0.328360, 0.376258],
                # Class 1
                [0.258084, 0.279329, 0.448885, 0.671640, 0.623742],
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
                [0.870921, 0.857105, 0.781170, 0.544729, 0.476710],
                # Class 1
                [0.129078, 0.142894, 0.218830, 0.455271, 0.523290],
            ],
            # Batch 1
            [
                # Class 0
                [0.867234, 0.852610, 0.648074, 0.334584, 0.386766],
                # Class 1
                [0.132766, 0.147390, 0.351926, 0.665416, 0.613234],
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
                    [0.0, 0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 1.0, 1.0, 0.0],
                    [1.0, 1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0, 0.0],
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
                    [0.159525, 0.161449, 0.270907, 0.152424, 0.152515],
                    [0.161763, 0.163849, 0.154026, 0.154187, 0.154360],
                    [0.273231, 0.154715, 0.155208, 0.155677, 0.275885],
                    [0.155076, 0.155748, 0.156349, 0.598796, 0.600179],
                    [0.156186, 0.156858, 0.277928, 0.598459, 0.600289],
                ],
                # Class 1
                [
                    [0.647632, 0.639540, 0.276122, 0.155184, 0.155117],
                    [0.638555, 0.629703, 0.155613, 0.155552, 0.155509],
                    [0.276475, 0.156138, 0.156061, 0.155919, 0.275726],
                    [0.156109, 0.156397, 0.156575, 0.172626, 0.172270],
                    [0.156380, 0.156690, 0.277053, 0.172495, 0.172123],
                ],
                # Class 2
                [
                    [0.192843, 0.199011, 0.452971, 0.692392, 0.692368],
                    [0.199682, 0.206448, 0.690361, 0.690261, 0.690130],
                    [0.450294, 0.689147, 0.688731, 0.688403, 0.448389],
                    [0.688815, 0.687855, 0.687076, 0.228579, 0.227552],
                    [0.687434, 0.686453, 0.445019, 0.229047, 0.227588],
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
                        [0.775729, 0.774871, 0.557369, 0.501589, 0.501239],
                        [0.774804, 0.774011, 0.556061, 0.501171, 0.500821],
                        [0.557136, 0.556079, 0.554716, 0.500764, 0.500415],
                        [0.501416, 0.501049, 0.500709, 0.500370, 0.500021],
                        [0.500989, 0.500631, 0.500300, 0.499986, 0.499665],
                    ],
                    # Slice 1
                    [
                        [0.774559, 0.773821, 0.555753, 0.501108, 0.500757],
                        [0.773701, 0.772905, 0.554399, 0.500680, 0.500342],
                        [0.555462, 0.554443, 0.553025, 0.500300, 0.499967],
                        [0.500892, 0.500562, 0.500256, 0.499931, 0.499666],
                        [0.500477, 0.500156, 0.499859, 0.499572, 0.499355],
                    ],
                    # Slice 2
                    [
                        [0.556395, 0.555530, 0.554037, 0.500641, 0.500290],
                        [0.555370, 0.554400, 0.552711, 0.500238, 0.499967],
                        [0.553709, 0.552798, 0.459696, 0.449011, 0.448406],
                        [0.500418, 0.500123, 0.448768, 0.448438, 0.447680],
                        [0.500064, 0.499770, 0.448217, 0.447788, 0.446945],
                    ],
                    # Slice 3
                    [
                        [0.500963, 0.500754, 0.500531, 0.500187, 0.499956],
                        [0.500662, 0.500394, 0.500144, 0.499822, 0.499657],
                        [0.500353, 0.500090, 0.448429, 0.448021, 0.447234],
                        [0.499966, 0.499724, 0.447893, 0.229453, 0.228867],
                        [0.499779, 0.499514, 0.447548, 0.229087, 0.228434],
                    ],
                    # Slice 4
                    [
                        [0.500406, 0.500208, 0.500018, 0.499775, 0.499615],
                        [0.500126, 0.499892, 0.499725, 0.499501, 0.499322],
                        [0.499869, 0.499645, 0.447670, 0.446978, 0.446165],
                        [0.499609, 0.499403, 0.447168, 0.228777, 0.228153],
                        [0.499467, 0.499255, 0.446656, 0.228424, 0.227778],
                    ],
                ],
                # Class 1
                [
                    # Slice 0
                    [
                        [0.224271, 0.225129, 0.442631, 0.498411, 0.498761],
                        [0.225196, 0.225989, 0.443939, 0.498829, 0.499179],
                        [0.442864, 0.443921, 0.445284, 0.499236, 0.499585],
                        [0.498584, 0.498951, 0.499291, 0.499630, 0.499979],
                        [0.499011, 0.499369, 0.499700, 0.500014, 0.500335],
                    ],
                    # Slice 1
                    [
                        [0.225441, 0.226179, 0.444247, 0.498892, 0.499243],
                        [0.226299, 0.227095, 0.445601, 0.499320, 0.499658],
                        [0.444538, 0.445557, 0.446975, 0.499700, 0.500033],
                        [0.499108, 0.499438, 0.499744, 0.500069, 0.500334],
                        [0.499523, 0.499844, 0.500141, 0.500428, 0.500645],
                    ],
                    # Slice 2
                    [
                        [0.443605, 0.444470, 0.445963, 0.499359, 0.499710],
                        [0.444630, 0.445600, 0.447289, 0.499762, 0.500033],
                        [0.446291, 0.447202, 0.540304, 0.550989, 0.551594],
                        [0.499582, 0.499877, 0.551232, 0.551562, 0.552320],
                        [0.499936, 0.500230, 0.551783, 0.552212, 0.553055],
                    ],
                    # Slice 3
                    [
                        [0.499037, 0.499246, 0.499469, 0.499813, 0.500044],
                        [0.499338, 0.499606, 0.499856, 0.500178, 0.500343],
                        [0.499647, 0.499910, 0.551571, 0.551979, 0.552766],
                        [0.500034, 0.500276, 0.552106, 0.770547, 0.771133],
                        [0.500221, 0.500486, 0.552452, 0.770913, 0.771566],
                    ],
                    # Slice 4
                    [
                        [0.499594, 0.499792, 0.499982, 0.500225, 0.500385],
                        [0.499874, 0.500108, 0.500275, 0.500499, 0.500678],
                        [0.500131, 0.500355, 0.552330, 0.553022, 0.553835],
                        [0.500391, 0.500597, 0.552832, 0.771223, 0.771847],
                        [0.500533, 0.500745, 0.553344, 0.771576, 0.772222],
                    ],
                ],
            ]
        ],
    ],
]


@skip_if_no_cpp_extension
class CRFTestCaseCpu(unittest.TestCase):
    @parameterized.expand(TEST_CASES)
    def test(self, test_case_description, params, input, features, expected):
        # Create input tensors
        input_tensor = torch.from_numpy(np.array(input)).to(dtype=torch.float, device=torch.device("cpu"))
        feature_tensor = torch.from_numpy(np.array(features)).to(dtype=torch.float, device=torch.device("cpu"))

        # apply filter
        crf = CRF(*params)
        output = crf(input_tensor, feature_tensor).cpu().numpy()

        # Ensure result are as expected
        np.testing.assert_allclose(output, expected, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
