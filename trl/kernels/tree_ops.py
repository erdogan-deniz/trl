# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .tree_gated_delta_rule import tree_causal_conv1d, tree_chunk_gated_delta_rule
from .tree_mamba2 import tree_mamba2_chunk_scan, tree_mamba2_split_conv1d_scan_combined


TREE_OPS = {
    "gated_deltanet": {
        "causal_conv1d_fn": tree_causal_conv1d,
        "torch_chunk_gated_delta_rule": tree_chunk_gated_delta_rule,
    },
    "mamba2": {
        "causal_conv1d_fn": tree_causal_conv1d,
        "mamba2_chunk_scan": tree_mamba2_chunk_scan,
        "mamba2_split_conv1d_scan_combined": tree_mamba2_split_conv1d_scan_combined,
    },
}
