# coding=utf-8
# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Namespace package that exposes the legacy module layout.

The original DeepMind release packages the code under
`emergent_in_context_learning.*`. Our repository keeps the original
sub-packages (`datasets`, `experiment`, `modules`) at the repo root, so we
bridge them here to preserve the expected import paths.
"""

import importlib as _importlib
import sys as _sys

_SUBMODULES = ('datasets', 'experiment', 'modules')

for _name in _SUBMODULES:
  _module = _importlib.import_module(_name)
  _sys.modules[f'{__name__}.{_name}'] = _module

del _importlib
del _sys
del _SUBMODULES
