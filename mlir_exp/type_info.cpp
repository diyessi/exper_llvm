//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ast/type_info.hpp"

namespace {
size_t hash_combine(const std::vector<size_t> &list) {
  size_t seed = 0;
  for (size_t v : list) {
    seed ^= v + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}
}

namespace std {
size_t std::hash<mlga::type_info::DiscreteTypeInfo>::
operator()(const mlga::type_info::DiscreteTypeInfo &k) const {
  size_t NameHash = hash<string>()(string(k.Name));
  size_t VersionHash = hash<decltype(k.Version)>()(k.Version);
  return hash_combine(vector<size_t>{NameHash, VersionHash});
}
}
