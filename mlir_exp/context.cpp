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

#include "ast/context.hpp"

mlga::core::AstContext::~AstContext() {
  for (auto Value : Managed) {
    delete Value;
  }
}

void mlga::core::AstContext::manage(AstContextManaged *Value) {
  Managed.insert(Value);
}

mlga::core::AstContextManaged::AstContextManaged(AstContext &Context)
    : Context(Context) {
  Context.manage(this);
}
