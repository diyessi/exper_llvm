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

#ifndef MLGA_AST_CONTEXT_HPP
#define MLGA_AST_CONTEXT_HPP

#include <unordered_set>

namespace mlga {
namespace core {
class AstContextManaged;

/// \brief Manages lifetime of managed objects
class AstContext {
public:
  virtual ~AstContext();
  void manage(AstContextManaged *Value);

protected:
  std::unordered_set<AstContextManaged *> Managed;
};

/// \brief An object manged by an AstContext
class AstContextManaged {
public:
  AstContextManaged(AstContext &Context);
  virtual ~AstContextManaged() {}
  AstContext &getContext() const { return Context; }

protected:
  AstContext &Context;
};
}
}

#endif