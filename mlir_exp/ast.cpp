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

#include "ast/ast.hpp"

AstContext::~AstContext() {
  for (auto Value : Managed) {
    delete Value;
  }
}

void AstContext::manage(AstContextManaged *Value) { Managed.insert(Value); }

AstContextManaged::AstContextManaged(AstContext &Context) : Context(Context) {
  Context.manage(this);
}

AstContext &AstNodeValue::getContext() const { return AstNode->getContext(); }

const AstKind ParameterOperation::Kind;

const AstKind AddOperation::Kind;

const AstKind MultiplyOperation::Kind;

AstNode::AstNode(AstContext &Context, const std::vector<AstNodeValue> &Operands)
    : AstContextManaged(Context), Operands(Operands) {}

void AstNode::setResultsSize(size_t size) {
  for (size_t i = 0; i < size; ++i) {
    Results.push_back(AstNodeValue{this, i});
  }
}

AstOperation::AstOperation(const AstOperation &Operation)
    : AstNode(Operation.AstNode) {}
