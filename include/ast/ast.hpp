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

#ifndef MLGA_AST_AST_HPP
#define MLGA_AST_AST_HPP

#include "ast/context.hpp"
#include "ast/exports.hpp"
#include "ast/type_info.hpp"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mlga {
namespace ast {

class AstNode;

class AstOp {
public:
  using type_info_t = mlga::core::DiscreteTypeInfo;
  AstOp(const AstOp &Op);
  AstOp(AstNode *AstNode) : AstNode(AstNode) {}
  AstNode *getNode() const { return AstNode; }

protected:
  AstNode *AstNode{nullptr};
};

/// \brief A handle to a particular result of an AstNode.
class AstResult {
public:
  AstResult(AstNode *Node, size_t Index) : AstNode(Node), Index(Index) {}
  AstNode *getNode() const { return AstNode; }
  size_t getIndex() const { return Index; }

protected:
  AstNode *AstNode;
  size_t Index;
};

class AstNode : public mlga::core::AstContextManaged {
public:
  virtual ~AstNode() {}
  AstNode(mlga::core::AstContext &Context,
          const std::vector<AstResult> &Operands);
  virtual const AstOp::type_info_t &getTypeInfo() = 0;
  void setResultsSize(size_t size);
  const std::vector<AstResult> &getOperands() const { return Operands; };
  const std::vector<AstResult> &getResults() const { return Results; }
  operator AstOp() { return AstOp(this); }

protected:
  std::vector<AstResult> Operands;
  std::vector<AstResult> Results;
};

struct DefaultAttributes {};

template <typename AstOpType, typename AttributesType = DefaultAttributes>
class AstOpImp {
public:
  using type_info_t = AstOp::type_info_t;
  class AstNodeImp : public AstNode {

  public:
    using AstNode::AstNode;
    const AstOp::type_info_t &getTypeInfo() { return AstOpType::TypeInfo; }
    static const AstOp::type_info_t &getStaticTypeInfo() {
      return AstOpType::TypeInfo;
    }
    AttributesType Attributes;
  };

  AstOpImp() {}

  AstOpImp(const AstOp &Op) : AstOpImp(Op.getNode()) {}

  AstOpImp(AstNode *Node)
      : AstNode(mlga::core::asType<AstNodeImp>(Node)),
        Attributes(AstNode ? &AstNode->Attributes : nullptr) {}

  AstOpImp(const AstOpType &Op)
      : AstNode(Op->AstNode), Attributes(*AstNode->Attributes) {}

  AstOpImp(mlga::core::AstContext &Context,
           const std::vector<AstResult> &Operands)
      : AstNode(create(Context, Operands)), Attributes(&AstNode->Attributes) {}

  operator AstOp() const { return AstOp(AstNode); }

  static AstNodeImp *create(mlga::core::AstContext &Context,
                            const std::vector<AstResult> &Operands) {
    AstNodeImp *Value = new AstNodeImp(Context, Operands);
    AstOpType::initialize(Value);
    return Value;
  }
  const std::vector<AstResult> &getResults() { return AstNode->getResults(); }
  static void initialize(AstNodeImp *node) { node->setResultsSize(1); }

protected:
  AstNodeImp *AstNode{nullptr};
  AttributesType *Attributes{nullptr};
};

struct ParameterAttributes {
  std::string Name;
};

class ParameterOp : public AstOpImp<ParameterOp, ParameterAttributes> {
public:
  using AstOpImp<ParameterOp, ParameterAttributes>::AstOpImp;
  ParameterOp(mlga::core::AstContext &Context, const std::string &Name)
      : ParameterOp(Context, std::vector<AstResult>{}) {
    Attributes->Name = Name;
  }
  static constexpr AstOp::type_info_t TypeInfo{"Parameter", 0};
};

class AddOp : public AstOpImp<AddOp> {
public:
  static constexpr type_info_t TypeInfo{"Add", 0};
};

class MultiplyOp : public AstOpImp<MultiplyOp> {
public:
  static constexpr type_info_t TypeInfo{"Multiply"};
};

inline AstResult operator+(const AstResult &x, const AstResult &y) {
  return AddOp::create(x.getNode()->getContext(), {x, y})->getResults().at(0);
}

inline AstResult operator*(const AstResult &x, const AstResult &y) {
  return MultiplyOp::create(x.getNode()->getContext(), {x, y})
      ->getResults()
      .at(0);
}
}
}
#endif
