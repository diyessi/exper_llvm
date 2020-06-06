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

#ifndef MY_MY_AST_HPP
#define MY_MY_AST_HPP

#include "ast/ast.hpp"

#include <string>

namespace mlga {
namespace my {

using mlga::ast::asOps;
using mlga::ast::Op;
using mlga::ast::OpImp;
using mlga::ast::Ops;
using mlga::ast::TypeInfo;

class Variable : public OpImp<Variable, std::string> {
public:
  Variable(const std::string &Name) : OpImp<Variable, std::string>(Ops{}) {
    getName() = Name;
  }
  std::string &getName() { return getAttributes(); }
  static constexpr TypeInfo TypeInfo{"Variable", 0};
};

class Parameters : public OpImp<Parameters> {
public:
  Parameters(const std::vector<Variable> &Parameters)
      : OpImp<class Parameters>(asOps(Parameters)) {}
  static constexpr TypeInfo TypeInfo{"Parameters", 0};
};

class Body : public OpImp<Body> {
public:
  Body(const Ops &Body) : OpImp<class Body>(Body) {}
  static constexpr TypeInfo TypeInfo{"Body", 0};
};

class Function : public OpImp<Function, std::string> {
public:
  Function(const std::string &Name, const Parameters &Parameters,
           const Body &Body)
      : OpImp<Function, std::string>(Ops{Parameters, Body}) {
    getName() = Name;
  }
  std::string &getName() { return getAttributes(); }
  static constexpr TypeInfo TypeInfo{"Function", 0};
};

class Add : public OpImp<Add> {
public:
  Add(const Op &X, const Op &Y) : OpImp<Add>({X, Y}) {}

  static constexpr TypeInfo TypeInfo{"Add", 0};
};

inline Op operator+(const Op &X, const Op &Y) { return Add(X, Y); }

class Multiply : public OpImp<Multiply> {
public:
  Multiply(const Op &X, const Op &Y) : OpImp<Multiply>({X, Y}) {}
  static constexpr TypeInfo TypeInfo{"Multiply"};
};

inline Op operator*(const Op &X, const Op &Y) { return Multiply(X, Y); }

class Return : public OpImp<Return> {
public:
  Return(const Op &Arg) : OpImp(Ops{Arg}) {}
  static constexpr TypeInfo TypeInfo{"Return"};
};
}
}

#endif
