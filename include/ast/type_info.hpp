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

#ifndef MLGA_AST_TYPE_INFO_HPP
#define MLGA_AST_TYPE_INFO_HPP

#include "ast/exports.hpp"

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace mlga {
namespace core {
/// Supports three functions, isType<Type>, asType<Type>, and
/// asType_ptr<Type> for type-safe
/// dynamic conversions via static_cast/static_ptr_cast without using C++ RTTI.
/// Type must have a static constexpr type_info member and a virtual
/// getTypeInfo() member that
/// returns a reference to its type_info member.

/// Type information for a type system without inheritance; instances have
/// exactly one type not
/// related to any other type.
struct MLGA_CORE_EXPORT DiscreteTypeInfo {
  const char *Name;
  uint64_t Version;

  bool IsCastable(const DiscreteTypeInfo &target_type) const {
    return *this == target_type;
  }
  // For use as a key
  bool operator<(const DiscreteTypeInfo &b) const {
    return Version < b.Version ||
           (Version == b.Version && strcmp(Name, b.Name) < 0);
  }
  bool operator<=(const DiscreteTypeInfo &b) const {
    return Version < b.Version ||
           (Version == b.Version && strcmp(Name, b.Name) <= 0);
  }
  bool operator>(const DiscreteTypeInfo &b) const {
    return Version < b.Version ||
           (Version == b.Version && strcmp(Name, b.Name) > 0);
  }
  bool operator>=(const DiscreteTypeInfo &b) const {
    return Version < b.Version ||
           (Version == b.Version && strcmp(Name, b.Name) >= 0);
  }
  bool operator==(const DiscreteTypeInfo &b) const {
    return Version == b.Version && strcmp(Name, b.Name) == 0;
  }
  bool operator!=(const DiscreteTypeInfo &b) const {
    return Version != b.Version || strcmp(Name, b.Name) != 0;
  }
};

/// \brief Tests if value is a pointer/shared_ptr that can be statically cast to
/// a
/// Type*/shared_ptr<Type>
template <typename Type, typename Value>
typename std::enable_if<
    std::is_convertible<decltype(std::declval<Value>()
                                     ->getTypeInfo()
                                     .IsCastable(Type::getStaticTypeInfo())),
                        bool>::value,
    bool>::type
isType(Value value) {
  return value->getTypeInfo().IsCastable(Type::getStaticTypeInfo());
}

/// Casts a Value* to a Type* if it is of type Type, nullptr otherwise
template <typename Type, typename Value>
typename std::enable_if<
    std::is_convertible<decltype(static_cast<Type *>(std::declval<Value>())),
                        Type *>::value,
    Type *>::type
asType(Value value) {
  return isType<Type>(value) ? static_cast<Type *>(value) : nullptr;
}

/// Casts a std::shared_ptr<Value> to a std::shared_ptr<Type> if it is of type
/// Type, nullptr otherwise
template <typename Type, typename Value>
typename std::enable_if<
    std::is_convertible<
        decltype(std::static_pointer_cast<Type>(std::declval<Value>())),
        std::shared_ptr<Type>>::value,
    std::shared_ptr<Type>>::type
asType_ptr(Value value) {
  return isType<Type>(value) ? std::static_pointer_cast<Type>(value)
                             : std::shared_ptr<Type>();
}
}
}

namespace std {
template <> struct MLGA_CORE_EXPORT hash<mlga::core::DiscreteTypeInfo> {
  size_t operator()(const mlga::core::DiscreteTypeInfo &k) const;
};
}
#endif
