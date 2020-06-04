#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

class AstNode;
class AstContextManaged;

/// \brief metaclass information
struct AstKind {
  const char *Name;
  uint8_t Version;
  bool operator==(const AstKind &Kind) const {
    return this == &Kind || (Version == Kind.Version &&
                             std::string(Name) == std::string(Kind.Name));
  }
};

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

/// \brief A handle to a particular result of an AstNode.
class AstNodeValue {
public:
  AstNodeValue(AstNode *Node, size_t Index) : AstNode(Node), Index(Index) {}

  AstContext &getContext() const;

protected:
  AstNode *AstNode;
  size_t Index;
};

class AstNode : public AstContextManaged {
public:
  virtual ~AstNode() {}
  AstNode(AstContext &Context, const std::vector<AstNodeValue> &Operands);
  virtual AstKind getKind() = 0;
  void *asKind(const AstKind &Kind) {
    return getKind() == Kind ? this : nullptr;
  }
  void setResultsSize(size_t size);
  const std::vector<AstNodeValue> &getOperands() const { return Operands; };
  const std::vector<AstNodeValue> &getResults() const { return Results; }

protected:
  std::vector<AstNodeValue> Operands;
  std::vector<AstNodeValue> Results;
};

struct DefaultAttributes {};

class AstOperation {
public:
  AstOperation(const AstOperation &Operation);
  AstOperation(AstNode *AstNode) : AstNode(AstNode) {}
  AstNode *getNode() const { return AstNode; }

protected:
  AstNode *AstNode{nullptr};
};

template <typename AstOperationType,
          typename AttributesType = DefaultAttributes>
class AstOperationImp {
public:
  class AstNodeImp : public AstNode {

  public:
    using AstNode::AstNode;
    AstKind getKind() { return AstOperationType::Kind; }
    AttributesType Attributes;
  };

  AstOperationImp() {}

  AstOperationImp(const AstOperation &Operation)
      : AstOperationImp(Operation.getNode()) {}

  AstOperationImp(AstNode *Node)
      : AstNode(
            static_cast<AstNodeImp *>(Node->asKind(AstOperationType::Kind))),
        Attributes(AstNode ? &AstNode->Attributes : nullptr) {}

  AstOperationImp(const AstOperationType &Operation)
      : AstNode(Operation->AstNode), Attributes(*AstNode->Attributes) {}

  AstOperationImp(AstContext &Context,
                  const std::vector<AstNodeValue> &Operands)
      : AstNode(create(Context, Operands)), Attributes(&AstNode->Attributes) {}

  operator AstOperation() const { return AstOperation(AstNode); }

  static AstNodeImp *create(AstContext &Context,
                            const std::vector<AstNodeValue> &Operands) {
    AstNodeImp *Value = new AstNodeImp(Context, Operands);
    AstOperationType::initialize(Value);
    return Value;
  }
  const std::vector<AstNodeValue> &getResults() {
    return AstNode->getResults();
  }
  static void initialize(AstNodeImp *node) { node->setResultsSize(1); }

protected:
  AstNodeImp *AstNode{nullptr};
  AttributesType *Attributes{nullptr};
};

struct ParameterAttributes {
  std::string Name;
};

class ParameterOperation
    : public AstOperationImp<ParameterOperation, ParameterAttributes> {
public:
  using AstOperationImp<ParameterOperation,
                        ParameterAttributes>::AstOperationImp;
  ParameterOperation(AstContext &Context, const std::string &Name)
      : ParameterOperation(Context, std::vector<AstNodeValue>{}) {
    Attributes->Name = Name;
  }
  static constexpr AstKind Kind{"Parameter", 0};
};

class AddOperation : public AstOperationImp<AddOperation> {
public:
  static constexpr AstKind Kind{"Add", 0};
};

class MultiplyOperation : public AstOperationImp<MultiplyOperation> {
public:
  static constexpr AstKind Kind{"Multiply"};
};

inline AstNodeValue operator+(const AstNodeValue &x, const AstNodeValue &y) {
  return AddOperation::create(x.getContext(), {x, y})->getResults().at(0);
}

inline AstNodeValue operator*(const AstNodeValue &x, const AstNodeValue &y) {
  return MultiplyOperation::create(x.getContext(), {x, y})->getResults().at(0);
}
