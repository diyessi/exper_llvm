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
