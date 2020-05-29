#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace cl = llvm::cl;

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "MLIR experiments\n");

  return 0;
}
