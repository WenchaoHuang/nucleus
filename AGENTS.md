# Nucleus Development Guidelines

1. Use UTF-8 without BOM and follow `.gitattributes` for line endings: LF by
   default, with explicit exceptions such as `.bat` and `.cmd` files. Preserve
   the encoding and line endings of untouched content.

2. Reuse existing Nucleus types, `NS_*` macros, logging, and error-handling
   patterns. Preserve API and CUDA semantics unless the task requires a change.

3. Follow the established naming, include ordering, file organization, and
   documentation conventions in the same module. When adding tests, follow
   nearby test structure, naming, and fixture patterns. Keep changes narrowly
   scoped: avoid unrelated reformatting, renaming, or refactoring.

4. Use the local `.clang-format` as the source of truth for C++ and CUDA
   formatting, including indentation, braces, and spacing, even when nearby
   code differs. Run clang-format 19 or newer on modified code, limiting the
   formatting range when needed to keep the diff scoped. Review the resulting
   diff and report if the formatter could not be run.
