# Nucleus Development Guidelines

1. Follow `.gitattributes` for line endings. Repository text files use LF by
   default; preserve explicit exceptions such as `.bat` and `.cmd` files. Use
   UTF-8 without BOM and tabs for indentation. Do not alter the encoding,
   indentation, or line endings of untouched lines.

2. Keep API and CUDA behavior consistent with adjacent code. Reuse existing
   Nucleus types, `NS_*` macros, logging, and error-handling patterns instead of
   creating parallel helpers or changing error semantics without a clear need.

3. Match the established style of nearby code in the same module before editing.
   Preserve its naming, include ordering, file layout, comments, control flow,
   brace and spacing conventions, and local helper usage. Keep changes narrowly
   scoped: avoid unrelated reformatting, renaming, or refactoring. When adding
   tests, follow the structure, naming, and fixture patterns of nearby tests.
