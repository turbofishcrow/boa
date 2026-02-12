## Plan: Build Boa Programming Language Interpreter

Implement a Boa interpreter starting with a calculator, then add variables/assignments, and finally control flow and functions with head-final (postfix) syntax as defined in BASICS.md.

### Completed Steps ✓
1. Basic arithmetic calculator (numbers, +, -, *, /, ^, parentheses)
2. Parser using `pest` library with proper operator precedence (PEMDAS)
3. Evaluator with recursive evaluation
4. REPL with interactive input/output
5. Variables and assignments with environment/symbol table
6. Identifier support and statement parsing (assignments vs expressions)
7. Booleans: `true`/`false` literals, `&&`, `||`, `!` operators with type checking
8. Comparison operators: `==`, `!=`, `<`, `>`, `<=`, `>=`
9. Control flow - if-else: `condition if { ... } else { ... }` with head-final syntax, chained else-if, block expressions, keyword protection, `Value::Unit`
10. Loops: `loop { ... }`, `condition while { ... }` with head-final syntax, `break`, `continue` control flow
11. Type annotations and declarations: `const`/`mut` keywords, optional `: type` annotations (`int32`, `uint32`, `bool`), mutability enforcement, `Value::UInt(u32)`, type checking and int→uint coercion
12. Additional operators: `%` (Python-style modulo), compound assignment `+=`, `-=`, `*=`, `/=`, `%=` (desugared in parser)
13. Functions: `(params)name: ReturnType fn { body }` declarations, `(args)name` calls, `expr return` early return, recursive calls, lexical scoping, block scoping, immutable parameters
14. Strings: `str` type, string literals with escape sequences, `+` concatenation, `+=` compound assignment, lexicographic comparisons, built-in `print`/`lnprint` functions
15. Mutable function arguments: The syntax is `(mut arg: Type)fn_name: ReturnType fn`; the variable is passed by reference. Only mutable variables can be passed as mutable arguments. Used to mutate non-primitive objects like strings, collections and structs.
16. Function types and lambdas: Lambda syntax `\ x(: T), y(: T) => expr`, function type annotations `(int32): int32`, copy-capture closures (`env.clone()` at creation), mutable params on lambdas with writeback, higher-order functions, nested lambdas/currying, standalone block expressions in `Primary`
17. Generic types and built-in List: `<T>GenericType` annotations, generic functions `<T>(x: T)identity: T fn { x }` with type variable inference, PascalCase type identifiers, built-in `List` type with `[1, 2, 3]` literal syntax, homogeneous element validation, `len` builtin for strings and lists
18. Structs and enums: Struct declarations `Point struct { x: int32, y: int32 }`, struct literals `(x: 1, y: 2)Point`, field access `p.x`, enum declarations `Color enum { Red, (int32)Custom }`, variant construction `Red`/`(42)Custom`, generic structs/enums `<T>Wrapper struct { value: T }` / `<T>Maybe enum { (T)Some, None }`, zero-field structs, equality/display, type validation
19. Type-level methods: `TypeName methods { (self: TypeName, args...)method: ReturnType fn { body } }` declarations, `var.(args)method` call syntax, explicit `self` parameter, generic method inference from self type, method chaining with field access, validation (self name/type/mutability, arg count), methods on structs and enums
20. Optional types: Built-in `<T>Maybe enum { (T)Exists, DoesNotExist }` with `T?` syntax sugar, built-in `<T, E>Attempt enum { (T)Success, (E)Failure }`, unresolved generic enum variants with annotation-based coercion
21. `compconst` declarations: `compconst` keyword treated as immutable `const` for now (will matter for compilation later), keyword protection
22. Floating-point numbers: `fl64` type (f64), float literals (`3.14`), Rust-style `/` (int/int → int, fl64/fl64 → fl64), float arithmetic, comparisons, negation, modulo, power
23. Numeric casts: `(TargetType cast expr)` syntax with `cast` keyword, supports int32/uint32/fl64 conversions, range checking for negative→uint32, truncation for fl64→int
24. Destructuring declarations: `const (field: var)StructName = expr` for structs with shorthand `const (field)StructName`, `const (var)VariantName = expr` for enum variant payload extraction, `mut` support for mutable bindings
25. Match expressions: `scrutinee match { pattern -> { body }, ... }` with head-final syntax, literal patterns (int/float/bool/string), enum variant patterns `(v)Exists`/`DoesNotExist`, wildcard `_`, expression semantics (returns value), first-match-wins, runtime error on non-exhaustive match
26. Pattern binding with `=:` operator: `scrutinee =: pattern if { ... } else { ... }` (if-let), `scrutinee =: pattern while { ... }` (while-let), standalone `scrutinee =: pattern` (returns bool), else-if chains with mixed pattern/condition, `try_match_pattern` helper shared with `eval_match`
27. Collections and iterators: Range literals `1..5` (exclusive) and `1..=5` (inclusive) with int32 bounds, indexing `list[i]`/`str[i]`/`range[i]` with Python-style negative indices, built-in collection methods `.()map`, `.()filter`, `.()reverse`, `.()enumerate`, `.()take`, `.()take_while` on lists/strings/ranges (all return List), `len` extended for ranges
28. For-loops: `collection elem var for { body }` syntax with head-final `elem`/`for` keywords, works on lists/strings/ranges via `collection_to_list`, immutable loop variable with fresh scope per iteration, `break`/`continue` support, returns `Unit`
29. Collection slicing: `list[1..4]` (exclusive) and `list[1..=3]` (inclusive) return a sub-list, works on lists/strings/ranges via `collection_to_list`, out-of-bounds clamped to collection length
30. Format strings and traits: `f"hello {name}"` format strings with `{expr}` (Display) and `{expr:?}` (Debug) interpolation, escape sequences `\{`/`\}`, nested brace support; Trait system with `TypeName TraitName impl { ... }` syntax, four built-in traits: Display (custom `display: str` for print/f-strings), Debug (custom `debug: str` for `{:?}`, auto-derive fallback with quoted strings), PartialEq (custom `eq: bool` for `==`/`!=` with structural equality fallback), Ord (custom `cmp: Ordering` for `<`/`>`/`<=`/`>=`); built-in `Ordering` enum (Less, Equal, Greater); validation of trait method signatures
31. `[T]` syntax sugar for `<T>List` type annotations (mirrors `T?` for `<T>Maybe`), desugars to `TypeAnn::Generic { name: "List" }` in parser
### Next Steps
32. Move to compiler implementation

### Language Syntax Notes (from BASICS.md)
- **Head-final syntax**: Keywords and operators come *after* operands (postfix)
  - Example: `condition if { ... }` instead of `if condition { ... }`
  - Example: `collection elem variable for { ... }` instead of `for variable in collection { ... }`
- **Boolean operators**: `&&`, `||`, `!` (C/Rust-style)
- **Optional types**: `T?` (Exists/DoesNotExist), `<T>Attempt` (Success/Failure)
- **Pattern matching**: `=:` operator for binding matches in conditions

### Current Architecture
- **AST** (`src/ast.rs`): Node/Statement enums for expressions and assignments
- **Parser** (`src/parser.rs`): Pest-based PEG parser with grammar rules
- **Grammar** (`src/grammar.pest`): Multi-statement program support
- **Evaluator** (`src/evaluator.rs`): Recursive tree-walk interpreter
- **Environment** (`src/env.rs`): Symbol table for variable storage
- **REPL** (`src/main.rs`): Interactive input/output loop

### Testing
- 383 passing unit tests for parser, evaluator, booleans, comparisons, control flow, loops, type annotations, modulo, compound assignment, functions, strings, lambdas, generic types, lists, structs, enums, methods, optional types, destructuring, match expressions, pattern binding, ranges, indexing, collection methods, for-loops, slicing, format strings, and traits
- All tests pass with no warnings
