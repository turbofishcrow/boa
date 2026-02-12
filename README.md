**Boa** (placeholder name) is a programming language inspired by head-final human languages (like Japanese and Tamil). The most important difference from programming languages common in our world is that function names go after their arguments, and keywords such as variable declarations tend to come last too:

```rs
(n: int32, d: int32)my_checked_div: int32? fn {
    d == 0 if {
        DoesNotExist
    } else {
        (n / d)Exists
    }
}

// `const` declares immutable variables and `mut` mutable ones
const result = (10, 3)my_checked_div
result =: (k)Exists if { // `=:` attempts pattern match and creates a binding if it succeeds
    k % 2 == 0 if {
        ("Value exists and is even")lnprint
    } else {
        ("Value exists and is odd")lnprint
    }
} else {
    ("Value does not exist")lnprint
}
```
corresponds to the Rust code
```rs
fn my_checked_div(n: i32, d: i32) -> Option<i32> {
    if d == 0 {
        None
    } else {
        Some(n / d)
    }
}

let result = my_checked_div(10, 3);
if let Some(k) = result {
    if k % 2 == 0 {
        println!("Value exists and is even")
    } else {
        println!("Value exists and is odd")
    }
} else {
    println!("Value does not exist")
}
```

Boa is statically typed and prefers functional-ish idioms like Rust. It uses a garbage collector for memory management. This implementation will eventually compile code into an LLVM backend; earlier versions will interpret code into Rust.

Boa is a worldbuilding exercise for a world where a predominantly head-final language occupies a role much like English does in our timeline (cf. Dolittle in our world which has Japanese keywords and uses Japanese-inspired syntax). Keywords are in English, which you should consider a translation convention. Common programming constructs such as `for`, `while`, `if`, and `else` just use the common English word (only their syntaxes change), but higher-level concepts such as option types (denoted `T?` for type `T`) are translated more literally: e.g. optional values are `(t)Exists` (equivalent to Rust `Some(t)`) and `DoesNotExist` (equivalent to Rust `None`).

## Current Status

The interpreter currently supports:

- **Arithmetic**: `+`, `-`, `*`, `/`, `%` (Python-style modulo), `^` (exponentiation), compound assignment (`+=`, `-=`, `*=`, `/=`, `%=`)
- **Types**: `int32`, `uint32`, `fl64`, `bool`, `str`, with type annotations, int-to-uint coercion, and `(TargetType cast expr)` numeric casts
- **Variables**: `const` (immutable), `compconst` (compile-time constant), and `mut` (mutable) declarations with optional type annotations
- **Booleans**: `true`/`false` literals, `&&`, `||`, `!`, comparison operators (`==`, `!=`, `<`, `>`, `<=`, `>=`)
- **Control flow**: `condition if { ... } else { ... }`, chained else-if, `loop { ... }`, `condition while { ... }`, `break`, `continue`
- **Functions**: declarations with `(params)name: ReturnType fn { body }`, calls with `(args)name`, early return with `expr return`, recursive calls, lexical scoping
- **Lambdas**: `\ x, y => expr` syntax, function type annotations `(int32): int32`, copy-capture closures, mutable params with writeback, higher-order functions
- **Strings**: string literals with escape sequences, `+` concatenation, format strings `f"hello {name}"` with `{expr}` and `{expr:?}` interpolation
- **Generics**: `<T>` type parameters on functions, structs, and enums with type variable inference
- **Structs**: `Point struct { x: int32, y: int32 }`, literals `(x: 1, y: 2)Point`, field access `p.x`
- **Enums**: `Color enum { Red, (int32)Custom }`, variant construction, built-in `Maybe` (`T?`), `Attempt`, and `Ordering`
- **Methods**: `TypeName methods { ... }` declarations, `var.(args)method` call syntax, method chaining
- **Traits**: `TypeName TraitName impl { ... }` syntax with built-in Display, Debug, PartialEq, and Ord traits
- **Pattern matching**: `scrutinee match { pattern -> { body }, ... }`, `=:` operator for if-let/while-let
- **Destructuring**: `const (field: var)StructName = expr`, `const (var)VariantName = expr`
- **Collections**: `[1, 2, 3]` list literals, `1..5` / `1..=5` ranges, indexing with negative indices, slicing, `.()map`, `.()filter`, `.()reverse`, `.()enumerate`, `.()take`, `.()take_while`
- **For-loops**: `collection elem var for { body }`
- **Block scoping**: variables declared in blocks are local to that block; functions see only globals and their own parameters

### Example: Factorial

```
(n: uint32)fact: uint32 fn {
    n <= 1 if {
        1
    } else {
        n * (n - 1)fact
    }
}
(5)fact
```

### Example: Structs with Traits

```
Point struct {
    x: fl64,
    y: fl64
}

Point Display impl {
    (self: Point)display: str fn {
        f"({self.x}, {self.y})"
    }
}

Point PartialEq impl {
    (self: Point, other: Point)eq: bool fn {
        self.x == other.x && self.y == other.y
    }
}

const p = (x: 1.0, y: 2.0)Point
(f"Point is: {p}")lnprint
```

## Building and Running

```sh
cargo run    # start the REPL
cargo test   # run the test suite (380 tests)
```
