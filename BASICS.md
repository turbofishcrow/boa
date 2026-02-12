# Statements
Statements are required to end in a line break. To continue a line of code past the next line, use `\` (as in Python).

# Block expressions and functions
Block expressions and functions return the type of the last line or the value used by the `foo return` statement.

Function declarations:
```rs
(arg1: Type1, ..., argn: Typen)fn_name: ReturnType fn {
    // body
}
```

# Booleans
AND, OR, and NOT are `&&`, `||`, and `!` as in C and Rust.

`if` keywords come *after* the condition, which requires no parens:
```rs
const x = ()user_input
x % 15 == 0 if {
    ("fizzbuzz")lnprint
} else x % 3 == 0 if {
    ("fizz")lnprint
} else x % 5 == 0 if {
    ("buzz")lnprint
} else {
    (x)lnprint
}
```
`if` can be used as in Rust as a substitute for the conditional operator:
```rs
boolean if {
    1
} else {
    0
}
```
# Loops
## `loop`
`loop` works as in Rust:
```rs
loop {
    ("This line never stops printing")lnprint
}
```
## `while` loops
`while` loops do not require parentheses:
```rs
mut attempt: uint32 = ()user_input
attempt == 0 while {
    ("Positive numbers only")lnprint
    attempt = ()user_input
}

```
## `for` loops
`for` loops use the keywords `elem` and `for`:
```rs
collection elem variable for {
    // ...
}
```

Example:
```rs
(n: uint32)factorial: uint32 fn {
    mut result: uint32 = 1
    1..=n elem i for {
        result *= i
    }
    result
}
```

# `match` statement, `=:` operator
`match` statements:
```rs
const x: int32 = ()user_input
x % 2 match {
    0 -> {
        ("x is even")lnprint
    },
    1 -> {
        ("x is odd")lnprint
    },
}
```

`=:` attempts a pattern match and creates a binding if successful in `if` and `while` statements. `scrutinee =: pattern if/while` corresponds to `if/while let pattern = scrutinee` in Rust. `=:` also coerces to a boolean indicating success/failure if needed.

# Optional types and `Attempt`

`T?` corresponds to Rust `Option<T>` and `<T, E>Attempt` corresponds to Rust `Result<T, E>`.

The enum values of `T?` are `DoesNotExist` and `(t)Exists`.

The enum values of `<T, E>Attempt` are `(t)Success` and `(e)Failure`.