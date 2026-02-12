pub mod ast;
pub mod env;
pub mod evaluator;
pub mod parser;

use ast::Value;
use env::Environment;
use evaluator::eval_stmt;
use parser::parse;
use std::io::{self, Write};

fn brace_depth(s: &str) -> i32 {
    let mut depth = 0i32;
    for ch in s.chars() {
        match ch {
            '{' => depth += 1,
            '}' => depth -= 1,
            _ => {}
        }
    }
    depth
}

fn main() {
    println!("Boa Calculator v0.1.0");
    println!("Type expressions to evaluate, or 'exit' to quit\n");

    let mut environment = Environment::new();
    let mut input = String::new();
    let mut line = String::new();
    loop {
        if input.is_empty() {
            print!("> ");
        } else {
            print!("  ");
        }
        io::stdout().flush().unwrap();

        line.clear();
        match io::stdin().read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                continue;
            }
        }

        input.push_str(&line);

        // Keep reading if braces are unclosed
        if brace_depth(&input) > 0 {
            continue;
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            input.clear();
            continue;
        }
        if trimmed.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }

        match parse(trimmed) {
            Ok(statements) => {
                for stmt in statements {
                    match eval_stmt(&stmt, &mut environment) {
                        Ok(Some(Value::Unit)) => {}
                        Ok(Some(result)) => println!("{}", result),
                        Ok(None) => {}
                        Err(e) => eprintln!("Evaluation error: {e}"),
                    }
                }
            }
            Err(e) => eprintln!("Parse error: {e}"),
        }
        input.clear();
    }
}
