use crate::ast::{BinaryOp, LambdaParam, MatchArm, Node, Pattern, PrefixOp, Statement, TypeAnn, Value};
use crate::env::Environment;
use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq)]
pub enum EvalError {
    DivisionByZero,
    UndefinedVariable(String),
    TypeError(String),
    Break,
    Continue,
    Return(Box<Value>),
}

impl std::fmt::Display for EvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            EvalError::DivisionByZero => write!(f, "Division by zero"),
            EvalError::UndefinedVariable(name) => write!(f, "Undefined variable: {}", name),
            EvalError::TypeError(msg) => write!(f, "Type error: {}", msg),
            EvalError::Break => write!(f, "break outside of loop"),
            EvalError::Continue => write!(f, "continue outside of loop"),
            EvalError::Return(_) => write!(f, "return outside of function"),
        }
    }
}

pub fn eval(node: &Node, env: &mut Environment) -> Result<Value, EvalError> {
    match node {
        Node::Int(n) => Ok(Value::Int(*n)),
        Node::Float(n) => Ok(Value::Float(*n)),
        Node::Bool(b) => Ok(Value::Bool(*b)),
        Node::Str(s) => Ok(Value::Str(s.clone())),
        Node::FormatString { parts } => {
            let mut result = String::new();
            for part in parts {
                match part {
                    crate::ast::FormatPart::Text(s) => result.push_str(s),
                    crate::ast::FormatPart::Display(expr) => {
                        let val = eval(expr, env)?;
                        result.push_str(&value_to_display_string(&val, env)?);
                    }
                    crate::ast::FormatPart::Debug(expr) => {
                        let val = eval(expr, env)?;
                        result.push_str(&value_to_debug_string(&val, env)?);
                    }
                }
            }
            Ok(Value::Str(result))
        }
        Node::Identifier(name) => env
            .get(name)
            .ok_or(EvalError::UndefinedVariable(name.clone())),
        Node::PrefixOp { op, child } => {
            let val = eval(child, env)?;
            match op {
                PrefixOp::Negative => match val {
                    Value::Int(n) => Ok(Value::Int(-n)),
                    Value::Float(n) => Ok(Value::Float(-n)),
                    _ => Err(EvalError::TypeError(
                        "cannot negate a non-numeric value".to_string(),
                    )),
                },
                PrefixOp::Not => match val {
                    Value::Bool(b) => Ok(Value::Bool(!b)),
                    _ => Err(EvalError::TypeError(
                        "cannot apply ! to a non-boolean".to_string(),
                    )),
                },
            }
        }
        Node::BinaryOp { lhs, rhs, op } => {
            let left = eval(lhs, env)?;
            let right = eval(rhs, env)?;
            match op {
                BinaryOp::Add => match (left, right) {
                    (Value::Int(l), Value::Int(r)) => Ok(Value::Int(l.wrapping_add(r))),
                    (Value::UInt(l), Value::UInt(r)) => Ok(Value::UInt(l.wrapping_add(r))),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l + r)),
                    (Value::Str(l), Value::Str(r)) => Ok(Value::Str(l + &r)),
                    _ => Err(EvalError::TypeError(
                        "+ requires matching numeric types or strings".to_string(),
                    )),
                },
                BinaryOp::Sub => match (left, right) {
                    (Value::Int(l), Value::Int(r)) => Ok(Value::Int(l.wrapping_sub(r))),
                    (Value::UInt(l), Value::UInt(r)) => Ok(Value::UInt(l.wrapping_sub(r))),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l - r)),
                    _ => Err(EvalError::TypeError(
                        "- requires matching numeric types".to_string(),
                    )),
                },
                BinaryOp::Mul => match (left, right) {
                    (Value::Int(l), Value::Int(r)) => Ok(Value::Int(l.wrapping_mul(r))),
                    (Value::UInt(l), Value::UInt(r)) => Ok(Value::UInt(l.wrapping_mul(r))),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l * r)),
                    _ => Err(EvalError::TypeError(
                        "* requires matching numeric types".to_string(),
                    )),
                },
                BinaryOp::Div => match (left, right) {
                    (Value::Int(l), Value::Int(r)) => {
                        if r == 0 {
                            Err(EvalError::DivisionByZero)
                        } else {
                            Ok(Value::Int(l / r))
                        }
                    }
                    (Value::UInt(l), Value::UInt(r)) => {
                        if r == 0 {
                            Err(EvalError::DivisionByZero)
                        } else {
                            Ok(Value::UInt(l / r))
                        }
                    }
                    (Value::Float(l), Value::Float(r)) => {
                        if r == 0.0 {
                            Err(EvalError::DivisionByZero)
                        } else {
                            Ok(Value::Float(l / r))
                        }
                    }
                    _ => Err(EvalError::TypeError(
                        "/ requires matching numeric types".to_string(),
                    )),
                },
                BinaryOp::Mod => {
                    match (left, right) {
                        (Value::Int(l), Value::Int(r)) => {
                            if r == 0 {
                                Err(EvalError::DivisionByZero)
                            } else {
                                // Python-style modulo: result has sign of divisor
                                let rem = l.wrapping_rem(r);
                                if rem != 0 && (rem ^ r) < 0 {
                                    Ok(Value::Int(rem.wrapping_add(r)))
                                } else {
                                    Ok(Value::Int(rem))
                                }
                            }
                        }
                        (Value::UInt(l), Value::UInt(r)) => {
                            if r == 0 {
                                Err(EvalError::DivisionByZero)
                            } else {
                                Ok(Value::UInt(l % r))
                            }
                        }
                        (Value::Float(l), Value::Float(r)) => {
                            if r == 0.0 {
                                Err(EvalError::DivisionByZero)
                            } else {
                                // Python-style modulo: result has sign of divisor
                                let rem = l % r;
                                if rem != 0.0 && rem.signum() != r.signum() {
                                    Ok(Value::Float(rem + r))
                                } else {
                                    Ok(Value::Float(rem))
                                }
                            }
                        }
                        _ => Err(EvalError::TypeError(
                            "% requires matching numeric types".to_string(),
                        )),
                    }
                }
                BinaryOp::Pow => match (left, right) {
                    (Value::Int(l), Value::Int(r)) => {
                        if r < 0 {
                            Ok(Value::Int(1 / l))
                        } else {
                            Ok(Value::Int(l.pow(r as u32)))
                        }
                    }
                    (Value::UInt(l), Value::UInt(r)) => Ok(Value::UInt(l.pow(r))),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l.powf(r))),
                    _ => Err(EvalError::TypeError(
                        "^ requires matching numeric types".to_string(),
                    )),
                },
                BinaryOp::Eq => {
                    if let Some(result) = try_partial_eq(&left, &right, env)? {
                        Ok(Value::Bool(result))
                    } else {
                        Ok(Value::Bool(left == right))
                    }
                }
                BinaryOp::Neq => {
                    if let Some(result) = try_partial_eq(&left, &right, env)? {
                        Ok(Value::Bool(!result))
                    } else {
                        Ok(Value::Bool(left != right))
                    }
                }
                BinaryOp::Lt => match (&left, &right) {
                    (Value::Int(l), Value::Int(r)) => Ok(Value::Bool(l < r)),
                    (Value::UInt(l), Value::UInt(r)) => Ok(Value::Bool(l < r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Bool(l < r)),
                    (Value::Str(l), Value::Str(r)) => Ok(Value::Bool(l < r)),
                    _ => {
                        if let Some(ord) = try_ord_compare(&left, &right, env)? {
                            Ok(Value::Bool(ord == "Less"))
                        } else {
                            Err(EvalError::TypeError(
                                "< requires matching comparable types".to_string(),
                            ))
                        }
                    }
                },
                BinaryOp::Gt => match (&left, &right) {
                    (Value::Int(l), Value::Int(r)) => Ok(Value::Bool(l > r)),
                    (Value::UInt(l), Value::UInt(r)) => Ok(Value::Bool(l > r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Bool(l > r)),
                    (Value::Str(l), Value::Str(r)) => Ok(Value::Bool(l > r)),
                    _ => {
                        if let Some(ord) = try_ord_compare(&left, &right, env)? {
                            Ok(Value::Bool(ord == "Greater"))
                        } else {
                            Err(EvalError::TypeError(
                                "> requires matching comparable types".to_string(),
                            ))
                        }
                    }
                },
                BinaryOp::Lte => match (&left, &right) {
                    (Value::Int(l), Value::Int(r)) => Ok(Value::Bool(l <= r)),
                    (Value::UInt(l), Value::UInt(r)) => Ok(Value::Bool(l <= r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Bool(l <= r)),
                    (Value::Str(l), Value::Str(r)) => Ok(Value::Bool(l <= r)),
                    _ => {
                        if let Some(ord) = try_ord_compare(&left, &right, env)? {
                            Ok(Value::Bool(ord != "Greater"))
                        } else {
                            Err(EvalError::TypeError(
                                "<= requires matching comparable types".to_string(),
                            ))
                        }
                    }
                },
                BinaryOp::Gte => match (&left, &right) {
                    (Value::Int(l), Value::Int(r)) => Ok(Value::Bool(l >= r)),
                    (Value::UInt(l), Value::UInt(r)) => Ok(Value::Bool(l >= r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Bool(l >= r)),
                    (Value::Str(l), Value::Str(r)) => Ok(Value::Bool(l >= r)),
                    _ => {
                        if let Some(ord) = try_ord_compare(&left, &right, env)? {
                            Ok(Value::Bool(ord != "Less"))
                        } else {
                            Err(EvalError::TypeError(
                                ">= requires matching comparable types".to_string(),
                            ))
                        }
                    }
                },
                BinaryOp::And => match (left, right) {
                    (Value::Bool(l), Value::Bool(r)) => Ok(Value::Bool(l && r)),
                    _ => Err(EvalError::TypeError("&& requires booleans".to_string())),
                },
                BinaryOp::Or => match (left, right) {
                    (Value::Bool(l), Value::Bool(r)) => Ok(Value::Bool(l || r)),
                    _ => Err(EvalError::TypeError("|| requires booleans".to_string())),
                },
            }
        }
        Node::Block(stmts) => {
            env.push_scope();
            let mut result = Value::Unit;
            for stmt in stmts {
                match eval_stmt_inner(stmt, env) {
                    Ok(val) => result = val,
                    Err(e) => {
                        env.pop_scope();
                        return Err(e);
                    }
                }
            }
            env.pop_scope();
            Ok(result)
        }
        Node::IfElse {
            condition,
            then_block,
            else_block,
        } => {
            let cond = eval(condition, env)?;
            match cond {
                Value::Bool(true) => eval(then_block, env),
                Value::Bool(false) => {
                    if let Some(else_blk) = else_block {
                        eval(else_blk, env)
                    } else {
                        Ok(Value::Unit)
                    }
                }
                _ => Err(EvalError::TypeError(
                    "if condition must be a boolean".to_string(),
                )),
            }
        }
        Node::Loop(body) => loop {
            match eval(body, env) {
                Ok(_) => {}
                Err(EvalError::Break) => return Ok(Value::Unit),
                Err(EvalError::Continue) => continue,
                Err(e) => return Err(e),
            }
        },
        Node::While { condition, body } => loop {
            let cond = eval(condition, env)?;
            match cond {
                Value::Bool(true) => match eval(body, env) {
                    Ok(_) => {}
                    Err(EvalError::Break) => return Ok(Value::Unit),
                    Err(EvalError::Continue) => continue,
                    Err(e) => return Err(e),
                },
                Value::Bool(false) => return Ok(Value::Unit),
                _ => {
                    return Err(EvalError::TypeError(
                        "while condition must be a boolean".to_string(),
                    ));
                }
            }
        },
        Node::ForLoop {
            collection,
            var_name,
            body,
        } => {
            let coll_val = eval(collection, env)?;
            let items = collection_to_list(&coll_val)?;
            for item in items {
                env.push_scope();
                let item_type = infer_type(&item, var_name)?;
                env.declare(var_name.clone(), item, false, item_type)
                    .map_err(EvalError::TypeError)?;
                match eval(body, env) {
                    Ok(_) => {}
                    Err(EvalError::Break) => {
                        env.pop_scope();
                        return Ok(Value::Unit);
                    }
                    Err(EvalError::Continue) => {
                        env.pop_scope();
                        continue;
                    }
                    Err(e) => {
                        env.pop_scope();
                        return Err(e);
                    }
                }
                env.pop_scope();
            }
            Ok(Value::Unit)
        }
        Node::Break => Err(EvalError::Break),
        Node::Continue => Err(EvalError::Continue),
        Node::FnCall { name, args } => eval_fn_call(name, args, env),
        Node::Return(expr) => {
            let val = eval(expr, env)?;
            Err(EvalError::Return(Box::new(val)))
        }
        Node::Lambda { params, body } => {
            let captured_env = env.clone();
            Ok(Value::Closure {
                params: params.clone(),
                body: body.clone(),
                env: Box::new(captured_env),
            })
        }
        Node::List(elements) => {
            let mut values = vec![];
            for elem in elements {
                values.push(eval(elem, env)?);
            }
            // Validate homogeneous types
            if values.len() > 1 {
                let first_type = infer_type(&values[0], "list element")?;
                for (i, val) in values.iter().enumerate().skip(1) {
                    validate_type(val, &first_type, &format!("list element [{}]", i)).map_err(
                        |e| match e {
                            EvalError::TypeError(msg) => {
                                EvalError::TypeError(format!("Heterogeneous list: {}", msg))
                            }
                            other => other,
                        },
                    )?;
                }
            }
            Ok(Value::List(values))
        }
        Node::StructLiteral {
            name,
            fields: field_exprs,
        } => eval_struct_literal(name, field_exprs, env),
        Node::VariantCall { type_name, payload } => {
            eval_variant_call(type_name, payload.as_deref(), env)
        }
        Node::FieldAccess { object, field } => {
            let val = eval(object, env)?;
            match val {
                Value::Struct {
                    ref fields,
                    ref name,
                    ..
                } => fields
                    .iter()
                    .find(|(n, _)| n == field)
                    .map(|(_, v)| v.clone())
                    .ok_or_else(|| {
                        EvalError::TypeError(format!("Struct '{}' has no field '{}'", name, field))
                    }),
                _ => Err(EvalError::TypeError(format!(
                    "Cannot access field '{}' on non-struct value of type {}",
                    field,
                    val.type_name()
                ))),
            }
        }
        Node::MethodCall {
            receiver,
            method,
            args,
        } => eval_method_call(receiver, method, args, env),
        Node::Cast { target_type, expr } => {
            let val = eval(expr, env)?;
            eval_cast(&val, target_type)
        }
        Node::Match { scrutinee, arms } => eval_match(scrutinee, arms, env),
        Node::IfLet {
            scrutinee,
            pattern,
            then_block,
            else_block,
        } => eval_if_let(scrutinee, pattern, then_block, else_block, env),
        Node::WhileLet {
            scrutinee,
            pattern,
            body,
        } => eval_while_let(scrutinee, pattern, body, env),
        Node::PatternTest {
            scrutinee,
            pattern,
        } => eval_pattern_test(scrutinee, pattern, env),
        Node::Range {
            start,
            end,
            inclusive,
        } => {
            let s = eval(start, env)?;
            let e = eval(end, env)?;
            let to_i32 = |v: Value| -> Result<i32, EvalError> {
                match v {
                    Value::Int(i) => Ok(i),
                    Value::UInt(u) => i32::try_from(u).map_err(|_| {
                        EvalError::TypeError(format!(
                            "Range bound {} exceeds int32 range",
                            u
                        ))
                    }),
                    other => Err(EvalError::TypeError(format!(
                        "Range bounds must be int32 or uint32, got {}",
                        other.type_name()
                    ))),
                }
            };
            let start_val = to_i32(s)?;
            let end_val = to_i32(e)?;
            Ok(Value::Range {
                start: start_val,
                end: end_val,
                inclusive: *inclusive,
            })
        }
        Node::IndexAccess { object, index } => {
            let obj = eval(object, env)?;
            let idx = eval(index, env)?;
            eval_index_access(&obj, &idx)
        }
    }
}

enum MatchResult {
    Matched(Vec<(String, Value, TypeAnn)>), // (binding_name, value, type_ann)
    NoMatch,
}

fn try_match_pattern(
    val: &Value,
    pattern: &Pattern,
    env: &mut Environment,
) -> Result<MatchResult, EvalError> {
    match pattern {
        Pattern::Wildcard => Ok(MatchResult::Matched(vec![])),
        Pattern::Literal(lit_node) => {
            let lit_val = eval(lit_node, env)?;
            if *val == lit_val {
                Ok(MatchResult::Matched(vec![]))
            } else {
                Ok(MatchResult::NoMatch)
            }
        }
        Pattern::BareVariant(variant_name) => {
            if let Value::EnumVariant {
                variant_name: vn,
                payload: None,
                ..
            } = val
                && vn == variant_name
            {
                Ok(MatchResult::Matched(vec![]))
            } else {
                Ok(MatchResult::NoMatch)
            }
        }
        Pattern::Variant {
            variant_name,
            binding,
        } => {
            if let Value::EnumVariant {
                variant_name: vn,
                payload: Some(payload),
                ..
            } = val
                && vn == variant_name
            {
                let payload_type = infer_type(payload, binding)?;
                Ok(MatchResult::Matched(vec![(
                    binding.clone(),
                    *payload.clone(),
                    payload_type,
                )]))
            } else {
                Ok(MatchResult::NoMatch)
            }
        }
    }
}

fn apply_bindings(
    bindings: Vec<(String, Value, TypeAnn)>,
    env: &mut Environment,
) -> Result<(), EvalError> {
    for (name, value, type_ann) in bindings {
        env.declare(name, value, false, type_ann)
            .map_err(EvalError::TypeError)?;
    }
    Ok(())
}

fn eval_match(
    scrutinee: &Node,
    arms: &[MatchArm],
    env: &mut Environment,
) -> Result<Value, EvalError> {
    let val = eval(scrutinee, env)?;

    for arm in arms {
        match try_match_pattern(&val, &arm.pattern, env)? {
            MatchResult::Matched(bindings) => {
                if bindings.is_empty() {
                    return eval(&arm.body, env);
                }
                env.push_scope();
                apply_bindings(bindings, env)?;
                let result = eval(&arm.body, env);
                env.pop_scope();
                return result;
            }
            MatchResult::NoMatch => continue,
        }
    }

    Err(EvalError::TypeError(format!(
        "Non-exhaustive match: no pattern matched value '{}'",
        val
    )))
}

fn eval_if_let(
    scrutinee: &Node,
    pattern: &Pattern,
    then_block: &Node,
    else_block: &Option<Box<Node>>,
    env: &mut Environment,
) -> Result<Value, EvalError> {
    let val = eval(scrutinee, env)?;

    match try_match_pattern(&val, pattern, env)? {
        MatchResult::Matched(bindings) => {
            env.push_scope();
            apply_bindings(bindings, env)?;
            let result = eval(then_block, env);
            env.pop_scope();
            result
        }
        MatchResult::NoMatch => {
            if let Some(else_blk) = else_block {
                eval(else_blk, env)
            } else {
                Ok(Value::Unit)
            }
        }
    }
}

fn eval_while_let(
    scrutinee: &Node,
    pattern: &Pattern,
    body: &Node,
    env: &mut Environment,
) -> Result<Value, EvalError> {
    loop {
        let val = eval(scrutinee, env)?;
        match try_match_pattern(&val, pattern, env)? {
            MatchResult::Matched(bindings) => {
                env.push_scope();
                apply_bindings(bindings, env)?;
                match eval(body, env) {
                    Ok(_) => {}
                    Err(EvalError::Break) => {
                        env.pop_scope();
                        return Ok(Value::Unit);
                    }
                    Err(EvalError::Continue) => {
                        env.pop_scope();
                        continue;
                    }
                    Err(e) => {
                        env.pop_scope();
                        return Err(e);
                    }
                }
                env.pop_scope();
            }
            MatchResult::NoMatch => return Ok(Value::Unit),
        }
    }
}

fn eval_pattern_test(
    scrutinee: &Node,
    pattern: &Pattern,
    env: &mut Environment,
) -> Result<Value, EvalError> {
    let val = eval(scrutinee, env)?;
    match try_match_pattern(&val, pattern, env)? {
        MatchResult::Matched(_) => Ok(Value::Bool(true)),
        MatchResult::NoMatch => Ok(Value::Bool(false)),
    }
}

fn resolve_index(i: i32, len: usize) -> Result<usize, EvalError> {
    let resolved = if i < 0 { i + len as i32 } else { i };
    if resolved < 0 || resolved as usize >= len {
        return Err(EvalError::TypeError(format!(
            "Index {} out of bounds for length {}",
            i, len
        )));
    }
    Ok(resolved as usize)
}

fn range_len(start: i32, end: i32, inclusive: bool) -> usize {
    let len = if inclusive { end - start + 1 } else { end - start };
    len.max(0) as usize
}

fn eval_index_access(obj: &Value, idx: &Value) -> Result<Value, EvalError> {
    // Range index (slicing)
    if let Value::Range {
        start: rs,
        end: re,
        inclusive: ri,
    } = idx
    {
        let items = collection_to_list(obj)?;
        let len = items.len();
        let start = (*rs).max(0) as usize;
        let end = if *ri { *re + 1 } else { *re };
        let end = (end.max(0) as usize).min(len);
        let start = start.min(end);
        return Ok(Value::List(items[start..end].to_vec()));
    }

    // Scalar index
    let i = match idx {
        Value::Int(i) => *i,
        _ => {
            return Err(EvalError::TypeError(format!(
                "Index must be int32 or range, got {}",
                idx.type_name()
            )));
        }
    };
    match obj {
        Value::List(items) => {
            let pos = resolve_index(i, items.len())?;
            Ok(items[pos].clone())
        }
        Value::Str(s) => {
            let chars: Vec<char> = s.chars().collect();
            let pos = resolve_index(i, chars.len())?;
            Ok(Value::Str(chars[pos].to_string()))
        }
        Value::Range {
            start,
            end,
            inclusive,
        } => {
            let len = range_len(*start, *end, *inclusive);
            let pos = resolve_index(i, len)?;
            Ok(Value::Int(start + pos as i32))
        }
        _ => Err(EvalError::TypeError(format!(
            "Cannot index into value of type '{}'",
            obj.type_name()
        ))),
    }
}

fn eval_cast(val: &Value, target: &TypeAnn) -> Result<Value, EvalError> {
    match (val, target) {
        // int32 -> fl64
        (Value::Int(n), TypeAnn::Fl64) => Ok(Value::Float(*n as f64)),
        // int32 -> uint32
        (Value::Int(n), TypeAnn::UInt32) => {
            if *n < 0 {
                Err(EvalError::TypeError(format!(
                    "Cannot cast negative value {} to uint32",
                    n
                )))
            } else {
                Ok(Value::UInt(*n as u32))
            }
        }
        // uint32 -> int32
        (Value::UInt(n), TypeAnn::Int32) => Ok(Value::Int(*n as i32)),
        // uint32 -> fl64
        (Value::UInt(n), TypeAnn::Fl64) => Ok(Value::Float(*n as f64)),
        // fl64 -> int32 (truncate)
        (Value::Float(n), TypeAnn::Int32) => Ok(Value::Int(*n as i32)),
        // fl64 -> uint32 (truncate, must be non-negative)
        (Value::Float(n), TypeAnn::UInt32) => {
            if *n < 0.0 {
                Err(EvalError::TypeError(format!(
                    "Cannot cast negative value {} to uint32",
                    n
                )))
            } else {
                Ok(Value::UInt(*n as u32))
            }
        }
        // identity casts
        (Value::Int(_), TypeAnn::Int32)
        | (Value::UInt(_), TypeAnn::UInt32)
        | (Value::Float(_), TypeAnn::Fl64) => Ok(val.clone()),
        _ => Err(EvalError::TypeError(format!(
            "Cannot cast {} to {}",
            val.type_name(),
            target
        ))),
    }
}

/// Evaluate a statement, returning its value (used inside blocks)
fn eval_stmt_inner(stmt: &Statement, env: &mut Environment) -> Result<Value, EvalError> {
    match stmt {
        Statement::Expr(node) => eval(node, env),
        Statement::Declaration {
            mutable,
            name,
            type_ann,
            value,
        } => eval_declaration(name, *mutable, type_ann.as_ref(), value, env),
        Statement::Reassignment { name, value } => eval_reassignment(name, value, env),
        Statement::FnDeclaration {
            name,
            type_params,
            params,
            return_type,
            body,
        } => eval_fn_declaration(name, type_params, params, return_type, body, env),
        Statement::StructDeclaration {
            name,
            type_params,
            fields,
        } => eval_struct_declaration(name, type_params, fields, env),
        Statement::EnumDeclaration {
            name,
            type_params,
            variants,
        } => eval_enum_declaration(name, type_params, variants, env),
        Statement::MethodsDeclaration {
            type_name,
            type_params,
            methods,
        } => eval_methods_declaration(type_name, type_params, methods, env),
        Statement::TraitImplDeclaration {
            type_name,
            type_params,
            trait_name,
            methods,
        } => eval_trait_impl_declaration(type_name, type_params, trait_name, methods, env),
        Statement::DestructuringDecl {
            mutable,
            bindings,
            type_name,
            value,
        } => eval_destructuring_decl(*mutable, bindings, type_name, value, env),
    }
}

fn eval_fn_declaration(
    name: &str,
    type_params: &[String],
    params: &[(String, TypeAnn, bool)],
    return_type: &Option<TypeAnn>,
    body: &Node,
    env: &mut Environment,
) -> Result<Value, EvalError> {
    // Extract statements from the Block node
    let body_stmts = match body {
        Node::Block(stmts) => stmts.clone(),
        _ => panic!("Function body must be a block"),
    };
    let info = crate::env::FuncInfo {
        type_params: type_params.to_vec(),
        params: params.to_vec(),
        return_type: return_type.clone(),
        body: body_stmts,
    };
    env.declare_fn(name.to_string(), info);
    Ok(Value::Unit)
}

fn eval_struct_declaration(
    name: &str,
    type_params: &[String],
    fields: &[(String, TypeAnn)],
    env: &mut Environment,
) -> Result<Value, EvalError> {
    let def = crate::env::StructDef {
        name: name.to_string(),
        type_params: type_params.to_vec(),
        fields: fields.to_vec(),
    };
    env.declare_struct(def).map_err(EvalError::TypeError)?;
    Ok(Value::Unit)
}

fn eval_enum_declaration(
    name: &str,
    type_params: &[String],
    variants: &[crate::ast::EnumVariantDef],
    env: &mut Environment,
) -> Result<Value, EvalError> {
    let def = crate::env::EnumDef {
        name: name.to_string(),
        type_params: type_params.to_vec(),
        variants: variants.to_vec(),
    };
    env.declare_enum(def).map_err(EvalError::TypeError)?;
    Ok(Value::Unit)
}

fn eval_methods_declaration(
    type_name: &str,
    block_type_params: &[String],
    methods: &[crate::ast::MethodDef],
    env: &mut Environment,
) -> Result<Value, EvalError> {
    for method in methods {
        // Validate first param is "self"
        if method.params.is_empty() {
            return Err(EvalError::TypeError(format!(
                "Method '{}' on type '{}' must have 'self' as first parameter",
                method.name, type_name
            )));
        }
        let (ref self_name, ref self_type, self_mut) = method.params[0];
        if self_name != "self" {
            return Err(EvalError::TypeError(format!(
                "First parameter of method '{}' must be named 'self', got '{}'",
                method.name, self_name
            )));
        }
        if self_mut {
            return Err(EvalError::TypeError(format!(
                "Method '{}': 'self' cannot be mutable (mutable self not yet supported)",
                method.name
            )));
        }
        // Validate self type matches the type being implemented
        validate_self_type(self_type, type_name, block_type_params)?;

        // Extract body statements from Block node
        let body_stmts = match method.body.as_ref() {
            Node::Block(stmts) => stmts.clone(),
            _ => panic!("Method body must be a block"),
        };

        // Merge block type params into method's FuncInfo
        // so generic inference works automatically from the self parameter
        let merged_type_params = if !block_type_params.is_empty() {
            let mut tp = block_type_params.to_vec();
            for mtp in &method.type_params {
                if !tp.contains(mtp) {
                    tp.push(mtp.clone());
                }
            }
            tp
        } else {
            method.type_params.clone()
        };

        let info = crate::env::FuncInfo {
            type_params: merged_type_params,
            params: method.params.clone(),
            return_type: method.return_type.clone(),
            body: body_stmts,
        };
        env.declare_method(type_name, method.name.clone(), info)
            .map_err(EvalError::TypeError)?;
    }
    Ok(Value::Unit)
}

fn validate_self_type(
    self_type: &TypeAnn,
    type_name: &str,
    block_type_params: &[String],
) -> Result<(), EvalError> {
    if block_type_params.is_empty() {
        match self_type {
            TypeAnn::Named(n) if n == type_name => Ok(()),
            _ => Err(EvalError::TypeError(format!(
                "self type must be '{}', got '{}'",
                type_name, self_type
            ))),
        }
    } else {
        match self_type {
            TypeAnn::Generic { name, type_params } if name == type_name => {
                if type_params.len() != block_type_params.len() {
                    return Err(EvalError::TypeError(format!(
                        "self type has {} type params, but {} has {}",
                        type_params.len(),
                        type_name,
                        block_type_params.len()
                    )));
                }
                Ok(())
            }
            _ => Err(EvalError::TypeError(format!(
                "self type must be '<...>{}', got '{}'",
                type_name, self_type
            ))),
        }
    }
}

fn eval_trait_impl_declaration(
    type_name: &str,
    block_type_params: &[String],
    trait_name: &str,
    methods: &[crate::ast::MethodDef],
    env: &mut Environment,
) -> Result<Value, EvalError> {
    // Validate trait name and get expected signature
    let (expected_method, expected_params, _expected_return) = match trait_name {
        "Display" => ("display", 1usize, "str"),
        "Debug" => ("debug", 1, "str"),
        "PartialEq" => ("eq", 2, "bool"),
        "Ord" => ("cmp", 2, "Ordering"),
        _ => {
            return Err(EvalError::TypeError(format!(
                "Unknown trait '{}'. Known traits: Display, Debug, PartialEq, Ord",
                trait_name
            )))
        }
    };

    if methods.len() != 1 {
        return Err(EvalError::TypeError(format!(
            "Trait '{}' impl must contain exactly 1 method, got {}",
            trait_name,
            methods.len()
        )));
    }
    let method = &methods[0];
    if method.name != expected_method {
        return Err(EvalError::TypeError(format!(
            "Trait '{}' requires method '{}', got '{}'",
            trait_name, expected_method, method.name
        )));
    }
    if method.params.len() != expected_params {
        return Err(EvalError::TypeError(format!(
            "Trait '{}' method '{}' requires {} parameter(s), got {}",
            trait_name, expected_method, expected_params, method.params.len()
        )));
    }

    // Validate first param is "self", immutable, correct type
    let (ref self_name, ref self_type, self_mut) = method.params[0];
    if self_name != "self" {
        return Err(EvalError::TypeError(format!(
            "First parameter of '{}' must be named 'self', got '{}'",
            expected_method, self_name
        )));
    }
    if self_mut {
        return Err(EvalError::TypeError(format!(
            "Trait method '{}': 'self' cannot be mutable",
            expected_method
        )));
    }
    validate_self_type(self_type, type_name, block_type_params)?;

    let body_stmts = match method.body.as_ref() {
        Node::Block(stmts) => stmts.clone(),
        _ => panic!("Method body must be a block"),
    };

    let merged_type_params = if !block_type_params.is_empty() {
        let mut tp = block_type_params.to_vec();
        for mtp in &method.type_params {
            if !tp.contains(mtp) {
                tp.push(mtp.clone());
            }
        }
        tp
    } else {
        method.type_params.clone()
    };

    let info = crate::env::FuncInfo {
        type_params: merged_type_params,
        params: method.params.clone(),
        return_type: method.return_type.clone(),
        body: body_stmts,
    };
    env.declare_trait_impl(type_name, trait_name.to_string(), info)
        .map_err(EvalError::TypeError)?;
    Ok(Value::Unit)
}

fn call_trait_method(
    receiver: &Value,
    func: &crate::env::FuncInfo,
    extra_args: &[Value],
    env: &mut Environment,
) -> Result<Value, EvalError> {
    let mut arg_vals = vec![receiver.clone()];
    arg_vals.extend(extra_args.iter().cloned());

    let resolved_params: Vec<(String, TypeAnn, bool)>;
    let resolved_return_type: Option<TypeAnn>;

    if !func.type_params.is_empty() {
        let mut bindings = HashMap::new();
        for (i, (_, param_type, _)) in func.params.iter().enumerate() {
            if i < arg_vals.len() {
                let arg_type =
                    infer_type(&arg_vals[i], &format!("trait method arg {}", i))?;
                infer_type_var_bindings(param_type, &arg_type, &mut bindings)
                    .map_err(EvalError::TypeError)?;
            }
        }
        resolved_params = func
            .params
            .iter()
            .map(|(n, t, m)| (n.clone(), substitute_type(t, &bindings), *m))
            .collect();
        resolved_return_type = func
            .return_type
            .as_ref()
            .map(|rt| substitute_type(rt, &bindings));
    } else {
        resolved_params = func.params.clone();
        resolved_return_type = func.return_type.clone();
    }

    let saved = env.save_caller_scopes();
    env.push_scope();

    for (i, (param_name, param_type, param_mut)) in resolved_params.iter().enumerate() {
        let val = if i < arg_vals.len() {
            coerce_if_needed(arg_vals[i].clone(), param_type)
        } else {
            Value::Unit
        };
        env.declare(param_name.clone(), val, *param_mut, param_type.clone())
            .map_err(EvalError::TypeError)?;
    }

    let mut result = Value::Unit;
    for body_stmt in &func.body {
        match eval_stmt_inner(body_stmt, env) {
            Ok(val) => result = val,
            Err(EvalError::Return(val)) => {
                result = *val;
                break;
            }
            Err(e) => {
                env.pop_scope();
                env.restore_caller_scopes(saved);
                return Err(e);
            }
        }
    }

    // Validate return type
    if let Some(ref rt) = resolved_return_type {
        validate_type(&result, rt, "trait method return")?;
    }

    env.pop_scope();
    env.restore_caller_scopes(saved);
    Ok(result)
}

fn value_to_display_string(val: &Value, env: &mut Environment) -> Result<String, EvalError> {
    let type_name = match val {
        Value::Struct { name, .. } => Some(name.as_str()),
        Value::EnumVariant { enum_name, .. } => Some(enum_name.as_str()),
        _ => None,
    };

    if let Some(tn) = type_name
        && let Some(func) = env.get_trait_impl(tn, "Display").cloned()
    {
        let result = call_trait_method(val, &func, &[], env)?;
        return match result {
            Value::Str(s) => Ok(s),
            _ => Err(EvalError::TypeError(
                "Display.display must return a string".to_string(),
            )),
        };
    }
    Ok(format!("{}", val))
}

fn value_to_debug_string(val: &Value, env: &mut Environment) -> Result<String, EvalError> {
    let type_name = match val {
        Value::Struct { name, .. } => Some(name.as_str()),
        Value::EnumVariant { enum_name, .. } => Some(enum_name.as_str()),
        _ => None,
    };

    if let Some(tn) = type_name
        && let Some(func) = env.get_trait_impl(tn, "Debug").cloned()
    {
        let result = call_trait_method(val, &func, &[], env)?;
        return match result {
            Value::Str(s) => Ok(s),
            _ => Err(EvalError::TypeError(
                "Debug.debug must return a string".to_string(),
            )),
        };
    }
    auto_debug_string(val, env)
}

#[allow(clippy::only_used_in_recursion)]
fn auto_debug_string(val: &Value, env: &mut Environment) -> Result<String, EvalError> {
    match val {
        Value::Str(s) => Ok(format!("\"{}\"", s)),
        Value::List(elems) => {
            let mut parts = vec![];
            for elem in elems {
                parts.push(auto_debug_string(elem, env)?);
            }
            Ok(format!("[{}]", parts.join(", ")))
        }
        Value::Struct { name, fields, .. } => {
            let mut field_strs = vec![];
            for (fname, fval) in fields {
                field_strs.push(format!("{}: {}", fname, auto_debug_string(fval, env)?));
            }
            Ok(format!("({}){}", field_strs.join(", "), name))
        }
        Value::EnumVariant {
            variant_name,
            payload,
            ..
        } => match payload {
            Some(p) => Ok(format!(
                "({}){}", auto_debug_string(p, env)?, variant_name
            )),
            None => Ok(variant_name.clone()),
        },
        _ => Ok(format!("{}", val)),
    }
}

fn try_partial_eq(
    left: &Value,
    right: &Value,
    env: &mut Environment,
) -> Result<Option<bool>, EvalError> {
    let type_name = match (left, right) {
        (Value::Struct { name: ln, .. }, Value::Struct { name: rn, .. }) if ln == rn => {
            ln.clone()
        }
        (
            Value::EnumVariant {
                enum_name: ln, ..
            },
            Value::EnumVariant {
                enum_name: rn, ..
            },
        ) if ln == rn => ln.clone(),
        _ => return Ok(None),
    };
    if let Some(func) = env.get_trait_impl(&type_name, "PartialEq").cloned() {
        let result = call_trait_method(left, &func, std::slice::from_ref(right), env)?;
        match result {
            Value::Bool(b) => Ok(Some(b)),
            _ => Err(EvalError::TypeError(
                "PartialEq.eq must return bool".to_string(),
            )),
        }
    } else {
        Ok(None) // No PartialEq impl, use structural equality
    }
}

fn try_ord_compare(
    left: &Value,
    right: &Value,
    env: &mut Environment,
) -> Result<Option<String>, EvalError> {
    let type_name = match (left, right) {
        (Value::Struct { name: ln, .. }, Value::Struct { name: rn, .. }) if ln == rn => {
            ln.clone()
        }
        (
            Value::EnumVariant {
                enum_name: ln, ..
            },
            Value::EnumVariant {
                enum_name: rn, ..
            },
        ) if ln == rn => ln.clone(),
        _ => return Ok(None),
    };
    if let Some(func) = env.get_trait_impl(&type_name, "Ord").cloned() {
        let result = call_trait_method(left, &func, std::slice::from_ref(right), env)?;
        match result {
            Value::EnumVariant {
                enum_name,
                variant_name,
                ..
            } if enum_name == "Ordering" => Ok(Some(variant_name)),
            _ => Err(EvalError::TypeError(
                "Ord.cmp must return an Ordering variant".to_string(),
            )),
        }
    } else {
        Ok(None)
    }
}

fn collection_to_list(val: &Value) -> Result<Vec<Value>, EvalError> {
    match val {
        Value::List(items) => Ok(items.clone()),
        Value::Str(s) => Ok(s.chars().map(|c| Value::Str(c.to_string())).collect()),
        Value::Range {
            start,
            end,
            inclusive,
        } => {
            let end_val = if *inclusive { *end + 1 } else { *end };
            Ok((*start..end_val).map(Value::Int).collect())
        }
        _ => Err(EvalError::TypeError(format!(
            "Cannot iterate over value of type '{}'",
            val.type_name()
        ))),
    }
}

fn call_closure(closure: &Value, args: Vec<Value>) -> Result<Value, EvalError> {
    match closure {
        Value::Closure { params, body, env } => {
            if args.len() != params.len() {
                return Err(EvalError::TypeError(format!(
                    "Closure expects {} argument(s), got {}",
                    params.len(),
                    args.len()
                )));
            }
            let mut closure_env = *env.clone();
            closure_env.push_scope();
            for (param, arg) in params.iter().zip(args.into_iter()) {
                let type_ann = if let Some(ref t) = param.type_ann {
                    t.clone()
                } else {
                    infer_type(&arg, &param.name)?
                };
                closure_env
                    .declare(param.name.clone(), arg, param.mutable, type_ann)
                    .map_err(EvalError::TypeError)?;
            }
            let result = eval(body, &mut closure_env);
            closure_env.pop_scope();
            result
        }
        _ => Err(EvalError::TypeError(format!(
            "Expected a closure, got {}",
            closure.type_name()
        ))),
    }
}

fn is_collection(val: &Value) -> bool {
    matches!(val, Value::List(_) | Value::Str(_) | Value::Range { .. })
}

fn ordering_variant(name: &str) -> Value {
    Value::EnumVariant {
        enum_name: "Ordering".to_string(),
        variant_name: name.to_string(),
        type_params: vec![],
        payload: None,
    }
}

fn eval_ordering_method(
    variant_name: &str,
    method_name: &str,
    arg_nodes: &[Node],
    env: &mut Environment,
) -> Result<Option<Value>, EvalError> {
    match method_name {
        "rev" => {
            if !arg_nodes.is_empty() {
                return Err(EvalError::TypeError(format!(
                    "'rev' expects 0 arguments, got {}",
                    arg_nodes.len()
                )));
            }
            let result = match variant_name {
                "Less" => ordering_variant("Greater"),
                "Greater" => ordering_variant("Less"),
                _ => ordering_variant("Equal"),
            };
            Ok(Some(result))
        }
        "then" => {
            if arg_nodes.len() != 1 {
                return Err(EvalError::TypeError(format!(
                    "'then' expects 1 argument (an Ordering), got {}",
                    arg_nodes.len()
                )));
            }
            if variant_name != "Equal" {
                // Self is not Equal — return self
                Ok(Some(ordering_variant(variant_name)))
            } else {
                // Self is Equal — return the other
                let other = eval(&arg_nodes[0], env)?;
                if let Value::EnumVariant { enum_name, .. } = &other
                    && enum_name == "Ordering"
                {
                    Ok(Some(other))
                } else {
                    Err(EvalError::TypeError(format!(
                        "'then' argument must be an Ordering, got {}",
                        other.type_name()
                    )))
                }
            }
        }
        _ => Ok(None),
    }
}

fn eval_maybe_attempt_method(
    receiver: &Value,
    method_name: &str,
    arg_nodes: &[Node],
    env: &mut Environment,
) -> Result<Option<Value>, EvalError> {
    let Value::EnumVariant {
        enum_name,
        variant_name,
        payload,
        ..
    } = receiver
    else {
        return Ok(None);
    };

    // Determine if this is the "happy path" variant (Exists/Success)
    let is_happy = match enum_name.as_str() {
        "Maybe" => variant_name == "Exists",
        "Attempt" => variant_name == "Success",
        _ => return Ok(None),
    };

    match method_name {
        "map" => {
            if arg_nodes.len() != 1 {
                return Err(EvalError::TypeError(format!(
                    "'map' expects 1 argument (a closure), got {}",
                    arg_nodes.len()
                )));
            }
            if is_happy {
                let closure = eval(&arg_nodes[0], env)?;
                let inner = payload.as_ref().unwrap();
                let result = call_closure(&closure, vec![*inner.clone()])?;
                // Wrap result in the same happy variant
                let wrap_variant = variant_name.clone();
                Ok(Some(Value::EnumVariant {
                    enum_name: enum_name.clone(),
                    variant_name: wrap_variant,
                    type_params: vec![],
                    payload: Some(Box::new(result)),
                }))
            } else {
                // DoesNotExist/Failure — pass through unchanged
                Ok(Some(receiver.clone()))
            }
        }
        "and_then" => {
            if arg_nodes.len() != 1 {
                return Err(EvalError::TypeError(format!(
                    "'and_then' expects 1 argument (a closure), got {}",
                    arg_nodes.len()
                )));
            }
            if is_happy {
                let closure = eval(&arg_nodes[0], env)?;
                let inner = payload.as_ref().unwrap();
                let result = call_closure(&closure, vec![*inner.clone()])?;
                // Closure must return the same enum type
                if let Value::EnumVariant {
                    enum_name: ref rn, ..
                } = result
                {
                    if rn != enum_name {
                        return Err(EvalError::TypeError(format!(
                            "'and_then' closure must return {}, got {}",
                            enum_name, rn
                        )));
                    }
                } else {
                    return Err(EvalError::TypeError(format!(
                        "'and_then' closure must return {}, got {}",
                        enum_name,
                        result.type_name()
                    )));
                }
                Ok(Some(result))
            } else {
                Ok(Some(receiver.clone()))
            }
        }
        "or_else" => {
            if arg_nodes.len() != 1 {
                return Err(EvalError::TypeError(format!(
                    "'or_else' expects 1 argument (a closure), got {}",
                    arg_nodes.len()
                )));
            }
            if is_happy {
                // Already happy — return self unchanged
                Ok(Some(receiver.clone()))
            } else {
                let closure = eval(&arg_nodes[0], env)?;
                // For Attempt's Failure, pass the error payload to the closure
                let result = if enum_name == "Attempt" {
                    let err = payload.as_ref().unwrap();
                    call_closure(&closure, vec![*err.clone()])?
                } else {
                    // Maybe's DoesNotExist has no payload — call with zero args
                    call_closure(&closure, vec![])?
                };
                // Closure must return the same enum type
                if let Value::EnumVariant {
                    enum_name: ref rn, ..
                } = result
                {
                    if rn != enum_name {
                        return Err(EvalError::TypeError(format!(
                            "'or_else' closure must return {}, got {}",
                            enum_name, rn
                        )));
                    }
                } else {
                    return Err(EvalError::TypeError(format!(
                        "'or_else' closure must return {}, got {}",
                        enum_name,
                        result.type_name()
                    )));
                }
                Ok(Some(result))
            }
        }
        "or_panic" => {
            if !arg_nodes.is_empty() {
                return Err(EvalError::TypeError(format!(
                    "'or_panic' expects 0 arguments, got {}",
                    arg_nodes.len()
                )));
            }
            if is_happy {
                Ok(Some(*payload.as_ref().unwrap().clone()))
            } else if enum_name == "Attempt" {
                let err = payload.as_ref().unwrap();
                Err(EvalError::TypeError(format!(
                    "or_panic called on Failure: {}",
                    err
                )))
            } else {
                Err(EvalError::TypeError(
                    "or_panic called on DoesNotExist".to_string(),
                ))
            }
        }
        "exists_and" => {
            if enum_name != "Maybe" {
                return Ok(None);
            }
            if arg_nodes.len() != 1 {
                return Err(EvalError::TypeError(format!(
                    "'exists_and' expects 1 argument (a closure), got {}",
                    arg_nodes.len()
                )));
            }
            if is_happy {
                let closure = eval(&arg_nodes[0], env)?;
                let inner = payload.as_ref().unwrap();
                let result = call_closure(&closure, vec![*inner.clone()])?;
                if !matches!(result, Value::Bool(_)) {
                    return Err(EvalError::TypeError(format!(
                        "'exists_and' closure must return a bool, got {}",
                        result.type_name()
                    )));
                }
                Ok(Some(result))
            } else {
                Ok(Some(Value::Bool(false)))
            }
        }
        "succeeded_and" => {
            if enum_name != "Attempt" {
                return Ok(None);
            }
            if arg_nodes.len() != 1 {
                return Err(EvalError::TypeError(format!(
                    "'succeeded_and' expects 1 argument (a closure), got {}",
                    arg_nodes.len()
                )));
            }
            if is_happy {
                let closure = eval(&arg_nodes[0], env)?;
                let inner = payload.as_ref().unwrap();
                let result = call_closure(&closure, vec![*inner.clone()])?;
                if !matches!(result, Value::Bool(_)) {
                    return Err(EvalError::TypeError(format!(
                        "'succeeded_and' closure must return a bool, got {}",
                        result.type_name()
                    )));
                }
                Ok(Some(result))
            } else {
                Ok(Some(Value::Bool(false)))
            }
        }
        "dne_or" => {
            if enum_name != "Maybe" {
                return Ok(None);
            }
            if arg_nodes.len() != 1 {
                return Err(EvalError::TypeError(format!(
                    "'dne_or' expects 1 argument (a closure), got {}",
                    arg_nodes.len()
                )));
            }
            if is_happy {
                let closure = eval(&arg_nodes[0], env)?;
                let inner = payload.as_ref().unwrap();
                let result = call_closure(&closure, vec![*inner.clone()])?;
                if !matches!(result, Value::Bool(_)) {
                    return Err(EvalError::TypeError(format!(
                        "'dne_or' closure must return a bool, got {}",
                        result.type_name()
                    )));
                }
                Ok(Some(result))
            } else {
                Ok(Some(Value::Bool(true)))
            }
        }
        "failed_and" => {
            if enum_name != "Attempt" {
                return Ok(None);
            }
            if arg_nodes.len() != 1 {
                return Err(EvalError::TypeError(format!(
                    "'failed_and' expects 1 argument (a closure), got {}",
                    arg_nodes.len()
                )));
            }
            if !is_happy {
                let closure = eval(&arg_nodes[0], env)?;
                let err = payload.as_ref().unwrap();
                let result = call_closure(&closure, vec![*err.clone()])?;
                if !matches!(result, Value::Bool(_)) {
                    return Err(EvalError::TypeError(format!(
                        "'failed_and' closure must return a bool, got {}",
                        result.type_name()
                    )));
                }
                Ok(Some(result))
            } else {
                Ok(Some(Value::Bool(false)))
            }
        }
        "or_default" => {
            if arg_nodes.len() != 1 {
                return Err(EvalError::TypeError(format!(
                    "'or_default' expects 1 argument (a default value), got {}",
                    arg_nodes.len()
                )));
            }
            if is_happy {
                Ok(Some(*payload.as_ref().unwrap().clone()))
            } else {
                let default = eval(&arg_nodes[0], env)?;
                Ok(Some(default))
            }
        }
        _ => Ok(None), // Not a built-in Maybe/Attempt method, fall through
    }
}

fn eval_builtin_method(
    receiver: &Value,
    method_name: &str,
    arg_nodes: &[Node],
    env: &mut Environment,
) -> Result<Option<Value>, EvalError> {
    if let Value::EnumVariant { enum_name, .. } = receiver
        && (enum_name == "Maybe" || enum_name == "Attempt")
    {
        return eval_maybe_attempt_method(receiver, method_name, arg_nodes, env);
    }

    if let Value::EnumVariant {
        enum_name,
        variant_name,
        ..
    } = receiver
        && enum_name == "Ordering"
    {
        return eval_ordering_method(variant_name, method_name, arg_nodes, env);
    }

    if !is_collection(receiver) {
        return Ok(None);
    }

    match method_name {
        "reverse" => {
            if !arg_nodes.is_empty() {
                return Err(EvalError::TypeError(format!(
                    "'reverse' expects 0 arguments, got {}",
                    arg_nodes.len()
                )));
            }
            let mut items = collection_to_list(receiver)?;
            items.reverse();
            Ok(Some(Value::List(items)))
        }
        "map" => {
            if arg_nodes.len() != 1 {
                return Err(EvalError::TypeError(format!(
                    "'map' expects 1 argument (a closure), got {}",
                    arg_nodes.len()
                )));
            }
            let closure = eval(&arg_nodes[0], env)?;
            let items = collection_to_list(receiver)?;
            let mut result = Vec::with_capacity(items.len());
            for item in items {
                result.push(call_closure(&closure, vec![item])?);
            }
            Ok(Some(Value::List(result)))
        }
        "filter" => {
            if arg_nodes.len() != 1 {
                return Err(EvalError::TypeError(format!(
                    "'filter' expects 1 argument (a closure), got {}",
                    arg_nodes.len()
                )));
            }
            let closure = eval(&arg_nodes[0], env)?;
            let items = collection_to_list(receiver)?;
            let mut result = vec![];
            for item in items {
                let keep = call_closure(&closure, vec![item.clone()])?;
                if keep == Value::Bool(true) {
                    result.push(item);
                } else if keep != Value::Bool(false) {
                    return Err(EvalError::TypeError(format!(
                        "'filter' closure must return a bool, got {}",
                        keep.type_name()
                    )));
                }
            }
            Ok(Some(Value::List(result)))
        }
        "enumerate" => {
            if !arg_nodes.is_empty() {
                return Err(EvalError::TypeError(format!(
                    "'enumerate' expects 0 arguments, got {}",
                    arg_nodes.len()
                )));
            }
            let items = collection_to_list(receiver)?;
            let result: Vec<Value> = items
                .into_iter()
                .enumerate()
                .map(|(i, v)| Value::List(vec![Value::UInt(i as u32), v]))
                .collect();
            Ok(Some(Value::List(result)))
        }
        "take" => {
            if arg_nodes.len() != 1 {
                return Err(EvalError::TypeError(format!(
                    "'take' expects 1 argument, got {}",
                    arg_nodes.len()
                )));
            }
            let n_val = eval(&arg_nodes[0], env)?;
            let n = match n_val {
                Value::Int(i) if i >= 0 => i as usize,
                Value::UInt(u) => u as usize,
                _ => {
                    return Err(EvalError::TypeError(format!(
                        "'take' argument must be a non-negative integer, got {}",
                        n_val.type_name()
                    )));
                }
            };
            let items = collection_to_list(receiver)?;
            let taken: Vec<Value> = items.into_iter().take(n).collect();
            Ok(Some(Value::List(taken)))
        }
        "while_take" => {
            if arg_nodes.len() != 1 {
                return Err(EvalError::TypeError(format!(
                    "'while_take' expects 1 argument (a closure), got {}",
                    arg_nodes.len()
                )));
            }
            let closure = eval(&arg_nodes[0], env)?;
            let items = collection_to_list(receiver)?;
            let mut result = vec![];
            for item in items {
                let keep = call_closure(&closure, vec![item.clone()])?;
                if keep == Value::Bool(true) {
                    result.push(item);
                } else if keep == Value::Bool(false) {
                    break;
                } else {
                    return Err(EvalError::TypeError(format!(
                        "'while_take' closure must return a bool, got {}",
                        keep.type_name()
                    )));
                }
            }
            Ok(Some(Value::List(result)))
        }
        _ => Ok(None), // Not a built-in method, fall through
    }
}

fn eval_method_call(
    receiver_node: &Node,
    method_name: &str,
    arg_nodes: &[Node],
    env: &mut Environment,
) -> Result<Value, EvalError> {
    // 1. Evaluate the receiver
    let receiver_val = eval(receiver_node, env)?;

    // 2. Try built-in collection methods first
    if let Some(result) = eval_builtin_method(&receiver_val, method_name, arg_nodes, env)? {
        return Ok(result);
    }

    // 3. Determine type name from the receiver value
    let type_name = match &receiver_val {
        Value::Struct { name, .. } => name.clone(),
        Value::EnumVariant { enum_name, .. } => enum_name.clone(),
        _ => {
            return Err(EvalError::TypeError(format!(
                "Cannot call method '{}' on value of type '{}'",
                method_name,
                receiver_val.type_name()
            )));
        }
    };

    // 3. Look up the method
    let method_info = env
        .get_method(&type_name, method_name)
        .ok_or_else(|| {
            EvalError::TypeError(format!(
                "Type '{}' has no method '{}'",
                type_name, method_name
            ))
        })?
        .clone();

    // 4. Check arg count (params include self, so actual args = params - 1)
    let non_self_param_count = method_info.params.len() - 1;
    if arg_nodes.len() != non_self_param_count {
        return Err(EvalError::TypeError(format!(
            "Method '{}.{}' expects {} argument(s), got {}",
            type_name,
            method_name,
            non_self_param_count,
            arg_nodes.len()
        )));
    }

    // 5. Evaluate arguments in caller scope; self is first
    let mut arg_vals = vec![receiver_val];
    for arg_node in arg_nodes {
        let val = eval(arg_node, env)?;
        arg_vals.push(val);
    }

    // 6. Handle generics: infer type variable bindings
    let resolved_params: Vec<(String, TypeAnn, bool)>;
    let resolved_return_type: Option<TypeAnn>;

    if !method_info.type_params.is_empty() {
        let mut bindings = HashMap::new();
        for (i, (_, param_type, _)) in method_info.params.iter().enumerate() {
            let arg_type = infer_type(
                &arg_vals[i],
                &format!("arg {} of '{}.{}'", i, type_name, method_name),
            )?;
            infer_type_var_bindings(param_type, &arg_type, &mut bindings)
                .map_err(EvalError::TypeError)?;
        }
        for tp in &method_info.type_params {
            if !bindings.contains_key(tp) {
                return Err(EvalError::TypeError(format!(
                    "Could not infer type variable '{}' in call to '{}.{}'",
                    tp, type_name, method_name
                )));
            }
        }
        resolved_params = method_info
            .params
            .iter()
            .map(|(n, t, m)| (n.clone(), substitute_type(t, &bindings), *m))
            .collect();
        resolved_return_type = method_info
            .return_type
            .as_ref()
            .map(|rt| substitute_type(rt, &bindings));
    } else {
        resolved_params = method_info.params.clone();
        resolved_return_type = method_info.return_type.clone();
    }

    // 7. Validate and coerce args
    for (i, (_, param_type, _)) in resolved_params.iter().enumerate() {
        validate_type(&arg_vals[i], param_type, &resolved_params[i].0)?;
        arg_vals[i] = coerce_if_needed(arg_vals[i].clone(), param_type);
    }

    // 8. Execute method body (same scoping as function calls)
    let saved = env.save_caller_scopes();
    env.push_scope();

    for (i, (pname, ptype, is_mut)) in resolved_params.iter().enumerate() {
        env.declare(pname.clone(), arg_vals[i].clone(), *is_mut, ptype.clone())
            .map_err(EvalError::TypeError)?;
    }

    let mut result = Value::Unit;
    let mut err = None;
    for stmt in &method_info.body {
        match eval_stmt_inner(stmt, env) {
            Ok(val) => result = val,
            Err(EvalError::Return(val)) => {
                result = *val;
                break;
            }
            Err(e) => {
                err = Some(e);
                break;
            }
        }
    }

    env.pop_scope();
    env.restore_caller_scopes(saved);

    if let Some(e) = err {
        return Err(e);
    }

    // 9. Validate return type
    if let Some(ref rt) = resolved_return_type {
        validate_type(
            &result,
            rt,
            &format!("return value of '{}.{}'", type_name, method_name),
        )?;
        let result = coerce_if_needed(result, rt);
        return Ok(result);
    }

    Ok(result)
}

fn eval_struct_literal(
    name: &str,
    field_exprs: &[(String, Node)],
    env: &mut Environment,
) -> Result<Value, EvalError> {
    let struct_def = env
        .get_struct(name)
        .ok_or_else(|| EvalError::TypeError(format!("Unknown struct type '{}'", name)))?
        .clone();

    // Check field count
    if field_exprs.len() != struct_def.fields.len() {
        return Err(EvalError::TypeError(format!(
            "Struct '{}' has {} field(s), got {}",
            name,
            struct_def.fields.len(),
            field_exprs.len()
        )));
    }

    // Evaluate field values and check field names
    let mut field_values: Vec<(String, Value)> = vec![];
    let mut used_fields = std::collections::HashSet::new();

    for (fname, fexpr) in field_exprs {
        if !used_fields.insert(fname.clone()) {
            return Err(EvalError::TypeError(format!(
                "Duplicate field '{}' in struct literal for '{}'",
                fname, name
            )));
        }
        if !struct_def.fields.iter().any(|(n, _)| n == fname) {
            return Err(EvalError::TypeError(format!(
                "Unknown field '{}' for struct '{}'",
                fname, name
            )));
        }
        let val = eval(fexpr, env)?;
        field_values.push((fname.clone(), val));
    }

    // Resolve generics and validate types
    let resolved_type_params = if !struct_def.type_params.is_empty() {
        let mut bindings = HashMap::new();
        for (fname, val) in &field_values {
            let def_field_type = &struct_def
                .fields
                .iter()
                .find(|(n, _)| n == fname)
                .unwrap()
                .1;
            let val_type = infer_type(val, &format!("{}.{}", name, fname))?;
            infer_type_var_bindings(def_field_type, &val_type, &mut bindings)
                .map_err(EvalError::TypeError)?;
        }
        for tp in &struct_def.type_params {
            if !bindings.contains_key(tp) {
                return Err(EvalError::TypeError(format!(
                    "Could not infer type variable '{}' in struct literal for '{}'",
                    tp, name
                )));
            }
        }
        // Validate against resolved types
        for (fname, val) in &field_values {
            let def_field_type = &struct_def
                .fields
                .iter()
                .find(|(n, _)| n == fname)
                .unwrap()
                .1;
            let resolved = substitute_type(def_field_type, &bindings);
            validate_type(val, &resolved, &format!("{}.{}", name, fname))?;
        }
        struct_def
            .type_params
            .iter()
            .map(|tp| bindings[tp].clone())
            .collect()
    } else {
        for (fname, val) in &field_values {
            let def_field_type = &struct_def
                .fields
                .iter()
                .find(|(n, _)| n == fname)
                .unwrap()
                .1;
            validate_type(val, def_field_type, &format!("{}.{}", name, fname))?;
        }
        vec![]
    };

    // Coerce field values and normalize to declaration order
    let final_fields: Vec<(String, Value)> = struct_def
        .fields
        .iter()
        .map(|(def_fname, def_ftype)| {
            let val = field_values
                .iter()
                .find(|(n, _)| n == def_fname)
                .unwrap()
                .1
                .clone();
            let resolved_type = if !struct_def.type_params.is_empty() {
                let bindings: HashMap<String, TypeAnn> = struct_def
                    .type_params
                    .iter()
                    .zip(resolved_type_params.iter())
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                substitute_type(def_ftype, &bindings)
            } else {
                def_ftype.clone()
            };
            (def_fname.clone(), coerce_if_needed(val, &resolved_type))
        })
        .collect();

    Ok(Value::Struct {
        name: name.to_string(),
        type_params: resolved_type_params,
        fields: final_fields,
    })
}

fn eval_variant_call(
    variant_name: &str,
    payload_node: Option<&Node>,
    env: &mut Environment,
) -> Result<Value, EvalError> {
    // Check if this is a known enum variant
    if let Some((enum_name, enum_def)) = env.get_enum_for_variant(variant_name) {
        let enum_name = enum_name.to_string();
        let enum_def = enum_def.clone();
        let variant_def = enum_def
            .variants
            .iter()
            .find(|v| v.name == variant_name)
            .unwrap();

        match (&variant_def.payload_type, payload_node) {
            (None, None) => {
                // For generic enums, produce value with empty type_params.
                // A type annotation context will fill them in via coercion.
                Ok(Value::EnumVariant {
                    enum_name,
                    variant_name: variant_name.to_string(),
                    type_params: vec![],
                    payload: None,
                })
            }
            (Some(expected_type), Some(payload_expr)) => {
                let val = eval(payload_expr, env)?;
                let resolved_type_params = if !enum_def.type_params.is_empty() {
                    let mut bindings = HashMap::new();
                    let val_type = infer_type(&val, &format!("payload of '{}'", variant_name))?;
                    infer_type_var_bindings(expected_type, &val_type, &mut bindings)
                        .map_err(EvalError::TypeError)?;
                    let all_bound = enum_def
                        .type_params
                        .iter()
                        .all(|tp| bindings.contains_key(tp));
                    if all_bound {
                        let resolved = substitute_type(expected_type, &bindings);
                        validate_type(&val, &resolved, &format!("payload of '{}'", variant_name))?;
                        enum_def
                            .type_params
                            .iter()
                            .map(|tp| bindings[tp].clone())
                            .collect()
                    } else {
                        // Partial inference (e.g. (42)Success where T binds but E doesn't).
                        // Validate what we can, leave type_params empty for annotation to fill in.
                        let resolved = substitute_type(expected_type, &bindings);
                        validate_type(&val, &resolved, &format!("payload of '{}'", variant_name))?;
                        vec![]
                    }
                } else {
                    validate_type(
                        &val,
                        expected_type,
                        &format!("payload of '{}'", variant_name),
                    )?;
                    vec![]
                };
                let coerced = if !resolved_type_params.is_empty() {
                    let bindings: HashMap<String, TypeAnn> = enum_def
                        .type_params
                        .iter()
                        .zip(resolved_type_params.iter())
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    coerce_if_needed(val, &substitute_type(expected_type, &bindings))
                } else if enum_def.type_params.is_empty() {
                    coerce_if_needed(val, expected_type)
                } else {
                    val // Partial inference — can't coerce payload without full bindings
                };
                Ok(Value::EnumVariant {
                    enum_name,
                    variant_name: variant_name.to_string(),
                    type_params: resolved_type_params,
                    payload: Some(Box::new(coerced)),
                })
            }
            (None, Some(_)) => Err(EvalError::TypeError(format!(
                "Variant '{}' of enum '{}' takes no payload",
                variant_name, enum_name
            ))),
            (Some(_), None) => Err(EvalError::TypeError(format!(
                "Variant '{}' of enum '{}' requires a payload",
                variant_name, enum_name
            ))),
        }
    }
    // Check if this is a zero-field struct
    else if payload_node.is_none() {
        if let Some(struct_def) = env.get_struct(variant_name).cloned()
            && struct_def.fields.is_empty()
        {
            if !struct_def.type_params.is_empty() {
                return Err(EvalError::TypeError(format!(
                    "Cannot infer type parameters for zero-field struct '{}'",
                    variant_name
                )));
            }
            return Ok(Value::Struct {
                name: variant_name.to_string(),
                type_params: vec![],
                fields: vec![],
            });
        }
        Err(EvalError::TypeError(format!(
            "Unknown variant or zero-field struct '{}'",
            variant_name
        )))
    } else {
        Err(EvalError::TypeError(format!(
            "Unknown variant '{}'",
            variant_name
        )))
    }
}

fn eval_fn_call(name: &str, arg_nodes: &[Node], env: &mut Environment) -> Result<Value, EvalError> {
    // Built-in functions
    if name == "print" || name == "lnprint" {
        if arg_nodes.len() != 1 {
            return Err(EvalError::TypeError(format!(
                "'{}' expects 1 argument, got {}",
                name,
                arg_nodes.len()
            )));
        }
        let val = eval(&arg_nodes[0], env)?;
        let display_str = value_to_display_string(&val, env)?;
        if name == "lnprint" {
            println!("{}", display_str);
        } else {
            use std::io::Write;
            print!("{}", display_str);
            std::io::stdout().flush().ok();
        }
        return Ok(Value::Unit);
    }

    if name == "len" {
        if arg_nodes.len() != 1 {
            return Err(EvalError::TypeError(format!(
                "'len' expects 1 argument, got {}",
                arg_nodes.len()
            )));
        }
        let val = eval(&arg_nodes[0], env)?;
        return match val {
            Value::Str(s) => Ok(Value::UInt(s.len() as u32)),
            Value::List(l) => Ok(Value::UInt(l.len() as u32)),
            Value::Range {
                start,
                end,
                inclusive,
            } => Ok(Value::UInt(range_len(start, end, inclusive) as u32)),
            _ => Err(EvalError::TypeError(format!(
                "'len' requires a string, list, or range, got {}",
                val.type_name()
            ))),
        };
    }

    // Look up function
    if let Some(func) = env.get_fn(name).cloned() {
        return eval_named_fn_call(name, &func, arg_nodes, env);
    }

    // Look up closure variable
    if let Some(Value::Closure {
        params,
        body,
        env: captured_env,
    }) = env.get(name)
    {
        return eval_closure_call(name, &params, &body, &captured_env, arg_nodes, env);
    }

    Err(EvalError::UndefinedVariable(format!("function '{}'", name)))
}

fn eval_named_fn_call(
    name: &str,
    func: &crate::env::FuncInfo,
    arg_nodes: &[Node],
    env: &mut Environment,
) -> Result<Value, EvalError> {
    // Check arg count
    if arg_nodes.len() != func.params.len() {
        return Err(EvalError::TypeError(format!(
            "Function '{}' expects {} argument(s), got {}",
            name,
            func.params.len(),
            arg_nodes.len()
        )));
    }

    // Phase 1: Validate mutable args and evaluate all arguments in caller's scope
    let mut arg_vals = vec![];
    let mut caller_var_names: Vec<Option<String>> = vec![];
    for (i, arg_node) in arg_nodes.iter().enumerate() {
        let (_, _, is_mut) = &func.params[i];
        if *is_mut {
            match arg_node {
                Node::Identifier(var_name) => {
                    let info = env
                        .get_info(var_name)
                        .ok_or_else(|| EvalError::UndefinedVariable(var_name.clone()))?;
                    if !info.mutable {
                        return Err(EvalError::TypeError(format!(
                            "cannot pass immutable variable '{}' as mutable argument",
                            var_name
                        )));
                    }
                    caller_var_names.push(Some(var_name.clone()));
                }
                _ => {
                    return Err(EvalError::TypeError(
                        "mutable argument must be a variable".to_string(),
                    ));
                }
            }
        } else {
            caller_var_names.push(None);
        }
        let val = eval(arg_node, env)?;
        arg_vals.push(val);
    }

    // Phase 2: If generic, infer type variable bindings and substitute
    let resolved_params: Vec<(String, TypeAnn, bool)>;
    let resolved_return_type: Option<TypeAnn>;

    if !func.type_params.is_empty() {
        let mut bindings = HashMap::new();
        for (i, (_, param_type, _)) in func.params.iter().enumerate() {
            let arg_type = infer_type(&arg_vals[i], &format!("arg {} of '{}'", i, name))?;
            infer_type_var_bindings(param_type, &arg_type, &mut bindings)
                .map_err(EvalError::TypeError)?;
        }
        // Check all type params are bound
        for tp in &func.type_params {
            if !bindings.contains_key(tp) {
                return Err(EvalError::TypeError(format!(
                    "Could not infer type variable '{}' in call to '{}'",
                    tp, name
                )));
            }
        }
        resolved_params = func
            .params
            .iter()
            .map(|(n, t, m)| (n.clone(), substitute_type(t, &bindings), *m))
            .collect();
        resolved_return_type = func
            .return_type
            .as_ref()
            .map(|rt| substitute_type(rt, &bindings));
    } else {
        resolved_params = func.params.clone();
        resolved_return_type = func.return_type.clone();
    }

    // Phase 3: Validate and coerce args against resolved types
    for (i, (_, param_type, _)) in resolved_params.iter().enumerate() {
        validate_type(&arg_vals[i], param_type, &resolved_params[i].0)?;
        arg_vals[i] = coerce_if_needed(arg_vals[i].clone(), param_type);
    }

    // Save caller's non-global scopes for lexical scoping
    let saved = env.save_caller_scopes();

    // Push function scope and bind parameters
    env.push_scope();
    for (i, (pname, ptype, is_mut)) in resolved_params.iter().enumerate() {
        env.declare(pname.clone(), arg_vals[i].clone(), *is_mut, ptype.clone())
            .map_err(EvalError::TypeError)?;
    }

    // Evaluate function body
    let mut result = Value::Unit;
    let mut err = None;
    for stmt in &func.body {
        match eval_stmt_inner(stmt, env) {
            Ok(val) => result = val,
            Err(EvalError::Return(val)) => {
                result = *val;
                break;
            }
            Err(e) => {
                err = Some(e);
                break;
            }
        }
    }

    // Collect writeback data for mutable params before popping scope
    let mut writeback: Vec<(String, Value)> = vec![];
    if err.is_none() {
        for (i, (pname, _, is_mut)) in resolved_params.iter().enumerate() {
            if *is_mut
                && let Some(val) = env.get(pname)
                && let Some(ref var_name) = caller_var_names[i]
            {
                writeback.push((var_name.clone(), val));
            }
        }
    }

    // Pop function scope and restore caller scopes
    env.pop_scope();
    env.restore_caller_scopes(saved);

    if let Some(e) = err {
        return Err(e);
    }

    // Write back mutable args to caller's variables
    for (var_name, val) in writeback {
        env.reassign(&var_name, val).map_err(EvalError::TypeError)?;
    }

    // Validate return type
    if let Some(ref rt) = resolved_return_type {
        validate_type(&result, rt, &format!("return value of '{}'", name))?;
        let result = coerce_if_needed(result, rt);
        return Ok(result);
    }

    Ok(result)
}

fn eval_closure_call(
    name: &str,
    params: &[LambdaParam],
    body: &Node,
    captured_env: &Environment,
    arg_nodes: &[Node],
    caller_env: &mut Environment,
) -> Result<Value, EvalError> {
    // Check arg count
    if arg_nodes.len() != params.len() {
        return Err(EvalError::TypeError(format!(
            "Closure '{}' expects {} argument(s), got {}",
            name,
            params.len(),
            arg_nodes.len()
        )));
    }

    // Validate mutable args and evaluate arguments in caller's scope
    let mut arg_vals = vec![];
    let mut caller_var_names: Vec<Option<String>> = vec![];
    for (i, arg_node) in arg_nodes.iter().enumerate() {
        let param = &params[i];
        if param.mutable {
            match arg_node {
                Node::Identifier(var_name) => {
                    let info = caller_env
                        .get_info(var_name)
                        .ok_or_else(|| EvalError::UndefinedVariable(var_name.clone()))?;
                    if !info.mutable {
                        return Err(EvalError::TypeError(format!(
                            "cannot pass immutable variable '{}' as mutable argument",
                            var_name
                        )));
                    }
                    caller_var_names.push(Some(var_name.clone()));
                }
                _ => {
                    return Err(EvalError::TypeError(
                        "mutable argument must be a variable".to_string(),
                    ));
                }
            }
        } else {
            caller_var_names.push(None);
        }
        let val = eval(arg_node, caller_env)?;
        if let Some(ref type_ann) = param.type_ann {
            validate_type(&val, type_ann, &param.name)?;
        }
        let val = if let Some(ref type_ann) = param.type_ann {
            coerce_if_needed(val, type_ann)
        } else {
            val
        };
        arg_vals.push(val);
    }

    // Clone captured environment for this call
    let mut closure_env = captured_env.clone();

    // Push scope and bind parameters
    closure_env.push_scope();
    for (i, param) in params.iter().enumerate() {
        let type_ann = param
            .type_ann
            .clone()
            .unwrap_or_else(|| infer_type(&arg_vals[i], &param.name).unwrap_or(TypeAnn::Int32));
        closure_env
            .declare(
                param.name.clone(),
                arg_vals[i].clone(),
                param.mutable,
                type_ann,
            )
            .map_err(EvalError::TypeError)?;
    }

    // Evaluate body
    let result = match eval(body, &mut closure_env) {
        Ok(val) => val,
        Err(EvalError::Return(val)) => *val,
        Err(e) => {
            return Err(e);
        }
    };

    // Collect writeback data for mutable params
    let mut writeback: Vec<(String, Value)> = vec![];
    for (i, param) in params.iter().enumerate() {
        if param.mutable
            && let Some(val) = closure_env.get(&param.name)
            && let Some(ref var_name) = caller_var_names[i]
        {
            writeback.push((var_name.clone(), val));
        }
    }

    // Write back mutable args to caller's variables
    for (var_name, val) in writeback {
        caller_env
            .reassign(&var_name, val)
            .map_err(EvalError::TypeError)?;
    }

    Ok(result)
}

fn eval_declaration(
    name: &str,
    mutable: bool,
    type_ann: Option<&TypeAnn>,
    value_node: &Node,
    env: &mut Environment,
) -> Result<Value, EvalError> {
    let val = eval(value_node, env)?;

    // Determine the effective type
    let effective_type = match type_ann {
        Some(ann) => {
            validate_type(&val, ann, name)?;
            ann.clone()
        }
        None => infer_type(&val, name)?,
    };

    // Coerce int literal to uint32 if annotated as uint32
    let final_val = coerce_if_needed(val, &effective_type);

    env.declare(name.to_string(), final_val, mutable, effective_type)
        .map_err(EvalError::TypeError)?;

    Ok(Value::Unit)
}

fn eval_destructuring_decl(
    mutable: bool,
    bindings: &[(String, Option<String>)],
    type_name: &str,
    value_node: &Node,
    env: &mut Environment,
) -> Result<Value, EvalError> {
    let val = eval(value_node, env)?;

    // Check if type_name is a struct
    if let Some(_struct_def) = env.get_struct(type_name).cloned() {
        // Struct destructuring
        let (struct_name, fields) = match &val {
            Value::Struct { name, fields, .. } if name == type_name => {
                (name.clone(), fields.clone())
            }
            _ => {
                return Err(EvalError::TypeError(format!(
                    "Cannot destructure: expected {}, got {}",
                    type_name,
                    val.type_name()
                )));
            }
        };

        for (field_name, rename) in bindings {
            let var_name = rename.as_deref().unwrap_or(field_name);
            let field_val = fields
                .iter()
                .find(|(n, _)| n == field_name)
                .ok_or_else(|| {
                    EvalError::TypeError(format!(
                        "Struct '{}' has no field '{}'",
                        struct_name, field_name
                    ))
                })?
                .1
                .clone();
            let field_type = infer_type(&field_val, var_name)?;
            env.declare(var_name.to_string(), field_val, mutable, field_type)
                .map_err(EvalError::TypeError)?;
        }
        Ok(Value::Unit)
    }
    // Check if type_name is an enum variant
    else if env.get_enum_for_variant(type_name).is_some() {
        // Enum variant destructuring
        if bindings.len() != 1 {
            return Err(EvalError::TypeError(format!(
                "Enum variant destructuring requires exactly 1 binding, got {}",
                bindings.len()
            )));
        }
        let var_name = bindings[0].1.as_deref().unwrap_or(&bindings[0].0);

        match &val {
            Value::EnumVariant {
                variant_name,
                payload,
                ..
            } if variant_name == type_name => {
                let payload_val = payload.as_ref().ok_or_else(|| {
                    EvalError::TypeError(format!(
                        "Variant '{}' has no payload to destructure",
                        type_name
                    ))
                })?;
                let payload_type = infer_type(payload_val, var_name)?;
                env.declare(
                    var_name.to_string(),
                    *payload_val.clone(),
                    mutable,
                    payload_type,
                )
                .map_err(EvalError::TypeError)?;
                Ok(Value::Unit)
            }
            Value::EnumVariant { variant_name, .. } => Err(EvalError::TypeError(format!(
                "Cannot destructure: expected variant '{}', got '{}'",
                type_name, variant_name
            ))),
            _ => Err(EvalError::TypeError(format!(
                "Cannot destructure as '{}': value is not an enum variant",
                type_name
            ))),
        }
    } else {
        Err(EvalError::TypeError(format!(
            "Unknown type '{}' in destructuring",
            type_name
        )))
    }
}

fn eval_reassignment(
    name: &str,
    value_node: &Node,
    env: &mut Environment,
) -> Result<Value, EvalError> {
    let val = eval(value_node, env)?;

    // Get target type for coercion
    let target_type = env
        .get_info(name)
        .ok_or_else(|| EvalError::UndefinedVariable(name.to_string()))?
        .type_ann
        .clone();

    validate_type(&val, &target_type, name)?;
    let final_val = coerce_if_needed(val, &target_type);

    env.reassign(name, final_val)
        .map_err(EvalError::TypeError)?;

    Ok(Value::Unit)
}

fn validate_type(val: &Value, ann: &TypeAnn, var_name: &str) -> Result<(), EvalError> {
    match (val, ann) {
        (Value::Int(_), TypeAnn::Int32) => Ok(()),
        (Value::Int(n), TypeAnn::UInt32) => {
            if *n < 0 {
                Err(EvalError::TypeError(format!(
                    "Cannot assign negative value {} to uint32 variable '{}'",
                    n, var_name
                )))
            } else {
                Ok(())
            }
        }
        (Value::UInt(_), TypeAnn::UInt32) => Ok(()),
        (Value::Float(_), TypeAnn::Fl64) => Ok(()),
        (Value::Bool(_), TypeAnn::Bool) => Ok(()),
        (Value::Str(_), TypeAnn::Str) => Ok(()),
        (Value::Closure { params, .. }, TypeAnn::Fn { param_types, .. }) => {
            if params.len() != param_types.len() {
                return Err(EvalError::TypeError(format!(
                    "Type mismatch for '{}': expected function with {} param(s), got closure with {}",
                    var_name,
                    param_types.len(),
                    params.len()
                )));
            }
            for (i, (param, expected_type)) in params.iter().zip(param_types.iter()).enumerate() {
                if let Some(ref actual_type) = param.type_ann
                    && actual_type != expected_type
                {
                    return Err(EvalError::TypeError(format!(
                        "Type mismatch for '{}' param {}: expected {}, got {}",
                        var_name, i, expected_type, actual_type
                    )));
                }
            }
            Ok(())
        }
        (Value::List(elems), TypeAnn::Generic { name, type_params }) if name == "List" => {
            if type_params.len() != 1 {
                return Err(EvalError::TypeError(format!(
                    "List type requires exactly 1 type parameter, got {}",
                    type_params.len()
                )));
            }
            let elem_type = &type_params[0];
            for (i, elem) in elems.iter().enumerate() {
                validate_type(elem, elem_type, &format!("{}[{}]", var_name, i))?;
            }
            Ok(())
        }
        (Value::Struct { name: val_name, .. }, TypeAnn::Named(ann_name)) => {
            if val_name == ann_name {
                Ok(())
            } else {
                Err(EvalError::TypeError(format!(
                    "Type mismatch for '{}': expected {}, got {}",
                    var_name, ann_name, val_name
                )))
            }
        }
        (Value::Struct { name: val_name, .. }, TypeAnn::Generic { name: ann_name, .. }) => {
            if val_name == ann_name {
                Ok(())
            } else {
                Err(EvalError::TypeError(format!(
                    "Type mismatch for '{}': expected <...>{}, got {}",
                    var_name, ann_name, val_name
                )))
            }
        }
        (Value::EnumVariant { enum_name, .. }, TypeAnn::Named(ann_name)) => {
            if enum_name == ann_name {
                Ok(())
            } else {
                Err(EvalError::TypeError(format!(
                    "Type mismatch for '{}': expected {}, got {}",
                    var_name, ann_name, enum_name
                )))
            }
        }
        (Value::EnumVariant { enum_name, .. }, TypeAnn::Generic { name: ann_name, .. }) => {
            if enum_name == ann_name {
                Ok(())
            } else {
                Err(EvalError::TypeError(format!(
                    "Type mismatch for '{}': expected <...>{}, got {}",
                    var_name, ann_name, enum_name
                )))
            }
        }
        (Value::Range { .. }, TypeAnn::Named(name)) if name == "Range" => Ok(()),
        (_, TypeAnn::Named(tv)) => Err(EvalError::TypeError(format!(
            "Unresolved type variable '{}' for '{}'",
            tv, var_name
        ))),
        (_, TypeAnn::Generic { name, .. }) => Err(EvalError::TypeError(format!(
            "Type mismatch for '{}': expected <...>{}, got {}",
            var_name,
            name,
            val.type_name()
        ))),
        _ => Err(EvalError::TypeError(format!(
            "Type mismatch for '{}': expected {}, got {}",
            var_name,
            ann,
            val.type_name()
        ))),
    }
}

fn infer_type(val: &Value, var_name: &str) -> Result<TypeAnn, EvalError> {
    match val {
        Value::Int(_) => Ok(TypeAnn::Int32),
        Value::UInt(_) => Ok(TypeAnn::UInt32),
        Value::Float(_) => Ok(TypeAnn::Fl64),
        Value::Bool(_) => Ok(TypeAnn::Bool),
        Value::Str(_) => Ok(TypeAnn::Str),
        Value::Unit => Err(EvalError::TypeError(format!(
            "Cannot infer type of unit value for variable '{}'",
            var_name
        ))),
        Value::Closure { params, .. } => {
            let mut param_types = vec![];
            for param in params {
                match &param.type_ann {
                    Some(t) => param_types.push(t.clone()),
                    None => {
                        return Err(EvalError::TypeError(format!(
                            "Cannot infer type of closure for '{}': param '{}' has no type annotation",
                            var_name, param.name
                        )));
                    }
                }
            }
            Ok(TypeAnn::Fn {
                param_types,
                return_type: None,
            })
        }
        Value::List(elems) => {
            if elems.is_empty() {
                Err(EvalError::TypeError(format!(
                    "Cannot infer type of empty list for '{}'; add a type annotation",
                    var_name
                )))
            } else {
                let elem_type = infer_type(&elems[0], &format!("{}[0]", var_name))?;
                Ok(TypeAnn::Generic {
                    name: "List".to_string(),
                    type_params: vec![elem_type],
                })
            }
        }
        Value::Struct {
            name, type_params, ..
        } => {
            if type_params.is_empty() {
                Ok(TypeAnn::Named(name.clone()))
            } else {
                Ok(TypeAnn::Generic {
                    name: name.clone(),
                    type_params: type_params.clone(),
                })
            }
        }
        Value::EnumVariant {
            enum_name,
            type_params,
            ..
        } => {
            if type_params.is_empty() {
                Ok(TypeAnn::Named(enum_name.clone()))
            } else {
                Ok(TypeAnn::Generic {
                    name: enum_name.clone(),
                    type_params: type_params.clone(),
                })
            }
        }
        Value::Range { .. } => Ok(TypeAnn::Named("Range".to_string())),
    }
}

fn coerce_if_needed(val: Value, target: &TypeAnn) -> Value {
    match (&val, target) {
        (Value::Int(n), TypeAnn::UInt32) => Value::UInt(*n as u32),
        (Value::List(elems), TypeAnn::Generic { name, type_params })
            if name == "List" && type_params.len() == 1 =>
        {
            let elem_type = &type_params[0];
            Value::List(
                elems
                    .iter()
                    .map(|e| coerce_if_needed(e.clone(), elem_type))
                    .collect(),
            )
        }
        // Fill in type_params for unresolved generic enum variants from annotation
        (
            Value::EnumVariant {
                enum_name,
                variant_name,
                type_params,
                payload,
            },
            TypeAnn::Generic {
                name,
                type_params: ann_tp,
            },
        ) if type_params.is_empty() && enum_name == name => Value::EnumVariant {
            enum_name: enum_name.clone(),
            variant_name: variant_name.clone(),
            type_params: ann_tp.clone(),
            payload: payload.clone(),
        },
        _ => val,
    }
}

fn substitute_type(type_ann: &TypeAnn, bindings: &HashMap<String, TypeAnn>) -> TypeAnn {
    match type_ann {
        TypeAnn::Named(name) => {
            if let Some(bound) = bindings.get(name) {
                bound.clone()
            } else {
                type_ann.clone()
            }
        }
        TypeAnn::Generic { name, type_params } => TypeAnn::Generic {
            name: name.clone(),
            type_params: type_params
                .iter()
                .map(|tp| substitute_type(tp, bindings))
                .collect(),
        },
        TypeAnn::Fn {
            param_types,
            return_type,
        } => TypeAnn::Fn {
            param_types: param_types
                .iter()
                .map(|pt| substitute_type(pt, bindings))
                .collect(),
            return_type: return_type
                .as_ref()
                .map(|rt| Box::new(substitute_type(rt, bindings))),
        },
        _ => type_ann.clone(),
    }
}

fn infer_type_var_bindings(
    param_type: &TypeAnn,
    arg_type: &TypeAnn,
    bindings: &mut HashMap<String, TypeAnn>,
) -> Result<(), String> {
    match (param_type, arg_type) {
        (TypeAnn::Named(tv), _) => {
            if let Some(existing) = bindings.get(tv) {
                if existing != arg_type {
                    return Err(format!(
                        "Type variable '{}' bound to both {} and {}",
                        tv, existing, arg_type
                    ));
                }
            } else {
                bindings.insert(tv.clone(), arg_type.clone());
            }
            Ok(())
        }
        (
            TypeAnn::Generic {
                name: pn,
                type_params: ptp,
            },
            TypeAnn::Generic {
                name: an,
                type_params: atp,
            },
        ) => {
            if pn != an || ptp.len() != atp.len() {
                return Err(format!(
                    "Generic type mismatch: expected <...>{}, got <...>{}",
                    pn, an
                ));
            }
            for (pt, at) in ptp.iter().zip(atp.iter()) {
                infer_type_var_bindings(pt, at, bindings)?;
            }
            Ok(())
        }
        (
            TypeAnn::Fn {
                param_types: ppt,
                return_type: prt,
            },
            TypeAnn::Fn {
                param_types: apt,
                return_type: art,
            },
        ) => {
            if ppt.len() != apt.len() {
                return Err("Function type param count mismatch".to_string());
            }
            for (pt, at) in ppt.iter().zip(apt.iter()) {
                infer_type_var_bindings(pt, at, bindings)?;
            }
            if let (Some(pr), Some(ar)) = (prt, art) {
                infer_type_var_bindings(pr, ar, bindings)?;
            }
            Ok(())
        }
        (TypeAnn::Int32, TypeAnn::Int32)
        | (TypeAnn::UInt32, TypeAnn::UInt32)
        | (TypeAnn::Fl64, TypeAnn::Fl64)
        | (TypeAnn::Bool, TypeAnn::Bool)
        | (TypeAnn::Str, TypeAnn::Str) => Ok(()),
        // Allow int->uint coercion during inference
        (TypeAnn::UInt32, TypeAnn::Int32) => Ok(()),
        _ => Err(format!(
            "Type structure mismatch: expected {}, got {}",
            param_type, arg_type
        )),
    }
}

/// Execute a statement and return optional result (declarations/reassignments return None)
pub fn eval_stmt(stmt: &Statement, env: &mut Environment) -> Result<Option<Value>, EvalError> {
    match stmt {
        Statement::Expr(node) => eval(node, env).map(Some),
        Statement::Declaration {
            mutable,
            name,
            type_ann,
            value,
        } => {
            eval_declaration(name, *mutable, type_ann.as_ref(), value, env)?;
            Ok(None)
        }
        Statement::Reassignment { name, value } => {
            eval_reassignment(name, value, env)?;
            Ok(None)
        }
        Statement::FnDeclaration {
            name,
            type_params,
            params,
            return_type,
            body,
        } => {
            eval_fn_declaration(name, type_params, params, return_type, body, env)?;
            Ok(None)
        }
        Statement::StructDeclaration {
            name,
            type_params,
            fields,
        } => {
            eval_struct_declaration(name, type_params, fields, env)?;
            Ok(None)
        }
        Statement::EnumDeclaration {
            name,
            type_params,
            variants,
        } => {
            eval_enum_declaration(name, type_params, variants, env)?;
            Ok(None)
        }
        Statement::MethodsDeclaration {
            type_name,
            type_params,
            methods,
        } => {
            eval_methods_declaration(type_name, type_params, methods, env)?;
            Ok(None)
        }
        Statement::TraitImplDeclaration {
            type_name,
            type_params,
            trait_name,
            methods,
        } => {
            eval_trait_impl_declaration(type_name, type_params, trait_name, methods, env)?;
            Ok(None)
        }
        Statement::DestructuringDecl {
            mutable,
            bindings,
            type_name,
            value,
        } => {
            eval_destructuring_decl(*mutable, bindings, type_name, value, env)?;
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer() {
        let mut env = Environment::new();
        assert_eq!(eval(&Node::Int(42), &mut env), Ok(Value::Int(42)));
    }

    #[test]
    fn test_negative() {
        let mut env = Environment::new();
        assert_eq!(
            eval(
                &Node::PrefixOp {
                    op: PrefixOp::Negative,
                    child: Box::new(Node::Int(5))
                },
                &mut env
            ),
            Ok(Value::Int(-5))
        );
    }

    #[test]
    fn test_addition() {
        let mut env = Environment::new();
        assert_eq!(
            eval(
                &Node::BinaryOp {
                    lhs: Box::new(Node::Int(3)),
                    rhs: Box::new(Node::Int(4)),
                    op: BinaryOp::Add
                },
                &mut env
            ),
            Ok(Value::Int(7))
        );
    }

    #[test]
    fn test_division_by_zero() {
        let mut env = Environment::new();
        let result = eval(
            &Node::BinaryOp {
                lhs: Box::new(Node::Int(5)),
                rhs: Box::new(Node::Int(0)),
                op: BinaryOp::Div,
            },
            &mut env,
        );
        assert!(matches!(result, Err(EvalError::DivisionByZero)));
    }

    #[test]
    fn test_power() {
        let mut env = Environment::new();
        assert_eq!(
            eval(
                &Node::BinaryOp {
                    lhs: Box::new(Node::Int(2)),
                    rhs: Box::new(Node::Int(3)),
                    op: BinaryOp::Pow
                },
                &mut env
            ),
            Ok(Value::Int(8))
        );
    }

    #[test]
    fn test_complex_expression() {
        // (3 + 4) * 2 = 14
        let expr = Node::BinaryOp {
            lhs: Box::new(Node::BinaryOp {
                lhs: Box::new(Node::Int(3)),
                rhs: Box::new(Node::Int(4)),
                op: BinaryOp::Add,
            }),
            rhs: Box::new(Node::Int(2)),
            op: BinaryOp::Mul,
        };
        let mut env = Environment::new();
        assert_eq!(eval(&expr, &mut env), Ok(Value::Int(14)));
    }

    #[test]
    fn test_bool_literal() {
        let mut env = Environment::new();
        assert_eq!(eval(&Node::Bool(true), &mut env), Ok(Value::Bool(true)));
        assert_eq!(eval(&Node::Bool(false), &mut env), Ok(Value::Bool(false)));
    }

    #[test]
    fn test_not() {
        let mut env = Environment::new();
        assert_eq!(
            eval(
                &Node::PrefixOp {
                    op: PrefixOp::Not,
                    child: Box::new(Node::Bool(true))
                },
                &mut env
            ),
            Ok(Value::Bool(false))
        );
        assert_eq!(
            eval(
                &Node::PrefixOp {
                    op: PrefixOp::Not,
                    child: Box::new(Node::Bool(false))
                },
                &mut env
            ),
            Ok(Value::Bool(true))
        );
    }

    #[test]
    fn test_and() {
        let mut env = Environment::new();
        assert_eq!(
            eval(
                &Node::BinaryOp {
                    lhs: Box::new(Node::Bool(true)),
                    rhs: Box::new(Node::Bool(true)),
                    op: BinaryOp::And
                },
                &mut env
            ),
            Ok(Value::Bool(true))
        );
        assert_eq!(
            eval(
                &Node::BinaryOp {
                    lhs: Box::new(Node::Bool(true)),
                    rhs: Box::new(Node::Bool(false)),
                    op: BinaryOp::And
                },
                &mut env
            ),
            Ok(Value::Bool(false))
        );
    }

    #[test]
    fn test_or() {
        let mut env = Environment::new();
        assert_eq!(
            eval(
                &Node::BinaryOp {
                    lhs: Box::new(Node::Bool(false)),
                    rhs: Box::new(Node::Bool(false)),
                    op: BinaryOp::Or
                },
                &mut env
            ),
            Ok(Value::Bool(false))
        );
        assert_eq!(
            eval(
                &Node::BinaryOp {
                    lhs: Box::new(Node::Bool(false)),
                    rhs: Box::new(Node::Bool(true)),
                    op: BinaryOp::Or
                },
                &mut env
            ),
            Ok(Value::Bool(true))
        );
    }

    #[test]
    fn test_type_error_not_on_int() {
        let mut env = Environment::new();
        let result = eval(
            &Node::PrefixOp {
                op: PrefixOp::Not,
                child: Box::new(Node::Int(5)),
            },
            &mut env,
        );
        assert!(matches!(result, Err(EvalError::TypeError(_))));
    }

    #[test]
    fn test_type_error_negative_on_bool() {
        let mut env = Environment::new();
        let result = eval(
            &Node::PrefixOp {
                op: PrefixOp::Negative,
                child: Box::new(Node::Bool(true)),
            },
            &mut env,
        );
        assert!(matches!(result, Err(EvalError::TypeError(_))));
    }

    #[test]
    fn test_type_error_and_on_ints() {
        let mut env = Environment::new();
        let result = eval(
            &Node::BinaryOp {
                lhs: Box::new(Node::Int(1)),
                rhs: Box::new(Node::Int(2)),
                op: BinaryOp::And,
            },
            &mut env,
        );
        assert!(matches!(result, Err(EvalError::TypeError(_))));
    }

    #[test]
    fn test_eq() {
        let mut env = Environment::new();
        assert_eq!(
            eval(
                &Node::BinaryOp {
                    lhs: Box::new(Node::Int(3)),
                    rhs: Box::new(Node::Int(3)),
                    op: BinaryOp::Eq
                },
                &mut env
            ),
            Ok(Value::Bool(true))
        );
        assert_eq!(
            eval(
                &Node::BinaryOp {
                    lhs: Box::new(Node::Int(3)),
                    rhs: Box::new(Node::Int(4)),
                    op: BinaryOp::Eq
                },
                &mut env
            ),
            Ok(Value::Bool(false))
        );
        // Bool equality
        assert_eq!(
            eval(
                &Node::BinaryOp {
                    lhs: Box::new(Node::Bool(true)),
                    rhs: Box::new(Node::Bool(true)),
                    op: BinaryOp::Eq
                },
                &mut env
            ),
            Ok(Value::Bool(true))
        );
        // Cross-type inequality
        assert_eq!(
            eval(
                &Node::BinaryOp {
                    lhs: Box::new(Node::Int(1)),
                    rhs: Box::new(Node::Bool(true)),
                    op: BinaryOp::Eq
                },
                &mut env
            ),
            Ok(Value::Bool(false))
        );
    }

    #[test]
    fn test_ordering_comparisons() {
        let mut env = Environment::new();
        assert_eq!(
            eval(
                &Node::BinaryOp {
                    lhs: Box::new(Node::Int(3)),
                    rhs: Box::new(Node::Int(5)),
                    op: BinaryOp::Lt
                },
                &mut env
            ),
            Ok(Value::Bool(true))
        );
        assert_eq!(
            eval(
                &Node::BinaryOp {
                    lhs: Box::new(Node::Int(5)),
                    rhs: Box::new(Node::Int(3)),
                    op: BinaryOp::Gt
                },
                &mut env
            ),
            Ok(Value::Bool(true))
        );
        assert_eq!(
            eval(
                &Node::BinaryOp {
                    lhs: Box::new(Node::Int(3)),
                    rhs: Box::new(Node::Int(3)),
                    op: BinaryOp::Lte
                },
                &mut env
            ),
            Ok(Value::Bool(true))
        );
        assert_eq!(
            eval(
                &Node::BinaryOp {
                    lhs: Box::new(Node::Int(3)),
                    rhs: Box::new(Node::Int(3)),
                    op: BinaryOp::Gte
                },
                &mut env
            ),
            Ok(Value::Bool(true))
        );
    }

    #[test]
    fn test_type_error_lt_on_bools() {
        let mut env = Environment::new();
        let result = eval(
            &Node::BinaryOp {
                lhs: Box::new(Node::Bool(true)),
                rhs: Box::new(Node::Bool(false)),
                op: BinaryOp::Lt,
            },
            &mut env,
        );
        assert!(matches!(result, Err(EvalError::TypeError(_))));
    }

    #[test]
    fn test_if_true() {
        let mut env = Environment::new();
        let node = Node::IfElse {
            condition: Box::new(Node::Bool(true)),
            then_block: Box::new(Node::Block(vec![Statement::Expr(Node::Int(42))])),
            else_block: Some(Box::new(Node::Block(vec![Statement::Expr(Node::Int(0))]))),
        };
        assert_eq!(eval(&node, &mut env), Ok(Value::Int(42)));
    }

    #[test]
    fn test_if_false() {
        let mut env = Environment::new();
        let node = Node::IfElse {
            condition: Box::new(Node::Bool(false)),
            then_block: Box::new(Node::Block(vec![Statement::Expr(Node::Int(42))])),
            else_block: Some(Box::new(Node::Block(vec![Statement::Expr(Node::Int(0))]))),
        };
        assert_eq!(eval(&node, &mut env), Ok(Value::Int(0)));
    }

    #[test]
    fn test_if_no_else_false() {
        let mut env = Environment::new();
        let node = Node::IfElse {
            condition: Box::new(Node::Bool(false)),
            then_block: Box::new(Node::Block(vec![Statement::Expr(Node::Int(42))])),
            else_block: None,
        };
        assert_eq!(eval(&node, &mut env), Ok(Value::Unit));
    }

    #[test]
    fn test_block_returns_last() {
        let mut env = Environment::new();
        let node = Node::Block(vec![
            Statement::Expr(Node::Int(1)),
            Statement::Expr(Node::Int(2)),
            Statement::Expr(Node::Int(3)),
        ]);
        assert_eq!(eval(&node, &mut env), Ok(Value::Int(3)));
    }

    #[test]
    fn test_block_with_declaration() {
        let mut env = Environment::new();
        let node = Node::Block(vec![
            Statement::Declaration {
                mutable: false,
                name: "x".to_string(),
                type_ann: None,
                value: Node::Int(10),
            },
            Statement::Expr(Node::Identifier("x".to_string())),
        ]);
        assert_eq!(eval(&node, &mut env), Ok(Value::Int(10)));
    }

    #[test]
    fn test_if_type_error() {
        let mut env = Environment::new();
        let node = Node::IfElse {
            condition: Box::new(Node::Int(1)),
            then_block: Box::new(Node::Block(vec![Statement::Expr(Node::Int(42))])),
            else_block: None,
        };
        assert!(matches!(
            eval(&node, &mut env),
            Err(EvalError::TypeError(_))
        ));
    }

    #[test]
    fn test_loop_break() {
        let mut env = Environment::new();
        env.set("i".to_string(), Value::Int(0));
        // loop { i == 3 if { break }; i = i + 1 }
        let node = Node::Loop(Box::new(Node::Block(vec![
            Statement::Expr(Node::IfElse {
                condition: Box::new(Node::BinaryOp {
                    lhs: Box::new(Node::Identifier("i".to_string())),
                    rhs: Box::new(Node::Int(3)),
                    op: BinaryOp::Eq,
                }),
                then_block: Box::new(Node::Block(vec![Statement::Expr(Node::Break)])),
                else_block: None,
            }),
            Statement::Reassignment {
                name: "i".to_string(),
                value: Node::BinaryOp {
                    lhs: Box::new(Node::Identifier("i".to_string())),
                    rhs: Box::new(Node::Int(1)),
                    op: BinaryOp::Add,
                },
            },
        ])));
        assert_eq!(eval(&node, &mut env), Ok(Value::Unit));
        assert_eq!(env.get("i"), Some(Value::Int(3)));
    }

    #[test]
    fn test_while_loop() {
        let mut env = Environment::new();
        env.set("x".to_string(), Value::Int(5));
        env.set("sum".to_string(), Value::Int(0));
        // x > 0 while { sum = sum + x; x = x - 1 }
        let node = Node::While {
            condition: Box::new(Node::BinaryOp {
                lhs: Box::new(Node::Identifier("x".to_string())),
                rhs: Box::new(Node::Int(0)),
                op: BinaryOp::Gt,
            }),
            body: Box::new(Node::Block(vec![
                Statement::Reassignment {
                    name: "sum".to_string(),
                    value: Node::BinaryOp {
                        lhs: Box::new(Node::Identifier("sum".to_string())),
                        rhs: Box::new(Node::Identifier("x".to_string())),
                        op: BinaryOp::Add,
                    },
                },
                Statement::Reassignment {
                    name: "x".to_string(),
                    value: Node::BinaryOp {
                        lhs: Box::new(Node::Identifier("x".to_string())),
                        rhs: Box::new(Node::Int(1)),
                        op: BinaryOp::Sub,
                    },
                },
            ])),
        };
        assert_eq!(eval(&node, &mut env), Ok(Value::Unit));
        assert_eq!(env.get("sum"), Some(Value::Int(15)));
        assert_eq!(env.get("x"), Some(Value::Int(0)));
    }

    #[test]
    fn test_while_type_error() {
        let mut env = Environment::new();
        let node = Node::While {
            condition: Box::new(Node::Int(1)),
            body: Box::new(Node::Block(vec![])),
        };
        assert!(matches!(
            eval(&node, &mut env),
            Err(EvalError::TypeError(_))
        ));
    }

    #[test]
    fn test_break_outside_loop() {
        let mut env = Environment::new();
        assert_eq!(eval(&Node::Break, &mut env), Err(EvalError::Break));
    }

    #[test]
    fn test_continue_outside_loop() {
        let mut env = Environment::new();
        assert_eq!(eval(&Node::Continue, &mut env), Err(EvalError::Continue));
    }

    #[test]
    fn test_loop_continue() {
        let mut env = Environment::new();
        env.set("i".to_string(), Value::Int(0));
        env.set("sum".to_string(), Value::Int(0));
        // loop { i = i + 1; i == 5 if { break }; i == 3 if { continue }; sum = sum + i }
        // Should sum 1+2+4 = 7 (skips 3)
        let node = Node::Loop(Box::new(Node::Block(vec![
            Statement::Reassignment {
                name: "i".to_string(),
                value: Node::BinaryOp {
                    lhs: Box::new(Node::Identifier("i".to_string())),
                    rhs: Box::new(Node::Int(1)),
                    op: BinaryOp::Add,
                },
            },
            Statement::Expr(Node::IfElse {
                condition: Box::new(Node::BinaryOp {
                    lhs: Box::new(Node::Identifier("i".to_string())),
                    rhs: Box::new(Node::Int(5)),
                    op: BinaryOp::Eq,
                }),
                then_block: Box::new(Node::Block(vec![Statement::Expr(Node::Break)])),
                else_block: None,
            }),
            Statement::Expr(Node::IfElse {
                condition: Box::new(Node::BinaryOp {
                    lhs: Box::new(Node::Identifier("i".to_string())),
                    rhs: Box::new(Node::Int(3)),
                    op: BinaryOp::Eq,
                }),
                then_block: Box::new(Node::Block(vec![Statement::Expr(Node::Continue)])),
                else_block: None,
            }),
            Statement::Reassignment {
                name: "sum".to_string(),
                value: Node::BinaryOp {
                    lhs: Box::new(Node::Identifier("sum".to_string())),
                    rhs: Box::new(Node::Identifier("i".to_string())),
                    op: BinaryOp::Add,
                },
            },
        ])));
        assert_eq!(eval(&node, &mut env), Ok(Value::Unit));
        assert_eq!(env.get("sum"), Some(Value::Int(7))); // 1+2+4
    }

    #[test]
    fn test_const_declaration() {
        let mut env = Environment::new();
        let stmt = Statement::Declaration {
            mutable: false,
            name: "x".to_string(),
            type_ann: Some(TypeAnn::Int32),
            value: Node::Int(42),
        };
        assert_eq!(eval_stmt(&stmt, &mut env), Ok(None));
        assert_eq!(env.get("x"), Some(Value::Int(42)));
    }

    #[test]
    fn test_mut_and_reassign() {
        let mut env = Environment::new();
        let decl = Statement::Declaration {
            mutable: true,
            name: "x".to_string(),
            type_ann: Some(TypeAnn::Int32),
            value: Node::Int(5),
        };
        eval_stmt(&decl, &mut env).unwrap();

        let reassign = Statement::Reassignment {
            name: "x".to_string(),
            value: Node::Int(10),
        };
        eval_stmt(&reassign, &mut env).unwrap();
        assert_eq!(env.get("x"), Some(Value::Int(10)));
    }

    #[test]
    fn test_const_reassign_error() {
        let mut env = Environment::new();
        let decl = Statement::Declaration {
            mutable: false,
            name: "x".to_string(),
            type_ann: None,
            value: Node::Int(5),
        };
        eval_stmt(&decl, &mut env).unwrap();

        let reassign = Statement::Reassignment {
            name: "x".to_string(),
            value: Node::Int(10),
        };
        assert!(eval_stmt(&reassign, &mut env).is_err());
    }

    #[test]
    fn test_reassign_undeclared_error() {
        let mut env = Environment::new();
        let reassign = Statement::Reassignment {
            name: "x".to_string(),
            value: Node::Int(10),
        };
        assert!(eval_stmt(&reassign, &mut env).is_err());
    }

    #[test]
    fn test_redeclaration_error() {
        let mut env = Environment::new();
        let decl = Statement::Declaration {
            mutable: false,
            name: "x".to_string(),
            type_ann: None,
            value: Node::Int(5),
        };
        eval_stmt(&decl, &mut env).unwrap();

        let decl2 = Statement::Declaration {
            mutable: false,
            name: "x".to_string(),
            type_ann: None,
            value: Node::Int(10),
        };
        assert!(eval_stmt(&decl2, &mut env).is_err());
    }

    #[test]
    fn test_uint32_declaration() {
        let mut env = Environment::new();
        let stmt = Statement::Declaration {
            mutable: false,
            name: "x".to_string(),
            type_ann: Some(TypeAnn::UInt32),
            value: Node::Int(42), // int literal coerced to uint32
        };
        eval_stmt(&stmt, &mut env).unwrap();
        assert_eq!(env.get("x"), Some(Value::UInt(42)));
    }

    #[test]
    fn test_uint32_negative_error() {
        let mut env = Environment::new();
        let stmt = Statement::Declaration {
            mutable: false,
            name: "x".to_string(),
            type_ann: Some(TypeAnn::UInt32),
            value: Node::PrefixOp {
                op: PrefixOp::Negative,
                child: Box::new(Node::Int(5)),
            },
        };
        assert!(eval_stmt(&stmt, &mut env).is_err());
    }

    #[test]
    fn test_uint32_arithmetic() {
        let mut env = Environment::new();
        env.declare("a".to_string(), Value::UInt(10), false, TypeAnn::UInt32)
            .unwrap();
        env.declare("b".to_string(), Value::UInt(3), false, TypeAnn::UInt32)
            .unwrap();

        let node = Node::BinaryOp {
            lhs: Box::new(Node::Identifier("a".to_string())),
            rhs: Box::new(Node::Identifier("b".to_string())),
            op: BinaryOp::Add,
        };
        assert_eq!(eval(&node, &mut env), Ok(Value::UInt(13)));
    }

    #[test]
    fn test_mixed_type_arithmetic_error() {
        let mut env = Environment::new();
        env.declare("a".to_string(), Value::Int(10), false, TypeAnn::Int32)
            .unwrap();
        env.declare("b".to_string(), Value::UInt(3), false, TypeAnn::UInt32)
            .unwrap();

        let node = Node::BinaryOp {
            lhs: Box::new(Node::Identifier("a".to_string())),
            rhs: Box::new(Node::Identifier("b".to_string())),
            op: BinaryOp::Add,
        };
        assert!(matches!(
            eval(&node, &mut env),
            Err(EvalError::TypeError(_))
        ));
    }

    #[test]
    fn test_type_mismatch_reassignment() {
        let mut env = Environment::new();
        let decl = Statement::Declaration {
            mutable: true,
            name: "x".to_string(),
            type_ann: Some(TypeAnn::Int32),
            value: Node::Int(5),
        };
        eval_stmt(&decl, &mut env).unwrap();

        let reassign = Statement::Reassignment {
            name: "x".to_string(),
            value: Node::Bool(true),
        };
        assert!(eval_stmt(&reassign, &mut env).is_err());
    }

    #[test]
    fn test_type_inference() {
        let mut env = Environment::new();
        let decl_int = Statement::Declaration {
            mutable: false,
            name: "x".to_string(),
            type_ann: None,
            value: Node::Int(5),
        };
        eval_stmt(&decl_int, &mut env).unwrap();
        assert_eq!(env.get("x"), Some(Value::Int(5)));

        let decl_bool = Statement::Declaration {
            mutable: false,
            name: "b".to_string(),
            type_ann: None,
            value: Node::Bool(true),
        };
        eval_stmt(&decl_bool, &mut env).unwrap();
        assert_eq!(env.get("b"), Some(Value::Bool(true)));
    }

    #[test]
    fn test_declaration_type_mismatch_error() {
        let mut env = Environment::new();
        let stmt = Statement::Declaration {
            mutable: false,
            name: "x".to_string(),
            type_ann: Some(TypeAnn::Int32),
            value: Node::Bool(true),
        };
        assert!(eval_stmt(&stmt, &mut env).is_err());
    }

    #[test]
    fn test_full_declaration_and_reassignment() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("mut x: int32 = 5\nx = x + 1").unwrap();
        for stmt in &stmts {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(env.get("x"), Some(Value::Int(6)));
    }

    #[test]
    fn test_uint32_end_to_end() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: uint32 = 42").unwrap();
        for stmt in &stmts {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(env.get("x"), Some(Value::UInt(42)));
    }

    #[test]
    fn test_while_with_declarations() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src =
            "mut x: int32 = 5\nmut sum: int32 = 0\nx > 0 while {\nsum = sum + x\nx = x - 1\n}";
        let stmts = parse(src).unwrap();
        for stmt in &stmts {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(env.get("sum"), Some(Value::Int(15)));
    }

    #[test]
    fn test_modulo() {
        let mut env = Environment::new();
        // 7 % 3 = 1
        let node = Node::BinaryOp {
            lhs: Box::new(Node::Int(7)),
            rhs: Box::new(Node::Int(3)),
            op: BinaryOp::Mod,
        };
        assert_eq!(eval(&node, &mut env), Ok(Value::Int(1)));

        // 10 % 5 = 0
        let node = Node::BinaryOp {
            lhs: Box::new(Node::Int(10)),
            rhs: Box::new(Node::Int(5)),
            op: BinaryOp::Mod,
        };
        assert_eq!(eval(&node, &mut env), Ok(Value::Int(0)));
    }

    #[test]
    fn test_modulo_python_style() {
        let mut env = Environment::new();
        // Python: -7 % 3 = 2 (not -1 like C/Rust)
        let node = Node::BinaryOp {
            lhs: Box::new(Node::PrefixOp {
                op: PrefixOp::Negative,
                child: Box::new(Node::Int(7)),
            }),
            rhs: Box::new(Node::Int(3)),
            op: BinaryOp::Mod,
        };
        assert_eq!(eval(&node, &mut env), Ok(Value::Int(2)));

        // Python: 7 % -3 = -2
        let node = Node::BinaryOp {
            lhs: Box::new(Node::Int(7)),
            rhs: Box::new(Node::PrefixOp {
                op: PrefixOp::Negative,
                child: Box::new(Node::Int(3)),
            }),
            op: BinaryOp::Mod,
        };
        assert_eq!(eval(&node, &mut env), Ok(Value::Int(-2)));

        // Python: -7 % -3 = -1
        let node = Node::BinaryOp {
            lhs: Box::new(Node::PrefixOp {
                op: PrefixOp::Negative,
                child: Box::new(Node::Int(7)),
            }),
            rhs: Box::new(Node::PrefixOp {
                op: PrefixOp::Negative,
                child: Box::new(Node::Int(3)),
            }),
            op: BinaryOp::Mod,
        };
        assert_eq!(eval(&node, &mut env), Ok(Value::Int(-1)));
    }

    #[test]
    fn test_modulo_division_by_zero() {
        let mut env = Environment::new();
        let node = Node::BinaryOp {
            lhs: Box::new(Node::Int(5)),
            rhs: Box::new(Node::Int(0)),
            op: BinaryOp::Mod,
        };
        assert!(matches!(
            eval(&node, &mut env),
            Err(EvalError::DivisionByZero)
        ));
    }

    #[test]
    fn test_modulo_uint32() {
        let mut env = Environment::new();
        env.declare("a".to_string(), Value::UInt(10), false, TypeAnn::UInt32)
            .unwrap();
        env.declare("b".to_string(), Value::UInt(3), false, TypeAnn::UInt32)
            .unwrap();

        let node = Node::BinaryOp {
            lhs: Box::new(Node::Identifier("a".to_string())),
            rhs: Box::new(Node::Identifier("b".to_string())),
            op: BinaryOp::Mod,
        };
        assert_eq!(eval(&node, &mut env), Ok(Value::UInt(1)));
    }

    #[test]
    fn test_compound_assignment_end_to_end() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("mut x: int32 = 10\nx += 5").unwrap();
        for stmt in &stmts {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(env.get("x"), Some(Value::Int(15)));
    }

    #[test]
    fn test_compound_sub_assign() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("mut x: int32 = 10\nx -= 3").unwrap();
        for stmt in &stmts {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(env.get("x"), Some(Value::Int(7)));
    }

    #[test]
    fn test_compound_mod_assign() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("mut x: int32 = 7\nx %= 3").unwrap();
        for stmt in &stmts {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(env.get("x"), Some(Value::Int(1)));
    }

    #[test]
    fn test_compound_assign_const_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: int32 = 10\nx += 5").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_modulo_end_to_end() {
        use crate::evaluator::eval;
        use crate::parser::parse;
        let mut env = Environment::new();
        // -7 % 3 = 2 (Python-style)
        let stmts = parse("-7 % 3").unwrap();
        match &stmts[0] {
            Statement::Expr(node) => assert_eq!(eval(node, &mut env), Ok(Value::Int(2))),
            _ => panic!("Expected Expr"),
        }
    }

    #[test]
    fn test_simple_function() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(x: int32)double: int32 fn { x + x }\n(5)double").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        let result = eval_stmt(&stmts[1], &mut env);
        assert_eq!(result, Ok(Some(Value::Int(10))));
    }

    #[test]
    fn test_function_no_params() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("()answer: int32 fn { 42 }\n()answer").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Int(42))));
    }

    #[test]
    fn test_recursive_factorial() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse(
            "(n: int32)fact: int32 fn {\nn <= 1 if { 1 } else { n * (n - 1)fact }\n}\n(5)fact",
        )
        .unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Int(120))));
    }

    #[test]
    fn test_return_keyword() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse(
            "(n: int32)early: int32 fn {\nn <= 0 if { 0 return }\nn + 1\n}\n(5)early\n(-1)early",
        )
        .unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Int(6))));
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Int(0))));
    }

    #[test]
    fn test_wrong_arg_count() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(x: int32)f fn { x }\n(1, 2)f").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_block_scoping() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Variable declared inside if block should not be visible outside
        let stmts = parse("true if {\nconst x = 10\nx\n}").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(env.get("x"), None);
    }

    #[test]
    fn test_function_scoping() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Function should not see caller's local variables
        let stmts = parse("()f: int32 fn { y }\nmut y: int32 = 99\n()f").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        // y is declared at global scope, so function CAN see it
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Int(99))));
    }

    #[test]
    fn test_function_cant_see_caller_locals() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Declare function, then call it from inside a block where a local var exists
        // The function should NOT see the caller's local scope
        let stmts = parse("()f: int32 fn { z }").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        // Create a local scope with z, then call f
        let stmts2 = parse("true if {\nmut z: int32 = 42\n()f\n}").unwrap();
        let result = eval_stmt(&stmts2[0], &mut env);
        assert!(result.is_err()); // f can't see z
    }

    #[test]
    fn test_function_modifies_global() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("mut g: int32 = 0\n()inc fn { g += 1 }\n()inc\n()inc").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        eval_stmt(&stmts[2], &mut env).unwrap();
        eval_stmt(&stmts[3], &mut env).unwrap();
        assert_eq!(env.get("g"), Some(Value::Int(2)));
    }

    #[test]
    fn test_function_calling_function() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(x: int32)double: int32 fn { x + x }\n(x: int32)quadruple: int32 fn { (x)double + (x)double }\n(3)quadruple").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Int(12))));
    }

    #[test]
    fn test_function_args_immutable() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Attempting to reassign a function parameter should fail
        let stmts = parse("(x: int32)f fn { x = 5 }\n(1)f").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_string_literal() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("\"hello\"").unwrap();
        match &stmts[0] {
            Statement::Expr(node) => {
                assert_eq!(eval(node, &mut env), Ok(Value::Str("hello".to_string())))
            }
            _ => panic!("Expected Expr"),
        }
    }

    #[test]
    fn test_string_concatenation() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("\"hello\" + \" world\"").unwrap();
        match &stmts[0] {
            Statement::Expr(node) => assert_eq!(
                eval(node, &mut env),
                Ok(Value::Str("hello world".to_string()))
            ),
            _ => panic!("Expected Expr"),
        }
    }

    #[test]
    fn test_string_comparison() {
        use crate::parser::parse;
        let mut env = Environment::new();

        let stmts = parse("\"abc\" == \"abc\"").unwrap();
        match &stmts[0] {
            Statement::Expr(node) => assert_eq!(eval(node, &mut env), Ok(Value::Bool(true))),
            _ => panic!("Expected Expr"),
        }

        let stmts = parse("\"a\" < \"b\"").unwrap();
        match &stmts[0] {
            Statement::Expr(node) => assert_eq!(eval(node, &mut env), Ok(Value::Bool(true))),
            _ => panic!("Expected Expr"),
        }

        let stmts = parse("\"abc\" != \"def\"").unwrap();
        match &stmts[0] {
            Statement::Expr(node) => assert_eq!(eval(node, &mut env), Ok(Value::Bool(true))),
            _ => panic!("Expected Expr"),
        }
    }

    #[test]
    fn test_string_type_annotation() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const s: str = \"hi\"").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(env.get("s"), Some(Value::Str("hi".to_string())));
    }

    #[test]
    fn test_string_type_mismatch() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const s: str = 42").unwrap();
        assert!(eval_stmt(&stmts[0], &mut env).is_err());
    }

    #[test]
    fn test_string_concat_type_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("\"hi\" + 5").unwrap();
        match &stmts[0] {
            Statement::Expr(node) => assert!(eval(node, &mut env).is_err()),
            _ => panic!("Expected Expr"),
        }
    }

    #[test]
    fn test_string_concat_assign() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("mut s: str = \"hello\"\ns += \" world\"").unwrap();
        for stmt in &stmts {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(env.get("s"), Some(Value::Str("hello world".to_string())));
    }

    #[test]
    fn test_lnprint_builtin() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(\"hello\")lnprint").unwrap();
        // lnprint returns Unit
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Unit)));
    }

    #[test]
    fn test_print_wrong_args() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(1, 2)print").unwrap();
        assert!(eval_stmt(&stmts[0], &mut env).is_err());
    }

    #[test]
    fn test_mutable_arg_writeback() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("mut x: int32 = 5\n(mut n: int32)inc fn { n += 1 }\n(x)inc").unwrap();
        for stmt in &stmts {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(env.get("x"), Some(Value::Int(6)));
    }

    #[test]
    fn test_mutable_arg_string() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts =
            parse("mut s: str = \"hello\"\n(mut t: str)append fn { t += \" world\" }\n(s)append")
                .unwrap();
        for stmt in &stmts {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(env.get("s"), Some(Value::Str("hello world".to_string())));
    }

    #[test]
    fn test_mutable_arg_rejects_literal() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(mut n: int32)inc fn { n += 1 }\n(5)inc").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_mutable_arg_rejects_immutable() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(mut n: int32)inc fn { n += 1 }\nconst x: int32 = 5\n(x)inc").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert!(eval_stmt(&stmts[2], &mut env).is_err());
    }

    #[test]
    fn test_mutable_arg_rejects_expression() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(mut n: int32)inc fn { n += 1 }\nmut x: int32 = 5\n(x + 1)inc").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert!(eval_stmt(&stmts[2], &mut env).is_err());
    }

    #[test]
    fn test_immutable_param_no_writeback() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Non-mut param should NOT write back (existing behavior)
        let stmts = parse("mut x: int32 = 5\n(n: int32)f: int32 fn { n + 1 }\n(x)f").unwrap();
        for stmt in &stmts {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(env.get("x"), Some(Value::Int(5)));
    }

    #[test]
    fn test_mixed_mut_immut_params() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("mut a: int32 = 10\nmut b: int32 = 20\n(mut x: int32, y: int32)add_to fn { x += y }\n(a, b)add_to").unwrap();
        for stmt in &stmts {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(env.get("a"), Some(Value::Int(30))); // a was mut param, written back
        assert_eq!(env.get("b"), Some(Value::Int(20))); // b was immut param, unchanged
    }

    #[test]
    fn test_basic_lambda_call() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const f = \\ x: int32 => x + 1\n(5)f").unwrap();
        for stmt in &stmts {
            eval_stmt(stmt, &mut env).unwrap();
        }
        // Last statement is the call, check result
        let stmts = parse("(5)f").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Int(6))));
    }

    #[test]
    fn test_lambda_zero_params() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const f = \\ => 42\n()f").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Int(42))));
    }

    #[test]
    fn test_lambda_multi_params() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const add = \\ x: int32, y: int32 => x + y\n(3, 4)add").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Int(7))));
    }

    #[test]
    fn test_lambda_block_body() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const f = \\ x: int32 => {\nconst y = x * 2\ny + 1\n}\n(5)f").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Int(11))));
    }

    #[test]
    fn test_lambda_copy_capture() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Lambda captures a=10, then we change a to 99, but closure still uses a=10
        let stmts =
            parse("mut a: int32 = 10\nconst g = \\ x: int32 => x + a\na = 99\n(5)g").unwrap();
        for stmt in &stmts[0..3] {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(eval_stmt(&stmts[3], &mut env), Ok(Some(Value::Int(15)))); // 5 + 10, not 5 + 99
    }

    #[test]
    fn test_lambda_mut_param_writeback() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts =
            parse("mut x: int32 = 5\nconst inc = \\ mut n: int32 => { n += 1 }\n(x)inc").unwrap();
        for stmt in &stmts {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(env.get("x"), Some(Value::Int(6)));
    }

    #[test]
    fn test_lambda_higher_order() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Pass a lambda to a function that takes a function type param
        let stmts = parse("(f: (int32): int32, x: int32)apply: int32 fn { (x)f }\nconst inc = \\ n: int32 => n + 1\n(inc, 5)apply").unwrap();
        for stmt in &stmts[0..2] {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Int(6))));
    }

    #[test]
    fn test_lambda_nested_currying() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // \ x: int32 => \ y: int32 => x + y
        let stmts =
            parse("const add = \\ x: int32 => \\ y: int32 => x + y\nconst add3 = (3)add\n(4)add3")
                .unwrap();
        for stmt in &stmts[0..2] {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Int(7))));
    }

    #[test]
    fn test_lambda_wrong_arg_count() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const f = \\ x: int32 => x\n(1, 2)f").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_lambda_type_mismatch() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const f = \\ x: int32 => x\n(true)f").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_call_non_closure_variable() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: int32 = 5\n(1)x").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_lambda_with_fn_type_annotation() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const f: (int32): int32 = \\ x: int32 => x + 1\n(5)f").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Int(6))));
    }

    #[test]
    fn test_lambda_fn_type_param_count_mismatch() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Annotated as taking 2 params, but lambda has 1
        let stmts = parse("const f: (int32, int32): int32 = \\ x: int32 => x");
        assert!(stmts.is_ok());
        let stmts = stmts.unwrap();
        assert!(eval_stmt(&stmts[0], &mut env).is_err());
    }

    // === List tests ===

    #[test]
    fn test_list_creation() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const xs = [1, 2, 3]").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(
            env.get("xs"),
            Some(Value::List(vec![
                Value::Int(1),
                Value::Int(2),
                Value::Int(3),
            ]))
        );
    }

    #[test]
    fn test_list_with_type_annotation() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const xs: <int32>List = [1, 2, 3]").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(
            env.get("xs"),
            Some(Value::List(vec![
                Value::Int(1),
                Value::Int(2),
                Value::Int(3),
            ]))
        );
    }

    #[test]
    fn test_list_type_sugar() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const xs: [int32] = [1, 2, 3]").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(
            env.get("xs"),
            Some(Value::List(vec![
                Value::Int(1),
                Value::Int(2),
                Value::Int(3),
            ]))
        );
    }

    #[test]
    fn test_list_mixed_types_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[1, true, 3]").unwrap();
        match &stmts[0] {
            Statement::Expr(node) => assert!(eval(node, &mut env).is_err()),
            _ => panic!("Expected Expr"),
        }
    }

    #[test]
    fn test_empty_list_with_type() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const xs: <int32>List = []").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(env.get("xs"), Some(Value::List(vec![])));
    }

    #[test]
    fn test_empty_list_no_type_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const xs = []").unwrap();
        assert!(eval_stmt(&stmts[0], &mut env).is_err());
    }

    #[test]
    fn test_list_type_mismatch() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Annotated as <str>List but contains ints
        let stmts = parse("const xs: <str>List = [1, 2]").unwrap();
        assert!(eval_stmt(&stmts[0], &mut env).is_err());
    }

    #[test]
    fn test_len_list() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("([1, 2, 3])len").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::UInt(3))));
    }

    #[test]
    fn test_len_empty_list() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("([])len").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::UInt(0))));
    }

    #[test]
    fn test_len_string() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(\"hello\")len").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::UInt(5))));
    }

    #[test]
    fn test_list_equality() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[1, 2] == [1, 2]").unwrap();
        match &stmts[0] {
            Statement::Expr(node) => assert_eq!(eval(node, &mut env), Ok(Value::Bool(true))),
            _ => panic!("Expected Expr"),
        }
        let stmts = parse("[1, 2] == [1, 3]").unwrap();
        match &stmts[0] {
            Statement::Expr(node) => assert_eq!(eval(node, &mut env), Ok(Value::Bool(false))),
            _ => panic!("Expected Expr"),
        }
    }

    #[test]
    fn test_print_list() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("([1, 2, 3])lnprint").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Unit)));
    }

    #[test]
    fn test_list_uint_coercion() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const xs: <uint32>List = [1, 2, 3]").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(
            env.get("xs"),
            Some(Value::List(vec![
                Value::UInt(1),
                Value::UInt(2),
                Value::UInt(3),
            ]))
        );
    }

    // === Generic function tests ===

    #[test]
    fn test_generic_identity() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts =
            parse("<T>(x: T)identity: T fn { x }\n(42)identity\n(true)identity\n(\"hi\")identity")
                .unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Int(42))));
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Bool(true))));
        assert_eq!(
            eval_stmt(&stmts[3], &mut env),
            Ok(Some(Value::Str("hi".to_string())))
        );
    }

    #[test]
    fn test_generic_multi_type_params() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("<T, U>(x: T, y: U)first: T fn { x }\n(1, true)first").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Int(1))));
    }

    #[test]
    fn test_generic_inconsistent_type_var() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // T is used for both params, but args have different types
        let stmts = parse("<T>(x: T, y: T)same: T fn { x }\n(1, true)same").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_generic_fn_with_list() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts =
            parse("<T>(xs: <T>List)listlen: uint32 fn { (xs)len }\n([1, 2, 3])listlen").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::UInt(3))));
    }

    #[test]
    fn test_generic_return_type_validation() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Return type T should be validated after substitution
        let stmts = parse("<T>(x: T)identity: T fn { x }\n(42)identity").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Int(42))));
    }

    #[test]
    fn test_generic_non_generic_fn_unchanged() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Non-generic function should still work exactly as before
        let stmts = parse("(x: int32)double: int32 fn { x + x }\n(5)double").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Int(10))));
    }

    #[test]
    fn test_struct_create_and_access() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts =
            parse("Point struct { x: int32, y: int32 }\nconst p = (x: 1, y: 2)Point\np.x").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Int(1))));
    }

    #[test]
    fn test_struct_field_access_y() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts =
            parse("Point struct { x: int32, y: int32 }\nconst p = (x: 10, y: 20)Point\np.y")
                .unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Int(20))));
    }

    #[test]
    fn test_struct_type_annotation() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts =
            parse("Point struct { x: int32, y: int32 }\nconst p: Point = (x: 1, y: 2)Point\np.x")
                .unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Int(1))));
    }

    #[test]
    fn test_struct_wrong_field_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Point struct { x: int32, y: int32 }\n(x: 1, z: 2)Point").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_struct_missing_field_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Point struct { x: int32, y: int32 }\n(x: 1)Point").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_struct_field_type_mismatch() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Point struct { x: int32, y: int32 }\n(x: 1, y: true)Point").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_struct_unknown_field_access() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts =
            parse("Point struct { x: int32, y: int32 }\nconst p = (x: 1, y: 2)Point\np.z").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert!(eval_stmt(&stmts[2], &mut env).is_err());
    }

    #[test]
    fn test_enum_no_payload_variant() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Color enum { Red, Green, Blue }\nRed").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(
            eval_stmt(&stmts[1], &mut env),
            Ok(Some(Value::EnumVariant {
                enum_name: "Color".to_string(),
                variant_name: "Red".to_string(),
                type_params: vec![],
                payload: None,
            }))
        );
    }

    #[test]
    fn test_enum_payload_variant() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Shape enum { (int32)Circle, (int32)Square }\n(5)Circle").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(
            eval_stmt(&stmts[1], &mut env),
            Ok(Some(Value::EnumVariant {
                enum_name: "Shape".to_string(),
                variant_name: "Circle".to_string(),
                type_params: vec![],
                payload: Some(Box::new(Value::Int(5))),
            }))
        );
    }

    #[test]
    fn test_enum_payload_type_mismatch() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Shape enum { (int32)Circle }\n(true)Circle").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_enum_wrong_payload_presence() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Variant takes no payload but we give one
        let stmts = parse("Color enum { Red }\n(42)Red").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_enum_missing_payload() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Variant requires payload but we give none
        let stmts = parse("Shape enum { (int32)Circle }\nCircle").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_generic_struct() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts =
            parse("<T>Wrapper struct { value: T }\nconst w = (value: 42)Wrapper\nw.value").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Int(42))));
    }

    #[test]
    fn test_generic_enum_with_payload() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("<T>Option enum { (T)Some, Nothing }\n(42)Some").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(
            eval_stmt(&stmts[1], &mut env),
            Ok(Some(Value::EnumVariant {
                enum_name: "Option".to_string(),
                variant_name: "Some".to_string(),
                type_params: vec![TypeAnn::Int32],
                payload: Some(Box::new(Value::Int(42))),
            }))
        );
    }

    #[test]
    fn test_generic_enum_no_payload_partial_type() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Generic enum variant with no payload produces value with empty type_params
        let stmts = parse("<T>Option enum { (T)Some, Nothing }\nNothing").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(
            eval_stmt(&stmts[1], &mut env),
            Ok(Some(Value::EnumVariant {
                enum_name: "Option".to_string(),
                variant_name: "Nothing".to_string(),
                type_params: vec![],
                payload: None,
            }))
        );
    }

    #[test]
    fn test_struct_equality() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts =
            parse("Point struct { x: int32, y: int32 }\n(x: 1, y: 2)Point == (x: 1, y: 2)Point")
                .unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Bool(true))));
    }

    #[test]
    fn test_struct_inequality() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts =
            parse("Point struct { x: int32, y: int32 }\n(x: 1, y: 2)Point == (x: 1, y: 3)Point")
                .unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Bool(false))));
    }

    #[test]
    fn test_enum_equality() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Color enum { Red, Green }\nRed == Red").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Bool(true))));
    }

    #[test]
    fn test_enum_inequality() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Color enum { Red, Green }\nRed == Green").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Bool(false))));
    }

    #[test]
    fn test_print_struct() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Point struct { x: int32, y: int32 }\n(x: 1, y: 2)Point").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        let result = eval_stmt(&stmts[1], &mut env).unwrap().unwrap();
        assert_eq!(format!("{}", result), "(x: 1, y: 2)Point");
    }

    #[test]
    fn test_print_enum() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Color enum { Red, (int32)Custom }\nRed").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        let result = eval_stmt(&stmts[1], &mut env).unwrap().unwrap();
        assert_eq!(format!("{}", result), "Red");
    }

    #[test]
    fn test_print_enum_with_payload() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Color enum { Red, (int32)Custom }\n(42)Custom").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        let result = eval_stmt(&stmts[1], &mut env).unwrap().unwrap();
        assert_eq!(format!("{}", result), "(42)Custom");
    }

    #[test]
    fn test_zero_field_struct() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Unit struct { }\n()Unit").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(
            eval_stmt(&stmts[1], &mut env),
            Ok(Some(Value::Struct {
                name: "Unit".to_string(),
                type_params: vec![],
                fields: vec![],
            }))
        );
    }

    // === Method tests ===

    #[test]
    fn test_basic_method_call() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "Point struct { x: int32, y: int32 }\nPoint methods {\n(self: Point)get_x: int32 fn { self.x }\n}\nconst p = (x: 10, y: 20)Point\np.()get_x";
        let stmts = parse(src).unwrap();
        for stmt in &stmts[..3] {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(eval_stmt(&stmts[3], &mut env), Ok(Some(Value::Int(10))));
    }

    #[test]
    fn test_method_with_args() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "Point struct { x: int32, y: int32 }\nPoint methods {\n(self: Point, n: int32)scale: Point fn {\n(x: self.x * n, y: self.y * n)Point\n}\n}\nconst p = (x: 3, y: 4)Point\np.(2)scale";
        let stmts = parse(src).unwrap();
        for stmt in &stmts[..3] {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(
            eval_stmt(&stmts[3], &mut env),
            Ok(Some(Value::Struct {
                name: "Point".to_string(),
                type_params: vec![],
                fields: vec![
                    ("x".to_string(), Value::Int(6)),
                    ("y".to_string(), Value::Int(8)),
                ],
            }))
        );
    }

    #[test]
    fn test_method_on_enum() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "Color enum { Red, Green, Blue }\nColor methods {\n(self: Color)is_red: bool fn {\nself == Red\n}\n}\nconst c = Red\nc.()is_red";
        let stmts = parse(src).unwrap();
        for stmt in &stmts[..3] {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(eval_stmt(&stmts[3], &mut env), Ok(Some(Value::Bool(true))));
    }

    #[test]
    fn test_method_on_generic_type() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "<T>Wrapper struct { value: T }\n<T>Wrapper methods {\n(self: <T>Wrapper)get: T fn { self.value }\n}\nconst w = (value: 42)Wrapper\nw.()get";
        let stmts = parse(src).unwrap();
        for stmt in &stmts[..3] {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(eval_stmt(&stmts[3], &mut env), Ok(Some(Value::Int(42))));
    }

    #[test]
    fn test_method_calling_another_method() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "Point struct { x: int32, y: int32 }\nPoint methods {\n(self: Point)get_x: int32 fn { self.x }\n(self: Point)get_x_doubled: int32 fn { self.()get_x * 2 }\n}\nconst p = (x: 5, y: 10)Point\np.()get_x_doubled";
        let stmts = parse(src).unwrap();
        for stmt in &stmts[..3] {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(eval_stmt(&stmts[3], &mut env), Ok(Some(Value::Int(10))));
    }

    #[test]
    fn test_method_chain_with_field() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "Point struct { x: int32, y: int32 }\nPoint methods {\n(self: Point, n: int32)scale: Point fn {\n(x: self.x * n, y: self.y * n)Point\n}\n}\nconst p = (x: 3, y: 4)Point\np.(2)scale.y";
        let stmts = parse(src).unwrap();
        for stmt in &stmts[..3] {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(eval_stmt(&stmts[3], &mut env), Ok(Some(Value::Int(8))));
    }

    #[test]
    fn test_methods_on_unknown_type_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "Foo methods {\n(self: Foo)bar fn { 1 }\n}";
        let stmts = parse(src).unwrap();
        assert!(eval_stmt(&stmts[0], &mut env).is_err());
    }

    #[test]
    fn test_method_first_param_not_self_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "Point struct { x: int32, y: int32 }\nPoint methods {\n(other: Point)get_x: int32 fn { other.x }\n}";
        let stmts = parse(src).unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_nonexistent_method_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src =
            "Point struct { x: int32, y: int32 }\nconst p = (x: 1, y: 2)Point\np.()nonexistent";
        let stmts = parse(src).unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert!(eval_stmt(&stmts[2], &mut env).is_err());
    }

    #[test]
    fn test_method_wrong_arg_count_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "Point struct { x: int32, y: int32 }\nPoint methods {\n(self: Point)get_x: int32 fn { self.x }\n}\nconst p = (x: 1, y: 2)Point\np.(42)get_x";
        let stmts = parse(src).unwrap();
        for stmt in &stmts[..3] {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert!(eval_stmt(&stmts[3], &mut env).is_err());
    }

    // === Optional type (Maybe / Attempt) tests ===

    #[test]
    fn test_exists_variant() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Exists").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::EnumVariant {
                enum_name: "Maybe".to_string(),
                variant_name: "Exists".to_string(),
                type_params: vec![TypeAnn::Int32],
                payload: Some(Box::new(Value::Int(42))),
            }))
        );
    }

    #[test]
    fn test_does_not_exist_with_annotation() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: int32? = DoesNotExist").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(
            env.get("x"),
            Some(Value::EnumVariant {
                enum_name: "Maybe".to_string(),
                variant_name: "DoesNotExist".to_string(),
                type_params: vec![TypeAnn::Int32],
                payload: None,
            })
        );
    }

    #[test]
    fn test_does_not_exist_no_annotation() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Without annotation, DoesNotExist is partially typed (empty type_params)
        let stmts = parse("const x = DoesNotExist").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(
            env.get("x"),
            Some(Value::EnumVariant {
                enum_name: "Maybe".to_string(),
                variant_name: "DoesNotExist".to_string(),
                type_params: vec![],
                payload: None,
            })
        );
    }

    #[test]
    fn test_maybe_equality() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Exists == (42)Exists").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Bool(true))));
        let stmts = parse("(42)Exists == (99)Exists").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Bool(false))));
    }

    #[test]
    fn test_success_variant() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: <int32, str>Attempt = (42)Success").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(
            env.get("x"),
            Some(Value::EnumVariant {
                enum_name: "Attempt".to_string(),
                variant_name: "Success".to_string(),
                type_params: vec![TypeAnn::Int32, TypeAnn::Str],
                payload: Some(Box::new(Value::Int(42))),
            })
        );
    }

    #[test]
    fn test_failure_variant() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: <int32, str>Attempt = (\"error\")Failure").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(
            env.get("x"),
            Some(Value::EnumVariant {
                enum_name: "Attempt".to_string(),
                variant_name: "Failure".to_string(),
                type_params: vec![TypeAnn::Int32, TypeAnn::Str],
                payload: Some(Box::new(Value::Str("error".to_string()))),
            })
        );
    }

    #[test]
    fn test_optional_type_sugar() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // int32? is sugar for <int32>Maybe
        let stmts = parse("const x: int32? = (5)Exists").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(
            env.get("x"),
            Some(Value::EnumVariant {
                enum_name: "Maybe".to_string(),
                variant_name: "Exists".to_string(),
                type_params: vec![TypeAnn::Int32],
                payload: Some(Box::new(Value::Int(5))),
            })
        );
    }

    #[test]
    fn test_compconst_basic() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("compconst pi = 42\npi + 1").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Int(43))));
    }

    #[test]
    fn test_compconst_immutable() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("compconst pi = 10\npi = 20").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_method_on_maybe() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "<T>Maybe methods {\n(self: <T>Maybe)is_exists: bool fn {\nself == DoesNotExist if { false } else { true }\n}\n}\nconst x: int32? = (42)Exists\nx.()is_exists";
        let stmts = parse(src).unwrap();
        for stmt in &stmts[..2] {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Bool(true))));
    }

    #[test]
    fn test_float_literal() {
        let mut env = Environment::new();
        assert_eq!(eval(&Node::Float(3.14), &mut env), Ok(Value::Float(3.14)));
    }

    #[test]
    fn test_float_arithmetic() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("1.5 + 2.5").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Float(4.0))));
        let stmts = parse("3.0 * 2.0").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Float(6.0))));
        let stmts = parse("5.0 - 1.5").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Float(3.5))));
    }

    #[test]
    fn test_float_division() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("7.0 / 2.0").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Float(3.5))));
    }

    #[test]
    fn test_int_division_unchanged() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("7 / 2").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Int(3))));
    }

    #[test]
    fn test_float_negation() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("-3.14").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::Float(-3.14)))
        );
    }

    #[test]
    fn test_float_comparison() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("1.5 < 2.5").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Bool(true))));
        let stmts = parse("3.0 == 3.0").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Bool(true))));
        let stmts = parse("2.0 >= 1.0").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Bool(true))));
    }

    #[test]
    fn test_mixed_arithmetic_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: fl64 = 1.5\nconst y: int32 = 1\nx + y").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert!(eval_stmt(&stmts[2], &mut env).is_err());
    }

    #[test]
    fn test_fl64_type_annotation() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: fl64 = 3.14\nx").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Float(3.14))));
    }

    #[test]
    fn test_float_division_by_zero() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("1.0 / 0.0").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Err(EvalError::DivisionByZero)
        );
    }

    #[test]
    fn test_float_modulo() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("5.5 % 2.0").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Float(1.5))));
    }

    #[test]
    fn test_float_power() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("2.0 ^ 3.0").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Float(8.0))));
    }

    #[test]
    fn test_cast_int_to_fl64() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x = 5\n(fl64 cast x)").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Float(5.0))));
    }

    #[test]
    fn test_cast_fl64_to_int() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(int32 cast 3.7)").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Int(3))));
    }

    #[test]
    fn test_cast_int_to_uint() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(uint32 cast 42)").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::UInt(42))));
    }

    #[test]
    fn test_cast_negative_to_uint_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x = -1\n(uint32 cast x)").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_cast_in_expression() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x = 5\n(fl64 cast x) * 2.0").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Float(10.0))));
    }

    #[test]
    fn test_cast_bool_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const b = true\n(int32 cast b)").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_cast_uint_to_fl64() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: uint32 = 10\n(fl64 cast x)").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Float(10.0))));
    }

    #[test]
    fn test_destructure_struct() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Point struct { x: int32, y: int32 }\nconst p = (x: 10, y: 20)Point\nconst (x: a, y: b)Point = p\na").unwrap();
        for s in &stmts[..3] {
            eval_stmt(s, &mut env).unwrap();
        }
        assert_eq!(eval_stmt(&stmts[3], &mut env), Ok(Some(Value::Int(10))));
    }

    #[test]
    fn test_destructure_struct_shorthand() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Point struct { x: int32, y: int32 }\nconst p = (x: 3, y: 7)Point\nconst (x, y)Point = p\nx + y").unwrap();
        for s in &stmts[..3] {
            eval_stmt(s, &mut env).unwrap();
        }
        assert_eq!(eval_stmt(&stmts[3], &mut env), Ok(Some(Value::Int(10))));
    }

    #[test]
    fn test_destructure_enum_payload() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const m: int32? = (42)Exists\nconst (v)Exists = m\nv").unwrap();
        for s in &stmts[..2] {
            eval_stmt(s, &mut env).unwrap();
        }
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Int(42))));
    }

    #[test]
    fn test_destructure_wrong_variant_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const m: int32? = DoesNotExist\nconst (v)Exists = m").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_destructure_unknown_field_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse(
            "Point struct { x: int32, y: int32 }\nconst p = (x: 1, y: 2)Point\nconst (z)Point = p",
        )
        .unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert!(eval_stmt(&stmts[2], &mut env).is_err());
    }

    #[test]
    fn test_destructure_mutable() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Point struct { x: int32, y: int32 }\nconst p = (x: 1, y: 2)Point\nmut (x, y)Point = p\nx = 99\nx").unwrap();
        for s in &stmts[..4] {
            eval_stmt(s, &mut env).unwrap();
        }
        assert_eq!(eval_stmt(&stmts[4], &mut env), Ok(Some(Value::Int(99))));
    }

    #[test]
    fn test_destructure_wrong_type_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts =
            parse("Point struct { x: int32, y: int32 }\nconst n = 42\nconst (x)Point = n").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert!(eval_stmt(&stmts[2], &mut env).is_err());
    }

    #[test]
    fn test_match_literal_int() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x = 2\nx match { 1 -> { \"one\" }, 2 -> { \"two\" }, _ -> { \"other\" } }").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Str("two".to_string()))));
    }

    #[test]
    fn test_match_literal_string() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const s = \"hi\"\ns match { \"hello\" -> { 1 }, \"hi\" -> { 2 }, _ -> { 0 } }").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Int(2))));
    }

    #[test]
    fn test_match_wildcard() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x = 99\nx match { _ -> { \"caught\" } }").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Str("caught".to_string()))));
    }

    #[test]
    fn test_match_enum_variant() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const m: int32? = (42)Exists\nm match { (v)Exists -> { v + 1 }, DoesNotExist -> { 0 } }").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Int(43))));
    }

    #[test]
    fn test_match_bare_variant() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const m: int32? = DoesNotExist\nm match { (v)Exists -> { v }, DoesNotExist -> { -1 } }").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Int(-1))));
    }

    #[test]
    fn test_match_expression_value() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x = 1\nconst r = x match { 1 -> { \"one\" }, _ -> { \"other\" } }\nr").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Str("one".to_string()))));
    }

    #[test]
    fn test_match_no_match_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x = 3\nx match { 1 -> { \"one\" }, 2 -> { \"two\" } }").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert!(eval_stmt(&stmts[1], &mut env).is_err());
    }

    #[test]
    fn test_match_first_arm_wins() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x = 1\nx match { 1 -> { \"first\" }, 1 -> { \"second\" }, _ -> { \"default\" } }").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Str("first".to_string()))));
    }

    // =: pattern binding tests

    #[test]
    fn test_if_let_match() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: int32? = (42)Exists\nx =: (v)Exists if { v } else { 0 }").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Int(42))));
    }

    #[test]
    fn test_if_let_no_match() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: int32? = DoesNotExist\nx =: (v)Exists if { v } else { 0 }").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Int(0))));
    }

    #[test]
    fn test_if_let_no_else_no_match() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: int32? = DoesNotExist\nx =: (v)Exists if { v }").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Unit)));
    }

    #[test]
    fn test_if_let_binding_scope() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // v should not be visible outside the if-let block
        let stmts = parse("const x: int32? = (42)Exists\nx =: (v)Exists if { v }\nv").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert!(eval_stmt(&stmts[2], &mut env).is_err());
    }

    #[test]
    fn test_while_let() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let program = "mut val: int32? = (3)Exists\nmut sum = 0\nval =: (n)Exists while {\nsum += n\nn > 1 if { val = (n - 1)Exists } else { val = DoesNotExist }\n}\nsum";
        let stmts = parse(program).unwrap();
        for stmt in &stmts[..stmts.len() - 1] {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(eval_stmt(stmts.last().unwrap(), &mut env), Ok(Some(Value::Int(6))));
    }

    #[test]
    fn test_pattern_test_match() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: int32? = (42)Exists\nx =: (v)Exists").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Bool(true))));
    }

    #[test]
    fn test_pattern_test_no_match() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: int32? = DoesNotExist\nx =: (v)Exists").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Bool(false))));
    }

    #[test]
    fn test_pattern_test_literal() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x = 5\nx =: 5").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Bool(true))));

        let stmts2 = parse("const y = 3\ny =: 5").unwrap();
        eval_stmt(&stmts2[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts2[1], &mut env), Ok(Some(Value::Bool(false))));
    }

    #[test]
    fn test_if_let_else_if_chain() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let program = "const x: int32? = DoesNotExist\nconst y: int32? = (99)Exists\nx =: (v)Exists if { v } else y =: (w)Exists if { w } else { 0 }";
        let stmts = parse(program).unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Int(99))));
    }

    // Range tests

    #[test]
    fn test_range_exclusive() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("1..5").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::Range {
                start: 1,
                end: 5,
                inclusive: false
            }))
        );
    }

    #[test]
    fn test_range_inclusive() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("1..=5").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::Range {
                start: 1,
                end: 5,
                inclusive: true
            }))
        );
    }

    #[test]
    fn test_range_len() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(1..5)len").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::UInt(4))));

        let stmts2 = parse("(1..=5)len").unwrap();
        assert_eq!(eval_stmt(&stmts2[0], &mut env), Ok(Some(Value::UInt(5))));
    }

    #[test]
    fn test_range_empty() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(5..3)len").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::UInt(0))));
    }

    #[test]
    fn test_range_equality() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(1..5) == (1..5)").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Bool(true))));

        let stmts2 = parse("(1..5) == (1..=4)").unwrap();
        assert_eq!(eval_stmt(&stmts2[0], &mut env), Ok(Some(Value::Bool(false))));
    }

    #[test]
    fn test_range_float_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("1.0..5.0").unwrap();
        assert!(eval_stmt(&stmts[0], &mut env).is_err());
    }

    #[test]
    fn test_range_uint32_bounds() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // uint32 bounds via len
        let stmts = parse("const xs = [10, 20, 30]\n0..(xs)len").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(
            eval_stmt(&stmts[1], &mut env),
            Ok(Some(Value::Range {
                start: 0,
                end: 3,
                inclusive: false,
            }))
        );
    }

    // Indexing tests

    #[test]
    fn test_list_index() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[10, 20, 30][1]").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Int(20))));
    }

    #[test]
    fn test_list_negative_index() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[10, 20, 30][-1]").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Int(30))));
    }

    #[test]
    fn test_list_out_of_bounds() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[10, 20, 30][3]").unwrap();
        assert!(eval_stmt(&stmts[0], &mut env).is_err());
    }

    #[test]
    fn test_string_index() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("\"hello\"[0]").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Str("h".to_string()))));
    }

    #[test]
    fn test_string_negative_index() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("\"hello\"[-1]").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Str("o".to_string()))));
    }

    #[test]
    fn test_range_index() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(1..=5)[2]").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Int(3))));
    }

    #[test]
    fn test_range_index_out_of_bounds() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(1..5)[4]").unwrap();
        assert!(eval_stmt(&stmts[0], &mut env).is_err());
    }

    // Slice tests

    #[test]
    fn test_list_slice_exclusive() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[10, 20, 30, 40, 50][1..4]").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![
                Value::Int(20),
                Value::Int(30),
                Value::Int(40),
            ])))
        );
    }

    #[test]
    fn test_list_slice_inclusive() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[10, 20, 30, 40, 50][1..=3]").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![
                Value::Int(20),
                Value::Int(30),
                Value::Int(40),
            ])))
        );
    }

    #[test]
    fn test_list_slice_from_start() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[10, 20, 30, 40][0..2]").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![Value::Int(10), Value::Int(20)])))
        );
    }

    #[test]
    fn test_list_slice_clamps() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // end beyond length clamps to length
        let stmts = parse("[10, 20, 30][1..100]").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![Value::Int(20), Value::Int(30)])))
        );
    }

    #[test]
    fn test_list_slice_empty() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[10, 20, 30][2..2]").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![])))
        );
    }

    #[test]
    fn test_string_slice() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("\"hello\"[1..4]").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![
                Value::Str("e".to_string()),
                Value::Str("l".to_string()),
                Value::Str("l".to_string()),
            ])))
        );
    }

    #[test]
    fn test_range_slice() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(0..10)[2..=4]").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![
                Value::Int(2),
                Value::Int(3),
                Value::Int(4),
            ])))
        );
    }

    // Collection method tests

    #[test]
    fn test_list_map() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[1, 2, 3].(\\x => x * 2)map").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![Value::Int(2), Value::Int(4), Value::Int(6)])))
        );
    }

    #[test]
    fn test_list_filter() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[1, 2, 3, 4].(\\x => x > 2)filter").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![Value::Int(3), Value::Int(4)])))
        );
    }

    #[test]
    fn test_list_reverse() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[1, 2, 3].()reverse").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![Value::Int(3), Value::Int(2), Value::Int(1)])))
        );
    }

    #[test]
    fn test_list_enumerate() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[10, 20].()enumerate").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![
                Value::List(vec![Value::UInt(0), Value::Int(10)]),
                Value::List(vec![Value::UInt(1), Value::Int(20)]),
            ])))
        );
    }

    #[test]
    fn test_string_map() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("\"abc\".(\\c => c)map").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![
                Value::Str("a".to_string()),
                Value::Str("b".to_string()),
                Value::Str("c".to_string()),
            ])))
        );
    }

    #[test]
    fn test_range_map() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(1..=3).(\\x => x * 10)map").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![Value::Int(10), Value::Int(20), Value::Int(30)])))
        );
    }

    #[test]
    fn test_method_chaining() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[1, 2, 3, 4].(\\x => x > 2)filter.(\\x => x * 10)map").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![Value::Int(30), Value::Int(40)])))
        );
    }

    #[test]
    fn test_method_on_non_collection() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("42.(\\x => x)map").unwrap();
        assert!(eval_stmt(&stmts[0], &mut env).is_err());
    }

    #[test]
    fn test_map_wrong_args() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[1, 2, 3].()map").unwrap();
        assert!(eval_stmt(&stmts[0], &mut env).is_err());
    }

    #[test]
    fn test_empty_list_reverse() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const xs: <int32>List = []\nxs.()reverse").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(
            eval_stmt(&stmts[1], &mut env),
            Ok(Some(Value::List(vec![])))
        );
    }

    #[test]
    fn test_list_take() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[10, 20, 30, 40, 50].(3)take").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![
                Value::Int(10),
                Value::Int(20),
                Value::Int(30),
            ])))
        );
    }

    #[test]
    fn test_take_more_than_length() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[1, 2].(5)take").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![Value::Int(1), Value::Int(2)])))
        );
    }

    #[test]
    fn test_take_zero() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[1, 2, 3].(0)take").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![])))
        );
    }

    #[test]
    fn test_range_take() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(1..=10).(3)take").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![
                Value::Int(1),
                Value::Int(2),
                Value::Int(3),
            ])))
        );
    }

    #[test]
    fn test_string_take() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("\"hello\".(3)take").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![
                Value::Str("h".to_string()),
                Value::Str("e".to_string()),
                Value::Str("l".to_string()),
            ])))
        );
    }

    #[test]
    fn test_list_while_take() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[1, 2, 3, 4, 5].(\\x => x < 4)while_take").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![
                Value::Int(1),
                Value::Int(2),
                Value::Int(3),
            ])))
        );
    }

    #[test]
    fn test_while_take_none_match() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[10, 20, 30].(\\x => x < 5)while_take").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![])))
        );
    }

    #[test]
    fn test_while_take_all_match() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("[1, 2, 3].(\\x => x < 100)while_take").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![
                Value::Int(1),
                Value::Int(2),
                Value::Int(3),
            ])))
        );
    }

    #[test]
    fn test_while_take_with_chain() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts =
            parse("[1, 2, 3, 4, 5].(\\x => x <= 3)while_take.(\\x => x * 10)map").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::List(vec![
                Value::Int(10),
                Value::Int(20),
                Value::Int(30),
            ])))
        );
    }

    // For-loop tests

    #[test]
    fn test_for_loop_list() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("mut sum: int32 = 0\n[10, 20, 30] elem x for { sum += x }\nsum").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Int(60))));
    }

    #[test]
    fn test_for_loop_range() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts =
            parse("mut result: int32 = 0\n1..=5 elem i for { result += i }\nresult").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Int(15))));
    }

    #[test]
    fn test_for_loop_string() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts =
            parse("mut result: str = \"\"\n\"abc\" elem c for { result += c }\nresult").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert_eq!(
            eval_stmt(&stmts[2], &mut env),
            Ok(Some(Value::Str("abc".to_string())))
        );
    }

    #[test]
    fn test_for_loop_break() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse(
            "mut sum: int32 = 0\n[1, 2, 3, 4, 5] elem x for {\nsum += x\nsum > 5 if { break }\n}\nsum",
        )
        .unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Int(6))));
    }

    #[test]
    fn test_for_loop_continue() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse(
            "mut sum: int32 = 0\n[1, 2, 3, 4, 5] elem x for {\nx % 2 == 0 if { continue }\nsum += x\n}\nsum",
        )
        .unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        // 1 + 3 + 5 = 9
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Int(9))));
    }

    #[test]
    fn test_for_loop_empty() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts =
            parse("mut sum: int32 = 0\nconst xs: <int32>List = []\nxs elem x for { sum += x }\nsum")
                .unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        eval_stmt(&stmts[2], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[3], &mut env), Ok(Some(Value::Int(0))));
    }

    #[test]
    fn test_for_loop_nested() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse(
            "mut sum: int32 = 0\n1..=3 elem i for {\n1..=3 elem j for {\nsum += i * j\n}\n}\nsum",
        )
        .unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        // (1+2+3)*(1+2+3) = 36
        assert_eq!(eval_stmt(&stmts[2], &mut env), Ok(Some(Value::Int(36))));
    }

    #[test]
    fn test_for_loop_non_collection_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("42 elem x for { x }").unwrap();
        assert!(eval_stmt(&stmts[0], &mut env).is_err());
    }

    // === Format String Tests ===

    #[test]
    fn test_format_string_no_interpolation() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("f\"hello world\"").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::Str("hello world".to_string())))
        );
    }

    #[test]
    fn test_format_string_simple() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x = 42\nf\"val: {x}\"").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(
            eval_stmt(&stmts[1], &mut env),
            Ok(Some(Value::Str("val: 42".to_string())))
        );
    }

    #[test]
    fn test_format_string_multiple() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x = 1\nconst y = 2\nf\"{x} + {y} = {x + y}\"").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        assert_eq!(
            eval_stmt(&stmts[2], &mut env),
            Ok(Some(Value::Str("1 + 2 = 3".to_string())))
        );
    }

    #[test]
    fn test_format_string_field_access() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "Point struct { x: int32, y: int32 }\nconst p = (x: 1, y: 2)Point\nf\"x={p.x}\"";
        let stmts = parse(src).unwrap();
        for s in &stmts[..2] {
            eval_stmt(s, &mut env).unwrap();
        }
        assert_eq!(
            eval_stmt(&stmts[2], &mut env),
            Ok(Some(Value::Str("x=1".to_string())))
        );
    }

    #[test]
    fn test_format_string_escape_braces() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("f\"use \\{ and \\}\"").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::Str("use { and }".to_string())))
        );
    }

    #[test]
    fn test_format_string_debug_string() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const s = \"hi\"\nf\"{s:?}\"").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(
            eval_stmt(&stmts[1], &mut env),
            Ok(Some(Value::Str("\"hi\"".to_string())))
        );
    }

    #[test]
    fn test_format_string_debug_number() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("f\"{42:?}\"").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::Str("42".to_string())))
        );
    }

    #[test]
    fn test_format_string_nested_braces() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("f\"{true if { 1 } else { 2 }}\"").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env),
            Ok(Some(Value::Str("1".to_string())))
        );
    }

    // === Display Trait Tests ===

    #[test]
    fn test_display_trait_basic() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "\
Point struct { x: int32, y: int32 }
Point Display impl {
    (self: Point)display: str fn {
        f\"({self.x}, {self.y})\"
    }
}
const p = (x: 1, y: 2)Point
f\"{p}\"";
        let stmts = parse(src).unwrap();
        for s in &stmts[..3] {
            eval_stmt(s, &mut env).unwrap();
        }
        assert_eq!(
            eval_stmt(&stmts[3], &mut env),
            Ok(Some(Value::Str("(1, 2)".to_string())))
        );
    }

    #[test]
    fn test_display_trait_in_fstring() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "\
Point struct { x: int32, y: int32 }
Point Display impl {
    (self: Point)display: str fn { f\"({self.x}, {self.y})\" }
}
const p = (x: 3, y: 4)Point
f\"point is {p}\"";
        let stmts = parse(src).unwrap();
        for s in &stmts[..3] {
            eval_stmt(s, &mut env).unwrap();
        }
        assert_eq!(
            eval_stmt(&stmts[3], &mut env),
            Ok(Some(Value::Str("point is (3, 4)".to_string())))
        );
    }

    #[test]
    fn test_debug_trait_custom() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "\
Point struct { x: int32, y: int32 }
Point Debug impl {
    (self: Point)debug: str fn { f\"Point(x={self.x}, y={self.y})\" }
}
const p = (x: 5, y: 6)Point
f\"{p:?}\"";
        let stmts = parse(src).unwrap();
        for s in &stmts[..3] {
            eval_stmt(s, &mut env).unwrap();
        }
        assert_eq!(
            eval_stmt(&stmts[3], &mut env),
            Ok(Some(Value::Str("Point(x=5, y=6)".to_string())))
        );
    }

    #[test]
    fn test_debug_auto_derive_struct() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "\
Pair struct { name: str, val: int32 }
const p = (name: \"hello\", val: 42)Pair
f\"{p:?}\"";
        let stmts = parse(src).unwrap();
        for s in &stmts[..2] {
            eval_stmt(s, &mut env).unwrap();
        }
        // Auto debug: strings get quotes
        assert_eq!(
            eval_stmt(&stmts[2], &mut env),
            Ok(Some(Value::Str("(name: \"hello\", val: 42)Pair".to_string())))
        );
    }

    #[test]
    fn test_display_trait_on_enum() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "\
Color enum { Red, Green, Blue }
Color Display impl {
    (self: Color)display: str fn {
        self match {
            Red -> { \"red\" },
            Green -> { \"green\" },
            Blue -> { \"blue\" },
        }
    }
}
const c = Red
f\"{c}\"";
        let stmts = parse(src).unwrap();
        for s in &stmts[..3] {
            eval_stmt(s, &mut env).unwrap();
        }
        assert_eq!(
            eval_stmt(&stmts[3], &mut env),
            Ok(Some(Value::Str("red".to_string())))
        );
    }

    // === PartialEq Trait Tests ===

    #[test]
    fn test_partial_eq_custom() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Custom PartialEq: only compare x field
        let src = "\
Point struct { x: int32, y: int32 }
Point PartialEq impl {
    (self: Point, other: Point)eq: bool fn { self.x == other.x }
}
const a = (x: 1, y: 2)Point
const b = (x: 1, y: 99)Point
a == b";
        let stmts = parse(src).unwrap();
        for s in &stmts[..4] {
            eval_stmt(s, &mut env).unwrap();
        }
        // Should be true because custom eq only checks x
        assert_eq!(
            eval_stmt(&stmts[4], &mut env),
            Ok(Some(Value::Bool(true)))
        );
    }

    #[test]
    fn test_partial_eq_neq() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "\
Point struct { x: int32, y: int32 }
Point PartialEq impl {
    (self: Point, other: Point)eq: bool fn { self.x == other.x }
}
const a = (x: 1, y: 2)Point
const b = (x: 2, y: 2)Point
a != b";
        let stmts = parse(src).unwrap();
        for s in &stmts[..4] {
            eval_stmt(s, &mut env).unwrap();
        }
        assert_eq!(
            eval_stmt(&stmts[4], &mut env),
            Ok(Some(Value::Bool(true)))
        );
    }

    #[test]
    fn test_partial_eq_fallback() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // No PartialEq impl, should use structural equality
        let src = "\
Point struct { x: int32, y: int32 }
const a = (x: 1, y: 2)Point
const b = (x: 1, y: 2)Point
a == b";
        let stmts = parse(src).unwrap();
        for s in &stmts[..3] {
            eval_stmt(s, &mut env).unwrap();
        }
        assert_eq!(
            eval_stmt(&stmts[3], &mut env),
            Ok(Some(Value::Bool(true)))
        );
    }

    // === Ord Trait Tests ===

    #[test]
    fn test_ord_basic() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "\
Score struct { val: int32 }
Score Ord impl {
    (self: Score, other: Score)cmp: Ordering fn {
        self.val < other.val if { Less } else self.val > other.val if { Greater } else { Equal }
    }
}
const a = (val: 1)Score
const b = (val: 2)Score
a < b";
        let stmts = parse(src).unwrap();
        for s in &stmts[..4] {
            eval_stmt(s, &mut env).unwrap();
        }
        assert_eq!(
            eval_stmt(&stmts[4], &mut env),
            Ok(Some(Value::Bool(true)))
        );
    }

    #[test]
    fn test_ord_all_operators() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "\
Score struct { val: int32 }
Score Ord impl {
    (self: Score, other: Score)cmp: Ordering fn {
        self.val < other.val if { Less } else self.val > other.val if { Greater } else { Equal }
    }
}
const a = (val: 5)Score
const b = (val: 3)Score
const c = (val: 5)Score";
        let stmts = parse(src).unwrap();
        for s in &stmts {
            eval_stmt(s, &mut env).unwrap();
        }

        let mut check = |expr: &str| -> Value {
            let stmts = parse(expr).unwrap();
            eval_stmt(&stmts[0], &mut env).unwrap().unwrap()
        };
        assert_eq!(check("a > b"), Value::Bool(true));
        assert_eq!(check("a < b"), Value::Bool(false));
        assert_eq!(check("a >= c"), Value::Bool(true));
        assert_eq!(check("a <= c"), Value::Bool(true));
        assert_eq!(check("b >= a"), Value::Bool(false));
    }

    #[test]
    fn test_ordering_enum() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Less").unwrap();
        let result = eval_stmt(&stmts[0], &mut env).unwrap().unwrap();
        match result {
            Value::EnumVariant {
                enum_name,
                variant_name,
                ..
            } => {
                assert_eq!(enum_name, "Ordering");
                assert_eq!(variant_name, "Less");
            }
            _ => panic!("Expected Ordering::Less variant"),
        }
    }

    // === Trait Validation Tests ===

    #[test]
    fn test_unknown_trait_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "\
Point struct { x: int32 }
Point Foo impl {
    (self: Point)display: str fn { \"x\" }
}";
        let stmts = parse(src).unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        let result = eval_stmt(&stmts[1], &mut env);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("Unknown trait"));
    }

    #[test]
    fn test_wrong_method_name_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "\
Point struct { x: int32 }
Point Display impl {
    (self: Point)show: str fn { \"x\" }
}";
        let stmts = parse(src).unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        let result = eval_stmt(&stmts[1], &mut env);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("requires method 'display'"));
    }

    #[test]
    fn test_duplicate_impl_error() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let src = "\
Point struct { x: int32 }
Point Display impl {
    (self: Point)display: str fn { \"x\" }
}
Point Display impl {
    (self: Point)display: str fn { \"y\" }
}";
        let stmts = parse(src).unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        eval_stmt(&stmts[1], &mut env).unwrap();
        let result = eval_stmt(&stmts[2], &mut env);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("already implemented"));
    }

    // --- Maybe built-in methods ---

    #[test]
    fn test_maybe_map_exists() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Exists.(\\x => x + 1)map").unwrap();
        let result = eval_stmt(&stmts[0], &mut env).unwrap().unwrap();
        assert_eq!(
            result,
            Value::EnumVariant {
                enum_name: "Maybe".to_string(),
                variant_name: "Exists".to_string(),
                type_params: vec![],
                payload: Some(Box::new(Value::Int(43))),
            }
        );
    }

    #[test]
    fn test_maybe_map_does_not_exist() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: int32? = DoesNotExist\nx.(\\v => v + 1)map").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        let result = eval_stmt(&stmts[1], &mut env).unwrap().unwrap();
        assert_eq!(
            result,
            Value::EnumVariant {
                enum_name: "Maybe".to_string(),
                variant_name: "DoesNotExist".to_string(),
                type_params: vec![],
                payload: None,
            }
        );
    }

    #[test]
    fn test_maybe_and_then_exists() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Exists.(\\x => (x + 1)Exists)and_then").unwrap();
        let result = eval_stmt(&stmts[0], &mut env).unwrap().unwrap();
        assert_eq!(
            result,
            Value::EnumVariant {
                enum_name: "Maybe".to_string(),
                variant_name: "Exists".to_string(),
                type_params: vec![],
                payload: Some(Box::new(Value::Int(43))),
            }
        );
    }

    #[test]
    fn test_maybe_and_then_returns_dne() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Exists.(\\x => DoesNotExist)and_then").unwrap();
        let result = eval_stmt(&stmts[0], &mut env).unwrap().unwrap();
        assert_eq!(
            result,
            Value::EnumVariant {
                enum_name: "Maybe".to_string(),
                variant_name: "DoesNotExist".to_string(),
                type_params: vec![],
                payload: None,
            }
        );
    }

    #[test]
    fn test_maybe_or_else_exists() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Exists.(\\=> (0)Exists)or_else").unwrap();
        let result = eval_stmt(&stmts[0], &mut env).unwrap().unwrap();
        assert_eq!(
            result,
            Value::EnumVariant {
                enum_name: "Maybe".to_string(),
                variant_name: "Exists".to_string(),
                type_params: vec![TypeAnn::Int32],
                payload: Some(Box::new(Value::Int(42))),
            }
        );
    }

    #[test]
    fn test_maybe_or_else_dne() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts =
            parse("const x: int32? = DoesNotExist\nx.(\\=> (99)Exists)or_else").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        let result = eval_stmt(&stmts[1], &mut env).unwrap().unwrap();
        assert_eq!(
            result,
            Value::EnumVariant {
                enum_name: "Maybe".to_string(),
                variant_name: "Exists".to_string(),
                type_params: vec![],
                payload: Some(Box::new(Value::Int(99))),
            }
        );
    }

    #[test]
    fn test_maybe_or_panic_exists() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Exists.()or_panic").unwrap();
        let result = eval_stmt(&stmts[0], &mut env).unwrap().unwrap();
        assert_eq!(result, Value::Int(42));
    }

    #[test]
    fn test_maybe_or_panic_dne() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: int32? = DoesNotExist\nx.()or_panic").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        let result = eval_stmt(&stmts[1], &mut env);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("DoesNotExist"));
    }

    #[test]
    fn test_maybe_map_chain_or_panic() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(10)Exists.(\\x => x * 2)map.()or_panic").unwrap();
        let result = eval_stmt(&stmts[0], &mut env).unwrap().unwrap();
        assert_eq!(result, Value::Int(20));
    }

    // --- Attempt built-in methods ---

    #[test]
    fn test_attempt_map_success() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Success.(\\x => x + 1)map").unwrap();
        let result = eval_stmt(&stmts[0], &mut env).unwrap().unwrap();
        assert_eq!(
            result,
            Value::EnumVariant {
                enum_name: "Attempt".to_string(),
                variant_name: "Success".to_string(),
                type_params: vec![],
                payload: Some(Box::new(Value::Int(43))),
            }
        );
    }

    #[test]
    fn test_attempt_map_failure() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(\"err\")Failure.(\\x => x + 1)map").unwrap();
        let result = eval_stmt(&stmts[0], &mut env).unwrap().unwrap();
        assert_eq!(
            result,
            Value::EnumVariant {
                enum_name: "Attempt".to_string(),
                variant_name: "Failure".to_string(),
                type_params: vec![],
                payload: Some(Box::new(Value::Str("err".to_string()))),
            }
        );
    }

    #[test]
    fn test_attempt_and_then_success() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Success.(\\x => (x + 1)Success)and_then").unwrap();
        let result = eval_stmt(&stmts[0], &mut env).unwrap().unwrap();
        assert_eq!(
            result,
            Value::EnumVariant {
                enum_name: "Attempt".to_string(),
                variant_name: "Success".to_string(),
                type_params: vec![],
                payload: Some(Box::new(Value::Int(43))),
            }
        );
    }

    #[test]
    fn test_attempt_or_else_failure() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(\"err\")Failure.(\\e => (0)Success)or_else").unwrap();
        let result = eval_stmt(&stmts[0], &mut env).unwrap().unwrap();
        assert_eq!(
            result,
            Value::EnumVariant {
                enum_name: "Attempt".to_string(),
                variant_name: "Success".to_string(),
                type_params: vec![],
                payload: Some(Box::new(Value::Int(0))),
            }
        );
    }

    #[test]
    fn test_attempt_or_else_success() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Success.(\\e => (0)Success)or_else").unwrap();
        let result = eval_stmt(&stmts[0], &mut env).unwrap().unwrap();
        assert_eq!(
            result,
            Value::EnumVariant {
                enum_name: "Attempt".to_string(),
                variant_name: "Success".to_string(),
                type_params: vec![],
                payload: Some(Box::new(Value::Int(42))),
            }
        );
    }

    #[test]
    fn test_attempt_or_panic_success() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Success.()or_panic").unwrap();
        let result = eval_stmt(&stmts[0], &mut env).unwrap().unwrap();
        assert_eq!(result, Value::Int(42));
    }

    #[test]
    fn test_attempt_or_panic_failure() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(\"something went wrong\")Failure.()or_panic").unwrap();
        let result = eval_stmt(&stmts[0], &mut env);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("something went wrong"));
    }

    // --- exists_and / dne_or ---

    #[test]
    fn test_exists_and_true() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Exists.(\\x => x > 0)exists_and").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Bool(true))));
    }

    #[test]
    fn test_exists_and_false() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Exists.(\\x => x < 0)exists_and").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Bool(false))));
    }

    #[test]
    fn test_exists_and_dne() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: int32? = DoesNotExist\nx.(\\v => true)exists_and").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Bool(false))));
    }

    #[test]
    fn test_dne_or_dne() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: int32? = DoesNotExist\nx.(\\v => false)dne_or").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Bool(true))));
    }

    #[test]
    fn test_dne_or_exists_true() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Exists.(\\x => x > 0)dne_or").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Bool(true))));
    }

    #[test]
    fn test_dne_or_exists_false() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Exists.(\\x => x < 0)dne_or").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Bool(false))));
    }

    // --- succeeded_and / failed_and ---

    #[test]
    fn test_succeeded_and_true() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Success.(\\x => x > 0)succeeded_and").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Bool(true))));
    }

    #[test]
    fn test_succeeded_and_failure() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(\"err\")Failure.(\\x => true)succeeded_and").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Bool(false))));
    }

    #[test]
    fn test_failed_and_true() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(\"timeout\")Failure.(\\e => e == \"timeout\")failed_and").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Bool(true))));
    }

    #[test]
    fn test_failed_and_success() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Success.(\\e => true)failed_and").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Bool(false))));
    }

    // --- or_default ---

    #[test]
    fn test_maybe_or_default_exists() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Exists.(0)or_default").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Int(42))));
    }

    #[test]
    fn test_maybe_or_default_dne() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("const x: int32? = DoesNotExist\nx.(0)or_default").unwrap();
        eval_stmt(&stmts[0], &mut env).unwrap();
        assert_eq!(eval_stmt(&stmts[1], &mut env), Ok(Some(Value::Int(0))));
    }

    #[test]
    fn test_attempt_or_default_success() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(42)Success.(0)or_default").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Int(42))));
    }

    #[test]
    fn test_attempt_or_default_failure() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("(\"err\")Failure.(0)or_default").unwrap();
        assert_eq!(eval_stmt(&stmts[0], &mut env), Ok(Some(Value::Int(0))));
    }

    // --- Ordering built-in methods ---

    #[test]
    fn test_ordering_rev_less() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Less.()rev").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env).unwrap().unwrap(),
            Value::EnumVariant {
                enum_name: "Ordering".to_string(),
                variant_name: "Greater".to_string(),
                type_params: vec![],
                payload: None,
            }
        );
    }

    #[test]
    fn test_ordering_rev_greater() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Greater.()rev").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env).unwrap().unwrap(),
            Value::EnumVariant {
                enum_name: "Ordering".to_string(),
                variant_name: "Less".to_string(),
                type_params: vec![],
                payload: None,
            }
        );
    }

    #[test]
    fn test_ordering_rev_equal() {
        use crate::parser::parse;
        let mut env = Environment::new();
        let stmts = parse("Equal.()rev").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env).unwrap().unwrap(),
            Value::EnumVariant {
                enum_name: "Ordering".to_string(),
                variant_name: "Equal".to_string(),
                type_params: vec![],
                payload: None,
            }
        );
    }

    #[test]
    fn test_ordering_then_not_equal() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Less.then(Greater) -> Less (self is not Equal, return self)
        let stmts = parse("Less.(Greater)then").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env).unwrap().unwrap(),
            Value::EnumVariant {
                enum_name: "Ordering".to_string(),
                variant_name: "Less".to_string(),
                type_params: vec![],
                payload: None,
            }
        );
    }

    #[test]
    fn test_ordering_then_equal() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Equal.then(Greater) -> Greater (self is Equal, return other)
        let stmts = parse("Equal.(Greater)then").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env).unwrap().unwrap(),
            Value::EnumVariant {
                enum_name: "Ordering".to_string(),
                variant_name: "Greater".to_string(),
                type_params: vec![],
                payload: None,
            }
        );
    }

    #[test]
    fn test_ordering_then_chain() {
        use crate::parser::parse;
        let mut env = Environment::new();
        // Equal.then(Equal).then(Less) -> Less
        let stmts = parse("Equal.(Equal)then.(Less)then").unwrap();
        assert_eq!(
            eval_stmt(&stmts[0], &mut env).unwrap().unwrap(),
            Value::EnumVariant {
                enum_name: "Ordering".to_string(),
                variant_name: "Less".to_string(),
                type_params: vec![],
                payload: None,
            }
        );
    }
}
