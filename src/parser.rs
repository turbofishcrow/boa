use pest::Parser;
use pest_derive::Parser;

use crate::ast::{
    BinaryOp, EnumVariantDef, FormatPart, LambdaParam, MatchArm, MethodDef, Node, Pattern,
    PrefixOp, Statement, TypeAnn,
};

// ANCHOR: parser
#[derive(Parser)]
#[grammar = "grammar.pest"] // relative to src
pub struct CalcParser;
// ANCHOR_END: parser

// ANCHOR: parse_source
pub fn parse(source: &str) -> std::result::Result<Vec<Statement>, pest::error::Error<Rule>> {
    let mut stmts = vec![];
    let pairs = CalcParser::parse(Rule::Program, source)?;
    for pair in pairs {
        if pair.as_rule() == Rule::Program {
            for inner in pair.into_inner() {
                match inner.as_rule() {
                    Rule::Stmt => {
                        stmts.push(build_statement(inner));
                    }
                    Rule::EOI => {}
                    _ => {}
                }
            }
        }
    }
    Ok(stmts)
}
// ANCHOR_END: parse_source

fn build_statement(pair: pest::iterators::Pair<Rule>) -> Statement {
    let mut inner = pair.into_inner();
    let first = inner.next().unwrap();

    match first.as_rule() {
        Rule::StructDeclaration => {
            let mut sd_inner = first.into_inner();
            let mut type_params = vec![];
            let next = sd_inner.next().unwrap();
            let name_pair = if next.as_rule() == Rule::TypeParams {
                for child in next.into_inner() {
                    for tp in child.into_inner() {
                        type_params.push(tp.as_str().to_string());
                    }
                }
                sd_inner.next().unwrap()
            } else {
                next
            };
            let name = name_pair.as_str().to_string();
            skip_keywords(&mut sd_inner); // skip StructKw
            let mut fields = vec![];
            for child in sd_inner {
                if child.as_rule() == Rule::StructFieldList {
                    for field in child.into_inner() {
                        if field.as_rule() == Rule::StructField {
                            let mut f_inner = field.into_inner();
                            let fname = f_inner.next().unwrap().as_str().to_string();
                            let ftype = parse_type(f_inner.next().unwrap());
                            fields.push((fname, ftype));
                        }
                    }
                }
            }
            Statement::StructDeclaration {
                name,
                type_params,
                fields,
            }
        }
        Rule::EnumDeclaration => {
            let mut ed_inner = first.into_inner();
            let mut type_params = vec![];
            let next = ed_inner.next().unwrap();
            let name_pair = if next.as_rule() == Rule::TypeParams {
                for child in next.into_inner() {
                    for tp in child.into_inner() {
                        type_params.push(tp.as_str().to_string());
                    }
                }
                ed_inner.next().unwrap()
            } else {
                next
            };
            let name = name_pair.as_str().to_string();
            skip_keywords(&mut ed_inner); // skip EnumKw
            let mut variants = vec![];
            for child in ed_inner {
                if child.as_rule() == Rule::EnumVariantList {
                    for variant in child.into_inner() {
                        if variant.as_rule() == Rule::EnumVariantDef {
                            let children: Vec<_> = variant.into_inner().collect();
                            if children.len() == 1 {
                                // No-payload variant: just TypeIdent
                                variants.push(EnumVariantDef {
                                    name: children[0].as_str().to_string(),
                                    payload_type: None,
                                });
                            } else {
                                // Payload variant: Type + TypeIdent
                                let payload_type = Some(parse_type(children[0].clone()));
                                let vname = children.last().unwrap().as_str().to_string();
                                variants.push(EnumVariantDef {
                                    name: vname,
                                    payload_type,
                                });
                            }
                        }
                    }
                }
            }
            Statement::EnumDeclaration {
                name,
                type_params,
                variants,
            }
        }
        Rule::MethodsDeclaration => {
            let mut md_inner = first.into_inner();
            let mut type_params = vec![];
            let next = md_inner.next().unwrap();
            let name_pair = if next.as_rule() == Rule::TypeParams {
                for child in next.into_inner() {
                    for tp in child.into_inner() {
                        type_params.push(tp.as_str().to_string());
                    }
                }
                md_inner.next().unwrap()
            } else {
                next
            };
            let type_name = name_pair.as_str().to_string();
            skip_keywords(&mut md_inner); // skip MethodsKw
            let mut methods = vec![];
            for child in md_inner {
                if child.as_rule() == Rule::FnDeclaration {
                    let fn_stmt = parse_fn_declaration(child);
                    if let Statement::FnDeclaration {
                        name,
                        type_params: fn_tp,
                        params,
                        return_type,
                        body,
                    } = fn_stmt
                    {
                        methods.push(MethodDef {
                            name,
                            type_params: fn_tp,
                            params,
                            return_type,
                            body,
                        });
                    }
                }
            }
            Statement::MethodsDeclaration {
                type_name,
                type_params,
                methods,
            }
        }
        Rule::TraitImplDeclaration => {
            let mut ti_inner = first.into_inner();
            let mut type_params = vec![];
            let next = ti_inner.next().unwrap();
            let name_pair = if next.as_rule() == Rule::TypeParams {
                for child in next.into_inner() {
                    for tp in child.into_inner() {
                        type_params.push(tp.as_str().to_string());
                    }
                }
                ti_inner.next().unwrap()
            } else {
                next
            };
            let type_name = name_pair.as_str().to_string();
            let trait_name = ti_inner.next().unwrap().as_str().to_string();
            skip_keywords(&mut ti_inner); // skip ImplKw
            let mut methods = vec![];
            for child in ti_inner {
                if child.as_rule() == Rule::FnDeclaration {
                    let fn_stmt = parse_fn_declaration(child);
                    if let Statement::FnDeclaration {
                        name,
                        type_params: fn_tp,
                        params,
                        return_type,
                        body,
                    } = fn_stmt
                    {
                        methods.push(MethodDef {
                            name,
                            type_params: fn_tp,
                            params,
                            return_type,
                            body,
                        });
                    }
                }
            }
            Statement::TraitImplDeclaration {
                type_name,
                type_params,
                trait_name,
                methods,
            }
        }
        Rule::FnDeclaration => parse_fn_declaration(first),
        Rule::DestructuringDecl => {
            let mut inner = first.into_inner();
            let kw = inner.next().unwrap();
            let mutable = kw.as_rule() == Rule::MutKw;
            let bindings_pair = inner.next().unwrap(); // DestructBindings
            let mut bindings = vec![];
            for binding in bindings_pair.into_inner() {
                if binding.as_rule() == Rule::DestructBinding {
                    let mut b_inner = binding.into_inner();
                    let name = b_inner.next().unwrap().as_str().to_string();
                    let rename = b_inner.next().map(|p| p.as_str().to_string());
                    bindings.push((name, rename));
                }
            }
            let type_name = inner.next().unwrap().as_str().to_string();
            let value = build_ast_from_expr(inner.next().unwrap());
            Statement::DestructuringDecl {
                mutable,
                bindings,
                type_name,
                value,
            }
        }
        Rule::Declaration => {
            let mut decl_inner = first.into_inner();
            // First child is CompConstKw, ConstKw, or MutKw
            let kw = decl_inner.next().unwrap();
            let mutable = kw.as_rule() == Rule::MutKw;
            // Next is Identifier
            let name = decl_inner.next().unwrap().as_str().to_string();
            // Optional TypeAnnotation, then Expr
            let next = decl_inner.next().unwrap();
            let (type_ann, value) = if next.as_rule() == Rule::TypeAnnotation {
                let type_ann = parse_type_annotation(next);
                let expr = build_ast_from_expr(decl_inner.next().unwrap());
                (Some(type_ann), expr)
            } else {
                (None, build_ast_from_expr(next))
            };
            Statement::Declaration {
                mutable,
                name,
                type_ann,
                value,
            }
        }
        Rule::CompoundReassignment => {
            let mut cr_inner = first.into_inner();
            let name = cr_inner.next().unwrap().as_str().to_string();
            let op_pair = cr_inner.next().unwrap();
            let binary_op = match op_pair.as_rule() {
                Rule::AddAssign => BinaryOp::Add,
                Rule::SubAssign => BinaryOp::Sub,
                Rule::MulAssign => BinaryOp::Mul,
                Rule::DivAssign => BinaryOp::Div,
                Rule::ModAssign => BinaryOp::Mod,
                _ => panic!("Unknown compound operator: {:?}", op_pair),
            };
            let expr = build_ast_from_expr(cr_inner.next().unwrap());
            // Desugar: x += e → x = x op e
            Statement::Reassignment {
                name: name.clone(),
                value: Node::BinaryOp {
                    lhs: Box::new(Node::Identifier(name)),
                    rhs: Box::new(expr),
                    op: binary_op,
                },
            }
        }
        Rule::Reassignment => {
            let mut assign_inner = first.into_inner();
            let name = assign_inner.next().unwrap().as_str().to_string();
            let expr = build_ast_from_expr(assign_inner.next().unwrap());
            Statement::Reassignment { name, value: expr }
        }
        Rule::Expr => Statement::Expr(build_ast_from_expr(first)),
        _ => panic!("Unknown statement: {:?}", first),
    }
}

fn parse_type_annotation(pair: pest::iterators::Pair<Rule>) -> TypeAnn {
    let type_pair = pair.into_inner().next().unwrap();
    parse_type(type_pair)
}

fn parse_fn_declaration(pair: pest::iterators::Pair<Rule>) -> Statement {
    let mut fn_inner = pair.into_inner();
    let mut type_params = vec![];
    let mut params = vec![];
    let next = fn_inner.next().unwrap();
    let next = if next.as_rule() == Rule::TypeParams {
        for child in next.into_inner() {
            for tp in child.into_inner() {
                type_params.push(tp.as_str().to_string());
            }
        }
        fn_inner.next().unwrap()
    } else {
        next
    };
    let name_pair = if next.as_rule() == Rule::ParamList {
        for param in next.into_inner() {
            let mut p_inner = param.into_inner();
            let first_tok = p_inner.next().unwrap();
            let (pmut, pname) = if first_tok.as_rule() == Rule::MutKw {
                (true, p_inner.next().unwrap().as_str().to_string())
            } else {
                (false, first_tok.as_str().to_string())
            };
            let ptype = parse_type(p_inner.next().unwrap());
            params.push((pname, ptype, pmut));
        }
        fn_inner.next().unwrap()
    } else {
        next
    };
    let name = name_pair.as_str().to_string();
    let next = fn_inner.next().unwrap();
    let (return_type, block_pair) = if next.as_rule() == Rule::TypeAnnotation {
        let rt = parse_type_annotation(next);
        skip_keywords(&mut fn_inner);
        (Some(rt), fn_inner.next().unwrap())
    } else if next.as_rule() == Rule::Block {
        (None, next)
    } else {
        skip_keywords(&mut fn_inner);
        let block = fn_inner.next().unwrap();
        (None, block)
    };
    let body = build_ast_from_expr(block_pair);
    Statement::FnDeclaration {
        name,
        type_params,
        params,
        return_type,
        body: Box::new(body),
    }
}

fn parse_type(pair: pest::iterators::Pair<Rule>) -> TypeAnn {
    match pair.as_rule() {
        Rule::TypeName => parse_type_name(pair),
        Rule::FunctionType => {
            let mut inner = pair.into_inner();
            let first = inner.next().unwrap();
            let (param_types, ret_pair) = if first.as_rule() == Rule::TypeList {
                let types: Vec<TypeAnn> = first.into_inner().map(parse_type).collect();
                (types, inner.next().unwrap())
            } else {
                // No params, first is the return Type
                (vec![], first)
            };
            TypeAnn::Fn {
                param_types,
                return_type: Some(Box::new(parse_type(ret_pair))),
            }
        }
        Rule::TypeIdent => TypeAnn::Named(pair.as_str().to_string()),
        Rule::GenericType => {
            let mut inner = pair.into_inner();
            let type_list = inner.next().unwrap();
            let type_params: Vec<TypeAnn> = type_list.into_inner().map(parse_type).collect();
            let name = inner.next().unwrap().as_str().to_string();
            TypeAnn::Generic { name, type_params }
        }
        Rule::OptionalType => {
            let inner = parse_type(pair.into_inner().next().unwrap());
            TypeAnn::Generic {
                name: "Maybe".to_string(),
                type_params: vec![inner],
            }
        }
        Rule::ListType => {
            let inner = parse_type(pair.into_inner().next().unwrap());
            TypeAnn::Generic {
                name: "List".to_string(),
                type_params: vec![inner],
            }
        }
        _ => panic!("Expected Type, got {:?}", pair.as_rule()),
    }
}

fn parse_type_name(pair: pest::iterators::Pair<Rule>) -> TypeAnn {
    match pair.as_str() {
        "int32" => TypeAnn::Int32,
        "uint32" => TypeAnn::UInt32,
        "fl64" => TypeAnn::Fl64,
        "bool" => TypeAnn::Bool,
        "str" => TypeAnn::Str,
        other => panic!("Unknown type: {}", other),
    }
}
// ANCHOR_END: parse_source

fn unescape_string(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('\\') => result.push('\\'),
                Some('"') => result.push('"'),
                Some(other) => {
                    result.push('\\');
                    result.push(other);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }
    result
}

fn parse_format_string(s: &str) -> Vec<FormatPart> {
    let mut parts = vec![];
    let mut text_buf = String::new();
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => text_buf.push('\n'),
                Some('t') => text_buf.push('\t'),
                Some('r') => text_buf.push('\r'),
                Some('\\') => text_buf.push('\\'),
                Some('"') => text_buf.push('"'),
                Some('{') => text_buf.push('{'),
                Some('}') => text_buf.push('}'),
                Some(other) => {
                    text_buf.push('\\');
                    text_buf.push(other);
                }
                None => text_buf.push('\\'),
            }
        } else if c == '{' {
            // Flush text buffer
            if !text_buf.is_empty() {
                parts.push(FormatPart::Text(text_buf.clone()));
                text_buf.clear();
            }
            // Collect until matching '}', tracking brace depth
            let mut expr_content = String::new();
            let mut depth = 1;
            for ic in chars.by_ref() {
                if ic == '{' {
                    depth += 1;
                    expr_content.push(ic);
                } else if ic == '}' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    expr_content.push(ic);
                } else {
                    expr_content.push(ic);
                }
            }
            // Check for :? suffix (Debug format)
            let (is_debug, actual_expr) = if expr_content.ends_with(":?") {
                (true, &expr_content[..expr_content.len() - 2])
            } else {
                (false, expr_content.as_str())
            };
            // Re-parse as Boa expression
            let parsed = CalcParser::parse(Rule::Expr, actual_expr).unwrap_or_else(|e| {
                panic!("Invalid expression in format string '{}': {}", actual_expr, e)
            });
            let expr_node = build_ast_from_expr(parsed.into_iter().next().unwrap());
            if is_debug {
                parts.push(FormatPart::Debug(expr_node));
            } else {
                parts.push(FormatPart::Display(expr_node));
            }
        } else {
            text_buf.push(c);
        }
    }
    // Flush remaining text
    if !text_buf.is_empty() {
        parts.push(FormatPart::Text(text_buf));
    }
    parts
}

fn skip_keywords(inner: &mut pest::iterators::Pairs<Rule>) {
    while let Some(next) = inner.peek() {
        match next.as_rule() {
            Rule::IfKw
            | Rule::ElseKw
            | Rule::LoopKw
            | Rule::WhileKw
            | Rule::BreakKw
            | Rule::ContinueKw
            | Rule::CompConstKw
            | Rule::ConstKw
            | Rule::MutKw
            | Rule::FnKw
            | Rule::ReturnKw
            | Rule::CastKw
            | Rule::StructKw
            | Rule::EnumKw
            | Rule::MethodsKw
            | Rule::MatchKw
            | Rule::ElemKw
            | Rule::ForKw
            | Rule::ImplKw => {
                inner.next();
            }
            _ => break,
        }
    }
}

fn is_pattern_rule(rule: Rule) -> bool {
    matches!(
        rule,
        Rule::VariantPattern | Rule::WildcardPattern | Rule::LiteralPattern
    )
}

fn build_else_body(inner: &mut pest::iterators::Pairs<Rule>) -> Option<Box<Node>> {
    skip_keywords(inner); // Skip ElseKw
    let else_part = inner.next()?;
    match else_part.as_rule() {
        Rule::Block => Some(Box::new(build_ast_from_expr(else_part))),
        Rule::OrExpr => {
            let cond = build_ast_from_expr(else_part);
            skip_keywords(inner);
            let next = inner.next().unwrap();
            if is_pattern_rule(next.as_rule()) {
                let pattern = parse_pattern(next);
                let if_tail = inner.next().unwrap();
                Some(Box::new(build_if_let(cond, pattern, if_tail)))
            } else {
                Some(Box::new(build_if_else(cond, next)))
            }
        }
        _ => panic!("Unexpected else body: {:?}", else_part),
    }
}

fn build_if_else(condition: Node, if_tail: pest::iterators::Pair<Rule>) -> Node {
    let mut inner = if_tail.into_inner();
    skip_keywords(&mut inner); // Skip IfKw
    let then_block = build_ast_from_expr(inner.next().unwrap());
    let else_block = build_else_body(&mut inner);

    Node::IfElse {
        condition: Box::new(condition),
        then_block: Box::new(then_block),
        else_block,
    }
}

fn build_if_let(scrutinee: Node, pattern: Pattern, if_tail: pest::iterators::Pair<Rule>) -> Node {
    let mut inner = if_tail.into_inner();
    skip_keywords(&mut inner); // Skip IfKw
    let then_block = build_ast_from_expr(inner.next().unwrap());
    let else_block = build_else_body(&mut inner);

    Node::IfLet {
        scrutinee: Box::new(scrutinee),
        pattern: Box::new(pattern),
        then_block: Box::new(then_block),
        else_block,
    }
}

fn parse_pattern(pair: pest::iterators::Pair<Rule>) -> Pattern {
    match pair.as_rule() {
        Rule::WildcardPattern => Pattern::Wildcard,
        Rule::VariantPattern => {
            let mut inner = pair.into_inner();
            let binding = inner.next().unwrap().as_str().to_string();
            let variant_name = inner.next().unwrap().as_str().to_string();
            Pattern::Variant { variant_name, binding }
        }
        Rule::LiteralPattern => {
            let inner = pair.into_inner().next().unwrap();
            match inner.as_rule() {
                Rule::TypeIdent => Pattern::BareVariant(inner.as_str().to_string()),
                _ => Pattern::Literal(build_ast_from_expr(inner)),
            }
        }
        _ => unreachable!("unexpected pattern rule: {:?}", pair.as_rule()),
    }
}

fn build_ast_from_expr(pair: pest::iterators::Pair<Rule>) -> Node {
    match pair.as_rule() {
        Rule::Expr => {
            let mut inner = pair.into_inner();
            let first = inner.next().unwrap();
            // Expr = { Lambda | RangeExpr ~ ExprSuffix? }
            if first.as_rule() == Rule::Lambda {
                return build_ast_from_expr(first);
            }
            let mut node = build_ast_from_expr(first); // First OrExpr from RangeExpr

            // Check for range operator (from silent RangeExpr/RangeOp)
            if let Some(next) = inner.peek()
                && matches!(next.as_rule(), Rule::RangeInclusive | Rule::RangeExclusive)
            {
                let range_op = inner.next().unwrap();
                let inclusive = range_op.as_rule() == Rule::RangeInclusive;
                let end_expr = build_ast_from_expr(inner.next().unwrap());
                node = Node::Range {
                    start: Box::new(node),
                    end: Box::new(end_expr),
                    inclusive,
                };
            }

            // Check for ExprSuffix (=:, if, while, return, match)
            if let Some(next) = inner.next() {
                if is_pattern_rule(next.as_rule()) {
                    // Pattern bind: =: was consumed (silent), pattern child appears directly
                    let pattern = parse_pattern(next);
                    if let Some(tail) = inner.next() {
                        match tail.as_rule() {
                            Rule::IfTail => {
                                node = build_if_let(node, pattern, tail);
                            }
                            Rule::WhileTail => {
                                let mut wh_inner = tail.into_inner();
                                skip_keywords(&mut wh_inner);
                                let body = build_ast_from_expr(wh_inner.next().unwrap());
                                node = Node::WhileLet {
                                    scrutinee: Box::new(node),
                                    pattern: Box::new(pattern),
                                    body: Box::new(body),
                                };
                            }
                            _ => panic!("Unexpected tail after pattern bind: {:?}", tail),
                        }
                    } else {
                        // Standalone pattern test: scrutinee =: pattern
                        node = Node::PatternTest {
                            scrutinee: Box::new(node),
                            pattern: Box::new(pattern),
                        };
                    }
                } else {
                    match next.as_rule() {
                        Rule::IfTail => {
                            node = build_if_else(node, next);
                        }
                        Rule::WhileTail => {
                            let mut wh_inner = next.into_inner();
                            skip_keywords(&mut wh_inner);
                            let body = build_ast_from_expr(wh_inner.next().unwrap());
                            node = Node::While {
                                condition: Box::new(node),
                                body: Box::new(body),
                            };
                        }
                        Rule::ReturnTail => {
                            node = Node::Return(Box::new(node));
                        }
                        Rule::MatchTail => {
                            let mut match_inner = next.into_inner();
                            skip_keywords(&mut match_inner);
                            let mut arms = vec![];
                            if let Some(arm_list) = match_inner.next() {
                                for arm_pair in arm_list.into_inner() {
                                    if arm_pair.as_rule() == Rule::MatchArm {
                                        let mut arm_inner = arm_pair.into_inner();
                                        let pattern = parse_pattern(arm_inner.next().unwrap());
                                        let body = build_ast_from_expr(arm_inner.next().unwrap());
                                        arms.push(MatchArm { pattern, body: Box::new(body) });
                                    }
                                }
                            }
                            node = Node::Match { scrutinee: Box::new(node), arms };
                        }
                        Rule::ForTail => {
                            let mut for_inner = next.into_inner();
                            skip_keywords(&mut for_inner); // skip ElemKw
                            let var_name = for_inner.next().unwrap().as_str().to_string();
                            skip_keywords(&mut for_inner); // skip ForKw
                            let body = build_ast_from_expr(for_inner.next().unwrap());
                            node = Node::ForLoop {
                                collection: Box::new(node),
                                var_name,
                                body: Box::new(body),
                            };
                        }
                        _ => panic!("Unexpected tail: {:?}", next),
                    }
                }
            }
            node
        }
        Rule::OrExpr => {
            let mut inner = pair.into_inner();
            let mut node = build_ast_from_expr(inner.next().unwrap());

            while let Some(op_pair) = inner.next() {
                if op_pair.as_rule() == Rule::Or {
                    let right = build_ast_from_expr(inner.next().unwrap());
                    node = Node::BinaryOp {
                        lhs: Box::new(node),
                        rhs: Box::new(right),
                        op: BinaryOp::Or,
                    };
                }
            }
            node
        }
        Rule::AndExpr => {
            let mut inner = pair.into_inner();
            let mut node = build_ast_from_expr(inner.next().unwrap());

            while let Some(op_pair) = inner.next() {
                if op_pair.as_rule() == Rule::And {
                    let right = build_ast_from_expr(inner.next().unwrap());
                    node = Node::BinaryOp {
                        lhs: Box::new(node),
                        rhs: Box::new(right),
                        op: BinaryOp::And,
                    };
                }
            }
            node
        }
        Rule::CmpExpr => {
            let mut inner = pair.into_inner();
            let mut node = build_ast_from_expr(inner.next().unwrap());

            while let Some(op_pair) = inner.next() {
                let binary_op = match op_pair.as_rule() {
                    Rule::Eq => BinaryOp::Eq,
                    Rule::Neq => BinaryOp::Neq,
                    Rule::Lt => BinaryOp::Lt,
                    Rule::Gt => BinaryOp::Gt,
                    Rule::Lte => BinaryOp::Lte,
                    Rule::Gte => BinaryOp::Gte,
                    _ => panic!("Unexpected comparison operator: {:?}", op_pair),
                };
                let right = build_ast_from_expr(inner.next().unwrap());
                node = Node::BinaryOp {
                    lhs: Box::new(node),
                    rhs: Box::new(right),
                    op: binary_op,
                };
            }
            node
        }
        Rule::AddSub | Rule::Term => {
            let mut inner = pair.into_inner();
            let mut node = build_ast_from_expr(inner.next().unwrap());

            while let Some(op_pair) = inner.next() {
                let binary_op = match op_pair.as_rule() {
                    Rule::Add => BinaryOp::Add,
                    Rule::Sub => BinaryOp::Sub,
                    Rule::Mul => BinaryOp::Mul,
                    Rule::Div => BinaryOp::Div,
                    Rule::Mod => BinaryOp::Mod,
                    _ => panic!("Unexpected operator: {:?}", op_pair),
                };
                let right = build_ast_from_expr(inner.next().unwrap());
                node = Node::BinaryOp {
                    lhs: Box::new(node),
                    rhs: Box::new(right),
                    op: binary_op,
                };
            }
            node
        }
        Rule::Power => {
            // Power is right-associative: 2^3^4 = 2^(3^4)
            // Grammar: Factor ~ (Pow ~ Factor)*
            // When we have 2^3^4, inner is [Factor(2), Pow, Factor(3), Pow, Factor(4)]
            let inner = pair.into_inner();
            let mut operands = vec![];

            // Collect all factors and operators
            for item in inner {
                if item.as_rule() == Rule::Pow {
                    // Skip the operator, next will be a factor
                } else {
                    operands.push(build_ast_from_expr(item));
                }
            }

            // Build from right to left for right-associativity
            if operands.is_empty() {
                panic!("Power rule has no operands");
            }

            let mut result = operands.pop().unwrap();
            while let Some(left) = operands.pop() {
                result = Node::BinaryOp {
                    lhs: Box::new(left),
                    rhs: Box::new(result),
                    op: BinaryOp::Pow,
                };
            }
            result
        }
        Rule::Factor => {
            let mut inner = pair.into_inner();

            // Handle prefix operators
            let mut prefixes = vec![];
            while let Some(next) = inner.clone().next() {
                match next.as_rule() {
                    Rule::Neg => {
                        prefixes.push(PrefixOp::Negative);
                        inner.next();
                    }
                    Rule::Not => {
                        prefixes.push(PrefixOp::Not);
                        inner.next();
                    }
                    _ => break,
                }
            }

            // Get the inner node (Power or Primary)
            let mut node = build_ast_from_expr(inner.next().unwrap());

            // Apply prefix operators in reverse order
            for prefix in prefixes.into_iter().rev() {
                node = Node::PrefixOp {
                    op: prefix,
                    child: Box::new(node),
                };
            }

            node
        }
        Rule::Bool => {
            let bool_str = pair.as_str();
            Node::Bool(bool_str == "true")
        }
        Rule::Float => Node::Float(pair.as_str().parse().unwrap()),
        Rule::Int => {
            let int_str = pair.as_str();
            Node::Int(int_str.parse().unwrap())
        }
        Rule::FormatString => {
            let raw = pair.as_str();
            let inner = &raw[2..raw.len() - 1]; // strip f" prefix and " suffix
            let parts = parse_format_string(inner);
            Node::FormatString { parts }
        }
        Rule::String => {
            let raw = pair.as_str();
            let inner = &raw[1..raw.len() - 1]; // strip surrounding quotes
            Node::Str(unescape_string(inner))
        }
        Rule::Identifier => {
            let name = pair.as_str().to_string();
            Node::Identifier(name)
        }
        Rule::CastExpr => {
            let mut inner = pair.into_inner();
            let target_type = parse_type_name(inner.next().unwrap());
            skip_keywords(&mut inner);
            let expr = build_ast_from_expr(inner.next().unwrap());
            Node::Cast {
                target_type,
                expr: Box::new(expr),
            }
        }
        Rule::Loop => {
            let mut inner = pair.into_inner();
            skip_keywords(&mut inner);
            let body = build_ast_from_expr(inner.next().unwrap());
            Node::Loop(Box::new(body))
        }
        Rule::FnCall => {
            let mut inner = pair.into_inner();
            let first = inner.next().unwrap();
            if first.as_rule() == Rule::ArgList {
                let args: Vec<Node> = first.into_inner().map(|e| build_ast_from_expr(e)).collect();
                let name = inner.next().unwrap().as_str().to_string();
                Node::FnCall { name, args }
            } else {
                // No args, first is the Identifier
                Node::FnCall {
                    name: first.as_str().to_string(),
                    args: vec![],
                }
            }
        }
        Rule::Lambda => {
            let mut inner = pair.into_inner();
            let mut params = vec![];
            let first = inner.next().unwrap();
            let body_pair = if first.as_rule() == Rule::LambdaParams {
                for param in first.into_inner() {
                    let mut p_inner = param.into_inner();
                    let first_tok = p_inner.next().unwrap();
                    let (is_mut, pname) = if first_tok.as_rule() == Rule::MutKw {
                        (true, p_inner.next().unwrap().as_str().to_string())
                    } else {
                        (false, first_tok.as_str().to_string())
                    };
                    let type_ann = p_inner.next().map(|t| parse_type(t));
                    params.push(LambdaParam {
                        name: pname,
                        type_ann,
                        mutable: is_mut,
                    });
                }
                inner.next().unwrap()
            } else {
                // No params, first is the body Expr
                first
            };
            Node::Lambda {
                params,
                body: Box::new(build_ast_from_expr(body_pair)),
            }
        }
        Rule::Postfix => {
            let mut inner = pair.into_inner();
            let mut node = build_ast_from_expr(inner.next().unwrap());
            for child in inner {
                match child.as_rule() {
                    Rule::FieldAccess => {
                        let field_name = child.into_inner().next().unwrap().as_str().to_string();
                        node = Node::FieldAccess {
                            object: Box::new(node),
                            field: field_name,
                        };
                    }
                    Rule::MethodCall => {
                        let mut mc_inner = child.into_inner();
                        let first = mc_inner.next().unwrap();
                        let (args, method_name) = if first.as_rule() == Rule::ArgList {
                            let args: Vec<Node> =
                                first.into_inner().map(build_ast_from_expr).collect();
                            let name = mc_inner.next().unwrap().as_str().to_string();
                            (args, name)
                        } else {
                            // No args, first is Identifier
                            (vec![], first.as_str().to_string())
                        };
                        node = Node::MethodCall {
                            receiver: Box::new(node),
                            method: method_name,
                            args,
                        };
                    }
                    Rule::IndexAccess => {
                        let index_expr =
                            build_ast_from_expr(child.into_inner().next().unwrap());
                        node = Node::IndexAccess {
                            object: Box::new(node),
                            index: Box::new(index_expr),
                        };
                    }
                    _ => {}
                }
            }
            node
        }
        Rule::StructLiteral => {
            let mut inner = pair.into_inner();
            let arg_list = inner.next().unwrap(); // StructArgList
            let mut fields = vec![];
            for arg in arg_list.into_inner() {
                if arg.as_rule() == Rule::StructArg {
                    let mut a_inner = arg.into_inner();
                    let fname = a_inner.next().unwrap().as_str().to_string();
                    let fexpr = build_ast_from_expr(a_inner.next().unwrap());
                    fields.push((fname, fexpr));
                }
            }
            let name = inner.next().unwrap().as_str().to_string();
            Node::StructLiteral { name, fields }
        }
        Rule::VariantCall => {
            let mut inner = pair.into_inner();
            let first = inner.next().unwrap();
            if first.as_rule() == Rule::TypeIdent {
                // ()TypeIdent — no payload
                Node::VariantCall {
                    type_name: first.as_str().to_string(),
                    payload: None,
                }
            } else {
                // (Expr)TypeIdent
                let payload = build_ast_from_expr(first);
                let name = inner.next().unwrap().as_str().to_string();
                Node::VariantCall {
                    type_name: name,
                    payload: Some(Box::new(payload)),
                }
            }
        }
        Rule::TypeIdent => {
            // Bare PascalCase name in expression context (enum variant without payload)
            Node::VariantCall {
                type_name: pair.as_str().to_string(),
                payload: None,
            }
        }
        Rule::ListLiteral => {
            let mut elements = vec![];
            for child in pair.into_inner() {
                if child.as_rule() == Rule::ListElements {
                    for elem in child.into_inner() {
                        elements.push(build_ast_from_expr(elem));
                    }
                }
            }
            Node::List(elements)
        }
        Rule::Break => Node::Break,
        Rule::Continue => Node::Continue,
        Rule::Block => {
            let stmts: Vec<Statement> = pair
                .into_inner()
                .filter(|p| p.as_rule() == Rule::Stmt)
                .map(build_statement)
                .collect();
            Node::Block(stmts)
        }
        unknown => panic!("Unknown expr: {:?}", unknown),
    }
}

#[cfg(test)]
mod tests {
    use crate::ast::{PrefixOp, Statement};

    fn test_expr(expected: &str, src: &str) {
        assert_eq!(
            expected,
            parse(src).unwrap().iter().fold(String::new(), |acc, arg| {
                let node_str = match arg {
                    Statement::Expr(node) => format!("{}", node),
                    Statement::Declaration {
                        mutable,
                        name,
                        type_ann,
                        value,
                    } => {
                        let kw = if *mutable { "mut" } else { "const" };
                        match type_ann {
                            Some(t) => format!("{} {}: {} = {}", kw, name, t, value),
                            None => format!("{} {} = {}", kw, name, value),
                        }
                    }
                    Statement::Reassignment { name, value } => format!("{} = {}", name, value),
                    Statement::FnDeclaration {
                        name,
                        type_params,
                        params,
                        return_type: _,
                        body,
                    } => {
                        let tp_str = if type_params.is_empty() {
                            String::new()
                        } else {
                            format!("<{}>", type_params.join(", "))
                        };
                        format!(
                            "{}({}){}fn {}",
                            tp_str,
                            params
                                .iter()
                                .map(|(n, t, m)| if *m {
                                    format!("mut {}: {}", n, t)
                                } else {
                                    format!("{}: {}", n, t)
                                })
                                .collect::<Vec<_>>()
                                .join(", "),
                            name,
                            body
                        )
                    }
                    Statement::StructDeclaration {
                        name,
                        type_params,
                        fields,
                    } => {
                        let tp_str = if type_params.is_empty() {
                            String::new()
                        } else {
                            format!("<{}>", type_params.join(", "))
                        };
                        let fields_str = fields
                            .iter()
                            .map(|(n, t)| format!("{}: {}", n, t))
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("{}{} struct {{ {} }}", tp_str, name, fields_str)
                    }
                    Statement::EnumDeclaration {
                        name,
                        type_params,
                        variants,
                    } => {
                        let tp_str = if type_params.is_empty() {
                            String::new()
                        } else {
                            format!("<{}>", type_params.join(", "))
                        };
                        let variants_str = variants
                            .iter()
                            .map(|v| {
                                if let Some(ref pt) = v.payload_type {
                                    format!("({}){}", pt, v.name)
                                } else {
                                    v.name.clone()
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("{}{} enum {{ {} }}", tp_str, name, variants_str)
                    }
                    Statement::MethodsDeclaration {
                        type_name,
                        type_params,
                        methods,
                    } => {
                        let tp_str = if type_params.is_empty() {
                            String::new()
                        } else {
                            format!("<{}>", type_params.join(", "))
                        };
                        let methods_str = methods
                            .iter()
                            .map(|m| {
                                let mtp = if m.type_params.is_empty() {
                                    String::new()
                                } else {
                                    format!("<{}>", m.type_params.join(", "))
                                };
                                let params_str = m
                                    .params
                                    .iter()
                                    .map(|(n, t, mu)| {
                                        if *mu {
                                            format!("mut {}: {}", n, t)
                                        } else {
                                            format!("{}: {}", n, t)
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                let rt = m
                                    .return_type
                                    .as_ref()
                                    .map(|r| format!(": {}", r))
                                    .unwrap_or_default();
                                format!("{}({}){}{} fn {}", mtp, params_str, m.name, rt, m.body)
                            })
                            .collect::<Vec<_>>()
                            .join("; ");
                        format!("{}{} methods {{ {} }}", tp_str, type_name, methods_str)
                    }
                    Statement::TraitImplDeclaration {
                        type_name,
                        type_params,
                        trait_name,
                        methods,
                    } => {
                        let tp_str = if type_params.is_empty() {
                            String::new()
                        } else {
                            format!("<{}>", type_params.join(", "))
                        };
                        let methods_str = methods
                            .iter()
                            .map(|m| {
                                let mtp = if m.type_params.is_empty() {
                                    String::new()
                                } else {
                                    format!("<{}>", m.type_params.join(", "))
                                };
                                let params_str = m
                                    .params
                                    .iter()
                                    .map(|(n, t, mu)| {
                                        if *mu {
                                            format!("mut {}: {}", n, t)
                                        } else {
                                            format!("{}: {}", n, t)
                                        }
                                    })
                                    .collect::<Vec<_>>()
                                    .join(", ");
                                let rt = m
                                    .return_type
                                    .as_ref()
                                    .map(|r| format!(": {}", r))
                                    .unwrap_or_default();
                                format!("{}({}){}{} fn {}", mtp, params_str, m.name, rt, m.body)
                            })
                            .collect::<Vec<_>>()
                            .join("; ");
                        format!(
                            "{}{} {} impl {{ {} }}",
                            tp_str, type_name, trait_name, methods_str
                        )
                    }
                    Statement::DestructuringDecl {
                        mutable,
                        bindings,
                        type_name,
                        value,
                    } => {
                        let kw = if *mutable { "mut" } else { "const" };
                        let bindings_str = bindings
                            .iter()
                            .map(|(n, r)| match r {
                                Some(r) => format!("{}: {}", n, r),
                                None => n.clone(),
                            })
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("{} ({}){} = {}", kw, bindings_str, type_name, value)
                    }
                };
                acc + &node_str
            })
        );
    }
    use super::*;

    // Helper to extract Node from Statement::Expr
    fn extract_node(stmts: Vec<Statement>) -> Node {
        match stmts.into_iter().next().unwrap() {
            Statement::Expr(node) => node,
            _ => panic!("Expected Expr, got Declaration or Reassignment"),
        }
    }

    #[test]
    fn basics() {
        // "b" is now a valid identifier expression
        let b = parse("b");
        assert!(b.is_ok());
        assert_eq!(extract_node(b.unwrap()), Node::Identifier("b".to_string()));

        let one = parse("1");
        assert!(one.is_ok());
        assert_eq!(extract_node(one.unwrap()), Node::Int(1));
    }

    #[test]
    fn prefix_ops() {
        let minus_one = parse("-1");
        assert!(minus_one.is_ok());
        assert_eq!(
            extract_node(minus_one.unwrap()),
            Node::PrefixOp {
                op: PrefixOp::Negative,
                child: Box::new(Node::Int(1))
            }
        );
    }

    #[test]
    fn nested_expr() {
        test_expr("1 + 2 + 3", "(1 + 2) + 3");
        test_expr("1 + (2 + 3)", "1 + (2 + 3)");
        test_expr("1 + (2 + (3 + 4))", "1 + (2 + (3 + 4))");
        test_expr("1 + 2 + (3 - 4)", "1 + 2 + (3 - 4)");
    }

    #[test]
    fn whitespace_handling() {
        // Issue #13: Parser should treat linefeed as whitespace
        let result = parse("1+2\n");
        assert!(result.is_ok());

        let result = parse("1 + 2\r\n");
        assert!(result.is_ok());
    }

    #[test]
    fn multiple_operators() {
        assert_eq!(
            extract_node(parse("1+2+3").unwrap()),
            Node::BinaryOp {
                lhs: Box::new(Node::BinaryOp {
                    lhs: Box::new(Node::Int(1)),
                    rhs: Box::new(Node::Int(2)),
                    op: BinaryOp::Add,
                }),
                rhs: Box::new(Node::Int(3)),
                op: BinaryOp::Add,
            }
        )
    }

    #[test]
    fn negative_first_number() {
        // Issue #17: First number in expression cannot be negative
        let result = parse("-1 + 2");
        assert!(result.is_ok());
        assert_eq!(
            extract_node(result.unwrap()),
            Node::BinaryOp {
                op: BinaryOp::Add,
                lhs: Box::new(Node::PrefixOp {
                    op: PrefixOp::Negative,
                    child: Box::new(Node::Int(1))
                }),
                rhs: Box::new(Node::Int(2))
            }
        );

        // Also test -2 + 5 = 3
        let result = parse("-2 + 5");
        assert!(result.is_ok());
    }

    #[test]
    fn pemdas_no_parens() {
        test_expr("3 + 4 * 2", "3 + 4 * 2");
        test_expr("3 * 4 + 2", "3 * 4 + 2");
        test_expr("3 + 4 * 2 / 5 - 1", "3 + 4 * 2 / 5 - 1");
        test_expr("3 + 4 * 2 ^ 3", "3 + 4 * 2 ^ 3");
    }

    #[test]
    fn pemdas_with_parens() {
        use crate::ast::Value;
        use crate::env::Environment;
        use crate::evaluator::eval;
        test_expr("3 + (4 * 2) ^ 3", "3 + (4 * 2) ^ 3");
        let node = extract_node(parse("3 + (4 * 2) ^ 3").unwrap());
        let mut env = Environment::new();
        assert_eq!(eval(&node, &mut env), Ok(Value::Int(515)));
    }

    #[test]
    fn bool_literals() {
        let t = parse("true");
        assert!(t.is_ok());
        assert_eq!(extract_node(t.unwrap()), Node::Bool(true));

        let f = parse("false");
        assert!(f.is_ok());
        assert_eq!(extract_node(f.unwrap()), Node::Bool(false));
    }

    #[test]
    fn not_operator() {
        let result = parse("!true");
        assert!(result.is_ok());
        assert_eq!(
            extract_node(result.unwrap()),
            Node::PrefixOp {
                op: PrefixOp::Not,
                child: Box::new(Node::Bool(true))
            }
        );

        // Double Negative
        let result = parse("!!false");
        assert!(result.is_ok());
        assert_eq!(
            extract_node(result.unwrap()),
            Node::PrefixOp {
                op: PrefixOp::Not,
                child: Box::new(Node::PrefixOp {
                    op: PrefixOp::Not,
                    child: Box::new(Node::Bool(false))
                })
            }
        );
    }

    #[test]
    fn and_operator() {
        let result = parse("true && false");
        assert!(result.is_ok());
        assert_eq!(
            extract_node(result.unwrap()),
            Node::BinaryOp {
                lhs: Box::new(Node::Bool(true)),
                rhs: Box::new(Node::Bool(false)),
                op: BinaryOp::And,
            }
        );
    }

    #[test]
    fn or_operator() {
        let result = parse("true || false");
        assert!(result.is_ok());
        assert_eq!(
            extract_node(result.unwrap()),
            Node::BinaryOp {
                lhs: Box::new(Node::Bool(true)),
                rhs: Box::new(Node::Bool(false)),
                op: BinaryOp::Or,
            }
        );
    }

    #[test]
    fn bool_precedence() {
        // && has higher precedence than ||
        // true || false && false should parse as true || (false && false)
        test_expr("true || false && false", "true || false && false");

        // With parens
        test_expr("(true || false) && false", "(true || false) && false");
    }

    #[test]
    fn bool_associativity() {
        // Both && and || are left-associative
        // true && false && true should be (true && false) && true
        test_expr("true && false && true", "true && false && true");
        test_expr("true || false || true", "true || false || true");
    }

    #[test]
    fn comparison_operators() {
        test_expr("1 == 2", "1 == 2");
        test_expr("1 != 2", "1 != 2");
        test_expr("1 < 2", "1 < 2");
        test_expr("1 > 2", "1 > 2");
        test_expr("1 <= 2", "1 <= 2");
        test_expr("1 >= 2", "1 >= 2");
    }

    #[test]
    fn comparison_precedence() {
        // Comparisons bind tighter than && but looser than +
        // 1 + 2 == 3 should parse as (1 + 2) == 3
        test_expr("1 + 2 == 3", "1 + 2 == 3");

        // true && 1 < 2 should parse as true && (1 < 2)
        test_expr("true && 1 < 2", "true && 1 < 2");
    }

    #[test]
    fn comparison_with_reassignment() {
        // x = 1 == 1 should parse as reassignment of (1 == 1) to x
        let result = parse("x = 1 == 1");
        assert!(result.is_ok());
        let stmts = result.unwrap();
        match &stmts[0] {
            Statement::Reassignment { name, value } => {
                assert_eq!(name, "x");
                assert_eq!(
                    *value,
                    Node::BinaryOp {
                        lhs: Box::new(Node::Int(1)),
                        rhs: Box::new(Node::Int(1)),
                        op: BinaryOp::Eq,
                    }
                );
            }
            _ => panic!("Expected reassignment"),
        }
    }

    #[test]
    fn if_expression() {
        let result = parse("true if { 1 }");
        assert!(result.is_ok());
        let node = extract_node(result.unwrap());
        assert_eq!(
            node,
            Node::IfElse {
                condition: Box::new(Node::Bool(true)),
                then_block: Box::new(Node::Block(vec![Statement::Expr(Node::Int(1))])),
                else_block: None,
            }
        );
    }

    #[test]
    fn if_else_expression() {
        let result = parse("true if { 1 } else { 0 }");
        assert!(result.is_ok());
        let node = extract_node(result.unwrap());
        assert_eq!(
            node,
            Node::IfElse {
                condition: Box::new(Node::Bool(true)),
                then_block: Box::new(Node::Block(vec![Statement::Expr(Node::Int(1))])),
                else_block: Some(Box::new(Node::Block(vec![Statement::Expr(Node::Int(0))]))),
            }
        );
    }

    #[test]
    fn chained_else_if() {
        // x == 1 if { 10 } else x == 2 if { 20 } else { 30 }
        let result = parse("x == 1 if { 10 } else x == 2 if { 20 } else { 30 }");
        assert!(result.is_ok());
        let node = extract_node(result.unwrap());
        assert_eq!(
            node,
            Node::IfElse {
                condition: Box::new(Node::BinaryOp {
                    lhs: Box::new(Node::Identifier("x".to_string())),
                    rhs: Box::new(Node::Int(1)),
                    op: BinaryOp::Eq,
                }),
                then_block: Box::new(Node::Block(vec![Statement::Expr(Node::Int(10))])),
                else_block: Some(Box::new(Node::IfElse {
                    condition: Box::new(Node::BinaryOp {
                        lhs: Box::new(Node::Identifier("x".to_string())),
                        rhs: Box::new(Node::Int(2)),
                        op: BinaryOp::Eq,
                    }),
                    then_block: Box::new(Node::Block(vec![Statement::Expr(Node::Int(20))])),
                    else_block: Some(Box::new(Node::Block(vec![Statement::Expr(Node::Int(30))]))),
                })),
            }
        );
    }

    #[test]
    fn if_with_block_body() {
        // Multiline block with reassignment and expression
        let result = parse("true if {\nx = 5\nx + 1\n}");
        assert!(result.is_ok());
        let node = extract_node(result.unwrap());
        assert_eq!(
            node,
            Node::IfElse {
                condition: Box::new(Node::Bool(true)),
                then_block: Box::new(Node::Block(vec![
                    Statement::Reassignment {
                        name: "x".to_string(),
                        value: Node::Int(5)
                    },
                    Statement::Expr(Node::BinaryOp {
                        lhs: Box::new(Node::Identifier("x".to_string())),
                        rhs: Box::new(Node::Int(1)),
                        op: BinaryOp::Add,
                    }),
                ])),
                else_block: None,
            }
        );
    }

    #[test]
    fn keyword_not_identifier() {
        // "if" should not be a valid identifier
        let result = parse("if = 5");
        assert!(result.is_err());

        // But "iffy" should be fine
        let result = parse("iffy = 5");
        assert!(result.is_ok());
    }

    #[test]
    fn if_else_end_to_end() {
        use crate::ast::Value;
        use crate::env::Environment;
        use crate::evaluator::eval;

        let mut env = Environment::new();

        // if-else as expression
        let stmts = parse("true if { 42 } else { 0 }").unwrap();
        let node = extract_node(stmts);
        assert_eq!(eval(&node, &mut env), Ok(Value::Int(42)));

        // false branch
        let stmts = parse("false if { 42 } else { 0 }").unwrap();
        let node = extract_node(stmts);
        assert_eq!(eval(&node, &mut env), Ok(Value::Int(0)));

        // Condition with comparison
        let stmts = parse("3 > 2 if { 1 } else { 0 }").unwrap();
        let node = extract_node(stmts);
        assert_eq!(eval(&node, &mut env), Ok(Value::Int(1)));

        // Chained else-if
        env.set("x".to_string(), Value::Int(2));
        let stmts = parse("x == 1 if { 10 } else x == 2 if { 20 } else { 30 }").unwrap();
        let node = extract_node(stmts);
        assert_eq!(eval(&node, &mut env), Ok(Value::Int(20)));
    }

    #[test]
    fn if_with_declaration_in_block() {
        use crate::ast::Value;
        use crate::env::Environment;
        use crate::evaluator::eval;

        let mut env = Environment::new();
        let stmts = parse("true if {\nconst x = 10\nx + 5\n}").unwrap();
        let node = extract_node(stmts);
        assert_eq!(eval(&node, &mut env), Ok(Value::Int(15)));
        // With block scoping, x is local to the if-block and not visible here
        assert_eq!(env.get("x"), None);
    }

    #[test]
    fn loop_expression() {
        let result = parse("loop { break }");
        assert!(result.is_ok());
        let node = extract_node(result.unwrap());
        assert_eq!(
            node,
            Node::Loop(Box::new(Node::Block(vec![Statement::Expr(Node::Break)])))
        );
    }

    #[test]
    fn while_expression() {
        let result = parse("x > 0 while { x = x - 1 }");
        assert!(result.is_ok());
        let node = extract_node(result.unwrap());
        assert_eq!(
            node,
            Node::While {
                condition: Box::new(Node::BinaryOp {
                    lhs: Box::new(Node::Identifier("x".to_string())),
                    rhs: Box::new(Node::Int(0)),
                    op: BinaryOp::Gt,
                }),
                body: Box::new(Node::Block(vec![Statement::Reassignment {
                    name: "x".to_string(),
                    value: Node::BinaryOp {
                        lhs: Box::new(Node::Identifier("x".to_string())),
                        rhs: Box::new(Node::Int(1)),
                        op: BinaryOp::Sub,
                    },
                },])),
            }
        );
    }

    #[test]
    fn break_continue_keywords() {
        // break and continue should not be valid identifiers
        assert!(parse("break = 5").is_err());
        assert!(parse("continue = 5").is_err());
        assert!(parse("loop = 5").is_err());
        assert!(parse("while = 5").is_err());

        // But prefixed names should be fine
        assert!(parse("breakpoint = 5").is_ok());
        assert!(parse("looping = 5").is_ok());
    }

    #[test]
    fn loop_while_end_to_end() {
        use crate::ast::Value;
        use crate::env::Environment;
        use crate::evaluator::eval_stmt;

        let mut env = Environment::new();

        // while loop: count down from 5 to 0
        env.set("x".to_string(), Value::Int(5));
        env.set("sum".to_string(), Value::Int(0));
        let stmts = parse("x > 0 while {\nsum = sum + x\nx = x - 1\n}").unwrap();
        for stmt in &stmts {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(env.get("sum"), Some(Value::Int(15))); // 5+4+3+2+1

        // loop with break
        env.set("n".to_string(), Value::Int(0));
        let stmts = parse("loop {\nn == 3 if { break }\nn = n + 1\n}").unwrap();
        for stmt in &stmts {
            eval_stmt(stmt, &mut env).unwrap();
        }
        assert_eq!(env.get("n"), Some(Value::Int(3)));
    }

    #[test]
    fn const_declaration_with_type() {
        let result = parse("const x: int32 = 5");
        assert!(result.is_ok());
        let stmts = result.unwrap();
        assert_eq!(
            stmts[0],
            Statement::Declaration {
                mutable: false,
                name: "x".to_string(),
                type_ann: Some(TypeAnn::Int32),
                value: Node::Int(5),
            }
        );
    }

    #[test]
    fn mut_declaration_with_type() {
        let result = parse("mut y: uint32 = 10");
        assert!(result.is_ok());
        let stmts = result.unwrap();
        assert_eq!(
            stmts[0],
            Statement::Declaration {
                mutable: true,
                name: "y".to_string(),
                type_ann: Some(TypeAnn::UInt32),
                value: Node::Int(10),
            }
        );
    }

    #[test]
    fn const_declaration_no_type() {
        let result = parse("const z = true");
        assert!(result.is_ok());
        let stmts = result.unwrap();
        assert_eq!(
            stmts[0],
            Statement::Declaration {
                mutable: false,
                name: "z".to_string(),
                type_ann: None,
                value: Node::Bool(true),
            }
        );
    }

    #[test]
    fn compconst_declaration() {
        let result = parse("compconst pi: int32 = 5");
        assert!(result.is_ok());
        let stmts = result.unwrap();
        assert_eq!(
            stmts[0],
            Statement::Declaration {
                mutable: false,
                name: "pi".to_string(),
                type_ann: Some(TypeAnn::Int32),
                value: Node::Int(5),
            }
        );
    }

    #[test]
    fn compconst_is_keyword() {
        // compconst should be a keyword, not usable as an identifier
        let result = parse("const compconst = 1");
        assert!(result.is_err());
    }

    #[test]
    fn reassignment_syntax() {
        let result = parse("x = 10");
        assert!(result.is_ok());
        let stmts = result.unwrap();
        assert_eq!(
            stmts[0],
            Statement::Reassignment {
                name: "x".to_string(),
                value: Node::Int(10),
            }
        );
    }

    #[test]
    fn const_mut_are_keywords() {
        assert!(parse("const = 5").is_err());
        assert!(parse("mut = 5").is_err());

        // Prefixed names should be fine
        assert!(parse("constant = 5").is_ok());
        assert!(parse("mutable = 5").is_ok());
    }

    #[test]
    fn type_names_are_keywords() {
        assert!(parse("int32 = 5").is_err());
        assert!(parse("uint32 = 5").is_err());
        assert!(parse("bool = 5").is_err());
    }

    #[test]
    fn declaration_with_complex_expr() {
        let result = parse("const result: int32 = 3 + 4 * 2");
        assert!(result.is_ok());
        match &result.unwrap()[0] {
            Statement::Declaration {
                mutable,
                name,
                type_ann,
                ..
            } => {
                assert_eq!(name, "result");
                assert!(!mutable);
                assert_eq!(*type_ann, Some(TypeAnn::Int32));
            }
            _ => panic!("Expected Declaration"),
        }
    }

    #[test]
    fn modulo_operator() {
        test_expr("7 % 3", "7 % 3");
        test_expr("10 % 5 + 1", "10 % 5 + 1"); // % binds tighter than +
    }

    #[test]
    fn modulo_precedence() {
        // % has same precedence as * and /
        test_expr("2 * 3 % 4", "2 * 3 % 4"); // left-assoc: (2*3) % 4
        test_expr("10 % 3 * 2", "10 % 3 * 2"); // left-assoc: (10%3) * 2
    }

    #[test]
    fn compound_add_assign() {
        let result = parse("x += 5");
        assert!(result.is_ok());
        let stmts = result.unwrap();
        // Desugars to x = x + 5
        assert_eq!(
            stmts[0],
            Statement::Reassignment {
                name: "x".to_string(),
                value: Node::BinaryOp {
                    lhs: Box::new(Node::Identifier("x".to_string())),
                    rhs: Box::new(Node::Int(5)),
                    op: BinaryOp::Add,
                },
            }
        );
    }

    #[test]
    fn compound_all_operators() {
        // All compound operators should parse
        assert!(parse("x += 1").is_ok());
        assert!(parse("x -= 1").is_ok());
        assert!(parse("x *= 2").is_ok());
        assert!(parse("x /= 2").is_ok());
        assert!(parse("x %= 3").is_ok());
    }

    #[test]
    fn compound_mod_assign() {
        let result = parse("x %= 3");
        assert!(result.is_ok());
        let stmts = result.unwrap();
        assert_eq!(
            stmts[0],
            Statement::Reassignment {
                name: "x".to_string(),
                value: Node::BinaryOp {
                    lhs: Box::new(Node::Identifier("x".to_string())),
                    rhs: Box::new(Node::Int(3)),
                    op: BinaryOp::Mod,
                },
            }
        );
    }

    #[test]
    fn fn_declaration() {
        let stmts = parse("(n: int32)double: int32 fn { n + n }").unwrap();
        assert_eq!(
            stmts[0],
            Statement::FnDeclaration {
                name: "double".to_string(),
                type_params: vec![],
                params: vec![("n".to_string(), TypeAnn::Int32, false)],
                return_type: Some(TypeAnn::Int32),
                body: Box::new(Node::Block(vec![Statement::Expr(Node::BinaryOp {
                    lhs: Box::new(Node::Identifier("n".to_string())),
                    rhs: Box::new(Node::Identifier("n".to_string())),
                    op: BinaryOp::Add,
                }),])),
            }
        );
    }

    #[test]
    fn fn_declaration_no_params() {
        let stmts = parse("()greet fn { 42 }").unwrap();
        assert_eq!(
            stmts[0],
            Statement::FnDeclaration {
                name: "greet".to_string(),
                type_params: vec![],
                params: vec![],
                return_type: None,
                body: Box::new(Node::Block(vec![Statement::Expr(Node::Int(42)),])),
            }
        );
    }

    #[test]
    fn fn_call() {
        let stmts = parse("(5)double").unwrap();
        assert_eq!(
            stmts[0],
            Statement::Expr(Node::FnCall {
                name: "double".to_string(),
                args: vec![Node::Int(5)],
            })
        );
    }

    #[test]
    fn fn_call_no_args() {
        let stmts = parse("()greet").unwrap();
        assert_eq!(
            stmts[0],
            Statement::Expr(Node::FnCall {
                name: "greet".to_string(),
                args: vec![],
            })
        );
    }

    #[test]
    fn fn_call_multi_args() {
        let stmts = parse("(1, 2)add").unwrap();
        assert_eq!(
            stmts[0],
            Statement::Expr(Node::FnCall {
                name: "add".to_string(),
                args: vec![Node::Int(1), Node::Int(2)],
            })
        );
    }

    #[test]
    fn return_syntax() {
        let stmts = parse("5 return").unwrap();
        assert_eq!(
            stmts[0],
            Statement::Expr(Node::Return(Box::new(Node::Int(5)),))
        );
    }

    #[test]
    fn fn_return_keywords() {
        // fn and return should not be valid identifiers
        assert!(parse("const fn = 5").is_err());
        assert!(parse("const return = 5").is_err());
    }

    #[test]
    fn string_literal() {
        let stmts = parse("\"hello\"").unwrap();
        assert_eq!(stmts[0], Statement::Expr(Node::Str("hello".to_string())));
    }

    #[test]
    fn string_escape_sequences() {
        let stmts = parse("\"line1\\nline2\"").unwrap();
        assert_eq!(
            stmts[0],
            Statement::Expr(Node::Str("line1\nline2".to_string()))
        );

        let stmts = parse("\"tab\\there\"").unwrap();
        assert_eq!(
            stmts[0],
            Statement::Expr(Node::Str("tab\there".to_string()))
        );

        let stmts = parse("\"escaped\\\\backslash\"").unwrap();
        assert_eq!(
            stmts[0],
            Statement::Expr(Node::Str("escaped\\backslash".to_string()))
        );

        let stmts = parse("\"say\\\"hi\\\"\"").unwrap();
        assert_eq!(
            stmts[0],
            Statement::Expr(Node::Str("say\"hi\"".to_string()))
        );
    }

    #[test]
    fn string_empty() {
        let stmts = parse("\"\"").unwrap();
        assert_eq!(stmts[0], Statement::Expr(Node::Str("".to_string())));
    }

    #[test]
    fn str_type_annotation() {
        let stmts = parse("const s: str = \"hi\"").unwrap();
        match &stmts[0] {
            Statement::Declaration { name, type_ann, .. } => {
                assert_eq!(name, "s");
                assert_eq!(*type_ann, Some(TypeAnn::Str));
            }
            _ => panic!("Expected Declaration"),
        }
    }

    #[test]
    fn fn_mutable_param() {
        let stmts = parse("(mut x: int32)inc fn { x += 1 }").unwrap();
        assert_eq!(
            stmts[0],
            Statement::FnDeclaration {
                name: "inc".to_string(),
                type_params: vec![],
                params: vec![("x".to_string(), TypeAnn::Int32, true)],
                return_type: None,
                body: Box::new(Node::Block(vec![Statement::Reassignment {
                    name: "x".to_string(),
                    value: Node::BinaryOp {
                        lhs: Box::new(Node::Identifier("x".to_string())),
                        rhs: Box::new(Node::Int(1)),
                        op: BinaryOp::Add,
                    },
                },])),
            }
        );
    }

    #[test]
    fn fn_mixed_params() {
        let stmts = parse("(mut x: int32, y: int32)foo fn { x += y }").unwrap();
        match &stmts[0] {
            Statement::FnDeclaration { params, .. } => {
                assert_eq!(params[0], ("x".to_string(), TypeAnn::Int32, true));
                assert_eq!(params[1], ("y".to_string(), TypeAnn::Int32, false));
            }
            _ => panic!("Expected FnDeclaration"),
        }
    }

    #[test]
    fn lambda_typed_params() {
        let stmts = parse("\\ x: int32 => x + 1").unwrap();
        assert_eq!(
            stmts[0],
            Statement::Expr(Node::Lambda {
                params: vec![LambdaParam {
                    name: "x".to_string(),
                    type_ann: Some(TypeAnn::Int32),
                    mutable: false
                }],
                body: Box::new(Node::BinaryOp {
                    lhs: Box::new(Node::Identifier("x".to_string())),
                    rhs: Box::new(Node::Int(1)),
                    op: BinaryOp::Add,
                }),
            })
        );
    }

    #[test]
    fn lambda_untyped_params() {
        let stmts = parse("\\ x => x").unwrap();
        assert_eq!(
            stmts[0],
            Statement::Expr(Node::Lambda {
                params: vec![LambdaParam {
                    name: "x".to_string(),
                    type_ann: None,
                    mutable: false
                }],
                body: Box::new(Node::Identifier("x".to_string())),
            })
        );
    }

    #[test]
    fn lambda_mut_param() {
        let stmts = parse("\\ mut x: int32 => x + 1").unwrap();
        match &stmts[0] {
            Statement::Expr(Node::Lambda { params, .. }) => {
                assert_eq!(params[0].mutable, true);
                assert_eq!(params[0].name, "x");
                assert_eq!(params[0].type_ann, Some(TypeAnn::Int32));
            }
            _ => panic!("Expected Lambda"),
        }
    }

    #[test]
    fn lambda_zero_params() {
        let stmts = parse("\\ => 42").unwrap();
        assert_eq!(
            stmts[0],
            Statement::Expr(Node::Lambda {
                params: vec![],
                body: Box::new(Node::Int(42)),
            })
        );
    }

    #[test]
    fn lambda_multi_params() {
        let stmts = parse("\\ x: int32, y: int32 => x + y").unwrap();
        match &stmts[0] {
            Statement::Expr(Node::Lambda { params, .. }) => {
                assert_eq!(params.len(), 2);
                assert_eq!(params[0].name, "x");
                assert_eq!(params[1].name, "y");
            }
            _ => panic!("Expected Lambda"),
        }
    }

    #[test]
    fn lambda_block_body() {
        let stmts = parse("\\ x: int32 => {\nconst y = x + 1\ny\n}").unwrap();
        match &stmts[0] {
            Statement::Expr(Node::Lambda { params, body }) => {
                assert_eq!(params.len(), 1);
                match body.as_ref() {
                    Node::Block(stmts) => assert_eq!(stmts.len(), 2),
                    _ => panic!("Expected Block body"),
                }
            }
            _ => panic!("Expected Lambda"),
        }
    }

    #[test]
    fn function_type_annotation() {
        let stmts = parse("const f: (int32): int32 = \\ x: int32 => x + 1").unwrap();
        match &stmts[0] {
            Statement::Declaration { type_ann, .. } => {
                assert_eq!(
                    *type_ann,
                    Some(TypeAnn::Fn {
                        param_types: vec![TypeAnn::Int32],
                        return_type: Some(Box::new(TypeAnn::Int32)),
                    })
                );
            }
            _ => panic!("Expected Declaration"),
        }
    }

    #[test]
    fn function_type_no_params() {
        let stmts = parse("const f: (): int32 = \\ => 42").unwrap();
        match &stmts[0] {
            Statement::Declaration { type_ann, .. } => {
                assert_eq!(
                    *type_ann,
                    Some(TypeAnn::Fn {
                        param_types: vec![],
                        return_type: Some(Box::new(TypeAnn::Int32)),
                    })
                );
            }
            _ => panic!("Expected Declaration"),
        }
    }

    #[test]
    fn fn_declaration_with_fn_type_param() {
        let stmts = parse("(f: (int32): int32, x: int32)apply: int32 fn { (x)f }").unwrap();
        match &stmts[0] {
            Statement::FnDeclaration { params, .. } => {
                assert_eq!(
                    params[0].1,
                    TypeAnn::Fn {
                        param_types: vec![TypeAnn::Int32],
                        return_type: Some(Box::new(TypeAnn::Int32)),
                    }
                );
                assert_eq!(params[1].1, TypeAnn::Int32);
            }
            _ => panic!("Expected FnDeclaration"),
        }
    }

    #[test]
    fn generic_type_annotation() {
        let stmts = parse("const xs: <int32>List = [1, 2, 3]").unwrap();
        match &stmts[0] {
            Statement::Declaration { type_ann, .. } => {
                assert_eq!(
                    *type_ann,
                    Some(TypeAnn::Generic {
                        name: "List".to_string(),
                        type_params: vec![TypeAnn::Int32],
                    })
                );
            }
            _ => panic!("Expected Declaration"),
        }
    }

    #[test]
    fn generic_fn_declaration() {
        let stmts = parse("<T>(x: T)identity: T fn { x }").unwrap();
        match &stmts[0] {
            Statement::FnDeclaration {
                name,
                type_params,
                params,
                return_type,
                ..
            } => {
                assert_eq!(name, "identity");
                assert_eq!(type_params, &vec!["T".to_string()]);
                assert_eq!(params[0].1, TypeAnn::Named("T".to_string()));
                assert_eq!(*return_type, Some(TypeAnn::Named("T".to_string())));
            }
            _ => panic!("Expected FnDeclaration"),
        }
    }

    #[test]
    fn generic_fn_multi_type_params() {
        let stmts = parse("<T, U>(x: T, y: U)first: T fn { x }").unwrap();
        match &stmts[0] {
            Statement::FnDeclaration { type_params, .. } => {
                assert_eq!(type_params, &vec!["T".to_string(), "U".to_string()]);
            }
            _ => panic!("Expected FnDeclaration"),
        }
    }

    #[test]
    fn list_literal() {
        let stmts = parse("[1, 2, 3]").unwrap();
        assert_eq!(
            stmts[0],
            Statement::Expr(Node::List(vec![Node::Int(1), Node::Int(2), Node::Int(3),]))
        );
    }

    #[test]
    fn empty_list_literal() {
        let stmts = parse("[]").unwrap();
        assert_eq!(stmts[0], Statement::Expr(Node::List(vec![])));
    }

    #[test]
    fn list_in_declaration() {
        let stmts = parse("const xs = [1, 2]").unwrap();
        match &stmts[0] {
            Statement::Declaration { name, value, .. } => {
                assert_eq!(name, "xs");
                assert_eq!(*value, Node::List(vec![Node::Int(1), Node::Int(2)]));
            }
            _ => panic!("Expected Declaration"),
        }
    }

    #[test]
    fn generic_type_with_fn_type_param() {
        // <(int32): int32>List — a list of functions
        let stmts = parse("const fs: <(int32): int32>List = []").unwrap();
        match &stmts[0] {
            Statement::Declaration { type_ann, .. } => {
                assert_eq!(
                    *type_ann,
                    Some(TypeAnn::Generic {
                        name: "List".to_string(),
                        type_params: vec![TypeAnn::Fn {
                            param_types: vec![TypeAnn::Int32],
                            return_type: Some(Box::new(TypeAnn::Int32)),
                        }],
                    })
                );
            }
            _ => panic!("Expected Declaration"),
        }
    }

    #[test]
    fn struct_declaration() {
        test_expr(
            "Point struct { x: int32, y: int32 }",
            "Point struct { x: int32, y: int32 }",
        );
    }

    #[test]
    fn generic_struct_declaration() {
        test_expr(
            "<T>Wrapper struct { value: T }",
            "<T>Wrapper struct { value: T }",
        );
    }

    #[test]
    fn enum_declaration() {
        test_expr(
            "Color enum { Red, Green, Blue }",
            "Color enum { Red, Green, Blue }",
        );
    }

    #[test]
    fn enum_with_payloads() {
        test_expr(
            "Shape enum { (int32)Circle, (int32)Square }",
            "Shape enum { (int32)Circle, (int32)Square }",
        );
    }

    #[test]
    fn generic_enum_declaration() {
        test_expr(
            "<T>Option enum { (T)Some, Nothing }",
            "<T>Option enum { (T)Some, Nothing }",
        );
    }

    #[test]
    fn struct_literal() {
        test_expr("(x: 1, y: 2)Point", "(x: 1, y: 2)Point");
    }

    #[test]
    fn variant_call_with_payload() {
        test_expr("(42)Some", "(42)Some");
    }

    #[test]
    fn bare_variant() {
        test_expr("None", "None");
    }

    #[test]
    fn field_access() {
        test_expr("p.x", "p.x");
    }

    #[test]
    fn chained_field_access() {
        test_expr("a.b.c", "a.b.c");
    }

    #[test]
    fn methods_declaration() {
        test_expr(
            "Point methods { (self: Point)get_x: int32 fn { self.x } }",
            "Point methods {\n(self: Point)get_x: int32 fn { self.x }\n}",
        );
    }

    #[test]
    fn generic_methods_declaration() {
        test_expr(
            "<T>Wrapper methods { (self: <T>Wrapper)get: T fn { self.value } }",
            "<T>Wrapper methods {\n(self: <T>Wrapper)get: T fn { self.value }\n}",
        );
    }

    #[test]
    fn method_call() {
        test_expr("p.(3)scale", "p.(3)scale");
    }

    #[test]
    fn method_call_no_args() {
        test_expr("p.()get_x", "p.()get_x");
    }

    #[test]
    fn method_call_chained_with_field_access() {
        test_expr("p.(4)scale.y", "p.(4)scale.y");
    }

    #[test]
    fn methods_keyword_protection() {
        assert!(parse("methods = 5").is_err());
        assert!(parse("methodology = 5").is_ok());
    }

    #[test]
    fn optional_type_annotation() {
        let stmts = parse("const x: int32? = DoesNotExist").unwrap();
        match &stmts[0] {
            Statement::Declaration { type_ann, .. } => {
                assert_eq!(
                    *type_ann,
                    Some(TypeAnn::Generic {
                        name: "Maybe".to_string(),
                        type_params: vec![TypeAnn::Int32],
                    })
                );
            }
            _ => panic!("Expected Declaration"),
        }
    }

    #[test]
    fn optional_generic_type() {
        let stmts = parse("const x: <int32>List? = DoesNotExist").unwrap();
        match &stmts[0] {
            Statement::Declaration { type_ann, .. } => {
                assert_eq!(
                    *type_ann,
                    Some(TypeAnn::Generic {
                        name: "Maybe".to_string(),
                        type_params: vec![TypeAnn::Generic {
                            name: "List".to_string(),
                            type_params: vec![TypeAnn::Int32],
                        }],
                    })
                );
            }
            _ => panic!("Expected Declaration"),
        }
    }

    #[test]
    fn list_type_sugar() {
        let stmts = parse("const xs: [int32] = [1, 2, 3]").unwrap();
        match &stmts[0] {
            Statement::Declaration { type_ann, .. } => {
                assert_eq!(
                    *type_ann,
                    Some(TypeAnn::Generic {
                        name: "List".to_string(),
                        type_params: vec![TypeAnn::Int32],
                    })
                );
            }
            _ => panic!("Expected Declaration"),
        }
    }

    #[test]
    fn list_type_sugar_nested() {
        // [str?] should parse as <Maybe<str>>List i.e. List of optional strings
        let stmts = parse("const xs: [str?] = []").unwrap();
        match &stmts[0] {
            Statement::Declaration { type_ann, .. } => {
                assert_eq!(
                    *type_ann,
                    Some(TypeAnn::Generic {
                        name: "List".to_string(),
                        type_params: vec![TypeAnn::Generic {
                            name: "Maybe".to_string(),
                            type_params: vec![TypeAnn::Str],
                        }],
                    })
                );
            }
            _ => panic!("Expected Declaration"),
        }
    }

    #[test]
    fn optional_type_in_fn_param() {
        let stmts = parse("(x: int32?)f fn { x }").unwrap();
        match &stmts[0] {
            Statement::FnDeclaration { params, .. } => {
                assert_eq!(
                    params[0].1,
                    TypeAnn::Generic {
                        name: "Maybe".to_string(),
                        type_params: vec![TypeAnn::Int32],
                    }
                );
            }
            _ => panic!("Expected FnDeclaration"),
        }
    }

    #[test]
    fn float_literal() {
        let stmts = parse("3.14").unwrap();
        assert_eq!(stmts[0], Statement::Expr(Node::Float(3.14)));
    }

    #[test]
    fn fl64_type_annotation() {
        let stmts = parse("const x: fl64 = 1.0").unwrap();
        assert_eq!(
            stmts[0],
            Statement::Declaration {
                mutable: false,
                name: "x".to_string(),
                type_ann: Some(TypeAnn::Fl64),
                value: Node::Float(1.0),
            }
        );
    }

    #[test]
    fn cast_expression() {
        let stmts = parse("(fl64 cast x)").unwrap();
        assert_eq!(
            stmts[0],
            Statement::Expr(Node::Cast {
                target_type: TypeAnn::Fl64,
                expr: Box::new(Node::Identifier("x".to_string())),
            })
        );
    }

    #[test]
    fn cast_in_arithmetic() {
        // (fl64 cast x) * 3.14 should parse
        let result = parse("(fl64 cast x) * 3.14");
        assert!(result.is_ok());
    }

    #[test]
    fn cast_is_keyword() {
        // cast should be a keyword, not usable as an identifier
        let result = parse("const cast = 1");
        assert!(result.is_err());
    }

    #[test]
    fn destructuring_struct() {
        let stmts = parse("const (x: a, y: b)Point = p").unwrap();
        assert_eq!(
            stmts[0],
            Statement::DestructuringDecl {
                mutable: false,
                bindings: vec![
                    ("x".to_string(), Some("a".to_string())),
                    ("y".to_string(), Some("b".to_string())),
                ],
                type_name: "Point".to_string(),
                value: Node::Identifier("p".to_string()),
            }
        );
    }

    #[test]
    fn destructuring_shorthand() {
        let stmts = parse("const (x, y)Point = p").unwrap();
        assert_eq!(
            stmts[0],
            Statement::DestructuringDecl {
                mutable: false,
                bindings: vec![("x".to_string(), None), ("y".to_string(), None),],
                type_name: "Point".to_string(),
                value: Node::Identifier("p".to_string()),
            }
        );
    }

    #[test]
    fn destructuring_enum() {
        let stmts = parse("const (val)Exists = x").unwrap();
        assert_eq!(
            stmts[0],
            Statement::DestructuringDecl {
                mutable: false,
                bindings: vec![("val".to_string(), None)],
                type_name: "Exists".to_string(),
                value: Node::Identifier("x".to_string()),
            }
        );
    }

    #[test]
    fn match_literal_arms() {
        let node = extract_node(parse("x match { 0 -> { \"zero\" }, 1 -> { \"one\" } }").unwrap());
        match node {
            Node::Match { scrutinee, arms } => {
                assert_eq!(*scrutinee, Node::Identifier("x".to_string()));
                assert_eq!(arms.len(), 2);
                assert_eq!(arms[0].pattern, Pattern::Literal(Node::Int(0)));
                assert_eq!(arms[1].pattern, Pattern::Literal(Node::Int(1)));
            }
            _ => panic!("expected Match, got {:?}", node),
        }
    }

    #[test]
    fn match_wildcard() {
        let node = extract_node(parse("x match { _ -> { 0 } }").unwrap());
        match node {
            Node::Match { arms, .. } => {
                assert_eq!(arms.len(), 1);
                assert_eq!(arms[0].pattern, Pattern::Wildcard);
            }
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn match_variant_pattern() {
        let node = extract_node(parse("m match { (v)Exists -> { v }, DoesNotExist -> { 0 } }").unwrap());
        match node {
            Node::Match { arms, .. } => {
                assert_eq!(arms.len(), 2);
                assert_eq!(
                    arms[0].pattern,
                    Pattern::Variant {
                        variant_name: "Exists".to_string(),
                        binding: "v".to_string(),
                    }
                );
                assert_eq!(arms[1].pattern, Pattern::BareVariant("DoesNotExist".to_string()));
            }
            _ => panic!("expected Match"),
        }
    }

    #[test]
    fn match_is_keyword() {
        assert!(parse("const match = 1").is_err());
    }

    #[test]
    fn pattern_bind_if() {
        let stmts = parse("x =: (v)Exists if { v }").unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Expr(Node::IfLet { scrutinee, pattern, then_block, else_block }) => {
                assert_eq!(**scrutinee, Node::Identifier("x".to_string()));
                assert_eq!(**pattern, Pattern::Variant { variant_name: "Exists".to_string(), binding: "v".to_string() });
                assert!(else_block.is_none());
                // then_block is a Block containing v
                match then_block.as_ref() {
                    Node::Block(stmts) => assert_eq!(stmts.len(), 1),
                    _ => panic!("Expected Block"),
                }
            }
            _ => panic!("Expected IfLet, got {:?}", stmts[0]),
        }
    }

    #[test]
    fn pattern_bind_if_else() {
        let stmts = parse("x =: (v)Exists if { v } else { 0 }").unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Expr(Node::IfLet { else_block, .. }) => {
                assert!(else_block.is_some());
            }
            _ => panic!("Expected IfLet with else"),
        }
    }

    #[test]
    fn pattern_bind_while() {
        let stmts = parse("x =: (v)Exists while { v }").unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Expr(Node::WhileLet { scrutinee, pattern, .. }) => {
                assert_eq!(**scrutinee, Node::Identifier("x".to_string()));
                assert_eq!(**pattern, Pattern::Variant { variant_name: "Exists".to_string(), binding: "v".to_string() });
            }
            _ => panic!("Expected WhileLet, got {:?}", stmts[0]),
        }
    }

    #[test]
    fn pattern_bind_standalone() {
        let stmts = parse("x =: (v)Exists").unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Expr(Node::PatternTest { scrutinee, pattern }) => {
                assert_eq!(**scrutinee, Node::Identifier("x".to_string()));
                assert_eq!(**pattern, Pattern::Variant { variant_name: "Exists".to_string(), binding: "v".to_string() });
            }
            _ => panic!("Expected PatternTest, got {:?}", stmts[0]),
        }
    }

    #[test]
    fn pattern_bind_else_if_chain() {
        let stmts = parse("x =: (v)Exists if { v } else y =: (w)Exists if { w } else { 0 }").unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Expr(Node::IfLet { else_block, .. }) => {
                match else_block.as_ref().unwrap().as_ref() {
                    Node::IfLet { scrutinee, pattern, else_block: inner_else, .. } => {
                        assert_eq!(**scrutinee, Node::Identifier("y".to_string()));
                        assert_eq!(**pattern, Pattern::Variant { variant_name: "Exists".to_string(), binding: "w".to_string() });
                        assert!(inner_else.is_some());
                    }
                    _ => panic!("Expected nested IfLet"),
                }
            }
            _ => panic!("Expected IfLet"),
        }
    }

    #[test]
    fn range_exclusive() {
        let stmts = parse("1..5").unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Expr(Node::Range { start, end, inclusive }) => {
                assert_eq!(**start, Node::Int(1));
                assert_eq!(**end, Node::Int(5));
                assert!(!inclusive);
            }
            _ => panic!("Expected Range, got {:?}", stmts[0]),
        }
    }

    #[test]
    fn range_inclusive() {
        let stmts = parse("1..=5").unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Expr(Node::Range { start, end, inclusive }) => {
                assert_eq!(**start, Node::Int(1));
                assert_eq!(**end, Node::Int(5));
                assert!(inclusive);
            }
            _ => panic!("Expected Range, got {:?}", stmts[0]),
        }
    }

    #[test]
    fn range_with_exprs() {
        let stmts = parse("(1 + 2)..=(3 * 4)").unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Expr(Node::Range { inclusive, .. }) => {
                assert!(inclusive);
            }
            _ => panic!("Expected Range"),
        }
    }

    #[test]
    fn index_access() {
        let stmts = parse("xs[0]").unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Expr(Node::IndexAccess { object, index }) => {
                assert_eq!(**object, Node::Identifier("xs".to_string()));
                assert_eq!(**index, Node::Int(0));
            }
            _ => panic!("Expected IndexAccess, got {:?}", stmts[0]),
        }
    }

    #[test]
    fn index_chain() {
        let stmts = parse("xs[0][1]").unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Expr(Node::IndexAccess { object, index }) => {
                assert_eq!(**index, Node::Int(1));
                match object.as_ref() {
                    Node::IndexAccess { index: inner_idx, .. } => {
                        assert_eq!(**inner_idx, Node::Int(0));
                    }
                    _ => panic!("Expected nested IndexAccess"),
                }
            }
            _ => panic!("Expected IndexAccess"),
        }
    }

    #[test]
    fn for_loop_basic() {
        let stmts = parse("xs elem x for { (x)lnprint }").unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Expr(Node::ForLoop {
                collection,
                var_name,
                ..
            }) => {
                assert_eq!(**collection, Node::Identifier("xs".to_string()));
                assert_eq!(var_name, "x");
            }
            _ => panic!("Expected ForLoop, got {:?}", stmts[0]),
        }
    }

    #[test]
    fn for_loop_range() {
        let stmts = parse("1..=5 elem i for { (i)lnprint }").unwrap();
        assert_eq!(stmts.len(), 1);
        match &stmts[0] {
            Statement::Expr(Node::ForLoop {
                collection,
                var_name,
                ..
            }) => {
                assert!(matches!(collection.as_ref(), Node::Range { inclusive: true, .. }));
                assert_eq!(var_name, "i");
            }
            _ => panic!("Expected ForLoop, got {:?}", stmts[0]),
        }
    }

    #[test]
    fn for_elem_is_keyword() {
        // "elem" should not be a valid identifier
        let result = parse("const elem = 5");
        assert!(result.is_err());

        // But "element" should be fine
        let result = parse("const element = 5");
        assert!(result.is_ok());

        // "for" should not be a valid identifier
        let result = parse("const for = 5");
        assert!(result.is_err());

        // But "formula" should be fine
        let result = parse("const formula = 5");
        assert!(result.is_ok());
    }
}
