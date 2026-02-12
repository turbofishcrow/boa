#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    And,
    Or,
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
}

impl std::fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &self {
            BinaryOp::Add => write!(f, "+"),
            BinaryOp::Sub => write!(f, "-"),
            BinaryOp::Mul => write!(f, "*"),
            BinaryOp::Div => write!(f, "/"),
            BinaryOp::Mod => write!(f, "%"),
            BinaryOp::Pow => write!(f, "^"),
            BinaryOp::And => write!(f, "&&"),
            BinaryOp::Or => write!(f, "||"),
            BinaryOp::Eq => write!(f, "=="),
            BinaryOp::Neq => write!(f, "!="),
            BinaryOp::Lt => write!(f, "<"),
            BinaryOp::Gt => write!(f, ">"),
            BinaryOp::Lte => write!(f, "<="),
            BinaryOp::Gte => write!(f, ">="),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PrefixOp {
    Negative,
    Not,
}

impl std::fmt::Display for PrefixOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &self {
            PrefixOp::Negative => write!(f, "-"),
            PrefixOp::Not => write!(f, "!"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct LambdaParam {
    pub name: String,
    pub type_ann: Option<TypeAnn>,
    pub mutable: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FormatPart {
    Text(String),
    Display(Node),
    Debug(Node),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Node {
    Int(i32),
    Float(f64),
    Bool(bool),
    Str(String),
    FormatString { parts: Vec<FormatPart> },
    Identifier(String),
    PrefixOp {
        op: PrefixOp,
        child: Box<Node>,
    },
    BinaryOp {
        lhs: Box<Node>,
        rhs: Box<Node>,
        op: BinaryOp,
    },
    Block(Vec<Statement>),
    IfElse {
        condition: Box<Node>,
        then_block: Box<Node>,
        else_block: Option<Box<Node>>,
    },
    Loop(Box<Node>),
    While {
        condition: Box<Node>,
        body: Box<Node>,
    },
    Break,
    Continue,
    FnCall {
        name: String,
        args: Vec<Node>,
    },
    Return(Box<Node>),
    Lambda {
        params: Vec<LambdaParam>,
        body: Box<Node>,
    },
    List(Vec<Node>),
    StructLiteral {
        name: String,
        fields: Vec<(String, Node)>,
    },
    VariantCall {
        type_name: String,
        payload: Option<Box<Node>>,
    },
    FieldAccess {
        object: Box<Node>,
        field: String,
    },
    MethodCall {
        receiver: Box<Node>,
        method: String,
        args: Vec<Node>,
    },
    Cast {
        target_type: TypeAnn,
        expr: Box<Node>,
    },
    Match {
        scrutinee: Box<Node>,
        arms: Vec<MatchArm>,
    },
    IfLet {
        scrutinee: Box<Node>,
        pattern: Box<Pattern>,
        then_block: Box<Node>,
        else_block: Option<Box<Node>>,
    },
    WhileLet {
        scrutinee: Box<Node>,
        pattern: Box<Pattern>,
        body: Box<Node>,
    },
    PatternTest {
        scrutinee: Box<Node>,
        pattern: Box<Pattern>,
    },
    Range {
        start: Box<Node>,
        end: Box<Node>,
        inclusive: bool,
    },
    IndexAccess {
        object: Box<Node>,
        index: Box<Node>,
    },
    ForLoop {
        collection: Box<Node>,
        var_name: String,
        body: Box<Node>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Literal(Node),
    Variant {
        variant_name: String,
        binding: String,
    },
    BareVariant(String),
    Wildcard,
}

impl std::fmt::Display for Pattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Pattern::Literal(node) => write!(f, "{}", node),
            Pattern::Variant { variant_name, binding } => write!(f, "({}){}", binding, variant_name),
            Pattern::BareVariant(name) => write!(f, "{}", name),
            Pattern::Wildcard => write!(f, "_"),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchArm {
    pub pattern: Pattern,
    pub body: Box<Node>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypeAnn {
    Int32,
    UInt32,
    Fl64,
    Bool,
    Str,
    Fn {
        param_types: Vec<TypeAnn>,
        return_type: Option<Box<TypeAnn>>,
    },
    Named(String),
    Generic {
        name: String,
        type_params: Vec<TypeAnn>,
    },
}

impl std::fmt::Display for TypeAnn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            TypeAnn::Int32 => write!(f, "int32"),
            TypeAnn::UInt32 => write!(f, "uint32"),
            TypeAnn::Fl64 => write!(f, "fl64"),
            TypeAnn::Bool => write!(f, "bool"),
            TypeAnn::Str => write!(f, "str"),
            TypeAnn::Fn {
                param_types,
                return_type,
            } => {
                write!(f, "(")?;
                for (i, pt) in param_types.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", pt)?;
                }
                write!(f, ")")?;
                if let Some(rt) = return_type {
                    write!(f, ": {}", rt)?;
                }
                Ok(())
            }
            TypeAnn::Named(name) => write!(f, "{}", name),
            TypeAnn::Generic { name, type_params } => {
                write!(f, "<")?;
                for (i, tp) in type_params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", tp)?;
                }
                write!(f, ">{}", name)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EnumVariantDef {
    pub name: String,
    pub payload_type: Option<TypeAnn>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MethodDef {
    pub name: String,
    pub type_params: Vec<String>,
    pub params: Vec<(String, TypeAnn, bool)>,
    pub return_type: Option<TypeAnn>,
    pub body: Box<Node>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Expr(Node),
    Declaration {
        mutable: bool,
        name: String,
        type_ann: Option<TypeAnn>,
        value: Node,
    },
    Reassignment {
        name: String,
        value: Node,
    },
    FnDeclaration {
        name: String,
        type_params: Vec<String>,
        params: Vec<(String, TypeAnn, bool)>,
        return_type: Option<TypeAnn>,
        body: Box<Node>,
    },
    StructDeclaration {
        name: String,
        type_params: Vec<String>,
        fields: Vec<(String, TypeAnn)>,
    },
    EnumDeclaration {
        name: String,
        type_params: Vec<String>,
        variants: Vec<EnumVariantDef>,
    },
    MethodsDeclaration {
        type_name: String,
        type_params: Vec<String>,
        methods: Vec<MethodDef>,
    },
    TraitImplDeclaration {
        type_name: String,
        type_params: Vec<String>,
        trait_name: String,
        methods: Vec<MethodDef>,
    },
    DestructuringDecl {
        mutable: bool,
        bindings: Vec<(String, Option<String>)>,
        type_name: String,
        value: Node,
    },
}

/// Runtime value
#[derive(Debug, Clone)]
pub enum Value {
    Int(i32),
    UInt(u32),
    Float(f64),
    Bool(bool),
    Str(String),
    Unit,
    Closure {
        params: Vec<LambdaParam>,
        body: Box<Node>,
        env: Box<crate::env::Environment>,
    },
    List(Vec<Value>),
    Struct {
        name: String,
        type_params: Vec<TypeAnn>,
        fields: Vec<(String, Value)>,
    },
    EnumVariant {
        enum_name: String,
        variant_name: String,
        type_params: Vec<TypeAnn>,
        payload: Option<Box<Value>>,
    },
    Range {
        start: i32,
        end: i32,
        inclusive: bool,
    },
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Int(a), Value::Int(b)) => a == b,
            (Value::UInt(a), Value::UInt(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Str(a), Value::Str(b)) => a == b,
            (Value::Unit, Value::Unit) => true,
            (Value::List(a), Value::List(b)) => a == b,
            (
                Value::Struct {
                    name: a,
                    fields: af,
                    ..
                },
                Value::Struct {
                    name: b,
                    fields: bf,
                    ..
                },
            ) => a == b && af == bf,
            (
                Value::EnumVariant {
                    enum_name: ae,
                    variant_name: av,
                    payload: ap,
                    ..
                },
                Value::EnumVariant {
                    enum_name: be,
                    variant_name: bv,
                    payload: bp,
                    ..
                },
            ) => ae == be && av == bv && ap == bp,
            (
                Value::Range {
                    start: as_,
                    end: ae,
                    inclusive: ai,
                },
                Value::Range {
                    start: bs,
                    end: be,
                    inclusive: bi,
                },
            ) => as_ == bs && ae == be && ai == bi,
            (Value::Closure { .. }, _) | (_, Value::Closure { .. }) => false,
            _ => false,
        }
    }
}

impl Eq for Value {}

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "int32",
            Value::UInt(_) => "uint32",
            Value::Float(_) => "fl64",
            Value::Bool(_) => "bool",
            Value::Str(_) => "str",
            Value::Unit => "()",
            Value::Closure { .. } => "closure",
            Value::List(_) => "list",
            Value::Struct { .. } => "struct",
            Value::EnumVariant { .. } => "enum",
            Value::Range { .. } => "range",
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::UInt(n) => write!(f, "{}", n),
            Value::Float(n) => {
                if n.fract() == 0.0 && n.is_finite() {
                    write!(f, "{:.1}", n)
                } else {
                    write!(f, "{}", n)
                }
            }
            Value::Bool(b) => write!(f, "{}", b),
            Value::Str(s) => write!(f, "{}", s),
            Value::Unit => write!(f, "()"),
            Value::Closure { params, .. } => {
                write!(f, "<closure(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p.name)?;
                }
                write!(f, ")>")
            }
            Value::List(elems) => {
                write!(f, "[")?;
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", elem)?;
                }
                write!(f, "]")
            }
            Value::Struct { name, fields, .. } => {
                write!(f, "(")?;
                for (i, (fname, fval)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", fname, fval)?;
                }
                write!(f, "){}", name)
            }
            Value::EnumVariant {
                variant_name,
                payload,
                ..
            } => match payload {
                Some(val) => write!(f, "({}){}", val, variant_name),
                None => write!(f, "{}", variant_name),
            },
            Value::Range {
                start,
                end,
                inclusive,
            } => {
                if *inclusive {
                    write!(f, "{}..={}", start, end)
                } else {
                    write!(f, "{}..{}", start, end)
                }
            }
        }
    }
}

fn precedence(op: &BinaryOp) -> i32 {
    match op {
        BinaryOp::Or => 0,
        BinaryOp::And => 1,
        BinaryOp::Eq
        | BinaryOp::Neq
        | BinaryOp::Lt
        | BinaryOp::Gt
        | BinaryOp::Lte
        | BinaryOp::Gte => 2,
        BinaryOp::Add | BinaryOp::Sub => 3,
        BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod => 4,
        BinaryOp::Pow => 5,
    }
}

impl Node {
    fn needs_parens(&self, parent_op: &BinaryOp, is_rhs: bool) -> bool {
        match self {
            Node::BinaryOp { op, .. } => {
                let child_prec = precedence(op);
                let parent_prec = precedence(parent_op);
                // Need parens if child has lower precedence
                if child_prec < parent_prec {
                    return true;
                }
                // For same precedence and right side, need parens for left-associative ops
                // Pow is right-associative, so 2^3^4 = 2^(3^4), no parens on RHS needed
                // All other ops are left-associative, so we need parens on RHS if same precedence
                // Comparison operators are non-chainable, all others except Pow are left-associative
                if child_prec == parent_prec && is_rhs && !matches!(parent_op, BinaryOp::Pow) {
                    return true;
                }
                false
            }
            _ => false,
        }
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match &self {
            Node::Int(n) => write!(f, "{}", n),
            Node::Float(n) => {
                if n.fract() == 0.0 && n.is_finite() {
                    write!(f, "{:.1}", n)
                } else {
                    write!(f, "{}", n)
                }
            }
            Node::Bool(b) => write!(f, "{}", b),
            Node::Str(s) => write!(f, "\"{}\"", s),
            Node::FormatString { parts } => {
                write!(f, "f\"")?;
                for part in parts {
                    match part {
                        FormatPart::Text(s) => write!(f, "{}", s)?,
                        FormatPart::Display(expr) => write!(f, "{{{}}}", expr)?,
                        FormatPart::Debug(expr) => write!(f, "{{{}:?}}", expr)?,
                    }
                }
                write!(f, "\"")
            }
            Node::Identifier(name) => write!(f, "{}", name),
            Node::PrefixOp { op, child } => write!(f, "{}{}", op, child),
            Node::BinaryOp { lhs, rhs, op } => {
                let lhs_str = if lhs.needs_parens(op, false) {
                    format!("({})", lhs)
                } else {
                    format!("{}", lhs)
                };
                let rhs_str = if rhs.needs_parens(op, true) {
                    format!("({})", rhs)
                } else {
                    format!("{}", rhs)
                };
                write!(f, "{} {} {}", lhs_str, op, rhs_str)
            }
            Node::Block(stmts) => {
                write!(f, "{{ ")?;
                for (i, stmt) in stmts.iter().enumerate() {
                    if i > 0 {
                        write!(f, "; ")?;
                    }
                    match stmt {
                        Statement::Expr(node) => write!(f, "{}", node)?,
                        Statement::Declaration {
                            mutable,
                            name,
                            type_ann,
                            value,
                        } => {
                            let kw = if *mutable { "mut" } else { "const" };
                            match type_ann {
                                Some(t) => write!(f, "{} {}: {} = {}", kw, name, t, value)?,
                                None => write!(f, "{} {} = {}", kw, name, value)?,
                            }
                        }
                        Statement::Reassignment { name, value } => {
                            write!(f, "{} = {}", name, value)?
                        }
                        Statement::FnDeclaration {
                            name,
                            type_params,
                            params,
                            return_type,
                            body,
                        } => {
                            if !type_params.is_empty() {
                                write!(f, "<")?;
                                for (i, tp) in type_params.iter().enumerate() {
                                    if i > 0 {
                                        write!(f, ", ")?;
                                    }
                                    write!(f, "{}", tp)?;
                                }
                                write!(f, ">")?;
                            }
                            write!(f, "(")?;
                            for (i, (pname, ptype, pmut)) in params.iter().enumerate() {
                                if i > 0 {
                                    write!(f, ", ")?;
                                }
                                if *pmut {
                                    write!(f, "mut ")?;
                                }
                                write!(f, "{}: {}", pname, ptype)?;
                            }
                            write!(f, "){}", name)?;
                            if let Some(rt) = return_type {
                                write!(f, ": {}", rt)?;
                            }
                            write!(f, " fn {}", body)?;
                        }
                        Statement::StructDeclaration {
                            name,
                            type_params,
                            fields,
                        } => {
                            if !type_params.is_empty() {
                                write!(f, "<{}>", type_params.join(", "))?;
                            }
                            write!(f, "{} struct {{ ", name)?;
                            for (i, (fname, ftype)) in fields.iter().enumerate() {
                                if i > 0 {
                                    write!(f, ", ")?;
                                }
                                write!(f, "{}: {}", fname, ftype)?;
                            }
                            write!(f, " }}")?;
                        }
                        Statement::EnumDeclaration {
                            name,
                            type_params,
                            variants,
                        } => {
                            if !type_params.is_empty() {
                                write!(f, "<{}>", type_params.join(", "))?;
                            }
                            write!(f, "{} enum {{ ", name)?;
                            for (i, v) in variants.iter().enumerate() {
                                if i > 0 {
                                    write!(f, ", ")?;
                                }
                                if let Some(ref pt) = v.payload_type {
                                    write!(f, "({}){}", pt, v.name)?;
                                } else {
                                    write!(f, "{}", v.name)?;
                                }
                            }
                            write!(f, " }}")?;
                        }
                        Statement::MethodsDeclaration {
                            type_name,
                            type_params,
                            methods,
                        } => {
                            if !type_params.is_empty() {
                                write!(f, "<{}>", type_params.join(", "))?;
                            }
                            write!(f, "{} methods {{ ", type_name)?;
                            for (i, m) in methods.iter().enumerate() {
                                if i > 0 {
                                    write!(f, "; ")?;
                                }
                                let tp_str = if m.type_params.is_empty() {
                                    String::new()
                                } else {
                                    format!("<{}>", m.type_params.join(", "))
                                };
                                write!(
                                    f,
                                    "{}({}){}",
                                    tp_str,
                                    m.params
                                        .iter()
                                        .map(|(n, t, mu)| if *mu {
                                            format!("mut {}: {}", n, t)
                                        } else {
                                            format!("{}: {}", n, t)
                                        })
                                        .collect::<Vec<_>>()
                                        .join(", "),
                                    m.name
                                )?;
                                if let Some(ref rt) = m.return_type {
                                    write!(f, ": {}", rt)?;
                                }
                                write!(f, " fn {}", m.body)?;
                            }
                            write!(f, " }}")?;
                        }
                        Statement::TraitImplDeclaration {
                            type_name,
                            type_params,
                            trait_name,
                            methods,
                        } => {
                            if !type_params.is_empty() {
                                write!(f, "<{}>", type_params.join(", "))?;
                            }
                            write!(f, "{} {} impl {{ ", type_name, trait_name)?;
                            for (i, m) in methods.iter().enumerate() {
                                if i > 0 {
                                    write!(f, "; ")?;
                                }
                                let tp_str = if m.type_params.is_empty() {
                                    String::new()
                                } else {
                                    format!("<{}>", m.type_params.join(", "))
                                };
                                write!(
                                    f,
                                    "{}({}){}",
                                    tp_str,
                                    m.params
                                        .iter()
                                        .map(|(n, t, mu)| if *mu {
                                            format!("mut {}: {}", n, t)
                                        } else {
                                            format!("{}: {}", n, t)
                                        })
                                        .collect::<Vec<_>>()
                                        .join(", "),
                                    m.name
                                )?;
                                if let Some(ref rt) = m.return_type {
                                    write!(f, ": {}", rt)?;
                                }
                                write!(f, " fn {}", m.body)?;
                            }
                            write!(f, " }}")?;
                        }
                        Statement::DestructuringDecl {
                            mutable,
                            bindings,
                            type_name,
                            value,
                        } => {
                            let kw = if *mutable { "mut" } else { "const" };
                            write!(f, "{} (", kw)?;
                            for (i, (name, rename)) in bindings.iter().enumerate() {
                                if i > 0 {
                                    write!(f, ", ")?;
                                }
                                write!(f, "{}", name)?;
                                if let Some(r) = rename {
                                    write!(f, ": {}", r)?;
                                }
                            }
                            write!(f, "){} = {}", type_name, value)?;
                        }
                    }
                }
                write!(f, " }}")
            }
            Node::IfElse {
                condition,
                then_block,
                else_block,
            } => {
                write!(f, "{} if {}", condition, then_block)?;
                if let Some(else_blk) = else_block {
                    match else_blk.as_ref() {
                        Node::IfElse { .. } => write!(f, " else {}", else_blk)?,
                        _ => write!(f, " else {}", else_blk)?,
                    }
                }
                Ok(())
            }
            Node::Loop(body) => write!(f, "loop {}", body),
            Node::While { condition, body } => write!(f, "{} while {}", condition, body),
            Node::Break => write!(f, "break"),
            Node::Continue => write!(f, "continue"),
            Node::FnCall { name, args } => {
                write!(f, "(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, "){}", name)
            }
            Node::Return(expr) => write!(f, "{} return", expr),
            Node::List(elements) => {
                write!(f, "[")?;
                for (i, elem) in elements.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", elem)?;
                }
                write!(f, "]")
            }
            Node::StructLiteral { name, fields } => {
                write!(f, "(")?;
                for (i, (fname, fval)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}: {}", fname, fval)?;
                }
                write!(f, "){}", name)
            }
            Node::VariantCall { type_name, payload } => match payload {
                Some(p) => write!(f, "({}){}", p, type_name),
                None => write!(f, "{}", type_name),
            },
            Node::FieldAccess { object, field } => {
                write!(f, "{}.{}", object, field)
            }
            Node::MethodCall {
                receiver,
                method,
                args,
            } => {
                write!(f, "{}.(", receiver)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, "){}", method)
            }
            Node::Lambda { params, body } => {
                write!(f, "\\")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, " ")?;
                    if p.mutable {
                        write!(f, "mut ")?;
                    }
                    write!(f, "{}", p.name)?;
                    if let Some(ref t) = p.type_ann {
                        write!(f, ": {}", t)?;
                    }
                }
                write!(f, " => {}", body)
            }
            Node::Cast { target_type, expr } => {
                write!(f, "({} cast {})", target_type, expr)
            }
            Node::Match { scrutinee, arms } => {
                write!(f, "{} match {{ ", scrutinee)?;
                for (i, arm) in arms.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{} -> {}", arm.pattern, arm.body)?;
                }
                write!(f, " }}")
            }
            Node::IfLet {
                scrutinee,
                pattern,
                then_block,
                else_block,
            } => {
                write!(f, "{} =: {} if {}", scrutinee, pattern, then_block)?;
                if let Some(else_blk) = else_block {
                    write!(f, " else {}", else_blk)?;
                }
                Ok(())
            }
            Node::WhileLet {
                scrutinee,
                pattern,
                body,
            } => write!(f, "{} =: {} while {}", scrutinee, pattern, body),
            Node::PatternTest {
                scrutinee,
                pattern,
            } => write!(f, "{} =: {}", scrutinee, pattern),
            Node::Range {
                start,
                end,
                inclusive,
            } => {
                if *inclusive {
                    write!(f, "{}..={}", start, end)
                } else {
                    write!(f, "{}..{}", start, end)
                }
            }
            Node::IndexAccess { object, index } => write!(f, "{}[{}]", object, index),
            Node::ForLoop {
                collection,
                var_name,
                body,
            } => write!(f, "{} elem {} for {}", collection, var_name, body),
        }
    }
}
