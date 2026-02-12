use crate::ast::{EnumVariantDef, Statement, TypeAnn, Value};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct VarInfo {
    pub value: Value,
    pub mutable: bool,
    pub type_ann: TypeAnn,
}

#[derive(Debug, Clone)]
pub struct FuncInfo {
    pub type_params: Vec<String>,
    pub params: Vec<(String, TypeAnn, bool)>,
    pub return_type: Option<TypeAnn>,
    pub body: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub struct StructDef {
    pub name: String,
    pub type_params: Vec<String>,
    pub fields: Vec<(String, TypeAnn)>,
}

#[derive(Debug, Clone)]
pub struct EnumDef {
    pub name: String,
    pub type_params: Vec<String>,
    pub variants: Vec<EnumVariantDef>,
}

/// Symbol table for storing variable bindings with lexical scoping
#[derive(Debug, Clone)]
pub struct Environment {
    scopes: Vec<HashMap<String, VarInfo>>,
    functions: HashMap<String, FuncInfo>,
    structs: HashMap<String, StructDef>,
    enums: HashMap<String, EnumDef>,
    variant_to_enum: HashMap<String, String>,
    methods: HashMap<String, HashMap<String, FuncInfo>>,
    trait_impls: HashMap<String, HashMap<String, FuncInfo>>,
}

impl Environment {
    /// Create a new environment with one empty global scope
    pub fn new() -> Self {
        let mut env = Environment {
            scopes: vec![HashMap::new()],
            functions: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            variant_to_enum: HashMap::new(),
            methods: HashMap::new(),
            trait_impls: HashMap::new(),
        };
        // Register built-in Maybe enum: <T>Maybe { (T)Exists, DoesNotExist }
        env.declare_enum(EnumDef {
            name: "Maybe".to_string(),
            type_params: vec!["T".to_string()],
            variants: vec![
                EnumVariantDef {
                    name: "Exists".to_string(),
                    payload_type: Some(TypeAnn::Named("T".to_string())),
                },
                EnumVariantDef {
                    name: "DoesNotExist".to_string(),
                    payload_type: None,
                },
            ],
        })
        .unwrap();
        // Register built-in Attempt enum: <T, E>Attempt { (T)Success, (E)Failure }
        env.declare_enum(EnumDef {
            name: "Attempt".to_string(),
            type_params: vec!["T".to_string(), "E".to_string()],
            variants: vec![
                EnumVariantDef {
                    name: "Success".to_string(),
                    payload_type: Some(TypeAnn::Named("T".to_string())),
                },
                EnumVariantDef {
                    name: "Failure".to_string(),
                    payload_type: Some(TypeAnn::Named("E".to_string())),
                },
            ],
        })
        .unwrap();
        // Register built-in Ordering enum: Ordering { Less, Equal, Greater }
        env.declare_enum(EnumDef {
            name: "Ordering".to_string(),
            type_params: vec![],
            variants: vec![
                EnumVariantDef {
                    name: "Less".to_string(),
                    payload_type: None,
                },
                EnumVariantDef {
                    name: "Equal".to_string(),
                    payload_type: None,
                },
                EnumVariantDef {
                    name: "Greater".to_string(),
                    payload_type: None,
                },
            ],
        })
        .unwrap();
        env
    }

    /// Push a new scope onto the stack
    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Pop the top scope off the stack
    pub fn pop_scope(&mut self) {
        if self.scopes.len() > 1 {
            self.scopes.pop();
        }
    }

    /// Get a variable value (searches top to bottom)
    pub fn get(&self, name: &str) -> Option<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(info) = scope.get(name) {
                return Some(info.value.clone());
            }
        }
        None
    }

    /// Get full variable info (searches top to bottom)
    pub fn get_info(&self, name: &str) -> Option<&VarInfo> {
        for scope in self.scopes.iter().rev() {
            if let Some(info) = scope.get(name) {
                return Some(info);
            }
        }
        None
    }

    /// Declare a new variable in the current (top) scope.
    /// Error if already declared in the current scope.
    pub fn declare(
        &mut self,
        name: String,
        value: Value,
        mutable: bool,
        type_ann: TypeAnn,
    ) -> Result<(), String> {
        let top = self.scopes.last_mut().unwrap();
        if top.contains_key(&name) {
            return Err(format!("Variable '{}' is already declared", name));
        }
        top.insert(
            name,
            VarInfo {
                value,
                mutable,
                type_ann,
            },
        );
        Ok(())
    }

    /// Reassign an existing mutable variable (searches top to bottom).
    pub fn reassign(&mut self, name: &str, value: Value) -> Result<(), String> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(info) = scope.get_mut(name) {
                if !info.mutable {
                    return Err(format!("Cannot reassign immutable variable '{}'", name));
                }
                info.value = value;
                return Ok(());
            }
        }
        Err(format!("Variable '{}' is not declared", name))
    }

    /// Convenience method for tests: creates/updates a mutable variable in the top scope
    pub fn set(&mut self, name: String, value: Value) {
        let type_ann = match &value {
            Value::Int(_) => TypeAnn::Int32,
            Value::UInt(_) => TypeAnn::UInt32,
            Value::Float(_) => TypeAnn::Fl64,
            Value::Bool(_) => TypeAnn::Bool,
            Value::Str(_) => TypeAnn::Str,
            Value::Unit => TypeAnn::Int32, // fallback
            Value::Closure { .. } => TypeAnn::Fn {
                param_types: vec![],
                return_type: None,
            },
            Value::List(_) => TypeAnn::Generic {
                name: "List".to_string(),
                type_params: vec![TypeAnn::Int32], // fallback for test helper
            },
            Value::Struct {
                name: sname,
                type_params,
                ..
            } => {
                if type_params.is_empty() {
                    TypeAnn::Named(sname.clone())
                } else {
                    TypeAnn::Generic {
                        name: sname.clone(),
                        type_params: type_params.clone(),
                    }
                }
            }
            Value::EnumVariant {
                enum_name,
                type_params,
                ..
            } => {
                if type_params.is_empty() {
                    TypeAnn::Named(enum_name.clone())
                } else {
                    TypeAnn::Generic {
                        name: enum_name.clone(),
                        type_params: type_params.clone(),
                    }
                }
            }
            Value::Range { .. } => TypeAnn::Named("Range".to_string()),
        };
        // Search existing scopes for update
        for scope in self.scopes.iter_mut().rev() {
            if let Some(info) = scope.get_mut(&name) {
                info.value = value;
                return;
            }
        }
        // Not found, insert into top scope
        let top = self.scopes.last_mut().unwrap();
        top.insert(
            name,
            VarInfo {
                value,
                mutable: true,
                type_ann,
            },
        );
    }

    /// Declare a function (global)
    pub fn declare_fn(&mut self, name: String, info: FuncInfo) {
        self.functions.insert(name, info);
    }

    /// Look up a function
    pub fn get_fn(&self, name: &str) -> Option<&FuncInfo> {
        self.functions.get(name)
    }

    /// Declare a struct type
    pub fn declare_struct(&mut self, def: StructDef) -> Result<(), String> {
        let name = def.name.clone();
        if self.structs.contains_key(&name) || self.enums.contains_key(&name) {
            return Err(format!("Type '{}' is already defined", name));
        }
        self.structs.insert(name, def);
        Ok(())
    }

    /// Look up a struct definition
    pub fn get_struct(&self, name: &str) -> Option<&StructDef> {
        self.structs.get(name)
    }

    /// Declare an enum type (also registers variant names globally)
    pub fn declare_enum(&mut self, def: EnumDef) -> Result<(), String> {
        let name = def.name.clone();
        if self.structs.contains_key(&name) || self.enums.contains_key(&name) {
            return Err(format!("Type '{}' is already defined", name));
        }
        for variant in &def.variants {
            if let Some(existing_enum) = self.variant_to_enum.get(&variant.name) {
                return Err(format!(
                    "Variant '{}' conflicts with variant from enum '{}'",
                    variant.name, existing_enum
                ));
            }
            self.variant_to_enum
                .insert(variant.name.clone(), name.clone());
        }
        self.enums.insert(name, def);
        Ok(())
    }

    /// Look up an enum definition
    pub fn get_enum(&self, name: &str) -> Option<&EnumDef> {
        self.enums.get(name)
    }

    /// Look up which enum a variant belongs to
    pub fn get_enum_for_variant(&self, variant_name: &str) -> Option<(&str, &EnumDef)> {
        self.variant_to_enum
            .get(variant_name)
            .and_then(|enum_name| {
                self.enums
                    .get(enum_name)
                    .map(|def| (enum_name.as_str(), def))
            })
    }

    /// Declare a method on a type
    pub fn declare_method(
        &mut self,
        type_name: &str,
        method_name: String,
        info: FuncInfo,
    ) -> Result<(), String> {
        if !self.structs.contains_key(type_name) && !self.enums.contains_key(type_name) {
            return Err(format!(
                "Cannot add methods to unknown type '{}'",
                type_name
            ));
        }
        let type_methods = self.methods.entry(type_name.to_string()).or_default();
        if type_methods.contains_key(&method_name) {
            return Err(format!(
                "Method '{}' already defined for type '{}'",
                method_name, type_name
            ));
        }
        type_methods.insert(method_name, info);
        Ok(())
    }

    /// Look up a method on a type
    pub fn get_method(&self, type_name: &str, method_name: &str) -> Option<&FuncInfo> {
        self.methods.get(type_name).and_then(|m| m.get(method_name))
    }

    /// Declare a trait implementation for a type
    pub fn declare_trait_impl(
        &mut self,
        type_name: &str,
        trait_name: String,
        info: FuncInfo,
    ) -> Result<(), String> {
        if !self.structs.contains_key(type_name) && !self.enums.contains_key(type_name) {
            return Err(format!(
                "Cannot implement trait for unknown type '{}'",
                type_name
            ));
        }
        let type_traits = self.trait_impls.entry(type_name.to_string()).or_default();
        if type_traits.contains_key(&trait_name) {
            return Err(format!(
                "Trait '{}' already implemented for type '{}'",
                trait_name, type_name
            ));
        }
        type_traits.insert(trait_name, info);
        Ok(())
    }

    /// Look up a trait implementation for a type
    pub fn get_trait_impl(&self, type_name: &str, trait_name: &str) -> Option<&FuncInfo> {
        self.trait_impls.get(type_name).and_then(|t| t.get(trait_name))
    }

    /// Save and remove all non-global scopes (for function call isolation).
    /// Returns the saved scopes to be restored later.
    pub fn save_caller_scopes(&mut self) -> Vec<HashMap<String, VarInfo>> {
        if self.scopes.len() <= 1 {
            return vec![];
        }
        self.scopes.drain(1..).collect()
    }

    /// Restore previously saved caller scopes after a function call.
    pub fn restore_caller_scopes(&mut self, saved: Vec<HashMap<String, VarInfo>>) {
        self.scopes.extend(saved);
    }
}

impl Default for Environment {
    fn default() -> Self {
        Self::new()
    }
}
