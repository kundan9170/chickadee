; C++, held by the harness (T-3.165): the wheel's query marks a function's declarator, not its
; definition, and marks no reference, which `tests_for` needs. This one replaces it whole.

(function_definition
  declarator: (function_declarator declarator: (identifier) @name)) @definition.function

(function_definition
  declarator: (function_declarator declarator: (field_identifier) @name)) @definition.function

(function_definition
  declarator: (function_declarator
    declarator: (qualified_identifier name: (identifier) @name))) @definition.method

(function_definition
  declarator: (pointer_declarator
    declarator: (function_declarator declarator: (identifier) @name))) @definition.function

(function_definition
  declarator: (reference_declarator
    (function_declarator declarator: (identifier) @name))) @definition.function

(class_specifier name: (type_identifier) @name body: (_)) @definition.class

(struct_specifier name: (type_identifier) @name body: (_)) @definition.class

(union_specifier name: (type_identifier) @name body: (_)) @definition.class

(enum_specifier name: (type_identifier) @name body: (_)) @definition.type

(type_definition declarator: (type_identifier) @name) @definition.type

(namespace_definition name: (namespace_identifier) @name) @definition.module

(call_expression function: (identifier) @name) @reference.call

(call_expression function: (qualified_identifier name: (identifier) @name)) @reference.call

(call_expression function: (field_expression field: (field_identifier) @name)) @reference.call

(call_expression
  function: (template_function name: (identifier) @name)) @reference.call
