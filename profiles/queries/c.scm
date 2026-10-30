; C, held by the harness (T-3.165): the wheel's query marks a function's declarator, not its
; definition, takes a prototype for a definition, and marks no reference, which `tests_for`
; needs. This one replaces it whole.

(function_definition
  declarator: (function_declarator declarator: (identifier) @name)) @definition.function

(function_definition
  declarator: (pointer_declarator
    declarator: (function_declarator declarator: (identifier) @name))) @definition.function

(struct_specifier name: (type_identifier) @name body: (_)) @definition.class

(union_specifier name: (type_identifier) @name body: (_)) @definition.class

(enum_specifier name: (type_identifier) @name body: (_)) @definition.type

(type_definition declarator: (type_identifier) @name) @definition.type

(call_expression function: (identifier) @name) @reference.call

(call_expression function: (field_expression field: (field_identifier) @name)) @reference.call
