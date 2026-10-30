; Swift, held by the harness (T-3.165): the wheel's query marks a method's whole class as the
; method's definition, and marks no reference, which `tests_for` needs. This one replaces it
; whole; a function inside a type is a method by where it is (`graph/parse`).

(class_declaration name: (type_identifier) @name) @definition.class

(protocol_declaration name: (type_identifier) @name) @definition.interface

(function_declaration name: (simple_identifier) @name) @definition.function

(protocol_function_declaration name: (simple_identifier) @name) @definition.method

(call_expression (simple_identifier) @name) @reference.call

(call_expression
  (navigation_expression
    suffix: (navigation_suffix suffix: (simple_identifier) @name))) @reference.call

(inheritance_specifier inherits_from: (user_type (type_identifier) @name)) @reference.implementation
