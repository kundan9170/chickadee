; C#, held by the harness (T-3.165): its wheel ships no tags query. A method is a method by
; where it is (`graph/parse`).

(namespace_declaration name: (identifier) @name) @definition.module

(namespace_declaration name: (qualified_name) @name) @definition.module

(file_scoped_namespace_declaration name: (_) @name) @definition.module

(class_declaration name: (identifier) @name) @definition.class

(struct_declaration name: (identifier) @name) @definition.class

(record_declaration name: (identifier) @name) @definition.class

(interface_declaration name: (identifier) @name) @definition.interface

(enum_declaration name: (identifier) @name) @definition.type

(method_declaration name: (identifier) @name) @definition.method

(constructor_declaration name: (identifier) @name) @definition.method

(local_function_statement name: (identifier) @name) @definition.function

(invocation_expression function: (identifier) @name) @reference.call

(invocation_expression
  function: (member_access_expression name: (identifier) @name)) @reference.call

(object_creation_expression type: (identifier) @name) @reference.class

(base_list (identifier) @name) @reference.implementation
