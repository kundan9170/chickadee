; PHP, held by the harness (T-3.165): the wheel's query (0.25.1) matches a call only by a
; qualified name, and its grammar parses `helper()` as a plain `name`, so it marked no call.
; This one replaces it whole.

(namespace_definition name: (namespace_name) @name) @definition.module

(interface_declaration name: (name) @name) @definition.interface

(trait_declaration name: (name) @name) @definition.interface

(class_declaration name: (name) @name) @definition.class

(enum_declaration name: (name) @name) @definition.class

(function_definition name: (name) @name) @definition.function

(method_declaration name: (name) @name) @definition.method

(class_interface_clause [(name) (qualified_name)] @name) @reference.implementation

(object_creation_expression [(name) @name (qualified_name (name) @name)]) @reference.class

(function_call_expression function: (name) @name) @reference.call

(function_call_expression function: (qualified_name (name) @name)) @reference.call

(scoped_call_expression name: (name) @name) @reference.call

(member_call_expression name: (name) @name) @reference.call

(nullsafe_member_call_expression name: (name) @name) @reference.call

(base_clause [(name) (qualified_name)] @name) @reference.class
