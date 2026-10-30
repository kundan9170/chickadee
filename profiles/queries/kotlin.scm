; Kotlin, held by the harness (T-3.165): its wheel ships no tags query. A function inside a
; class or object is a method by where it is (`graph/parse`).

(class_declaration name: (identifier) @name) @definition.class

(object_declaration name: (identifier) @name) @definition.class

(function_declaration name: (identifier) @name) @definition.function

(call_expression (identifier) @name) @reference.call

(call_expression
  (navigation_expression (_) (identifier) @name .)) @reference.call

(delegation_specifier (user_type (identifier) @name)) @reference.implementation

(delegation_specifier
  (constructor_invocation (user_type (identifier) @name))) @reference.implementation
