<!-- cargo-rdme start -->

A singly linked list intended to be chained along (program) stack frames.

This is useful for visitors and recursive functions.

# Example: a JSON visitor
```rust

enum Path<'a> {
    Key(&'a str),
    Index(usize),
}

impl std::fmt::Display for Path<'_> { ... }

/// Recursively visit JSON strings, recording their path and contents
fn collect_strs<'a>(
    v: &mut Vec<(String, &'a str)>,
    path: stackstack::Stack<Path>,
    //                ^^^^^^^^^^^ shared across recursive calls
    json: &'a serde_json::Value,
) {
    match json {
        Value::String(it) => v.push((itertools::join(&path, "."), it)),
        //    iterate the path to the current node ~~^
        Value::Array(arr) => {
            for (ix, child) in arr.iter().enumerate() {
                collect_strs(v, path.pushed(Path::Index(ix)), child)
                //              ^~~ recurse with an appended path
            }
        }
        Value::Object(obj) => {
            for (k, child) in obj {
                collect_strs(v, path.pushed(Path::Key(k)), child)
                //                          ^~~ the new node is allocated on
                //                              the current (program) stack frame
            }
        },
        _ => {}
    }
}

let mut v = vec![];
let json = json!({
    "mary": {
        "had": [
            {"a": "little lamb"},
            {"two": "yaks"}
        ]
    }
});
collect_strs(&mut v, Stack::new(), &json);
assert_eq! { v, [
    ("mary.had.0.a".into(), "little lamb"),
    ("mary.had.1.two".into(), "yaks")
]}
```

<!-- cargo-rdme end -->
