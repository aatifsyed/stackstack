//! A singly linked list intended to be chained along (program) stack frames.
//!
//! This is useful for visitors and recursive functions.
//!
//! # Example: a JSON visitor
//! ```
//! # use serde_json::{json, Value};
//! # use stackstack::Stack;
//!
//! enum Path<'a> {
//!     Key(&'a str),
//!     Index(usize),
//! }
//!
//! # const _: &str = stringify! {
//! impl std::fmt::Display for Path<'_> { ... }
//! # };
//! # impl std::fmt::Display for Path<'_> {
//! #     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//! #         match self {
//! #             Path::Key(it) => f.write_str(it),
//! #             Path::Index(it) => f.write_fmt(format_args!("{}", it)),
//! #         }
//! #     }
//! # }
//!
//! /// Recursively visit JSON strings, recording their path and contents
//! fn collect_strs<'a>(
//!     v: &mut Vec<(String, &'a str)>,
//!     path: stackstack::Stack<Path>,
//!     //                ^^^^^^^^^^^ shared across recursive calls
//!     json: &'a serde_json::Value,
//! ) {
//!     match json {
//!         Value::String(it) => v.push((itertools::join(&path, "."), it)),
//!         //    iterate the path to the current node ~~^
//!         Value::Array(arr) => {
//!             for (ix, child) in arr.iter().enumerate() {
//!                 collect_strs(v, path.pushed(Path::Index(ix)), child)
//!                 //              ^~~ recurse with an appended path
//!             }
//!         }
//!         Value::Object(obj) => {
//!             for (k, child) in obj {
//!                 collect_strs(v, path.pushed(Path::Key(k)), child)
//!                 //                          ^~~ the new node is allocated on
//!                 //                              the current (program) stack frame
//!             }
//!         },
//!         _ => {}
//!     }
//! }
//!
//! let mut v = vec![];
//! let json = json!({
//!     "mary": {
//!         "had": [
//!             {"a": "little lamb"},
//!             {"two": "yaks"}
//!         ]
//!     }
//! });
//! collect_strs(&mut v, Stack::new(), &json);
//! assert_eq! { v, [
//!     ("mary.had.0.a".into(), "little lamb"),
//!     ("mary.had.1.two".into(), "yaks")
//! ]}
//! ```

#![cfg_attr(not(test), no_std)]

use core::{fmt, hash::Hash, iter::FusedIterator, ops::Range};

/// A stack of items, where each item is stored on a program stack frame.
///
/// Implemented as a singly linked list.
///
/// See [module documentation](mod@self) for more.
#[derive(Clone, Copy)]
pub struct Stack<'a, T> {
    inner: _Stack<'a, T>,
}

impl<T, O> PartialEq<Stack<'_, O>> for Stack<'_, T>
where
    T: PartialEq<O>,
{
    fn eq(&self, other: &Stack<O>) -> bool {
        self.iter().eq(other.iter())
    }
}
/// As a convenience, you may directly compare [`Stack`]s with [slice](core::slice)s.
impl<T, O> PartialEq<[O]> for Stack<'_, T>
where
    T: PartialEq<O>,
{
    fn eq(&self, other: &[O]) -> bool {
        self.iter().eq(other.iter())
    }
}
/// As a convenience, you may directly compare [`Stack`]s with [slice](core::slice)s.
impl<T, O> PartialEq<Stack<'_, O>> for [T]
where
    T: PartialEq<O>,
{
    fn eq(&self, other: &Stack<O>) -> bool {
        self.iter().eq(other.iter())
    }
}
/// As a convenience, you may directly compare [`Stack`]s with [array](core::array)s.
impl<const N: usize, T, O> PartialEq<[O; N]> for Stack<'_, T>
where
    T: PartialEq<O>,
{
    fn eq(&self, other: &[O; N]) -> bool {
        self.iter().eq(other.iter())
    }
}
/// As a convenience, you may directly compare [`Stack`]s with [array](core::array)s.
impl<const N: usize, T, O> PartialEq<Stack<'_, O>> for [T; N]
where
    T: PartialEq<O>,
{
    fn eq(&self, other: &Stack<O>) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<T, O> PartialOrd<Stack<'_, O>> for Stack<'_, T>
where
    T: PartialOrd<O>,
{
    fn partial_cmp(&self, other: &Stack<'_, O>) -> Option<core::cmp::Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<T> Ord for Stack<'_, T>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<T> Eq for Stack<'_, T> where T: Eq {}
impl<T> Hash for Stack<'_, T>
where
    T: Hash,
{
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.len());
        for i in self {
            i.hash(state)
        }
    }
}

impl<T> fmt::Debug for Stack<'_, T>
where
    T: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self).finish()
    }
}

impl<'a, T> Stack<'a, T> {
    /// Create a new empty stack.
    /// ```
    /// # use stackstack::Stack;
    /// let stack = Stack::<&str>::new();
    /// assert!(stack.is_empty());
    /// ```
    pub const fn new() -> Self {
        Self {
            inner: _Stack::Bottom,
        }
    }
    /// Create a stack with a single element.
    /// ```
    /// # use stackstack::Stack;
    /// let stack = Stack::of("a");
    /// assert_eq!(stack.len(), 1);
    /// assert_eq!(stack.get(0), Some(&"a"));
    /// ```
    pub const fn of(item: T) -> Self {
        Self {
            inner: _Stack::Top {
                head: item,
                tail: Self::EMPTY_REF,
            },
        }
    }
    /// Return a **new** stack, with the given item at the top, which follows
    /// from `self`.
    /// ```
    /// # use stackstack::Stack;
    /// let a = Stack::of("a");
    /// let ab = a.pushed("b");
    ///
    /// assert_eq!(a.len(), 1);
    /// assert_eq!(ab.len(), 2);
    /// ```
    pub const fn pushed(&'a self, item: T) -> Self {
        Self {
            inner: _Stack::Top {
                head: item,
                tail: self,
            },
        }
    }
    /// Returns the number of items in the stack.
    /// ```
    /// # use stackstack::Stack;
    /// assert_eq!(Stack::<&str>::new().len(), 0);
    /// assert_eq!(Stack::of("a").len(), 1);
    /// ```
    pub const fn len(&self) -> usize {
        match self.inner {
            _Stack::Bottom => 0,
            _Stack::Top { head: _, tail } => 1 + tail.len(),
        }
    }
    /// Returns `true` if there are no items in the stack.
    /// ```
    /// # use stackstack::Stack;
    /// assert!(Stack::<&str>::new().is_empty());
    /// ```
    pub const fn is_empty(&self) -> bool {
        matches!(self.inner, _Stack::Bottom)
    }
    /// Get an item by 0-based index, starting at the bottom of the stack.
    /// ```
    /// # use stackstack::{Stack, stack};
    /// let mut storage;
    /// let abcd = stack!(["a", "b", "c", "d"] in storage);
    /// assert_eq!(abcd.get(0), Some(&"a"));
    /// assert_eq!(abcd.get(3), Some(&"d"));
    /// assert_eq!(abcd.get(4), None);
    /// ```
    pub const fn get(&self, ix: usize) -> Option<&T> {
        let mut current = self;
        let Some(mut depth) = self.len().checked_sub(ix + 1) else {
            return None;
        };
        while let Some(next_depth) = depth.checked_sub(1) {
            depth = next_depth;
            match current.inner {
                _Stack::Bottom => return None,
                _Stack::Top { head: _, tail } => current = tail,
            }
        }

        match &current.inner {
            _Stack::Bottom => None,
            _Stack::Top { head, tail: _ } => Some(head),
        }
    }
    /// Get the item at the bottom of the stack, or [`None`] if it is empty.
    /// ```
    /// # use stackstack::Stack;
    /// assert_eq!(Stack::<&str>::new().first(), None);
    /// assert_eq!(Stack::of("a").pushed("b").first(), Some(&"a"));
    /// ```
    pub const fn first(&self) -> Option<&T> {
        self.get(0)
    }
    /// Get the item at the top of the stack, or [`None`] if it is empty.
    /// ```
    /// # use stackstack::Stack;
    /// assert_eq!(Stack::<&str>::new().last(), None);
    /// assert_eq!(Stack::of("a").pushed("b").last(), Some(&"b"));
    /// ```
    pub const fn last(&self) -> Option<&T> {
        match &self.inner {
            _Stack::Bottom => None,
            _Stack::Top { head, tail: _ } => Some(head),
        }
    }
    /// Get a mutable reference to the item at the top of the stack, or [`None`]
    /// if it is empty.
    /// ```
    /// # use stackstack::Stack;
    /// let mut stack = Stack::of("a");
    /// *stack.last_mut().unwrap() = "b";
    /// assert_eq!(stack.last(), Some(&"b"));
    /// ```
    pub fn last_mut(&mut self) -> Option<&mut T> {
        match &mut self.inner {
            _Stack::Bottom => None,
            _Stack::Top { head, tail: _ } => Some(head),
        }
    }
    /// Return the top of the stack, and the rest, or [`None`] if it is empty.
    /// ```
    /// # use stackstack::Stack;
    /// assert_eq!(Stack::<&str>::new().split_last(), None);
    /// assert_eq!(Stack::of("a").split_last(), Some((&"a", &Stack::new())));
    /// ```
    pub fn split_last(&self) -> Option<(&T, &'a Self)> {
        match &self.inner {
            _Stack::Bottom => None,
            _Stack::Top { head, tail } => Some((head, tail)),
        }
    }
    /// Consumes the stack, returning the top item and its predecessor
    /// (or [`None`] if it was empty.)
    /// ```
    /// # use stackstack::Stack;
    /// assert_eq!(Stack::<&str>::new().into_split_last(), None);
    /// assert_eq!(Stack::of("a").into_split_last(), Some(("a", &Stack::new())));
    /// ```
    pub fn into_split_last(self) -> Option<(T, &'a Self)> {
        match self.inner {
            _Stack::Bottom => None,
            _Stack::Top { head, tail } => Some((head, tail)),
        }
    }
    /// Return an [`Iterator`] of items in the stack, from the bottom to the top.
    ///
    /// Note that the returned item implements [`DoubleEndedIterator`].
    /// ```
    /// # use stackstack::{Stack, stack};
    /// let mut storage;
    /// let abcd = stack!(["a", "b", "c", "d"] in storage);
    /// assert!(abcd.iter().eq(&["a", "b", "c", "d"]));
    /// assert!(abcd.iter().rev().eq(&["d", "c", "b", "a"]));
    /// ```
    pub fn iter(&'a self) -> Iter<'a, T> {
        Iter {
            inner: self,
            ix: 0..self.len(),
        }
    }
    /// Execute the given closure on the current list with the given item appended.
    /// ```
    /// # use stackstack::Stack;
    /// Stack::of("a").with("b", |ab| {
    ///     assert_eq!(ab, &["a", "b"])
    /// })
    /// ```
    pub fn with<R>(&self, item: T, f: impl FnOnce(&Stack<'_, T>) -> R) -> R {
        f(&self.pushed(item))
    }
    /// Extend the current stack with each item from the given iterator, calling
    /// the given closure on the result.
    ///
    /// This creates a program stack frame for each item in the iterator - beware
    /// when using with long iterators.
    ///
    /// ```
    /// use stackstack::Stack;
    /// let a = Stack::of("a");
    /// a.with_all(["b", "c", "d"], |abcd| {
    ///     assert_eq!(abcd, &["a", "b", "c", "d"])
    /// })
    /// ```
    pub fn with_all<R>(
        &self,
        items: impl IntoIterator<Item = T>,
        f: impl FnOnce(&Stack<'_, T>) -> R,
    ) -> R {
        let mut chain = items.into_iter();
        match chain.next() {
            Some(head) => self.pushed(head).with_all(chain, f),
            None => f(self),
        }
    }
    /// Edit all the non-empty stacks in the given iterator to follow from `self`,
    /// returning the new head of the stack.
    ///
    /// Note that *only the `last` items* are chained together.
    ///
    /// This is useful for appending items in some scratch space.
    ///
    /// ```
    /// # use stackstack::Stack;
    /// let mut scratch = ["b", "c", "d"].map(Stack::of);
    /// let a = Stack::of("a");
    /// let abcd = a.chained(&mut scratch);
    /// assert_eq!(abcd, &["a", "b", "c", "d"]);
    ///
    /// let e = Stack::of("f");
    /// let mut ef = e.pushed("f");
    /// //           ^~ will be ignored because it's not the last item
    /// let abcdf = abcd.chained([&mut ef]);
    /// assert_eq!(abcdf, &["a", "b", "c", "d", /* e */ "f"]);
    /// ```
    pub fn chained(&'a self, iter: impl IntoIterator<Item = &'a mut Self>) -> &'a Self {
        let mut curr = self;
        for it in iter {
            match &mut it.inner {
                _Stack::Bottom => {}
                _Stack::Top { head: _, tail } => *tail = curr,
            }
            curr = it;
        }
        curr
    }
    /// A reference to an empty stack, intended as a terminator for
    /// advanced usage.
    pub const EMPTY_REF: &'a Self = &Self {
        inner: _Stack::Bottom,
    };
}

impl<'a, T> Default for Stack<'a, T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy)]
enum _Stack<'a, T> {
    Bottom,
    Top { head: T, tail: &'a Stack<'a, T> },
}

/// Iterator over a [`Stack`]'s items.
///
/// See [`Stack::iter`].
pub struct Iter<'a, T> {
    inner: &'a Stack<'a, T>,
    ix: Range<usize>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.get(self.ix.next()?)
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.inner.get(self.ix.next_back()?)
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T> {}
impl<'a, T> FusedIterator for Iter<'a, T> {}

impl<'a, T> IntoIterator for &'a Stack<'a, T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Convenience macro for creating a [`Stack`] in a single stack frame.
///
/// ```
/// # use stackstack::stack;
/// let mut storage;
/// let abcd = stack![["a", "b", "c", "d"] in storage];
/// assert_eq!(abcd, &["a", "b", "c", "d"])
/// ```
#[macro_export]
macro_rules! stack {
    ([$($expr:expr),* $(,)?] in $ident:ident) => {{
        $ident = [$($expr),*].map($crate::Stack::of);
        $crate::Stack::EMPTY_REF.chained($ident.iter_mut())
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_chained() {
        Stack::of(String::from("mary")).with_all(
            ["had".into(), "a".into(), "little".into(), "lamb".into()],
            |actual| {
                let mut storage;
                let expected = stack![["mary", "had", "a", "little", "lamb"] in storage];
                assert_eq!(actual, expected)
            },
        );
    }

    #[test]
    fn get() {
        Stack::of(String::from("mary")).with_all(
            ["had".into(), "a".into(), "little".into(), "lamb".into()],
            |it| {
                assert_eq!(it.get(0).unwrap(), "mary");
                assert_eq!(it.get(1).unwrap(), "had");
                assert_eq!(it.get(2).unwrap(), "a");
                assert_eq!(it.get(3).unwrap(), "little");
                assert_eq!(it.get(4).unwrap(), "lamb");
                assert_eq!(it.get(5), None);
            },
        );
    }

    #[test]
    fn iter() {
        Stack::of(String::from("mary")).with_all(
            ["had".into(), "a".into(), "little".into(), "lamb".into()],
            |it| {
                assert_eq!(
                    it.iter().collect::<Vec<_>>(),
                    ["mary", "had", "a", "little", "lamb",]
                )
            },
        );
    }
    #[test]
    fn rev() {
        Stack::of(String::from("mary")).with_all(
            ["had".into(), "a".into(), "little".into(), "lamb".into()],
            |it| {
                assert_eq!(
                    it.iter().rev().collect::<Vec<_>>(),
                    ["lamb", "little", "a", "had", "mary",]
                )
            },
        );
    }
}
