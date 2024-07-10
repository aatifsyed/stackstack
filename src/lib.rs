//! A singly linked list intended to be chained along stack frames.

#![cfg_attr(not(test), no_std)]

use core::ops::Range;

/// A singly-linked list intended to be chained along stack frames.
///
/// See [module documentation](mod@self) for more.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stack<'a, T> {
    /// An empty stack
    Bottom,
    /// A successor to a previous stack
    Top { head: T, tail: &'a Self },
}

impl<'a, T> Stack<'a, T> {
    /// A utility terminator for any stack.
    /// ```
    /// # use stackstack::Stack;
    /// let b = Stack::BOTTOM;
    /// let _: &Stack::<&str> = b;
    /// assert!(b.is_empty());
    /// ```
    pub const BOTTOM: &'a Self = &Self::Bottom;
    /// Returns a stack of length 1, containing only the given item.
    /// ```
    /// # use stackstack::Stack;
    /// let a = Stack::of("a");
    /// assert_eq!(a.len(), 1);
    ///
    /// ```
    pub const fn of(head: T) -> Self {
        Self::Top {
            head,
            tail: Self::BOTTOM,
        }
    }
    /// Return a new stack with the given element appended.
    /// ```
    /// # use stackstack::Stack;
    /// let a = Stack::of("a");
    /// let ab = a.pushed("b");
    /// assert_eq!(a.len(), 1);
    /// assert_eq!(ab.len(), 2);
    /// assert!(ab.iter().eq(&["a", "b"]))
    /// ```
    pub const fn pushed(&'a self, head: T) -> Self {
        Self::Top { head, tail: self }
    }
    /// Edit all the non-empty items in the given iterator to follow from `self`,
    /// returning the head of the stack.
    ///
    /// This is useful for appending items with some scratch space.
    ///
    /// ```
    /// # use stackstack::Stack;
    /// let mut scratch = ["b", "c", "d"].map(Stack::of);
    /// let a = Stack::of("a");
    /// let abcd = a.chained(&mut scratch);
    /// assert!(abcd.iter().eq(&["a", "b", "c", "d"]))
    /// ```
    pub fn chained(&'a self, iter: impl IntoIterator<Item = &'a mut Self>) -> &'a Self {
        let mut curr = self;
        for it in iter {
            match it {
                Stack::Bottom => continue,
                Stack::Top { head: _, tail } => {
                    *tail = curr;
                }
            }
            curr = it;
        }
        curr
    }
    /// Extend the current stack with each item from the given iterator, calling
    /// the given closure on the result.
    ///
    /// This creates a stack frame for each item in the iterator.
    ///
    /// ```
    /// use stackstack::Stack;
    /// let a = Stack::of("a");
    /// a.on_chained(["b", "c", "d"], |abcd| {
    ///     assert!(abcd.iter().eq(&["a", "b", "c", "d"]))
    /// })
    /// ```
    pub fn on_chained<R>(
        &self,
        chain: impl IntoIterator<Item = T>,
        on: impl FnOnce(&Stack<'_, T>) -> R,
    ) -> R {
        let mut chain = chain.into_iter();
        match chain.next() {
            Some(head) => self.pushed(head).on_chained(chain, on),
            None => on(self),
        }
    }
    /// Return the number of items in the stack.
    pub fn len(&self) -> usize {
        match self {
            Stack::Bottom => 0,
            Stack::Top { head: _, tail } => tail.len() + 1,
        }
    }
    /// Returns `true` if there are no items in the stack.
    pub fn is_empty(&self) -> bool {
        matches!(self, Stack::Bottom)
    }
    /// Get an item by 0-based index, from the bottom of the stack.
    ///
    /// ```
    /// # use stackstack::{Stack, stack};
    /// let mut storage;
    /// let abcd = stack!(["a", "b", "c", "d"] in storage);
    /// assert_eq!(abcd.get(0), Some(&"a"));
    /// assert_eq!(abcd.get(3), Some(&"d"));
    /// assert_eq!(abcd.get(4), None);
    /// ```
    pub fn get(&self, ix: usize) -> Option<&T> {
        let mut current = self;
        let depth = self.len().checked_sub(ix + 1)?;
        for _ in 0..depth {
            match current {
                Stack::Bottom => return None,
                Stack::Top { head: _, tail } => current = tail,
            }
        }

        match current {
            Stack::Bottom => None,
            Stack::Top { head, tail: _ } => Some(head),
        }
    }

    /// Return an [`Iterator`] of items in the stack, from the bottom to the top.
    ///
    /// Note that the returned item implements [`DoubleEndedIterator`].
    ///
    /// ```
    /// # use stackstack::{Stack, stack};
    /// let mut storage;
    /// let abcd = stack!(["a", "b", "c", "d"] in storage);
    /// assert!(abcd.iter().eq(&["a", "b", "c", "d"]));
    /// assert!(abcd.iter().rev().eq(&["d", "c", "b", "a"]));
    /// ```
    pub fn iter(&'a self) -> Iter<'a, T> {
        Iter {
            head: self,
            ixs: 0..self.len(),
        }
    }
}

impl<'a, T> Default for Stack<'a, T> {
    fn default() -> Self {
        Self::Bottom
    }
}

pub struct Iter<'a, T> {
    head: &'a Stack<'a, T>,
    ixs: Range<usize>,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.head.get(self.ixs.next()?)
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.head.get(self.ixs.next_back()?)
    }
}

/// Convenience macro for creating a [`Stack`] in a single stack frame.
///
/// ```
/// # use stackstack::stack;
/// let mut storage;
/// let abcd = stack![["a", "b", "c", "d"] in storage];
/// assert!(abcd.iter().eq(&["a", "b", "c", "d"]))
/// ```
#[macro_export]
macro_rules! stack {
    ([$($expr:expr),* $(,)?] in $ident:ident) => {{
        $ident = [$($expr),*].map($crate::Stack::of);
        $crate::Stack::Bottom.chained($ident.iter_mut())
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_chained() {
        Stack::of(String::from("mary")).on_chained(
            ["had".into(), "a".into(), "little".into(), "lamb".into()],
            |it| {
                assert_eq!(
                    *it,
                    Stack::Top {
                        head: String::from("lamb"),
                        tail: &Stack::Top {
                            head: String::from("little"),
                            tail: &Stack::Top {
                                head: String::from("a"),
                                tail: &Stack::Top {
                                    head: String::from("had"),
                                    tail: &Stack::Top {
                                        head: String::from("mary"),
                                        tail: &Stack::Bottom
                                    }
                                }
                            }
                        }
                    }
                )
            },
        );
    }

    #[test]
    fn get() {
        Stack::of(String::from("mary")).on_chained(
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
        Stack::of(String::from("mary")).on_chained(
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
        Stack::of(String::from("mary")).on_chained(
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
