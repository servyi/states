//! The dynstate library: `Installed<S, T>` and the PRIVATE pairing.
//! The capability is a value (`Pairing<S, T>`) handed only to `mk` —
//! constructible nowhere else — so the projection it unlocks is not
//! reachable by naming a type. The client's entries are capturing
//! closures (they hold the pairing), which is what makes the privacy
//! possible: no fn-pointer entries that would need a nameable,
//! universally-callable trait.

use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use handle::Handle;

/// The installed pair: the context `T` (what the box derefs to) and
/// the state `S` it is paired with. Built solely by [`install`].
pub struct Installed<S, T> {
    ctx: T,
    state: S,
}

impl<S, T> Deref for Installed<S, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.ctx
    }
}

impl<S, T> DerefMut for Installed<S, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.ctx
    }
}

/// The erased box: what a chain's handle wraps. The box derefs to the
/// context `T`.
pub type Erased<T> = Box<dyn DerefMut<Target = T>>;

/// The pairing capability: a zero-sized value whose TYPE names the
/// `(S, T)` it may project. Constructible ONLY by [`install`] (the
/// field is private), which hands it to `mk` — there is no public
/// projection path that does not start from this value.
pub struct Pairing<S, T>(PhantomData<fn() -> (S, T)>);

impl<S, T> Clone for Pairing<S, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S, T> Copy for Pairing<S, T> {}

impl<S: 'static, T: 'static> Pairing<S, T> {
    /// Lend the state's own handle — `&mut Handle<'o, 'c, S>` —
    /// built as a TEMPORARY inside: the node pointer differs from the
    /// chain handle's (the state lives inside the box), so the handle
    /// is constructed, not cast; its BRANDS are the chain's (tree
    /// identity preserved); the closure scopes its lifetime, so no
    /// ownership is given out.
    ///
    /// Public METHOD, private CAPABILITY: calling it requires a
    /// `Pairing` value, and values exist only where `install` handed
    /// them to `mk` — the constructor is private, so a foreign
    /// pairing cannot be conjured by any code (client or not).
    pub fn project<'o, 'c, R>(
        self,
        h: &mut Handle<'o, 'c, Erased<T>>,
        f: impl FnOnce(&mut Handle<'o, 'c, S>) -> R,
    ) -> R {
        // dig the erased object: Handle derefs to its node (the
        // Erased<T> box); borrow the box, then the trait object.
        let b: &mut Erased<T> = &mut **h;
        let dyn_ref: &mut dyn DerefMut<Target = T> = &mut **b;
        // SAFETY: the re-type lands on the `Installed<S, T>` that the
        // SAME `install` call built which minted this pairing value —
        // the value cannot be obtained any other way, so the (S, T)
        // here names exactly the box under this handle.
        let installed = unsafe { &mut *(dyn_ref as *mut dyn DerefMut<Target = T> as *mut Installed<S, T>) };
        let node = std::ptr::from_mut(&mut installed.state);
        // SAFETY: the node is this Installed's state — live and
        // exclusively owned by the box this handle wraps; the brands
        // ride with the chain handle (no rebranding); the temporary
        // dies inside, so the borrow is scoped.
        unsafe { h.with_projected(node, f) }
    }
}

/// install(S, mk) — per the sketch: mk RECEIVES the capability; it
/// builds the context (the entries, capturing the capability); the
/// pairing leaves together with the state.
pub fn install<S: 'static, T: 'static>(
    state: S,
    mk: impl FnOnce(Pairing<S, T>) -> T,
) -> Installed<S, T> {
    let ctx = mk(Pairing(PhantomData));
    Installed { ctx, state }
}
