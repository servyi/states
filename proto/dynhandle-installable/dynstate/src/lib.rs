//! The dynstate library, per the two-trait sketch: the TRAIT is
//! public (client entries are generic over it, monomorphized into
//! plain fn pointers — static, no allocation, nothing captured);
//! the STRUCT implementing it is private, so the capability is only
//! reachable where `install` instantiates it for the matching
//! pairing. No values are handed out at all.

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

/// The capability trait — PUBLIC, but its method is only reachable
/// through a generic `U: UnwrapMut`, and the only implementor is the
/// private [`Unwrapper`], instantiated solely by [`install`] for the
/// pairing it is building. A client cannot name the implementor, so
/// it cannot call the method on a foreign chain; generic client code
/// receives `U` already pinned to the right pairing.
pub trait UnwrapMut {
    /// The state this capability may project.
    type Type: 'static;
    /// Lend the state's own handle — `&mut Handle<'o, 'c, Self::Type>`
    /// — as a TEMPORARY inside `f`: the node pointer differs from the
    /// chain handle's (the state lives inside the box), so the handle
    /// is constructed, not cast; its BRANDS are the chain's (tree
    /// identity preserved); the closure scopes its lifetime.
    fn unwrap_mut<'o, 'c, T, R>(
        h: &mut Handle<'o, 'c, Erased<T>>,
        f: impl FnOnce(&mut Handle<'o, 'c, Self::Type>) -> R,
    ) -> R;
}

/// The private implementor: zero-sized, per state type, constructed
/// nowhere but inside `install`.
struct Unwrapper<S> {
    _p: PhantomData<fn(S)>,
}

impl<S: 'static> UnwrapMut for Unwrapper<S> {
    type Type = S;
    fn unwrap_mut<'o, 'c, T, R>(
        h: &mut Handle<'o, 'c, Erased<T>>,
        f: impl FnOnce(&mut Handle<'o, 'c, S>) -> R,
    ) -> R {
        // dig the erased object: Handle derefs to its node (the
        // Erased<T> box); borrow the box, then the trait object.
        let b: &mut Erased<T> = &mut **h;
        let dyn_ref: &mut dyn DerefMut<Target = T> = &mut **b;
        // SAFETY: the re-type lands on the `Installed<S, T>` built by
        // the `install` that instantiated this impl for its `S` — the
        // struct is private, so no other instantiation path exists.
        let installed =
            unsafe { &mut *(dyn_ref as *mut dyn DerefMut<Target = T> as *mut Installed<S, T>) };
        let node = std::ptr::from_mut(&mut installed.state);
        // SAFETY: the node is this Installed's state — live and
        // exclusively owned by the box this handle wraps; the brands
        // ride with the chain handle (no rebranding); the temporary
        // dies inside, so the borrow is scoped.
        unsafe { h.with_projected(node, f) }
    }
}

/// The secondary trait on the client's context: `new<U>` is generic
/// over the capability, so the client's entries are monomorphized
/// plain fn pointers (static, zero-sized, no capture).
pub trait ContextData<S>: Sized {
    fn new<U: UnwrapMut<Type = S>>() -> Self;
}

/// install — per the sketch: `T::new::<Unwrapper<S>>()`. No values
/// are handed out; the pairing is the instantiation itself.
pub fn install<S: 'static, T: ContextData<S>>(state: S) -> Installed<S, T> {
    Installed { ctx: T::new::<Unwrapper<S>>(), state }
}
