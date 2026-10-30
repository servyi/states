//! The dynstate library, rebuilt around `Installed<S, T>` (per the
//! 15:50 comment): the erased-box pairing, generalized over the state
//! `S` and the context `T`. All the reusable parts — the wrapper, its
//! derefs, the projection, install — live here; the client carries
//! only its vocabulary and entries, zero unsafe.

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

/// The erased box's interface: derefs to the context `T`, and is
/// downcastable — so the pairing can be VERIFIED at runtime instead
/// of trusted (a capability applied to a box that is not its own
/// `Installed` fails the downcast instead of mis-typing memory).
pub trait Erasable<T>: DerefMut<Target = T> {
    fn as_any(&mut self) -> &mut dyn std::any::Any;
}

impl<S: 'static, T: 'static> Erasable<T> for Installed<S, T> {
    fn as_any(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// The erased box: what a chain's handle wraps. The box derefs to the
/// context `T`.
pub type Erased<T> = Box<dyn Erasable<T>>;

/// install(S, mk) — per the sketch: mk RECEIVES the capability (a
/// zero-sized value; its TYPE is the pairing); it builds the context
/// (the entries); the pairing leaves together.
pub fn install<S: 'static, T: 'static>(
    state: S,
    mk: impl FnOnce(Proj<S, T>) -> T,
) -> Installed<S, T> {
    let ctx = mk(Proj(PhantomData));
    Installed { ctx, state }
}

/// The projection's carrier: a zero-sized type per `(S, T)`, so the
/// (fn-pointer) entries reach the unwrapper statically without
/// capturing a value.
pub struct Proj<S, T>(PhantomData<fn() -> (S, T)>);

pub trait Unwrap<S, T> {
    /// Run `f` with the state's own handle — `&mut Handle<'o, 'c, S>`
    /// — built as a TEMPORARY inside: the node pointer differs from
    /// the chain handle's (the state lives inside the box), so the
    /// handle is constructed, not cast; its BRANDS are the chain's
    /// (tree identity preserved — no rebranding); the closure scopes
    /// its lifetime, so no ownership is given out.
    fn unwrap_mut<'o, 'c, R>(
        h: &mut Handle<'o, 'c, Erased<T>>,
        f: impl FnOnce(&mut Handle<'o, 'c, S>) -> R,
    ) -> R;
}

impl<S: 'static, T: 'static> Unwrap<S, T> for Proj<S, T> {
    fn unwrap_mut<'o, 'c, R>(
        h: &mut Handle<'o, 'c, Erased<T>>,
        f: impl FnOnce(&mut Handle<'o, 'c, S>) -> R,
    ) -> R {
        // dig the erased object (safe derefs)
        let b: &mut Erased<T> = &mut *h;
        // VERIFIED pairing, not trusted: the box must BE the
        // Installed<S, T> that install::<S, T> built — the downcast
        // is the machine check; a mismatched capability (a safe client
        // can name Proj::<Other, T> and call unwrap_mut on this box)
        // lands here and fails loudly instead of mis-typing memory.
        let Some(installed) = b.as_any().downcast_mut::<Installed<S, T>>() else {
            unreachable!(
                "pairing violated: the box is not Installed<{}> — the \
                 capability was applied to a foreign chain",
                std::any::type_name::<S>()
            );
        };
        let node = std::ptr::from_mut(&mut installed.state);
        // SAFETY: the node is this Installed's state — live and
        // exclusively owned by the box this handle wraps; the brands
        // ride with the chain handle (no rebranding); the temporary
        // dies inside, so the borrow is scoped.
        unsafe { h.with_projected(node, f) }
    }
}
