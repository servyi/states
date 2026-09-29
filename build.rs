fn main() {
    // Register the verus-only cfg the same way vstd's build script does,
    // so plain `cargo check/clippy` accepts `#[cfg_attr(verus_only, ...)]`
    // without a [lints] manifest table (Servyi lint policy: no tables).
    println!("cargo::rustc-check-cfg=cfg(verus_only)");
}
