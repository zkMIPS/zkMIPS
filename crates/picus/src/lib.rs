//! Determinism extraction for the Ziren machine AIRs.
//!
//! `picus_builder` evaluates a chip and derives a Picus module; `lean` renders the same program
//! as Lean 4 theorems.  See `README.md`.
pub mod lean;
pub mod lower;
pub mod pcl;
pub mod picus_builder;

pub use pcl::*;
