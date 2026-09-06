//! Lowering of Plonky3 symbolic expressions into Picus expressions.
//!
//! The extractor evaluates a chip's AIR with [`crate::picus_builder::PicusBuilder`], whose
//! `Expr` type is Plonky3's own `SymbolicExpression<KoalaBear>` (so the builder satisfies the
//! `Algebra` bounds of the current `AirBuilder` trait without re-implementing field arithmetic on
//! Picus syntax).  Once evaluation is done, every recorded constraint and lookup is lowered here
//! into [`PicusExpr`] trees over a fixed variable numbering ([`VarLayout`]).
//!
//! Two details matter for the output size:
//!
//! - Plonky3 shares sub-expressions through `Arc`s; the lowering memoizes on the node address so
//!   a shared sub-tree is lowered once.  This is only sound while every symbolic expression is
//!   still alive, which is why the builder keeps them all until the final lowering pass.
//! - Sub-trees larger than `reify_threshold` nodes are bound to a fresh variable
//!   (`fresh = expr`) and replaced by it, keeping the emitted constraints small for the solver and
//!   for Lean.

use std::collections::HashMap;

use p3_air::symbolic::{BaseEntry, BaseLeaf, SymbolicExpression, SymbolicVariable};
use p3_field::PrimeField32;

use crate::pcl::{fresh_picus_var_id, Felt, PicusConstraint, PicusExpr};

/// Column-to-variable numbering shared by the builder, the lowering pass and the naming table.
///
/// Main row 0 (the local row) occupies variables `0..main_width`, so a main column index IS its
/// variable id — the selector / `is_real` specialization environments are keyed by column index
/// and need no remapping.  Row 1 (the `next` row) follows; no current chip reads it (cross-row
/// sequencing lives on lookup buses now), but the window must still exist for `AirBuilder`.
#[derive(Clone, Debug)]
pub struct VarLayout {
    pub main_width: usize,
    pub prep_width: usize,
    pub num_public: usize,
}

impl VarLayout {
    pub fn main(&self, offset: usize, col: usize) -> usize {
        offset * self.main_width + col
    }

    pub fn preprocessed(&self, offset: usize, col: usize) -> usize {
        2 * self.main_width + offset * self.prep_width + col
    }

    pub fn public(&self, i: usize) -> usize {
        2 * self.main_width + 2 * self.prep_width + i
    }

    /// First variable id available for fresh (extractor-introduced) variables.
    pub fn fresh_base(&self) -> usize {
        2 * self.main_width + 2 * self.prep_width + self.num_public + 1
    }

    pub fn var_id(&self, v: &SymbolicVariable<Felt>) -> usize {
        match v.entry {
            BaseEntry::Main { offset } => self.main(offset, v.index),
            BaseEntry::Preprocessed { offset } => self.preprocessed(offset, v.index),
            BaseEntry::Public => self.public(v.index),
            BaseEntry::Periodic => panic!("periodic columns are not used by the machine AIRs"),
        }
    }

    /// Human-readable names for the non-main variables (main columns are named from the
    /// column struct by `PicusInfo`).
    pub fn extra_names(&self) -> HashMap<usize, String> {
        let mut names = HashMap::new();
        for col in 0..self.main_width {
            names.insert(self.main(1, col), format!("next_c{col}"));
        }
        for col in 0..self.prep_width {
            names.insert(self.preprocessed(0, col), format!("prep_{col}"));
            names.insert(self.preprocessed(1, col), format!("next_prep_{col}"));
        }
        for i in 0..self.num_public {
            names.insert(self.public(i), format!("pv_{i}"));
        }
        names
    }
}

/// Lowers `SymbolicExpression`s to `PicusExpr`s with sharing-aware memoization and reification.
pub struct Lowerer<'a> {
    layout: &'a VarLayout,
    memo: HashMap<*const SymbolicExpression<Felt>, PicusExpr>,
    /// `0` disables reification.
    reify_threshold: usize,
    /// `fresh = expr` bindings introduced by reification, as constraints.
    pub bindings: Vec<PicusConstraint>,
}

impl<'a> Lowerer<'a> {
    pub fn new(layout: &'a VarLayout, reify_threshold: usize) -> Self {
        Self { layout, memo: HashMap::new(), reify_threshold, bindings: Vec::new() }
    }

    pub fn lower(&mut self, e: &SymbolicExpression<Felt>) -> PicusExpr {
        match e {
            SymbolicExpression::Leaf(leaf) => self.leaf(leaf),
            SymbolicExpression::Add { x, y, .. } => {
                let key = e as *const _;
                if let Some(hit) = self.memo.get(&key) {
                    return hit.clone();
                }
                let r = self.lower(x) + self.lower(y);
                let r = self.reify(r);
                self.memo.insert(key, r.clone());
                r
            }
            SymbolicExpression::Sub { x, y, .. } => {
                let key = e as *const _;
                if let Some(hit) = self.memo.get(&key) {
                    return hit.clone();
                }
                let r = self.lower(x) - self.lower(y);
                let r = self.reify(r);
                self.memo.insert(key, r.clone());
                r
            }
            SymbolicExpression::Mul { x, y, .. } => {
                let key = e as *const _;
                if let Some(hit) = self.memo.get(&key) {
                    return hit.clone();
                }
                let r = self.lower(x) * self.lower(y);
                let r = self.reify(r);
                self.memo.insert(key, r.clone());
                r
            }
            SymbolicExpression::Neg { x, .. } => {
                let key = e as *const _;
                if let Some(hit) = self.memo.get(&key) {
                    return hit.clone();
                }
                let r = -self.lower(x);
                let r = self.reify(r);
                self.memo.insert(key, r.clone());
                r
            }
        }
    }

    fn leaf(&self, leaf: &BaseLeaf<Felt>) -> PicusExpr {
        match leaf {
            BaseLeaf::Variable(v) => PicusExpr::Var(self.layout.var_id(v)),
            BaseLeaf::Constant(c) => PicusExpr::Const(c.as_canonical_u32() as u64),
            // Every chip is extracted as a single real row: it is simultaneously the first and
            // the last row of its (one-row) trace and no transition is available.  No machine
            // chip uses these predicates any more; the constants keep legacy call sites sound.
            BaseLeaf::IsFirstRow | BaseLeaf::IsLastRow => PicusExpr::Const(1),
            BaseLeaf::IsTransition => PicusExpr::Const(0),
        }
    }

    fn reify(&mut self, e: PicusExpr) -> PicusExpr {
        if self.reify_threshold == 0 || e.size() <= self.reify_threshold {
            return e;
        }
        if matches!(e, PicusExpr::Const(_) | PicusExpr::Var(_)) {
            return e;
        }
        let id = fresh_picus_var_id();
        self.bindings.push(PicusConstraint::new_equality(PicusExpr::Var(id), e));
        PicusExpr::Var(id)
    }
}
