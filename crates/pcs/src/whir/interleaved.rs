//! WHIR with the interleaved commitment — the STIR-authenticated scheme.
//!
//! The flat commitment (`full_prover`) packs `2^ff` CONSECUTIVE codeword
//! positions per Merkle leaf, which folds by the RS *domain* coset-fold — a
//! coefficient-basis operation the Lagrange sumcheck tower cannot consume.
//! The interleaved commitment instead reshapes the polynomial into a
//! `[2^{v-ff} rows × 2^ff cols]` matrix — column `c` is the stride-`2^ff`
//! slice `f[h·2^ff + c]` (the low `ff` hypercube variables select the column,
//! LSB-first, exactly the variables `fix_last_variable` folds first) — and
//! RS-encodes each column independently (coefficients zero-padded, then a
//! per-column DFT).  A committed row `j` is then the vector
//!
//!   leaf_j[c] = Σ_h f(c, h) · x_j^h,      x_j = g^j
//!
//! i.e. the *evaluations over the low-variable hypercube* of the function
//! `c ↦ (f's high variables monomial-evaluated at x_j)`.  Two consequences,
//! each carried by a validated keystone (`monomial.rs`):
//!
//!   * folding the leaf with the round's Lagrange randomness `r` gives
//!     `Σ_c eq(r,c)·leaf_j[c] = mono_eval(f_folded, map_to_pow(x_j))` — the
//!     **stir value** is a monomial evaluation of the *folded* polynomial, so
//!     it joins the sumcheck as an ordinary constraint;
//!   * the constraint's weight table is the **monomial** basis at
//!     `map_to_pow(x_j)`, whose Lagrange fold has the closed form
//!     `Π_j ((1-r_j) + r_j·pt_j)` — evaluable by the verifier without the
//!     `2^v` table.
//!
//! The Lagrange (eval-claim + OOD) and monomial (STIR) constraints coexist in
//! one weight vector: the invariant is only `claim = Σ_x weight[x]·f[x]`, and
//! each constraint brings its own terminal closed form.

use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_dft::TwoAdicSubgroupDft;
use p3_field::{ExtensionField, PrimeField64, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

use crate::basefold::mle::Mle;
use crate::basefold::proof::{LeafOpening, MerkleOpening};
use crate::whir::monomial::monomial_partial_eq;
use crate::whir::proof::{ProofOfWork, SumcheckPoly, WhirProof};
use crate::whir::prover::WhirProver;
use crate::whir::sumcheck::{batched_eq_weight, WhirFolder};
use crate::whir::verifier::WhirVerifierError;

/// `[x, x², x⁴, …, x^{2^{len-1}}]` — LSB-first: element `j` is variable `j`'s
/// coordinate, so the table below aligns with the tower's hypercube indexing.
pub fn map_to_pow_lsb<EF: p3_field::Field>(x: EF, len: usize) -> Vec<EF> {
    let mut res = Vec::with_capacity(len);
    let mut e = x;
    for _ in 0..len {
        res.push(e);
        e = e.square();
    }
    res
}

/// The monomial weight table for an LSB-first point: `t[i] = Π_j pt_j^{bit_j(i)}`
/// with bit 0 the LSB — for `pt = map_to_pow_lsb(x)` this is `t[i] = x^i`.
pub fn mono_table_lsb<EF: p3_field::Field>(point_lsb: &[EF]) -> Vec<EF> {
    let rev: Vec<EF> = point_lsb.iter().rev().copied().collect();
    monomial_partial_eq(&rev)
}

/// Monomial evaluation of a value vector at an LSB-first point.
pub fn mono_eval_lsb<EF: p3_field::Field>(values: &[EF], point_lsb: &[EF]) -> EF {
    let t = mono_table_lsb(point_lsb);
    debug_assert_eq!(t.len(), values.len());
    values.iter().zip(&t).map(|(&v, &w)| v * w).sum()
}

/// Interleaved RS-encode of a BASE-field value vector: reshape to
/// `[2^{v-ff}, 2^ff]` (row-major — the low `ff` bits are the column), zero-pad
/// the rows by the blowup, per-column DFT.  Rows of the result are Merkle
/// leaves of width `2^ff`.
fn encode_interleaved_base<F, D>(
    values: &[F],
    ff: usize,
    log_inv_rate: usize,
    dft: &D,
) -> RowMajorMatrix<F>
where
    F: TwoAdicField,
    D: TwoAdicSubgroupDft<F>,
{
    let width = 1usize << ff;
    let mut padded = values.to_vec();
    padded.resize(values.len() << log_inv_rate, F::ZERO);
    dft.dft_batch(RowMajorMatrix::new(padded, width)).to_row_major_matrix()
}

/// Interleaved RS-encode of an EF value vector, flattened to base storage for
/// the base-field Mmcs: leaf width `2^ff · EF::DIMENSION`.
fn encode_interleaved_ef<F, EF, EFDft>(
    values: &[EF],
    ff: usize,
    log_inv_rate: usize,
    ef_dft: &EFDft,
) -> RowMajorMatrix<F>
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
    EFDft: TwoAdicSubgroupDft<EF>,
{
    let width = 1usize << ff;
    let mut padded = values.to_vec();
    padded.resize(values.len() << log_inv_rate, EF::ZERO);
    let ef_mat = ef_dft.dft_batch(RowMajorMatrix::new(padded, width)).to_row_major_matrix();
    let base: Vec<F> = ef_mat
        .values
        .iter()
        .flat_map(|e| e.as_basis_coefficients_slice().iter().copied())
        .collect();
    RowMajorMatrix::new(base, width * EF::DIMENSION)
}

/// Reinterpret one opened leaf row as `2^ff` EF values.
fn leaf_to_ef<F, EF>(leaf: &[F], ff: usize, base: bool) -> Vec<EF>
where
    F: TwoAdicField,
    EF: ExtensionField<F>,
{
    let out: Vec<EF> = if base {
        leaf.iter().map(|&v| EF::from(v)).collect()
    } else {
        leaf.chunks_exact(EF::DIMENSION)
            .map(|c| EF::from_basis_coefficients_iter(c.iter().copied()).unwrap())
            .collect()
    };
    debug_assert_eq!(out.len(), 1usize << ff);
    out
}

/// Fold an opened leaf (the low-`ff`-variable hypercube) at the round's
/// Lagrange folding randomness — the stir value.
fn leaf_stir_value<F, EF>(leaf: &[F], ff: usize, base: bool, folding_randomness: &[EF]) -> EF
where
    F: TwoAdicField,
    EF: ExtensionField<F> + TwoAdicField,
{
    let vals = leaf_to_ef::<F, EF>(leaf, ff, base);
    Mle::from_row_major(RowMajorMatrix::new(vals, 1)).eval_at::<EF>(folding_randomness)[0]
}

impl<EF: p3_field::Field> WhirFolder<EF> {
    /// Fold a round's STIR constraints into the running weight and claim, with
    /// batching powers CONTINUING from `start_coeff` (the OOD constraints of
    /// the same round consume the earlier powers of the same batch element).
    ///
    /// Each constraint is a MONOMIAL evaluation: `value = Σ_i t[i]·f[i]` with
    /// `t = mono_table_lsb(point)`, so adding `coeff·t` to the weight and
    /// `coeff·value` to the claim preserves `claim = Σ weight·f`.
    pub fn add_monomial_constraints(
        &mut self,
        points_lsb: &[Vec<EF>],
        values: &[EF],
        batch: EF,
        start_coeff: EF,
    ) -> EF {
        let (coeffs, next) = self.monomial_coeffs(values, batch, start_coeff);
        self.absorb_monomial_tables(points_lsb, &coeffs);
        next
    }

    /// The transcript half of [`Self::add_monomial_constraints`]: fold the
    /// constraint VALUES into the claimed sum and return the per-constraint
    /// batching coefficients (plus the next coefficient) — tiny, serial.
    /// Split out so a device backend can take over the weight absorption.
    pub fn monomial_coeffs(&mut self, values: &[EF], batch: EF, start_coeff: EF) -> (Vec<EF>, EF) {
        let mut coeffs = Vec::with_capacity(values.len());
        let mut coeff = start_coeff;
        for &val in values {
            coeffs.push(coeff);
            self.claimed_sum += coeff * val;
            coeff *= batch;
        }
        (coeffs, coeff)
    }

    /// The weight half of [`Self::add_monomial_constraints`]: absorb the
    /// batched monomial tables into the weight.
    ///
    /// This was the whir open's measured host hot spot: 84 round-0 STIR
    /// constraints x a 2^17 monomial table build + FMA each (~22M serial EF
    /// ops per shard, ~0.3-0.5 s x 304 shards on a combined reth).  Build
    /// the tables in parallel over constraints, then absorb in one parallel
    /// pass over the weight index.  Bitwise identical: EF addition is
    /// associative-exact (no floats), so any grouping gives the same value.
    pub fn absorb_monomial_tables(&mut self, points_lsb: &[Vec<EF>], coeffs: &[EF]) {
        use p3_maybe_rayon::prelude::*;
        let tables: Vec<Vec<EF>> = points_lsb.par_iter().map(|pt| mono_table_lsb(pt)).collect();
        for t in &tables {
            debug_assert_eq!(t.len(), self.weight.len());
        }
        self.weight.par_iter_mut().enumerate().for_each(|(i, w)| {
            let mut acc = EF::ZERO;
            for (c, t) in coeffs.iter().zip(&tables) {
                acc += *c * t[i];
            }
            *w += acc;
        });
    }
}

/// One accumulated terminal constraint on the verifier side.
enum Constraint<EF> {
    /// Lagrange constraint: full point (vars-when-added coords, LSB-first).
    Lagrange { point: Vec<EF>, coeff: EF, vars: usize },
    /// Monomial constraint: LSB-first point.
    Monomial { point: Vec<EF>, coeff: EF, vars: usize },
}

impl<F, EF, MT, D> WhirProver<F, EF, MT, D>
where
    F: TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F, Commitment: Clone, ProverData<RowMajorMatrix<F>>: 'static>,
    D: TwoAdicSubgroupDft<F>,
{
    /// The commit phase alone, returning what the verifier needs:
    /// `(commitment, ood_points, ood_answers)`.  Must consume the challenger
    /// exactly as the head of [`Self::prove_interleaved`] does.
    pub fn commit_interleaved_public<Challenger>(
        &self,
        challenger: &mut Challenger,
        mle: Arc<Mle<F>>,
    ) -> (MT::Commitment, Vec<Vec<EF>>, Vec<EF>)
    where
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
    {
        let n = mle.num_variables() as usize;
        let ff = self.config.round_parameters[0].folding_factor;
        let start_leaves = encode_interleaved_base(
            mle.guts().as_slice(),
            ff,
            self.config.starting_log_inv_rate,
            self.encoder.dft.as_ref(),
        );
        let (start_commit, _data) = self.mmcs.commit(alloc::vec![start_leaves]);
        challenger.observe(start_commit.clone());
        let mut pts = Vec::with_capacity(self.config.starting_ood_samples);
        let mut answers = Vec::with_capacity(self.config.starting_ood_samples);
        for _ in 0..self.config.starting_ood_samples {
            let pt: Vec<EF> = (0..n).map(|_| challenger.sample_algebra_element()).collect();
            let ans = mle.eval_at::<EF>(&pt)[0];
            challenger.observe_algebra_element(ans);
            pts.push(pt);
            answers.push(ans);
        }
        (start_commit, pts, answers)
    }

    /// The complete interleaved-commitment WHIR prover.
    pub fn prove_interleaved<EFDft, Challenger>(
        &self,
        ef_dft: Arc<EFDft>,
        challenger: &mut Challenger,
        mle: Arc<Mle<F>>,
        point: Vec<EF>,
        eval: EF,
    ) -> WhirProof<F, EF, MT>
    where
        EFDft: TwoAdicSubgroupDft<EF>,
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
    {
        let n = mle.num_variables() as usize;
        let ff = self.config.round_parameters[0].folding_factor;
        let num_rounds = self.config.round_parameters.len();

        // ---- Starting commitment: interleaved encode + commit + OOD. ----
        let start_rate = self.config.starting_log_inv_rate;
        let start_leaves = encode_interleaved_base(
            mle.guts().as_slice(),
            ff,
            start_rate,
            self.encoder.dft.as_ref(),
        );
        let start_domain_log = (n - ff) + start_rate;
        let (start_commit, start_data) = self.mmcs.commit(alloc::vec![start_leaves]);
        challenger.observe(start_commit.clone());
        let mut start_ood_points = Vec::with_capacity(self.config.starting_ood_samples);
        let mut start_ood_answers = Vec::with_capacity(self.config.starting_ood_samples);
        for _ in 0..self.config.starting_ood_samples {
            let pt: Vec<EF> = (0..n).map(|_| challenger.sample_algebra_element()).collect();
            let ans = mle.eval_at::<EF>(&pt)[0];
            challenger.observe_algebra_element(ans);
            start_ood_points.push(pt);
            start_ood_answers.push(ans);
        }

        // ---- Seat the folder (starting claim + OOD batched in). ----
        let batch: EF = challenger.sample_algebra_element();
        let mut claimed_sum = eval;
        let mut coeff = batch;
        for &a in &start_ood_answers {
            claimed_sum += coeff * a;
            coeff *= batch;
        }
        let mut folder = WhirFolder {
            f_vec: mle.guts().as_slice().iter().map(|&v| EF::from(v)).collect(),
            weight: batched_eq_weight(n, &point, &start_ood_points, batch),
            claimed_sum,
        };

        let mut prev_domain_log = start_domain_log;
        let mut prev_base = true;
        let mut prev_data = start_data;

        let mut round_sumcheck_polys: Vec<Vec<SumcheckPoly<EF>>> = Vec::new();
        let mut round_ood_answers: Vec<Vec<EF>> = Vec::new();
        let mut round_commitments: Vec<MT::Commitment> = Vec::new();
        let mut round_query_openings: Vec<MerkleOpening<F, MT>> = Vec::new();
        let mut folding_pow: Vec<ProofOfWork<F>> = Vec::new();

        for (r, round_cfg) in self.config.round_parameters.iter().enumerate() {
            // (1) Fold this round's variables (sumcheck + PoW).
            let mut this_round_randomness = Vec::new();
            let polys = folder.fold_variables::<F, _>(
                round_cfg.folding_factor,
                &round_cfg.pow_bits,
                challenger,
                &mut this_round_randomness,
            );
            round_sumcheck_polys.push(polys.iter().map(|(p, _)| p.clone()).collect());
            for (_, pow) in &polys {
                folding_pow.push(pow.clone());
            }

            if r + 1 == num_rounds {
                break;
            }

            // (2) Interleaved-encode the folded polynomial and commit it.
            // Leaf width follows the NEXT round's folding factor - that
            // round's stir fold consumes one leaf per query.
            let next_ff = self.config.round_parameters[r + 1].folding_factor;
            let rem = folder.f_vec.len().trailing_zeros() as usize;
            let leaves = encode_interleaved_ef::<F, EF, _>(
                &folder.f_vec,
                next_ff,
                round_cfg.log_inv_rate,
                ef_dft.as_ref(),
            );
            let this_domain_log = (rem - next_ff) + round_cfg.log_inv_rate;
            let (commitment, prover_data) = self.mmcs.commit(alloc::vec![leaves]);
            challenger.observe(commitment.clone());
            round_commitments.push(commitment);

            // (3) Fresh OOD on the folded polynomial (Lagrange).
            let folded = Mle::<EF>::from_row_major(RowMajorMatrix::new(folder.f_vec.clone(), 1));
            let mut ood_points = Vec::with_capacity(round_cfg.ood_samples);
            let mut ood_answers = Vec::with_capacity(round_cfg.ood_samples);
            for _ in 0..round_cfg.ood_samples {
                let pt: Vec<EF> = (0..rem).map(|_| challenger.sample_algebra_element()).collect();
                let ans = folded.eval_at::<EF>(&pt)[0];
                challenger.observe_algebra_element(ans);
                ood_points.push(pt);
                ood_answers.push(ans);
            }
            round_ood_answers.push(ood_answers.clone());

            // (4) Query PoW, then sample query indices into the PREVIOUS codeword.
            folding_pow.push(ProofOfWork(challenger.grind(round_cfg.queries_pow_bits)));
            let mask = (1usize << prev_domain_log) - 1;
            let indices: Vec<usize> = (0..round_cfg.num_queries)
                .map(|_| challenger.sample_bits(prev_domain_log) & mask)
                .collect();

            // (5) Open the previous codeword's ROWS at those indices (the leaf
            //     index IS the domain index) and fold each to a stir value.
            let g_prev = EF::two_adic_generator(prev_domain_log);
            let mut leaves_open = Vec::with_capacity(indices.len());
            let mut stir_points: Vec<Vec<EF>> = Vec::with_capacity(indices.len());
            let mut stir_values = Vec::with_capacity(indices.len());
            for &idx in &indices {
                let opening = self.mmcs.open_batch(idx, &prev_data);
                let leaf = &opening.opened_values[0];
                stir_values.push(leaf_stir_value::<F, EF>(
                    leaf,
                    round_cfg.folding_factor,
                    prev_base,
                    &this_round_randomness,
                ));
                stir_points.push(map_to_pow_lsb(g_prev.exp_u64(idx as u64), rem));
                leaves_open.push(LeafOpening {
                    values: opening.opened_values,
                    proof: opening.opening_proof,
                });
            }
            round_query_openings.push(MerkleOpening { leaves: leaves_open });

            // (6) Fold OOD (Lagrange) then STIR (monomial) constraints into the
            //     running claim and weight under ONE round batch, continuing
            //     powers across the two groups.
            let round_batch: EF = challenger.sample_algebra_element();
            folder.add_ood_constraints(&ood_points, &ood_answers, round_batch);
            let start_coeff = round_batch.exp_u64((ood_points.len() + 1) as u64);
            folder.add_monomial_constraints(&stir_points, &stir_values, round_batch, start_coeff);

            prev_domain_log = this_domain_log;
            prev_base = false;
            prev_data = prover_data;
        }

        // ---- Final round: reveal the final poly; final PoW + final queries. ----
        let final_poly = folder.f_vec.clone();
        let final_pow = ProofOfWork(challenger.grind(self.config.final_pow_bits));
        let final_mask = (1usize << prev_domain_log) - 1;
        let mut final_leaves = Vec::with_capacity(self.config.final_queries);
        for _ in 0..self.config.final_queries {
            let idx = challenger.sample_bits(prev_domain_log) & final_mask;
            let opening = self.mmcs.open_batch(idx, &prev_data);
            final_leaves
                .push(LeafOpening { values: opening.opened_values, proof: opening.opening_proof });
        }
        round_query_openings.push(MerkleOpening { leaves: final_leaves });

        let final_sumcheck_polys: Vec<SumcheckPoly<EF>> =
            round_sumcheck_polys.pop().unwrap_or_default();

        WhirProof {
            round_sumcheck_polys,
            round_ood_answers,
            round_commitments,
            round_query_openings,
            final_poly,
            final_sumcheck_polys,
            folding_pow,
            final_pow,
        }
    }
}

/// The interleaved-commitment WHIR verifier: replays the transcript, checks
/// every sumcheck message and PoW, Merkle-authenticates every STIR opening,
/// re-derives the stir values, and checks the terminal identity with the
/// per-basis closed forms.
pub struct WhirInterleavedVerifier<F: p3_field::Field, EF, MT: Mmcs<F>> {
    pub mmcs: MT,
    pub config: crate::whir::config::WhirConfig,
    pub start_commitment: MT::Commitment,
    pub start_ood_points: Vec<Vec<EF>>,
    pub start_ood_answers: Vec<EF>,
    _f: core::marker::PhantomData<F>,
}

impl<F, EF, MT> WhirInterleavedVerifier<F, EF, MT>
where
    F: TwoAdicField + PrimeField64,
    EF: ExtensionField<F> + TwoAdicField,
    MT: Mmcs<F, Commitment: Clone>,
{
    pub fn new(
        mmcs: MT,
        config: crate::whir::config::WhirConfig,
        start_commitment: MT::Commitment,
        start_ood_points: Vec<Vec<EF>>,
        start_ood_answers: Vec<EF>,
    ) -> Self {
        Self {
            mmcs,
            config,
            start_commitment,
            start_ood_points,
            start_ood_answers,
            _f: core::marker::PhantomData,
        }
    }

    fn lagrange_eq(a: &[EF], b: &[EF]) -> EF {
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(&x, &y)| x * y + (EF::ONE - x) * (EF::ONE - y)).product()
    }

    /// `Π_j ((1-b_j) + b_j·a_j)` — the Lagrange fold of a monomial table.
    fn monomial_fold(a: &[EF], b: &[EF]) -> EF {
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(&x, &y)| EF::ONE - y + y * x).product()
    }

    /// Verify `f(point) = eval` against the starting commitment.
    pub fn verify<Challenger>(
        &self,
        challenger: &mut Challenger,
        point: &[EF],
        eval: EF,
        proof: &WhirProof<F, EF, MT>,
    ) -> Result<(), WhirVerifierError>
    where
        Challenger:
            FieldChallenger<F> + GrindingChallenger<Witness = F> + CanObserve<MT::Commitment>,
    {
        let n = point.len();
        let ff = self.config.round_parameters[0].folding_factor;
        let num_rounds = self.config.round_parameters.len();
        let folds: alloc::vec::Vec<usize> =
            self.config.round_parameters.iter().map(|rp| rp.folding_factor).collect();
        let final_log = n - folds.iter().sum::<usize>();
        if proof.final_poly.len() != 1usize << final_log {
            return Err(WhirVerifierError::IncorrectShape(alloc::format!(
                "final_poly len {} != 2^{final_log}",
                proof.final_poly.len()
            )));
        }

        // ---- Replay the starting commit + OOD. ----
        challenger.observe(self.start_commitment.clone());
        for (k, ans) in self.start_ood_answers.iter().enumerate() {
            let pt: Vec<EF> = (0..n).map(|_| challenger.sample_algebra_element()).collect();
            if pt != self.start_ood_points[k] {
                return Err(WhirVerifierError::OodPointMismatch { round: 0, sample: k });
            }
            challenger.observe_algebra_element(*ans);
        }
        let batch: EF = challenger.sample_algebra_element();

        let mut claim = eval;
        let mut constraints: Vec<Constraint<EF>> =
            alloc::vec![Constraint::Lagrange { point: point.to_vec(), coeff: EF::ONE, vars: n }];
        let mut coeff = batch;
        for (a, p) in self.start_ood_answers.iter().zip(&self.start_ood_points) {
            claim += coeff * *a;
            constraints.push(Constraint::Lagrange { point: p.clone(), coeff, vars: n });
            coeff *= batch;
        }

        // ---- Per-round replay. ----
        let mut prev_domain_log = (n - ff) + self.config.starting_log_inv_rate;
        let mut prev_base = true;
        let mut prev_commitment = self.start_commitment.clone();
        let mut all_fr: Vec<EF> = Vec::with_capacity(n - final_log);
        let mut folded_vars = 0usize;
        let mut pow_flat = 0usize;
        for (r, round_cfg) in self.config.round_parameters.iter().enumerate() {
            let msgs: &[SumcheckPoly<EF>] = if r + 1 == num_rounds {
                &proof.final_sumcheck_polys
            } else {
                &proof.round_sumcheck_polys[r]
            };
            if msgs.len() != round_cfg.folding_factor {
                return Err(WhirVerifierError::IncorrectShape("round message count".into()));
            }
            let mut this_round_randomness: Vec<EF> = Vec::with_capacity(ff);
            for (var, poly) in msgs.iter().enumerate() {
                let c = &poly.0;
                if c.len() != 3 {
                    return Err(WhirVerifierError::IncorrectShape("degree-2 message".into()));
                }
                let (c0, c1, c2) = (c[0], c[1], c[2]);
                if c0 + (c0 + c1 + c2) != claim {
                    return Err(WhirVerifierError::SumcheckMismatch { round: r, var });
                }
                challenger.observe_algebra_element(c0);
                challenger.observe_algebra_element(c1);
                challenger.observe_algebra_element(c2);
                let pow = &proof.folding_pow[pow_flat];
                pow_flat += 1;
                if !challenger
                    .check_witness(round_cfg.pow_bits.get(var).copied().unwrap_or(0), pow.0)
                {
                    return Err(WhirVerifierError::PowMismatch { round: r, var });
                }
                let rc: EF = challenger.sample_algebra_element();
                claim = c0 + c1 * rc + c2 * rc * rc;
                all_fr.push(rc);
                this_round_randomness.push(rc);
            }
            folded_vars += round_cfg.folding_factor;

            if r + 1 == num_rounds {
                break;
            }

            // Commitment + OOD replay.
            challenger.observe(proof.round_commitments[r].clone());
            let rem = n - folded_vars;
            let ood_answers = &proof.round_ood_answers[r];
            let mut ood_points: Vec<Vec<EF>> = Vec::with_capacity(ood_answers.len());
            for ans in ood_answers.iter() {
                let pt: Vec<EF> = (0..rem).map(|_| challenger.sample_algebra_element()).collect();
                challenger.observe_algebra_element(*ans);
                ood_points.push(pt);
            }

            // Query PoW + indices, then AUTHENTICATE each opening and re-derive
            // its stir value.
            let query_pow = &proof.folding_pow[pow_flat];
            pow_flat += 1;
            if !challenger.check_witness(round_cfg.queries_pow_bits, query_pow.0) {
                return Err(WhirVerifierError::PowMismatch { round: r, var: usize::MAX });
            }
            let mask = (1usize << prev_domain_log) - 1;
            let indices: Vec<usize> = (0..round_cfg.num_queries)
                .map(|_| challenger.sample_bits(prev_domain_log) & mask)
                .collect();
            let openings = &proof.round_query_openings[r];
            if openings.leaves.len() != indices.len() {
                return Err(WhirVerifierError::IncorrectShape("query opening count".into()));
            }
            let leaf_width = if prev_base {
                1usize << round_cfg.folding_factor
            } else {
                (1usize << round_cfg.folding_factor) * EF::DIMENSION
            };
            let dims = alloc::vec![p3_matrix::Dimensions {
                width: leaf_width,
                height: 1usize << prev_domain_log,
            }];
            let g_prev = EF::two_adic_generator(prev_domain_log);
            let mut stir_points: Vec<Vec<EF>> = Vec::with_capacity(indices.len());
            let mut stir_values: Vec<EF> = Vec::with_capacity(indices.len());
            for (&idx, leaf) in indices.iter().zip(&openings.leaves) {
                let opened = p3_commit::BatchOpeningRef {
                    opened_values: &leaf.values,
                    opening_proof: &leaf.proof,
                };
                self.mmcs
                    .verify_batch(&prev_commitment, &dims, idx, opened)
                    .map_err(|_| WhirVerifierError::IncorrectShape("merkle".into()))?;
                stir_values.push(leaf_stir_value::<F, EF>(
                    &leaf.values[0],
                    round_cfg.folding_factor,
                    prev_base,
                    &this_round_randomness,
                ));
                stir_points.push(map_to_pow_lsb(g_prev.exp_u64(idx as u64), rem));
            }

            // Batch OOD then STIR under one round batch, continuing powers.
            let round_batch: EF = challenger.sample_algebra_element();
            let mut cc = round_batch;
            for (a, p) in ood_answers.iter().zip(&ood_points) {
                claim += cc * *a;
                constraints.push(Constraint::Lagrange { point: p.clone(), coeff: cc, vars: rem });
                cc *= round_batch;
            }
            for (v, p) in stir_values.iter().zip(&stir_points) {
                claim += cc * *v;
                constraints.push(Constraint::Monomial { point: p.clone(), coeff: cc, vars: rem });
                cc *= round_batch;
            }

            prev_domain_log =
                (rem - self.config.round_parameters[r + 1].folding_factor) + round_cfg.log_inv_rate;
            prev_base = false;
            prev_commitment = proof.round_commitments[r].clone();
        }

        // ---- Final PoW + final queries: the last committed codeword must fold
        //      (by the LAST round's randomness) to the revealed final_poly's
        //      monomial evaluation. ----
        if !challenger.check_witness(self.config.final_pow_bits, proof.final_pow.0) {
            return Err(WhirVerifierError::PowMismatch { round: num_rounds, var: usize::MAX });
        }
        let final_mask = (1usize << prev_domain_log) - 1;
        let final_openings = proof.round_query_openings.last().unwrap();
        if final_openings.leaves.len() != self.config.final_queries {
            return Err(WhirVerifierError::IncorrectShape("final query count".into()));
        }
        let last_ff = *folds.last().unwrap();
        let leaf_width =
            if prev_base { 1usize << last_ff } else { (1usize << last_ff) * EF::DIMENSION };
        let dims = alloc::vec![p3_matrix::Dimensions {
            width: leaf_width,
            height: 1usize << prev_domain_log,
        }];
        let g_final = EF::two_adic_generator(prev_domain_log);
        let last_randomness = &all_fr[all_fr.len() - last_ff..];
        for leaf in &final_openings.leaves {
            let idx = challenger.sample_bits(prev_domain_log) & final_mask;
            let opened = p3_commit::BatchOpeningRef {
                opened_values: &leaf.values,
                opening_proof: &leaf.proof,
            };
            self.mmcs
                .verify_batch(&prev_commitment, &dims, idx, opened)
                .map_err(|_| WhirVerifierError::IncorrectShape("final merkle".into()))?;
            let folded =
                leaf_stir_value::<F, EF>(&leaf.values[0], last_ff, prev_base, last_randomness);
            let expected = mono_eval_lsb(
                &proof.final_poly,
                &map_to_pow_lsb(g_final.exp_u64(idx as u64), final_log),
            );
            if folded != expected {
                return Err(WhirVerifierError::TerminalMismatch);
            }
        }

        // ---- Terminal identity with per-basis closed forms. ----
        let final_mle = Mle::from_row_major(RowMajorMatrix::new(proof.final_poly.clone(), 1));
        let mut total = EF::ZERO;
        for c in &constraints {
            match c {
                Constraint::Lagrange { point: p, coeff, vars } => {
                    let k = vars - final_log;
                    let eq_part = Self::lagrange_eq(&p[..k], &all_fr[(n - vars)..(n - vars) + k]);
                    let f_part = final_mle.eval_at::<EF>(&p[k..])[0];
                    total += *coeff * eq_part * f_part;
                }
                Constraint::Monomial { point: p, coeff, vars } => {
                    let k = vars - final_log;
                    let fold_part =
                        Self::monomial_fold(&p[..k], &all_fr[(n - vars)..(n - vars) + k]);
                    let f_part = mono_eval_lsb(&proof.final_poly, &p[k..]);
                    total += *coeff * fold_part * f_part;
                }
            }
        }
        if total != claim {
            return Err(WhirVerifierError::TerminalMismatch);
        }
        Ok(())
    }
}
