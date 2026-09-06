//! Lean 4 backend: renders an extracted [`PicusProgram`] as a Lean file whose theorems state the
//! determinism obligations Picus would check.
//!
//! For every module `M` the file contains
//!
//! ```lean
//! namespace M
//! structure W where v0 : F ... vn : F        -- every variable the module mentions
//! def constraints (w : W) : Prop := c₁ ∧ … ∧ cₖ ∧ Aux.rel [..] [..] ∧ …
//! def inputs (w : W) : List F := [...]      -- module inputs
//! def outputs (w : W) : List F := [...]     -- module outputs
//! def assumed (w : W) : List F := [...]     -- `assume-deterministic` expressions
//! def rel (ins outs : List F) : Prop := ∃ w, constraints w ∧ inputs w = ins ∧ outputs w = outs
//! theorem deterministic (h_Aux : ∀ i o o', Aux.rel i o → Aux.rel i o' → o = o') …
//!     (w w' : W) (hw : constraints w) (hw' : constraints w')
//!     (hin : inputs w = inputs w') (hassume : assumed w = assumed w') :
//!     outputs w = outputs w' := by picus_det
//! theorem postconditions (w : W) (hw : constraints w) : p₁ ∧ … := by picus_det
//! end M
//! ```
//!
//! Abstract helper modules (byte-table operations the extractor does not expand) get an
//! `opaque rel` and their determinism is a *hypothesis* of every theorem that calls them, never
//! an axiom.  `picus_det` (defined in `ZirenDet/Basic.lean`) tries the cheap closers and falls
//! back to `sorry`, so a file always elaborates and the remaining obligations are visible as
//! `sorry` warnings.

use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

use crate::pcl::{PicusCall, PicusConstraint, PicusExpr, PicusModule, PicusProgram};

/// Lean identifier for a Picus module / chip name.
pub fn lean_ident(name: &str) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    while out.contains("__") {
        out = out.replace("__", "_");
    }
    let out = out.trim_matches('_').to_string();
    if out.is_empty() || out.chars().next().unwrap().is_ascii_digit() {
        format!("M_{out}")
    } else {
        out
    }
}

fn collect_vars_expr(e: &PicusExpr, out: &mut BTreeSet<usize>) {
    match e {
        PicusExpr::Const(_) => {}
        PicusExpr::Var(v) => {
            out.insert(*v);
        }
        PicusExpr::Add(a, b)
        | PicusExpr::Sub(a, b)
        | PicusExpr::Mul(a, b)
        | PicusExpr::Div(a, b) => {
            collect_vars_expr(a, out);
            collect_vars_expr(b, out);
        }
        PicusExpr::Neg(a) | PicusExpr::Pow(_, a) => collect_vars_expr(a, out),
    }
}

fn collect_vars_constraint(c: &PicusConstraint, out: &mut BTreeSet<usize>) {
    match c {
        PicusConstraint::Lt(a, b)
        | PicusConstraint::Leq(a, b)
        | PicusConstraint::Gt(a, b)
        | PicusConstraint::Geq(a, b) => {
            collect_vars_expr(a, out);
            collect_vars_expr(b, out);
        }
        PicusConstraint::Implies(a, b)
        | PicusConstraint::Iff(a, b)
        | PicusConstraint::And(a, b)
        | PicusConstraint::Or(a, b) => {
            collect_vars_constraint(a, out);
            collect_vars_constraint(b, out);
        }
        PicusConstraint::Not(a) => collect_vars_constraint(a, out),
        PicusConstraint::Eq(e) => collect_vars_expr(e, out),
    }
}

fn render_expr(e: &PicusExpr) -> String {
    match e {
        PicusExpr::Const(c) => format!("({c} : F)"),
        PicusExpr::Var(v) => format!("w.v{v}"),
        PicusExpr::Add(a, b) => format!("({} + {})", render_expr(a), render_expr(b)),
        PicusExpr::Sub(a, b) => format!("({} - {})", render_expr(a), render_expr(b)),
        PicusExpr::Mul(a, b) => format!("({} * {})", render_expr(a), render_expr(b)),
        // `ZMod n` has an `Inv` instance for every `n`; rendering division as a product with
        // the inverse keeps the prelude free of a primality proof (a 31-bit `Nat.Prime`
        // certificate is too deep for the kernel's default recursion limit).
        PicusExpr::Div(a, b) => format!("({} * ({})⁻¹)", render_expr(a), render_expr(b)),
        PicusExpr::Neg(a) => format!("(-{})", render_expr(a)),
        PicusExpr::Pow(k, a) => format!("({} ^ {k})", render_expr(a)),
    }
}

fn render_constraint(c: &PicusConstraint) -> String {
    match c {
        PicusConstraint::Eq(e) => format!("{} = 0", render_expr(e)),
        PicusConstraint::Lt(a, b) => format!("({}).val < ({}).val", render_expr(a), render_expr(b)),
        PicusConstraint::Leq(a, b) => {
            format!("({}).val ≤ ({}).val", render_expr(a), render_expr(b))
        }
        PicusConstraint::Gt(a, b) => format!("({}).val > ({}).val", render_expr(a), render_expr(b)),
        PicusConstraint::Geq(a, b) => {
            format!("({}).val ≥ ({}).val", render_expr(a), render_expr(b))
        }
        PicusConstraint::Implies(a, b) => {
            format!("({} → {})", render_constraint(a), render_constraint(b))
        }
        PicusConstraint::Iff(a, b) => {
            format!("({} ↔ {})", render_constraint(a), render_constraint(b))
        }
        PicusConstraint::And(a, b) => {
            format!("({} ∧ {})", render_constraint(a), render_constraint(b))
        }
        PicusConstraint::Or(a, b) => {
            format!("({} ∨ {})", render_constraint(a), render_constraint(b))
        }
        PicusConstraint::Not(a) => format!("¬ {}", render_constraint(a)),
    }
}

fn render_list(exprs: &[PicusExpr]) -> String {
    format!("[{}]", exprs.iter().map(render_expr).collect::<Vec<_>>().join(", "))
}

fn render_call(call: &PicusCall) -> String {
    format!(
        "{}.rel {} {}",
        lean_ident(&call.mod_name),
        render_list(&call.inputs),
        render_list(&call.outputs)
    )
}

/// A module with no constraints and no calls is an abstract summary (a byte-table operation or
/// an unexpanded helper): its relation is opaque and its determinism is assumed by callers.
fn is_abstract(m: &PicusModule) -> bool {
    m.constraints.is_empty() && m.calls.is_empty() && m.postconditions.is_empty()
}

fn write_module(
    w: &mut impl Write,
    m: &PicusModule,
    names: &HashMap<usize, String>,
    all_modules: &BTreeMap<String, PicusModule>,
) -> io::Result<()> {
    let ident = lean_ident(&m.name);
    let mut vars = BTreeSet::new();
    for e in m.inputs.iter().chain(&m.outputs).chain(&m.assume_deterministic) {
        collect_vars_expr(e, &mut vars);
    }
    for c in m.constraints.iter().chain(&m.postconditions) {
        collect_vars_constraint(c, &mut vars);
    }
    for call in &m.calls {
        for e in call.inputs.iter().chain(&call.outputs) {
            collect_vars_expr(e, &mut vars);
        }
    }

    writeln!(w, "/-! ### Module `{}` -/", m.name)?;
    writeln!(w, "namespace {ident}\n")?;

    if is_abstract(m) {
        writeln!(
            w,
            "/-- Abstract helper: {} input(s), {} output(s).  Its relation is left opaque; every\n\
             caller takes its determinism as a hypothesis. -/",
            m.inputs.len(),
            m.outputs.len()
        )?;
        writeln!(w, "opaque rel : List F → List F → Prop\n")?;
        writeln!(w, "end {ident}\n")?;
        return Ok(());
    }

    // Variables.
    writeln!(w, "/-- The witness row: one field element per variable the module mentions. -/")?;
    writeln!(w, "structure W where")?;
    if vars.is_empty() {
        writeln!(w, "  dummy : F := 0")?;
    }
    for v in &vars {
        match names.get(v) {
            Some(n) => writeln!(w, "  /-- `{n}` -/\n  v{v} : F")?,
            None => writeln!(w, "  v{v} : F")?,
        }
    }
    writeln!(w)?;

    // Constraints, in chunks of `CHUNK` conjuncts: one right-nested conjunction of several
    // thousand terms exceeds Lean's default recursion depth while elaborating and while
    // `casesm` splits it.
    const CHUNK: usize = 48;
    let mut conj: Vec<String> = m.constraints.iter().map(render_constraint).collect();
    conj.extend(m.calls.iter().map(render_call));
    let chunks: Vec<&[String]> = conj.chunks(CHUNK).collect();
    let chunk_names: Vec<String> = (0..chunks.len()).map(|k| format!("constraints_{k}")).collect();
    writeln!(
        w,
        "/-- Every polynomial identity, range fact and helper-module call of the module. -/"
    )?;
    if conj.is_empty() {
        writeln!(w, "def constraints (_w : W) : Prop := True\n")?;
    } else {
        for (k, chunk) in chunks.iter().enumerate() {
            writeln!(w, "def constraints_{k} (w : W) : Prop :=")?;
            for (i, c) in chunk.iter().enumerate() {
                let sep = if i + 1 == chunk.len() { "" } else { " ∧" };
                writeln!(w, "  {c}{sep}")?;
            }
            writeln!(w)?;
        }
        writeln!(w, "def constraints (w : W) : Prop :=")?;
        for (k, name) in chunk_names.iter().enumerate() {
            let sep = if k + 1 == chunk_names.len() { "" } else { " ∧" };
            writeln!(w, "  {name} w{sep}")?;
        }
        writeln!(w)?;
    }
    let tactic = if chunk_names.is_empty() {
        "picus_det".to_string()
    } else {
        format!("picus_det [{}]", chunk_names.join(", "))
    };

    let list_def = |w: &mut dyn Write, name: &str, exprs: &[PicusExpr]| -> io::Result<()> {
        if exprs.is_empty() {
            writeln!(w, "def {name} (_w : W) : List F := []")
        } else {
            writeln!(w, "def {name} (w : W) : List F :=\n  {}", render_list(exprs))
        }
    };
    list_def(w, "inputs", &m.inputs)?;
    list_def(w, "outputs", &m.outputs)?;
    list_def(w, "assumed", &m.assume_deterministic)?;
    writeln!(w)?;
    writeln!(
        w,
        "/-- The module as a relation between its input and output lists. -/\n\
         def rel (ins outs : List F) : Prop :=\n  ∃ w : W, constraints w ∧ inputs w = ins ∧ outputs w = outs\n"
    )?;

    // Determinism hypotheses for every called module.
    let mut callees: BTreeSet<String> = BTreeSet::new();
    for call in &m.calls {
        callees.insert(call.mod_name.clone());
    }
    let mut hyps = Vec::new();
    for callee in &callees {
        let ci = lean_ident(callee);
        let note = match all_modules.get(callee) {
            Some(cm) if !is_abstract(cm) => format!(" -- discharged by `{ci}.deterministic`"),
            _ => " -- abstract helper: assumed".to_string(),
        };
        hyps.push(format!("    (h_{ci} : ∀ i o o', {ci}.rel i o → {ci}.rel i o' → o = o'){note}"));
    }

    writeln!(w, "/-- Determinism: equal inputs (and equal assumed-deterministic values) force equal outputs. -/")?;
    writeln!(w, "theorem deterministic")?;
    for h in &hyps {
        writeln!(w, "{h}")?;
    }
    writeln!(
        w,
        "    (w w' : W) (hw : constraints w) (hw' : constraints w')\n\
         \x20   (hin : inputs w = inputs w') (hassume : assumed w = assumed w') :\n\
         \x20   outputs w = outputs w' := by\n\
         \x20 {tactic}\n"
    )?;

    if !m.postconditions.is_empty() {
        writeln!(w, "/-- Selector-shape / bit postconditions implied by the constraints. -/")?;
        writeln!(w, "theorem postconditions (w : W) (hw : constraints w) :")?;
        let posts: Vec<String> = m.postconditions.iter().map(render_constraint).collect();
        for (i, p) in posts.iter().enumerate() {
            let sep = if i + 1 == posts.len() { " := by" } else { " ∧" };
            writeln!(w, "    {p}{sep}")?;
        }
        writeln!(w, "  {tactic}\n")?;
    }

    writeln!(w, "end {ident}\n")?;
    Ok(())
}

/// Writes `<out_dir>/ZirenDet/Chips/<Chip>.lean` for `program` and returns the path.
pub fn write_chip(
    program: &PicusProgram,
    chip: &str,
    out_dir: &Path,
    names: &HashMap<usize, String>,
) -> io::Result<PathBuf> {
    let chip_ident = lean_ident(chip);
    let dir = out_dir.join("ZirenDet").join("Chips");
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{chip_ident}.lean"));
    let mut f = fs::File::create(&path)?;
    writeln!(
        f,
        "/-\n  Generated by `cargo run -p zkm-picus -- --chip {chip} --format lean`.\n  \
         Do not edit: regenerate after any change to the chip's AIR.\n-/\n\
         import ZirenDet.Basic\n\nset_option maxRecDepth 4000000\nset_option maxHeartbeats 8000000\nset_option linter.dupNamespace false\n\nnamespace ZirenDet.Chips.{chip_ident}\n\nopen ZirenDet\n"
    )?;
    let modules = program.modules();
    // Abstract helpers first (callees), then everything else in the program's stable order.
    for m in modules.values().filter(|m| is_abstract(m)) {
        write_module(&mut f, m, names, modules)?;
    }
    for m in modules.values().filter(|m| !is_abstract(m)) {
        write_module(&mut f, m, names, modules)?;
    }
    writeln!(f, "end ZirenDet.Chips.{chip_ident}")?;
    Ok(path)
}

/// Writes the shared prelude (`ZirenDet/Basic.lean`) if it is missing and (re)writes the root
/// module `ZirenDet.lean` importing every generated chip file present on disk.
pub fn write_project_files(out_dir: &Path) -> io::Result<()> {
    let basic = out_dir.join("ZirenDet").join("Basic.lean");
    if !basic.exists() {
        fs::create_dir_all(basic.parent().unwrap())?;
        fs::write(
            &basic,
            r#"import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic

/-!
# Ziren determinism prelude

The AIRs are stated over the KoalaBear prime field.  `picus_det` is the closing tactic used by
the generated theorems: it tries the cheap closers and otherwise leaves a `sorry`, so generated
files always elaborate and the remaining obligations show up as warnings.
-/

namespace ZirenDet

/-- The KoalaBear prime `2^31 - 2^24 + 1`. -/
abbrev KB : ℕ := 2130706433

-- `KB` is prime; the `norm_num` certificate for a 31-bit prime needs a deeper kernel recursion
-- limit than the default (it checks in about 3 s).
set_option maxRecDepth 100000 in
instance : Fact (Nat.Prime KB) := ⟨by norm_num⟩

/-- The base field of every Ziren AIR: `ZMod KB` is a `Field` through the instance above.
Division in extracted constraints is rendered as multiplication by the inverse. -/
abbrev F := ZMod KB

end ZirenDet

/-- Closing tactic for generated determinism / postcondition theorems.

Strategy: split both witness records into their fields, unfold the generated definitions,
turn every `a - b = 0` into `a = b`, split conjunctions, substitute every variable that is
defined by an equation (`subst_vars` propagates the linear definitions and the input
equalities), and close what is left by `rfl` / `ring`.  Anything that needs case analysis on
guarded bits is left as `sorry` so the file still elaborates and the obligation is visible. -/
syntax "picus_det" (" [" term,* "]")? : tactic
-- Hygiene off: the tactic must see the use-site `constraints`, `inputs`, `W`, `w`, … of the
-- module it closes, not names resolved at this definition site.  `ring1` rather than `ring`:
-- `ring` falls back to `ring_nf` and *succeeds* with the goal still open.
set_option hygiene false in
macro_rules
  | `(tactic| picus_det) => `(tactic| picus_det [])
  | `(tactic| picus_det [$ts,*]) =>
    `(tactic| (
        (try intros)
        (try cases w)
        (try cases w')
        (try simp only [constraints, $[$ts:term],*, inputs, outputs, assumed, rel, List.cons.injEq,
               List.nil_eq, and_true, true_and, sub_eq_zero, W.mk.injEq] at *)
        (try casesm* _ ∧ _)
        (try subst_vars)
        (try constructorm* _ ∧ _)
        all_goals (first | rfl | ring1 | sorry)))
"#,
        )?;
    }
    let chips_dir = out_dir.join("ZirenDet").join("Chips");
    let mut chips: Vec<String> = Vec::new();
    if chips_dir.exists() {
        for entry in fs::read_dir(&chips_dir)? {
            let p = entry?.path();
            if p.extension().and_then(|e| e.to_str()) == Some("lean") {
                if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                    chips.push(stem.to_string());
                }
            }
        }
    }
    chips.sort();
    let mut root = String::from("import ZirenDet.Basic\n");
    for c in &chips {
        root.push_str(&format!("import ZirenDet.Chips.{c}\n"));
    }
    fs::write(out_dir.join("ZirenDet.lean"), root)?;
    Ok(())
}
