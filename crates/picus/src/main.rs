use std::{
    collections::{BTreeMap, HashMap},
    path::PathBuf,
};

use clap::{Parser, ValueEnum, ValueHint};
use p3_air::{Air, BaseAir};
use zkm_core_machine::MipsAir;
use zkm_pcs::{Chip, MachineAir, PicusInfo, PROOF_MAX_NUM_PVS};
use zkm_picus::{
    lean,
    lower::VarLayout,
    pcl::{
        initialize_fresh_var_ctr, set_field_modulus, set_picus_names, Felt, PicusConstraint,
        PicusExpr, PicusModule, PicusProgram,
    },
    picus_builder::{
        build_selector_env, extract_module, ColumnOutputMode, ExtractionConfig, PicusBuilder,
        ShrCarrySummaryMode, SubmoduleMode,
    },
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Chip name to extract (as returned by `MachineAir::name`).  Repeatable.
    #[arg(long)]
    pub chip: Vec<String>,

    /// Extract every chip of the machine.
    #[arg(long, default_value_t = false)]
    pub all: bool,

    /// List the chip names and exit.
    #[arg(long, default_value_t = false)]
    pub list: bool,

    /// Directory for `<Chip>.picus` files.  Can be overridden with PICUS_OUT_DIR.
    #[arg(long = "picus-out-dir", value_name = "DIR", value_hint = ValueHint::DirPath, env = "PICUS_OUT_DIR", default_value = "picus_out")]
    pub picus_out_dir: PathBuf,

    /// Root of the Lean project (`ZirenDet.lean` + `ZirenDet/Chips/<Chip>.lean` are written
    /// under it).  Can be overridden with LEAN_OUT_DIR.
    #[arg(long = "lean-out-dir", value_name = "DIR", value_hint = ValueHint::DirPath, env = "LEAN_OUT_DIR", default_value = "lean/ZirenDet")]
    pub lean_out_dir: PathBuf,

    /// Which back ends to write.
    #[arg(long, value_enum, default_value_t = Format::Both)]
    pub format: Format,

    /// Add `assume-deterministic` for the selector outputs of the top module.
    #[arg(long = "assume-selectors-deterministic", default_value_t = false)]
    pub assume_selectors_deterministic: bool,

    /// How to summarize `ByteOpcode::ShrCarry`.
    #[arg(long = "shrcarry-summary", value_enum, default_value_t = ShrCarrySummaryModeArg::Abstract)]
    pub shrcarry_summary: ShrCarrySummaryModeArg,

    /// Which columns become module outputs.
    #[arg(long = "column-output-mode", value_enum, default_value_t = ColumnOutputModeArg::InteractionsOnly)]
    pub column_output_mode: ColumnOutputModeArg,

    /// Expression size above which a sub-tree is bound to a fresh variable (0 disables).
    #[arg(long = "reify-threshold", default_value_t = 128)]
    pub reify_threshold: usize,

    /// Do not specialize `is_real = 1` (extract padding rows too).
    #[arg(long = "keep-padding", default_value_t = false)]
    pub keep_padding: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum Format {
    Picus,
    Lean,
    Both,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum ShrCarrySummaryModeArg {
    Abstract,
    Precise,
}

impl From<ShrCarrySummaryModeArg> for ShrCarrySummaryMode {
    fn from(value: ShrCarrySummaryModeArg) -> Self {
        match value {
            ShrCarrySummaryModeArg::Abstract => ShrCarrySummaryMode::AbstractModule,
            ShrCarrySummaryModeArg::Precise => ShrCarrySummaryMode::Precise,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum ColumnOutputModeArg {
    InteractionsOnly,
    AllNonInputsAreOutputs,
}

impl From<ColumnOutputModeArg> for ColumnOutputMode {
    fn from(value: ColumnOutputModeArg) -> Self {
        match value {
            ColumnOutputModeArg::InteractionsOnly => ColumnOutputMode::InteractionsOnly,
            ColumnOutputModeArg::AllNonInputsAreOutputs => ColumnOutputMode::AllNonInputsAreOutputs,
        }
    }
}

/// Selector-shape module: the chip's polynomial constraints alone (no lookups) must make the
/// selector columns boolean and mutually exclusive — or a partition of the real rows when the
/// chip declares `selectors_partition_real_rows`.
fn build_top_module<A>(
    chip: &Chip<Felt, A>,
    picus_info: &PicusInfo,
    cfg: ExtractionConfig,
    assume_selectors_deterministic: bool,
) -> Option<(PicusModule, BTreeMap<String, PicusModule>)>
where
    A: MachineAir<Felt> + BaseAir<Felt> + Air<PicusBuilder>,
{
    if picus_info.selector_indices.is_empty() {
        return None;
    }
    let partition = chip.selectors_partition_real_rows();
    let real_row_only = partition && picus_info.is_real_index.is_some();
    let env = build_selector_env(picus_info, None, real_row_only);
    let cfg = ExtractionConfig { submodule_mode: SubmoduleMode::Ignore, ..cfg };
    let (mut top, aux) = extract_module(chip, "top".to_string(), &env, cfg);
    top.inputs.clear();
    top.outputs.clear();
    top.assume_deterministic.clear();

    let mut one_hot_sum = PicusExpr::Const(0);
    for (selector_col, _) in &picus_info.selector_indices {
        let selector = PicusExpr::Var(*selector_col);
        one_hot_sum += selector.clone();
        top.outputs.push(selector.clone());
        top.postconditions.push(PicusConstraint::new_bit(selector.clone()));
        if assume_selectors_deterministic {
            top.assume_deterministic.push(selector);
        }
    }
    if partition {
        if real_row_only {
            top.postconditions.push(PicusConstraint::new_equality(one_hot_sum, 1.into()));
        } else if let Some(is_real) = picus_info.is_real_index {
            let is_real = PicusExpr::Var(is_real);
            top.outputs.push(is_real.clone());
            top.postconditions.push(PicusConstraint::new_bit(is_real.clone()));
            top.postconditions.push(PicusConstraint::new_equality(one_hot_sum, is_real));
        } else {
            top.postconditions.push(PicusConstraint::new_lt(one_hot_sum, 2.into()));
        }
    } else {
        top.postconditions.push(PicusConstraint::new_lt(one_hot_sum, 2.into()));
    }
    Some((top, aux))
}

/// Extracts one chip into a program: one module per (allowed) selector, or a single module when
/// the chip has no selectors, plus the `top` selector-shape module.
fn extract_chip<A>(chip: &Chip<Felt, A>, args: &Args) -> (PicusProgram, HashMap<usize, String>)
where
    A: MachineAir<Felt> + BaseAir<Felt> + Air<PicusBuilder>,
{
    let picus_info = chip.picus_info();
    let layout = VarLayout {
        main_width: chip.air.width(),
        prep_width: chip.preprocessed_width().max(1),
        num_public: PROOF_MAX_NUM_PVS,
    };
    let mut names = picus_info.col_to_name.clone();
    names.extend(layout.extra_names());
    set_picus_names(names.clone());
    initialize_fresh_var_ctr(layout.fresh_base());

    let koala_prime = 0x7f00_0001;
    let _ = set_field_modulus(koala_prime);
    let mut program = PicusProgram::new(koala_prime);

    let cfg = ExtractionConfig {
        submodule_mode: SubmoduleMode::Inline,
        shr_carry: args.shrcarry_summary.into(),
        column_output_mode: args.column_output_mode.into(),
        reify_threshold: args.reify_threshold,
    };
    let specialize_is_real = !args.keep_padding;

    let mut modules = BTreeMap::new();
    let mut aux_modules = BTreeMap::new();
    let allowed: Vec<&(usize, String)> = picus_info
        .selector_indices
        .iter()
        .filter(|(_, name)| chip.picus_selector_specialization_allowed(name))
        .collect();
    if allowed.is_empty() {
        let env = build_selector_env(&picus_info, None, specialize_is_real);
        println!("  module {} (env {})", chip.name(), format_env(&env, &names));
        let (m, mut aux) = extract_module(chip, chip.name(), &env, cfg);
        aux_modules.append(&mut aux);
        modules.insert(m.name.clone(), m);
    } else {
        for (col, sel_name) in allowed {
            let env = build_selector_env(&picus_info, Some(*col), specialize_is_real);
            let name = format!("{}__{}", chip.name(), sel_name);
            println!("  module {name} (env {})", format_env(&env, &names));
            let (m, mut aux) = extract_module(chip, name, &env, cfg);
            aux_modules.append(&mut aux);
            modules.insert(m.name.clone(), m);
        }
    }
    program.add_modules(&mut aux_modules);
    program.add_modules(&mut modules);
    if let Some((top, mut aux)) =
        build_top_module(chip, &picus_info, cfg, args.assume_selectors_deterministic)
    {
        println!("  module top (selector shape)");
        program.add_modules(&mut aux);
        program.add_module("top", top);
    }
    (program, names)
}

fn format_env(env: &BTreeMap<usize, u64>, names: &HashMap<usize, String>) -> String {
    if env.is_empty() {
        return "{}".to_string();
    }
    let entries = env
        .iter()
        .map(|(k, v)| format!("{} = {v}", names.get(k).cloned().unwrap_or_else(|| format!("x{k}"))))
        .collect::<Vec<_>>();
    format!("{{ {} }}", entries.join(", "))
}

fn main() {
    let args = Args::parse();
    let chips = MipsAir::<Felt>::chips();

    if args.list {
        for c in &chips {
            let info = c.picus_info();
            println!(
                "{:28} width={:5} selectors={:2} annotated={}",
                c.name(),
                c.air.width(),
                info.selector_indices.len(),
                !info.col_to_name.is_empty()
            );
        }
        return;
    }

    let selected: Vec<&Chip<Felt, MipsAir<Felt>>> = if args.all {
        chips.iter().collect()
    } else {
        if args.chip.is_empty() {
            panic!("pass --chip <NAME> (repeatable), --all, or --list");
        }
        args.chip
            .iter()
            .map(|name| {
                chips
                    .iter()
                    .find(|c| c.name() == *name)
                    .unwrap_or_else(|| panic!("No chip found named {name}; try --list"))
            })
            .collect()
    };

    let mut failures = Vec::new();
    for chip in selected {
        println!("Extracting {} .....", chip.name());
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| extract_chip(chip, &args)));
        let (program, names) = match result {
            Ok(x) => x,
            Err(_) => {
                failures.push(chip.name());
                continue;
            }
        };
        if matches!(args.format, Format::Picus | Format::Both) {
            let path = args.picus_out_dir.join(format!("{}.picus", chip.name()));
            program.write_to_path(&path).unwrap_or_else(|e| panic!("write {path:?}: {e}"));
            println!("  wrote {}", path.display());
        }
        if matches!(args.format, Format::Lean | Format::Both) {
            let path = lean::write_chip(&program, &chip.name(), &args.lean_out_dir, &names)
                .unwrap_or_else(|e| panic!("write lean for {}: {e}", chip.name()));
            println!("  wrote {}", path.display());
        }
    }
    if matches!(args.format, Format::Lean | Format::Both) {
        lean::write_project_files(&args.lean_out_dir).expect("write lean project files");
    }
    if !failures.is_empty() {
        eprintln!("extraction FAILED for {} chip(s): {}", failures.len(), failures.join(", "));
        std::process::exit(1);
    }
    println!("Done.");
}
