//! End-to-end smoke test: every machine chip extracts, and an instruction chip gets the expected
//! lookup-derived interface.

use std::collections::BTreeMap;

use p3_air::BaseAir;
use zkm_core_machine::MipsAir;
use zkm_pcs::MachineAir;
use zkm_picus::{
    pcl::{initialize_fresh_var_ctr, set_field_modulus, Felt},
    picus_builder::{
        build_selector_env, extract_module, ColumnOutputMode, ExtractionConfig,
        ShrCarrySummaryMode, SubmoduleMode,
    },
};

fn cfg() -> ExtractionConfig {
    ExtractionConfig {
        submodule_mode: SubmoduleMode::Inline,
        shr_carry: ShrCarrySummaryMode::AbstractModule,
        column_output_mode: ColumnOutputMode::InteractionsOnly,
        reify_threshold: 128,
    }
}

#[test]
fn add_sub_interface() {
    let _ = set_field_modulus(0x7f00_0001);
    let chips = MipsAir::<Felt>::chips();
    let chip = chips.iter().find(|c| c.name() == "AddSub").expect("AddSub chip");
    let info = chip.picus_info();
    assert_eq!(info.selector_indices.len(), 2, "is_add / is_sub");
    let (is_add, _) = info.selector_indices[0].clone();
    initialize_fresh_var_ctr(10 * chip.air.width());
    let env = build_selector_env(&info, Some(is_add), true);
    let (m, aux) = extract_module(chip, "AddSub__is_add".to_string(), &env, cfg());
    // pc, instruction fields, three register reads and the state receive come in; the state
    // send and the register write go out.
    assert!(m.inputs.len() >= 16, "inputs: {}", m.inputs.len());
    assert!(m.outputs.len() >= 8, "outputs: {}", m.outputs.len());
    assert!(!m.constraints.is_empty());
    assert!(aux.is_empty(), "AddSub needs no abstract helpers: {:?}", aux.keys());
}

#[test]
fn every_chip_extracts() {
    let _ = set_field_modulus(0x7f00_0001);
    let chips = MipsAir::<Felt>::chips();
    for chip in &chips {
        let info = chip.picus_info();
        initialize_fresh_var_ctr(10 * chip.air.width() + 1024);
        // Chips with selectors must be specialized per selector (byte-table opcodes only fold to
        // constants then); take the first allowed one, as the CLI does.
        let selector = info
            .selector_indices
            .iter()
            .find(|(_, name)| chip.picus_selector_specialization_allowed(name))
            .map(|(col, _)| *col);
        let env: BTreeMap<usize, u64> = build_selector_env(&info, selector, true);
        let (m, _) = extract_module(chip, chip.name(), &env, cfg());
        // Only the preprocessed tables (Byte, Program, Range) have nothing to say.
        let table = matches!(chip.name().as_str(), "Byte" | "Program" | "Range");
        assert_eq!(m.constraints.is_empty() && m.inputs.is_empty(), table, "chip {}", chip.name());
    }
}
