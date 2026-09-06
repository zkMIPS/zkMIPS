use std::collections::HashMap;

use crate::Word;
/// Information about Picus annotations on AIR columns.
#[derive(Debug, Clone, Default)]
pub struct PicusInfo {
    /// Column to name mapping. column i will get map to the string "f_i" where f is the field
    /// in the column struct that contains column i
    pub col_to_name: HashMap<usize, String>,
    /// Name to column ranges
    pub name_to_colrange: HashMap<String, (usize, usize)>,
    /// Ranges of columns marked as inputs.
    /// Each tuple contains (`start_index`, `end_index`, `field_name`) where:
    /// - `start_index` is the first column index (inclusive)
    /// - `end_index` is the last column index (exclusive)
    /// - `field_name` is the name of the field
    pub input_ranges: Vec<(usize, usize, String)>,

    /// Ranges of columns marked as outputs.
    /// Each tuple contains (`start_index`, `end_index`, `field_name`) where:
    /// - `start_index` is the first column index (inclusive)
    /// - `end_index` is the last column index (exclusive)
    /// - `field_name` is the name of the field
    pub output_ranges: Vec<(usize, usize, String)>,

    /// Ranges of columns whose current-row values should be exposed as inputs in transition-capable
    /// extraction phases.
    /// Each tuple contains (`start_index`, `end_index`, `field_name`) where:
    /// - `start_index` is the first column index (inclusive)
    /// - `end_index` is the last column index (exclusive)
    /// - `field_name` is the name of the field
    pub transition_input_ranges: Vec<(usize, usize, String)>,

    /// Ranges of columns whose next-row values should be exposed as outputs in transition-capable
    /// extraction phases.
    /// Each tuple contains (`start_index`, `end_index`, `field_name`) where:
    /// - `start_index` is the first column index (inclusive)
    /// - `end_index` is the last column index (exclusive)
    /// - `field_name` is the name of the field
    pub transition_output_ranges: Vec<(usize, usize, String)>,

    /// Indices of columns marked as selectors.
    /// Each tuple contains (`column_index`, `field_name`) where:
    /// - `column_index` is the index of the selector column
    /// - `field_name` is the name of the field
    pub selector_indices: Vec<(usize, String)>,

    /// Indices of columns marked as `is_real`
    pub is_real_index: Option<usize>,
}

/// Information about a semantic projection over a larger Picus-annotated witness layout.
///
/// Unlike [`PicusInfo`], which describes whole storage fields in a concrete trace column
/// struct, a projection describes only the semantically relevant slices that should appear
/// on an operation/module boundary. This is intended for operation-level submodules where
/// most witness columns are internal and should remain existential.
#[derive(Debug, Clone, Default)]
pub struct PicusProjectionInfo {
    /// Column to projected-name mapping for every byte covered by the projection.
    pub col_to_name: HashMap<usize, String>,
    /// Projected field names to concrete column ranges in the source layout.
    pub name_to_colrange: HashMap<String, (usize, usize)>,
    /// Projected ranges that should be treated as module inputs.
    pub input_ranges: Vec<(usize, usize, String)>,
    /// Projected ranges that should be treated as module outputs.
    pub output_ranges: Vec<(usize, usize, String)>,
}

/// Helper trait for projection metadata: recover the first concrete source
/// column from either a scalar column id or a nested array of column ids.
///
/// Projection annotations should be able to point at the semantic slice they
/// mean, for example `state.external_rounds_state[0]` rather than
/// `state.external_rounds_state[0][0]`. The derive computes the projected width
/// from the destination field type, and this trait supplies the starting source
/// column by recursively taking the first element of nested arrays.
pub trait PicusProjectionStart {
    fn projection_start(&self) -> usize;
}

impl PicusProjectionStart for usize {
    fn projection_start(&self) -> usize {
        *self
    }
}

impl<T: PicusProjectionStart, const N: usize> PicusProjectionStart for [T; N] {
    fn projection_start(&self) -> usize {
        self[0].projection_start()
    }
}

impl<T: PicusProjectionStart> PicusProjectionStart for Word<T> {
    fn projection_start(&self) -> usize {
        self.0[0].projection_start()
    }
}
