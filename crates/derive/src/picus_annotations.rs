use proc_macro::TokenStream;
use quote::quote;
use syn::Generics;
use syn::{parse_macro_input, DeriveInput};

use syn::{
    parse::{Parse, ParseStream},
    parse_quote,
    punctuated::Punctuated,
    GenericArgument, Path, PathArguments, Result, Token, Type, TypeArray, TypeReference, TypeSlice,
};

#[derive(Default, Clone)]
struct PicusArgs {
    input: bool,
    output: bool,
    transition_input: bool,
    transition_output: bool,
    selector: bool,
    path: Option<syn::Expr>,
}

enum Arg {
    Input,
    Output,
    TransitionInput,
    TransitionOutput,
    Selector,
    Path(syn::Expr),
}

impl Parse for Arg {
    // parses the arguments for the picus attribute
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let key: Path = input.parse()?;

        let is = |s: &str| key.is_ident(s);

        if is("input") {
            return Ok(Arg::Input);
        }
        if is("output") {
            return Ok(Arg::Output);
        }
        if is("transition_input") {
            return Ok(Arg::TransitionInput);
        }
        if is("transition_output") {
            return Ok(Arg::TransitionOutput);
        }
        if is("selector") {
            return Ok(Arg::Selector);
        }
        if is("path") {
            input.parse::<Token![=]>()?;
            return Ok(Arg::Path(input.parse()?));
        }

        Err(syn::Error::new_spanned(key, "unknown key in #[picus(...)]"))
    }
}

fn parse_picus_attr(attr: &syn::Attribute) -> syn::Result<Option<PicusArgs>> {
    // check that the attribute is a picus attribute
    if !attr.path.is_ident("picus") {
        return Ok(None);
    }
    // parse the attributes
    let items = attr.parse_args_with(Punctuated::<Arg, Token![,]>::parse_terminated)?;
    let mut out = PicusArgs::default();
    for it in items {
        match it {
            Arg::Input => out.input = true,
            Arg::Output => out.output = true,
            Arg::TransitionInput => out.transition_input = true,
            Arg::TransitionOutput => out.transition_output = true,
            Arg::Selector => out.selector = true,
            Arg::Path(expr) => out.path = Some(expr),
        }
    }
    Ok(Some(out))
}

#[derive(Clone)]
struct ProjectionStructArgs {
    source: Type,
    col_map: syn::Expr,
}

enum ProjectionStructArg {
    Source(Type),
    ColMap(syn::Expr),
}

impl Parse for ProjectionStructArg {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let key: Path = input.parse()?;
        if key.is_ident("source") {
            input.parse::<Token![=]>()?;
            return Ok(Self::Source(input.parse()?));
        }
        if key.is_ident("col_map") {
            input.parse::<Token![=]>()?;
            return Ok(Self::ColMap(input.parse()?));
        }
        Err(syn::Error::new_spanned(key, "unknown key in #[picus_projection(...)]"))
    }
}

fn parse_picus_projection_attr(
    attrs: &[syn::Attribute],
) -> syn::Result<Option<ProjectionStructArgs>> {
    let mut source: Option<Type> = None;
    let mut col_map: Option<syn::Expr> = None;
    let mut found = false;
    for attr in attrs {
        if !attr.path.is_ident("picus_projection") {
            continue;
        }
        found = true;
        let items =
            attr.parse_args_with(Punctuated::<ProjectionStructArg, Token![,]>::parse_terminated)?;
        for it in items {
            match it {
                ProjectionStructArg::Source(ty) => source = Some(ty),
                ProjectionStructArg::ColMap(expr) => col_map = Some(expr),
            }
        }
    }

    if !found {
        return Ok(None);
    }
    let source = source.ok_or_else(|| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            "missing `source = ...` in #[picus_projection(...)]",
        )
    })?;
    let col_map = col_map.ok_or_else(|| {
        syn::Error::new(
            proc_macro2::Span::call_site(),
            "missing `col_map = ...` in #[picus_projection(...)]",
        )
    })?;
    Ok(Some(ProjectionStructArgs { source, col_map }))
}

// ---------- type substitution: replace the primary column element type with `u8` ----------
fn first_type_param_ident(gens: &Generics) -> Option<syn::Ident> {
    gens.type_params().next().map(|tp| tp.ident.clone())
}

// column values are determined by computing the offset of the ColStruct when instantiated
// with the u8 parameter. Only the leading column element type should be rewritten; structural
// generics such as curve/field parameter types must be preserved.
fn ty_sub_u8(mut ty: Type, first_type_param: &Option<syn::Ident>) -> Type {
    match ty {
        Type::Path(ref mut tp) => {
            if tp.qself.is_none() && tp.path.segments.len() == 1 {
                let seg = &tp.path.segments[0];
                if first_type_param.as_ref().is_some_and(|ident| *ident == seg.ident) {
                    return parse_quote!(u8);
                }
            }
            for seg in tp.path.segments.iter_mut() {
                if let PathArguments::AngleBracketed(ref mut ab) = seg.arguments {
                    for arg in ab.args.iter_mut() {
                        if let GenericArgument::Type(ref mut inner) = arg {
                            *inner = ty_sub_u8(inner.clone(), first_type_param);
                        }
                    }
                }
            }
            ty
        }
        Type::Reference(TypeReference { ref mut elem, .. }) => {
            **elem = ty_sub_u8((**elem).clone(), first_type_param);
            ty
        }
        Type::Array(TypeArray { ref mut elem, .. })
        | Type::Slice(TypeSlice { ref mut elem, .. }) => {
            **elem = ty_sub_u8((**elem).clone(), first_type_param);
            ty
        }
        Type::Tuple(ref mut tup) => {
            for el in tup.elems.iter_mut() {
                *el = ty_sub_u8(el.clone(), first_type_param);
            }
            ty
        }
        _ => ty,
    }
}

// Build Self<u8, P, ...> actual type args; keep non-leading type params, lifetimes and consts as-is.
fn concrete_type_args(gens: &Generics) -> proc_macro2::TokenStream {
    let mut seen_type_param = false;
    let args = gens.params.iter().map(|p| match p {
        syn::GenericParam::Type(tp) => {
            if !seen_type_param {
                seen_type_param = true;
                quote!(u8)
            } else {
                let ident = &tp.ident;
                quote!(#ident)
            }
        }
        syn::GenericParam::Lifetime(lt) => {
            let lt = &lt.lifetime;
            quote!(#lt)
        }
        syn::GenericParam::Const(c) => {
            let id = &c.ident;
            quote!(#id)
        }
    });
    quote!(<#(#args),*>)
}

// impl generics = all generics except the leading element type parameter.
fn impl_generics_without_primary_type_param(gens: &Generics) -> proc_macro2::TokenStream {
    let mut parts: Vec<proc_macro2::TokenStream> = Vec::new();
    let mut skipped_primary_type = false;
    for param in gens.params.iter() {
        match param {
            syn::GenericParam::Type(tp) => {
                if !skipped_primary_type {
                    skipped_primary_type = true;
                } else {
                    parts.push(quote!(#tp));
                }
            }
            syn::GenericParam::Lifetime(lt) => parts.push(quote!(#lt)),
            syn::GenericParam::Const(c) => parts.push(quote!(#c)),
        }
    }
    if parts.is_empty() {
        quote!()
    } else {
        quote!(< #(#parts),* >)
    }
}

pub fn picus_annotations_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let ident = input.ident.clone();
    let gens = input.generics.clone();

    let data = match &input.data {
        syn::Data::Struct(s) => s,
        _ => {
            return syn::Error::new_spanned(&input, "PicusInfoGen only supports structs")
                .to_compile_error()
                .into()
        }
    };
    let fields = match &data.fields {
        syn::Fields::Named(f) => &f.named,
        _ => {
            return syn::Error::new_spanned(&input, "PicusInfoGen requires named fields")
                .to_compile_error()
                .into()
        }
    };

    let first_type_param = first_type_param_ident(&gens);
    let impl_gens = impl_generics_without_primary_type_param(&gens);
    let self_args = concrete_type_args(&gens);
    let self_conc = quote!(#ident #self_args);
    let where_clause = &gens.where_clause;

    // Per-field code
    let mut steps = Vec::new();
    for field in fields.iter() {
        let f_ident = field.ident.as_ref().unwrap();
        let f_name = f_ident.to_string();
        // Collect flags
        let mut flags = PicusArgs::default();
        for attr in &field.attrs {
            if attr.path.is_ident("picus") {
                match parse_picus_attr(attr) {
                    Ok(Some(a)) => {
                        flags.input |= a.input;
                        flags.output |= a.output;
                        flags.transition_input |= a.transition_input;
                        flags.transition_output |= a.transition_output;
                        flags.selector |= a.selector;
                    }
                    Ok(None) => {}
                    Err(e) => return e.to_compile_error().into(),
                }
            }
        }

        // Field type with all *type* params → u8
        let conc_ty: Type = ty_sub_u8(field.ty.clone(), &first_type_param);

        // Add name to id map
        let push_name = {
            quote! {
                if width > 0 {
                    info.name_to_colrange.insert((#f_name).to_string(), (cur, cur+width));
                    for x in cur..(cur+width) {
                        info.col_to_name.insert(x, format!("{}_{}", #f_name, x));
                    }
                }
            }
        };
        let push_in = if flags.input {
            quote! { if width > 0 { info.input_ranges.push((cur, cur + width, #f_name.to_string())); } }
        } else {
            quote!()
        };

        let push_out = if flags.output {
            quote! { if width > 0 { info.output_ranges.push((cur, cur + width, #f_name.to_string())); } }
        } else {
            quote!()
        };

        let push_transition_in = if flags.transition_input {
            quote! {
                if width > 0 {
                    info.transition_input_ranges.push((cur, cur + width, #f_name.to_string()));
                }
            }
        } else {
            quote!()
        };

        let push_transition_out = if flags.transition_output {
            quote! {
                if width > 0 {
                    info.transition_output_ranges.push((cur, cur + width, #f_name.to_string()));
                }
            }
        } else {
            quote!()
        };

        let push_sel = if flags.selector {
            quote! {
                if width > 0 {
                    for i in 0..width {
                        info.selector_indices.push((cur + i, #f_name.to_string()));
                    }
                }

            }
        } else {
            quote!()
        };
        // If the field name is "is_real" then add that mark it in PicusInfo
        let push_is_real = if f_name == "is_real" {
            quote! {
                if width > 0 {
                    info.is_real_index = Some(cur);
                }
            }
        } else {
            quote!()
        };

        steps.push(quote! {{
            let width: usize = ::core::mem::size_of::<#conc_ty>();
            #push_name
            #push_in
            #push_out
            #push_transition_in
            #push_transition_out
            #push_sel
            #push_is_real
            cur += width;
        }});
    }

    let expanded = quote! {
        // Implement on the concrete instantiation where *type* params are `u8`
        impl #impl_gens #self_conc #where_clause {
            pub fn picus_info() -> PicusInfo {
                let mut info = PicusInfo::default();
                let mut cur: usize = 0; // 1 column == 1 byte
                #(#steps)*
                info
            }
        }
    };
    expanded.into()
}

pub fn picus_projection_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let ident = input.ident.clone();
    let generics = input.generics.clone();
    let where_clause = &generics.where_clause;
    let (impl_generics, ty_generics, _) = generics.split_for_impl();

    let struct_args = match parse_picus_projection_attr(&input.attrs) {
        Ok(Some(args)) => args,
        Ok(None) => {
            return syn::Error::new_spanned(
                &input,
                "PicusProjection requires #[picus_projection(source = ..., col_map = ...)]",
            )
            .to_compile_error()
            .into()
        }
        Err(e) => return e.to_compile_error().into(),
    };

    let data = match &input.data {
        syn::Data::Struct(s) => s,
        _ => {
            return syn::Error::new_spanned(&input, "PicusProjection only supports structs")
                .to_compile_error()
                .into()
        }
    };
    let fields = match &data.fields {
        syn::Fields::Named(f) => &f.named,
        _ => {
            return syn::Error::new_spanned(&input, "PicusProjection requires named fields")
                .to_compile_error()
                .into()
        }
    };

    let source_ty = struct_args.source;
    let col_map = struct_args.col_map;

    let mut steps = Vec::new();
    for field in fields.iter() {
        let f_ident = field.ident.as_ref().unwrap();
        let f_name = f_ident.to_string();

        let mut flags = PicusArgs::default();
        for attr in &field.attrs {
            if attr.path.is_ident("picus") {
                match parse_picus_attr(attr) {
                    Ok(Some(a)) => {
                        flags.input |= a.input;
                        flags.output |= a.output;
                        flags.transition_input |= a.transition_input;
                        flags.transition_output |= a.transition_output;
                        flags.selector |= a.selector;
                        if let Some(path) = a.path {
                            flags.path = Some(path);
                        }
                    }
                    Ok(None) => {}
                    Err(e) => return e.to_compile_error().into(),
                }
            }
        }

        if flags.transition_input || flags.transition_output || flags.selector {
            return syn::Error::new_spanned(
                field,
                "PicusProjection fields currently support only #[picus(input, ...)] and #[picus(output, ...)]",
            )
            .to_compile_error()
            .into();
        }
        if !flags.input && !flags.output {
            return syn::Error::new_spanned(
                field,
                "PicusProjection fields must be marked with #[picus(input, path = ...)] or #[picus(output, path = ...)]",
            )
            .to_compile_error()
            .into();
        }
        let Some(path_expr) = flags.path.clone() else {
            return syn::Error::new_spanned(
                field,
                "PicusProjection fields require `path = ...` to identify the source slice",
            )
            .to_compile_error()
            .into();
        };

        let field_ty = &field.ty;
        let push_in = if flags.input {
            quote! { info.input_ranges.push((start, end, #f_name.to_string())); }
        } else {
            quote!()
        };
        let push_out = if flags.output {
            quote! { info.output_ranges.push((start, end, #f_name.to_string())); }
        } else {
            quote!()
        };

        steps.push(quote! {{
            let start: usize =
                zkm_pcs::PicusProjectionStart::projection_start(&((#col_map).#path_expr));
            let width: usize = ::core::mem::size_of::<#field_ty>();
            let end = start + width;
            info.name_to_colrange.insert(#f_name.to_string(), (start, end));
            for x in start..end {
                info.col_to_name.insert(x, format!("{}_{}", #f_name, x));
            }
            #push_in
            #push_out
        }});
    }

    let expanded = quote! {
        impl #impl_generics #ident #ty_generics #where_clause {
            pub fn picus_projection_info() -> zkm_pcs::PicusProjectionInfo {
                let mut info = zkm_pcs::PicusProjectionInfo::default();
                let _ = ::core::mem::size_of::<#source_ty>();
                #(#steps)*
                info
            }
        }
    };
    expanded.into()
}
