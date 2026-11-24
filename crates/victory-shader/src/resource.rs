//! Resource tracking (decision 0003): where each descriptor a program uses
//! comes from, in terms of the root signature.
//!
//! Root parameters arrive in user SGPRs ([`RootParam`]): a descriptor table
//! as the 32-bit heap address of its range (the code sets the high half to
//! 0), a root descriptor as a 64-bit buffer address, root constants as
//! themselves. A root signature with dynamic resources (SM 6.6) adds the
//! 32-bit base addresses of both heaps; a descriptor is then loaded from
//! one of them at an index the code computes (from a root constant, say). The code loads table descriptors from heap memory with
//! `s_load_dword*`, and builds a buffer descriptor around a root
//! descriptor's address itself. Each descriptor operand and each scalar
//! memory address is traced through the SSA values back to those
//! parameters, by data flow: copies, phis that agree, and additions.

use crate::ShaderError;
use crate::gfx10::{Fields, NULL, Operand};
use crate::ir::{Inst, Program, Reg, Value};
use crate::native::{HwStage, RootParam, RootType};

/// An instruction: block and index in [`crate::ir::Block::insts`].
pub type Site = (usize, usize);

/// A byte address: a root parameter's, plus a constant and at most one
/// value the program computes (an index times a stride, say).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Address {
    /// Index into the layout, or [`CODE`].
    pub param: usize,
    /// First dword within the parameter (0 except for a 64-bit address in
    /// `RootType::Constants`).
    pub dword: u32,
    /// Added modulo 2^32 (with carry into the high half of a root
    /// descriptor's address).
    pub offset: u32,
    pub dynamic: Option<Value>,
}

/// [`Address::param`] of the program's own code: a constant table the
/// compiler appends after the code and reads through `s_getpc_b64`; the
/// offset is then the byte offset in the code.
pub const CODE: usize = usize::MAX;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Source {
    /// Dwords `first..` of what the scalar load at `load` reads (see
    /// [`Resources::scalar`] for its address).
    Loaded { load: Site, first: u8 },
    /// A buffer descriptor built around an address: `bits` OR-ed into
    /// dword 1 (stride, swizzle), dwords 2 and 3 as given.
    Built {
        address: Address,
        bits: u32,
        words: [u32; 2],
    },
    /// Fixed in the code (an immediate sampler, say).
    Constant(Vec<u32>),
    /// Chosen by control flow: one source per predecessor of `block`, in
    /// [`crate::ir::Block::preds`] order.
    Phi { block: usize, args: Vec<Source> },
}

impl Source {
    /// The sources this one picks from (itself unless a phi).
    pub fn leaves(&self) -> Vec<&Source> {
        match self {
            Source::Phi { args, .. } => args.iter().flat_map(Source::leaves).collect(),
            s => vec![s],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Image, buffer or `s_buffer_load` resource.
    Resource,
    Sampler,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Descriptor {
    pub site: Site,
    pub role: Role,
    pub source: Source,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Resources {
    /// Every scalar memory access by address (`s_load_*`, `s_store_*`), and
    /// every global memory access (`global_*`, the dynamic term per lane).
    pub scalar: Vec<(Site, Address)>,
    pub descriptors: Vec<Descriptor>,
    /// User SGPRs and spill-table loads holding root parameters: value,
    /// layout index, dword within the parameter.
    pub params: Vec<(Value, usize, u32)>,
    /// Scalar loads from the user-data spill table (`user_data >= 16`).
    pub spill_loads: Vec<Site>,
}

/// Traces every descriptor and scalar address of `p`, which runs as
/// `stage`, to the root parameters in `layout`. Fails on the first one it
/// cannot trace.
pub fn track(p: &Program, layout: &[RootParam], stage: HwStage) -> Result<Resources, ShaderError> {
    let mut t = Tracer::new(p, layout, stage)?;
    let mut params: Vec<(Value, usize, u32)> = p
        .inputs
        .iter()
        .filter_map(|&(reg, v)| {
            let (param, dword) = t.param_of(reg)?;
            Some((v, param, dword))
        })
        .collect();
    params.extend(t.collect_spill_loads()?);
    let mut out = Resources {
        params,
        spill_loads: t.spill_loads.clone(),
        ..Resources::default()
    };
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            let site = (b, i);
            let fail = |what| ShaderError::Unsupported {
                offset: inst.machine.offset,
                what,
            };
            let mut descriptor = |op: &Operand, role| -> Result<(), ShaderError> {
                let source = t.descriptor(inst, op).map_err(fail)?;
                out.descriptors.push(Descriptor { site, role, source });
                Ok(())
            };
            match &inst.fields {
                Fields::Image { rsrc, sampler, .. } => {
                    descriptor(rsrc, Role::Resource)?;
                    if let Some(s) = sampler {
                        descriptor(s, Role::Sampler)?;
                    }
                }
                Fields::Buffer { rsrc, .. } => descriptor(rsrc, Role::Resource)?,
                Fields::Smem { base, .. } if inst.name().starts_with("s_buffer_") => {
                    descriptor(base, Role::Resource)?;
                }
                Fields::Smem {
                    base,
                    offset,
                    soffset,
                    ..
                } if inst.name().contains("_load_") || inst.name().contains("_store_") => {
                    if t.spill_loads.contains(&site) {
                        continue;
                    }
                    let address = t.smem_address(inst, base, *offset, soffset).map_err(fail)?;
                    out.scalar.push((site, address));
                }
                Fields::Flat {
                    addr,
                    saddr,
                    offset,
                    ..
                } if inst.name().starts_with("global_") => {
                    let address = t
                        .flat_address(inst, addr.as_ref(), saddr.as_ref(), *offset)
                        .map_err(fail)?;
                    out.scalar.push((site, address));
                }
                _ => {}
            }
        }
    }
    Ok(out)
}

/// A dword after copy propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Term {
    Const(u32),
    Value(Value),
}

/// A 32-bit sum: optional root parameter (as its first dword), constant,
/// optional computed term.
struct Sum {
    base: Option<Value>,
    offset: u32,
    dynamic: Option<Value>,
}

struct Tracer<'a> {
    p: &'a Program,
    layout: &'a [RootParam],
    stage: HwStage,
    /// Defining instruction per value (phis and inputs have none).
    defs: Vec<Option<Site>>,
    /// Block and index of each phi value.
    phis: Vec<Option<(usize, usize)>>,
    /// Register per input value.
    inputs: Vec<Option<Reg>>,
    origin: Vec<Term>,
    /// Entry SGPR values holding the user-data spill-table pointer (`user_sgpr(0..=1)`),
    /// when `layout` has parameters at `user_data >= 16`.
    spill_ptr: Option<(Value, Value)>,
    /// SSA values produced by spill-table loads: `(param_index, dword)`.
    spilled: std::collections::HashMap<Value, (usize, u32)>,
    /// Sites of resolved spill-table scalar loads.
    spill_loads: Vec<Site>,
}

const MAX_ROUNDS: usize = 64;
const MAX_JOINS: usize = 8;

type ValueTables = (
    Vec<Option<Site>>,
    Vec<Option<(usize, usize)>>,
    Vec<Option<Reg>>,
);

/// Indexes each SSA value's defining site, phi location, and input register.
fn index_values(p: &Program) -> ValueTables {
    let all = || {
        p.inputs
            .iter()
            .map(|i| i.1)
            .chain(p.blocks.iter().flat_map(|b| {
                b.phis
                    .iter()
                    .map(|phi| phi.value)
                    .chain(b.insts.iter().flat_map(|i| i.writes.iter().map(|w| w.1)))
            }))
    };
    let n = all().map(|v| v.0 as usize + 1).max().unwrap_or(0);
    let mut defs = vec![None; n];
    let mut phis = vec![None; n];
    let mut inputs = vec![None; n];
    for &(reg, v) in &p.inputs {
        inputs[v.0 as usize] = Some(reg);
    }
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, phi) in block.phis.iter().enumerate() {
            phis[phi.value.0 as usize] = Some((b, i));
        }
        for (i, inst) in block.insts.iter().enumerate() {
            for &(_, v) in &inst.writes {
                defs[v.0 as usize] = Some((b, i));
            }
        }
    }
    (defs, phis, inputs)
}

impl<'a> Tracer<'a> {
    fn new(p: &'a Program, layout: &'a [RootParam], stage: HwStage) -> Result<Self, ShaderError> {
        let (defs, phis, inputs) = index_values(p);
        let n = defs.len();
        let mut t = Tracer {
            p,
            layout,
            stage,
            defs,
            phis,
            inputs,
            origin: Vec::new(),
            spill_ptr: None,
            spilled: std::collections::HashMap::new(),
            spill_loads: Vec::new(),
        };
        t.origin = t.propagate_copies(n)?;
        let has_spill = layout.iter().any(|rp| {
            u32::from(rp.user_data) >= user_sgpr_limit(stage)
                || u32::from(rp.user_data) + rp.sgprs > user_sgpr_limit(stage)
        });
        if has_spill {
            let u0 = Reg::S(stage.user_sgpr(0));
            let u1 = Reg::S(stage.user_sgpr(1));
            if t.param_of(u0).is_none()
                && t.param_of(u1).is_none()
                && let (Some(lo), Some(hi)) = (
                    p.inputs.iter().find(|i| i.0 == u0).map(|i| i.1),
                    p.inputs.iter().find(|i| i.0 == u1).map(|i| i.1),
                )
            {
                t.spill_ptr = Some((lo, hi));
            }
        }
        Ok(t)
    }

    /// Optimistic copy propagation: a move's result is its source's
    /// origin, a phi's the one origin its arguments agree on.
    fn propagate_copies(&self, n: usize) -> Result<Vec<Term>, ShaderError> {
        let mut o: Vec<Option<Term>> = vec![None; n];
        for &(_, v) in &self.p.inputs {
            o[v.0 as usize] = Some(Term::Value(v));
        }
        let order = crate::ir::reverse_postorder(&self.p.blocks);
        for _ in 0..MAX_ROUNDS {
            let mut changed = false;
            for &b in &order {
                let block = &self.p.blocks[b];
                for phi in &block.phis {
                    let own = Term::Value(phi.value);
                    let mut args = phi
                        .args
                        .iter()
                        .filter_map(|a| o[a.0 as usize])
                        .filter(|&t| t != own);
                    let new = args
                        .next()
                        .map(|first| if args.all(|t| t == first) { first } else { own });
                    if new.is_some() && new != o[phi.value.0 as usize] {
                        o[phi.value.0 as usize] = new;
                        changed = true;
                    }
                }
                for inst in &block.insts {
                    for (k, &(_, v)) in inst.writes.iter().enumerate() {
                        let new = match copy_source(inst, k).or_else(|| self.restored(inst)) {
                            Some(Moved::Const(c)) => Some(Term::Const(c)),
                            Some(Moved::Value(s)) => o[s.0 as usize],
                            None => Some(Term::Value(v)),
                        };
                        if new.is_some() && new != o[v.0 as usize] {
                            o[v.0 as usize] = new;
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                return Ok(o
                    .into_iter()
                    .enumerate()
                    .map(|(v, t)| t.unwrap_or(Term::Value(Value(v as u32))))
                    .collect());
            }
        }
        Err(ShaderError::Unsupported {
            offset: 0,
            what: "copies that do not settle",
        })
    }

    fn inst(&self, (b, i): Site) -> &'a Inst {
        &self.p.blocks[b].insts[i]
    }

    fn def(&self, v: Value) -> Option<&'a Inst> {
        self.defs[v.0 as usize].map(|s| self.inst(s))
    }

    fn origin(&self, v: Value) -> Term {
        self.origin[v.0 as usize]
    }

    /// The origin of a one-dword operand as `inst` reads it.
    fn term(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Scalar { code: NULL, .. } => Some(Term::Const(0)),
            Operand::Scalar { code, .. } => Some(self.origin(inst.read(Reg::S(code))?)),
            Operand::Const(c) => inline(c).map(Term::Const),
            Operand::Literal(x) => Some(Term::Const(x)),
            _ => None,
        }
    }

    /// Layout index and dword of the root parameter `reg` holds on entry.
    fn param_of(&self, reg: Reg) -> Option<(usize, u32)> {
        let Reg::S(code) = reg else { return None };
        self.layout.iter().enumerate().find_map(|(i, p)| {
            if u32::from(p.user_data) >= user_sgpr_limit(self.stage) {
                return None;
            }
            let first = self.stage.user_sgpr(p.user_data);
            let dword = u32::from(code).checked_sub(first.into())?;
            (dword < p.sgprs && u32::from(p.user_data) + dword < user_sgpr_limit(self.stage))
                .then_some((i, dword))
        })
    }

    /// Layout index and dword of the root parameter `v` holds, either from
    /// an entry user SGPR or from a spill-table load.
    fn param_value(&self, v: Value) -> Option<(usize, u32)> {
        if let Some(reg) = self.inputs[v.0 as usize] {
            return self.param_of(reg);
        }
        self.spilled.get(&v).copied()
    }

    /// Layout index and dword covering spilled user-data slot `ud >= 16`.
    fn spilled_param_of(&self, ud: u32) -> Option<(usize, u32)> {
        if ud < user_sgpr_limit(self.stage) {
            return None;
        }
        self.layout.iter().enumerate().find_map(|(i, p)| {
            let dword = ud.checked_sub(u32::from(p.user_data))?;
            (dword < p.sgprs).then_some((i, dword))
        })
    }

    /// Strips a 40/48-bit pointer mask (`s_and_b32 x, 0xff/0xffff` or `s_bfe_u32`).
    fn strip_hi_mask(&self, v: Value) -> Option<Term> {
        if let Some((_, [a, Term::Const(mask)] | [Term::Const(mask), a])) =
            self.sop2(v, &["s_and_b32"])
            && mask & 0xff == 0xff
        {
            return Some(a);
        }
        if let Some((_, [a, Term::Const(ctrl)])) = self.sop2(v, &["s_bfe_u32"]) {
            let off = ctrl & 0x1F;
            let width = (ctrl >> 16) & 0x7F;
            if off == 0 && width >= 8 {
                return Some(a);
            }
        }
        None
    }

    fn is_spill_hi(&self, t: Term, hi_val: Value) -> bool {
        let Term::Value(v) = t else { return false };
        if v == hi_val {
            return true;
        }
        if let Some(a) = self.strip_hi_mask(v) {
            return self.is_spill_hi(a, hi_val);
        }
        if let Some((_, [a, Term::Const(0)] | [Term::Const(0), a])) = self.sop2(v, &["s_addc_u32"])
        {
            return self.is_spill_hi(a, hi_val);
        }
        false
    }

    /// Recognizes scalar loads from the user-data spill table (`s[0:1]` of the
    /// stage's user SGPRs) and resolves their destination dwords to `(param, dword)`.
    fn collect_spill_loads(&mut self) -> Result<Vec<(Value, usize, u32)>, ShaderError> {
        let Some((spill_lo, spill_hi)) = self.spill_ptr else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for (b, block) in self.p.blocks.iter().enumerate() {
            for (i, inst) in block.insts.iter().enumerate() {
                if !inst.name().starts_with("s_load_dword") {
                    continue;
                }
                let Fields::Smem {
                    data: Some(Operand::Scalar { code, dwords }),
                    base: Operand::Scalar { code: bcode, .. },
                    offset,
                    soffset,
                    ..
                } = &inst.fields
                else {
                    continue;
                };
                let (Some(lo_v), Some(hi_v)) =
                    (inst.read(Reg::S(*bcode)), inst.read(Reg::S(bcode + 1)))
                else {
                    continue;
                };
                let (lo, hi) = (self.origin(lo_v), self.origin(hi_v));
                if !self.is_spill_hi(hi, spill_hi) {
                    continue;
                }
                let s = self.sum(lo);
                if s.base.is_some() {
                    continue;
                }
                let fail = |what| ShaderError::Unsupported {
                    offset: inst.machine.offset,
                    what,
                };
                if s.dynamic != Some(spill_lo) {
                    if s.dynamic
                        .is_some_and(|v| self.leaves(v).any(|l| l == spill_lo))
                    {
                        return Err(fail("a spill-table load at a dynamic offset"));
                    }
                    continue;
                }
                let so_term = self
                    .term(inst, soffset)
                    .ok_or_else(|| fail("an unknown offset operand"))?;
                let extra = self.sum(so_term);
                if extra.base.is_some() || extra.dynamic.is_some() {
                    return Err(fail("a spill-table load at a dynamic offset"));
                }
                let byte_off = i64::from(s.offset) + i64::from(extra.offset) + i64::from(*offset);
                if byte_off < 0 || byte_off % 4 != 0 {
                    return Err(fail("an unaligned spill-table load"));
                }
                let Ok(first_ud) =
                    u32::try_from(i64::from(user_sgpr_limit(self.stage)) + byte_off / 4)
                else {
                    return Err(fail("a spill-table load outside the root layout"));
                };
                let mut matched = false;
                for k in 0..*dwords {
                    if let Some((param, dword)) = self.spilled_param_of(first_ud + u32::from(k))
                        && let Some(v) = inst.write(Reg::S(code + k))
                    {
                        self.spilled.insert(v, (param, dword));
                        out.push((v, param, dword));
                        matched = true;
                    }
                }
                if !matched {
                    return Err(fail("a spill-table load outside the root layout"));
                }
                self.spill_loads.push((b, i));
            }
        }
        Ok(out)
    }

    /// The SOP2 operands of `v` if an `s_add` with one of `names` wrote it.
    fn sop2(&self, v: Value, names: &[&str]) -> Option<(&'a Inst, [Term; 2])> {
        let inst = self.def(v)?;
        if !names.contains(&inst.name()) || inst.writes.first()?.1 != v {
            return None;
        }
        let Fields::Sop { src, .. } = &inst.fields else {
            return None;
        };
        Some((
            inst,
            [
                self.term(inst, src.first()?)?,
                self.term(inst, src.get(1)?)?,
            ],
        ))
    }

    fn add_terms(&self, v: Value) -> Option<[Term; 2]> {
        self.sop2(v, &["s_add_u32", "s_add_i32"])
            .map(|(_, t)| t)
            .or_else(|| self.vadd(v))
    }

    /// Operands of `s_addc_u32` or `v_add_co_ci_u32` writing `v`.
    fn carry_add(&self, v: Value) -> Option<[Term; 2]> {
        if let Some((_, t)) = self.sop2(v, &["s_addc_u32"]) {
            return Some(t);
        }
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if inst.name() != "v_add_co_ci_u32"
            || f.dst.is_none_or(|d| !matches!(d, Operand::Vector { .. }))
            || inst.write(Reg::V(match f.dst? {
                Operand::Vector { reg, .. } => reg,
                _ => return None,
            })) != Some(v)
            || f.abs != 0
            || f.neg != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        Some([
            self.vterm(inst, f.src.first()?)?,
            self.vterm(inst, f.src.get(1)?)?,
        ])
    }

    fn sum(&self, t: Term) -> Sum {
        let v = match t {
            Term::Const(c) => {
                return Sum {
                    base: None,
                    offset: c,
                    dynamic: None,
                };
            }
            Term::Value(v) => v,
        };
        let dynamic = Sum {
            base: None,
            offset: 0,
            dynamic: Some(v),
        };
        if let Some((param, 0)) = self.param_value(v)
            && is_address_root(self.layout[param].root)
        {
            return Sum {
                base: Some(v),
                offset: 0,
                dynamic: None,
            };
        }
        if self.inputs[v.0 as usize].is_some() {
            return dynamic;
        }
        if self.getpc(v).is_some() {
            return Sum {
                base: Some(v),
                offset: 0,
                dynamic: None,
            };
        }
        let Some([a, b]) = self.add_terms(v) else {
            return dynamic;
        };
        let (x, y) = (self.sum(a), self.sum(b));
        if x.base.is_some() && y.base.is_some() || x.dynamic.is_some() && y.dynamic.is_some() {
            return dynamic;
        }
        Sum {
            base: x.base.or(y.base),
            offset: x.offset.wrapping_add(y.offset),
            dynamic: x.dynamic.or(y.dynamic),
        }
    }

    /// The `s_getpc_b64` writing `lo` as its low dword.
    fn getpc(&self, lo: Value) -> Option<&'a Inst> {
        let inst = self.def(lo)?;
        (inst.name() == "s_getpc_b64" && inst.writes.first().is_some_and(|w| w.1 == lo))
            .then_some(inst)
    }

    /// The operands of a VALU addition (`v_add_co_u32`, `v_add_nc_u32`)
    /// writing `v`, through VGPR copies of scalars.
    fn vadd(&self, v: Value) -> Option<[Term; 2]> {
        let inst = self.def(v)?;
        let Fields::Valu(f) = &inst.fields else {
            return None;
        };
        if !matches!(inst.name(), "v_add_co_u32" | "v_add_nc_u32")
            || f.dst.is_none_or(|d| !matches!(d, Operand::Vector { .. }))
            || inst.write(Reg::V(match f.dst? {
                Operand::Vector { reg, .. } => reg,
                _ => return None,
            })) != Some(v)
            || f.abs != 0
            || f.neg != 0
            || f.sdwa.is_some()
            || f.dpp.is_some()
        {
            return None;
        }
        Some([
            self.vterm(inst, f.src.first()?)?,
            self.vterm(inst, f.src.get(1)?)?,
        ])
    }

    /// The origin of a VALU operand.
    fn vterm(&self, inst: &Inst, op: &Operand) -> Option<Term> {
        match *op {
            Operand::Vector { reg, .. } => Some(self.vorigin(inst.read(Reg::V(reg))?)),
            op => self.term(inst, &op),
        }
    }

    /// A VGPR value's origin through `v_mov_b32` copies.
    fn vorigin(&self, v: Value) -> Term {
        if let Some(inst) = self.def(v)
            && inst.name() == "v_mov_b32"
            && let Fields::Valu(f) = &inst.fields
            && f.abs == 0
            && f.neg == 0
            && f.sdwa.is_none()
            && f.dpp.is_none()
            && let Some(t) = f.src.first().and_then(|s| self.vterm(inst, s))
        {
            return t;
        }
        self.origin(v)
    }

    /// A global memory address: an SGPR pair plus a VGPR offset, or a VGPR
    /// pair.
    fn flat_address(
        &self,
        inst: &Inst,
        addr: Option<&Operand>,
        saddr: Option<&Operand>,
        offset: i16,
    ) -> Result<Address, &'static str> {
        let Some(&Operand::Vector { reg, .. }) = addr else {
            return Err("a global address not in VGPRs");
        };
        let off = offset as i32 as u32;
        match saddr {
            Some(&Operand::Scalar { code, .. }) if code != NULL => {
                let word = |k| Some(self.origin(inst.read(Reg::S(code + k))?));
                let (Some(lo), Some(hi)) = (word(0), word(1)) else {
                    return Err("an address not in SGPRs");
                };
                let v = inst.read(Reg::V(reg)).ok_or("a global address")?;
                let mut extra = self.sum(self.vorigin(v));
                if extra.base.is_some() {
                    return Err("an offset from a root parameter");
                }
                extra.offset = extra.offset.wrapping_add(off);
                self.address(lo, hi, extra)
            }
            _ => {
                let word = |k| inst.read(Reg::V(reg + k)).map(|v| self.vorigin(v));
                let (Some(lo), Some(hi)) = (word(0), word(1)) else {
                    return Err("a global address");
                };
                self.address(
                    lo,
                    hi,
                    Sum {
                        base: None,
                        offset: off,
                        dynamic: None,
                    },
                )
            }
        }
    }

    /// The address `lo` and `hi` hold, plus `extra` (an instruction's
    /// immediate and offset register).
    fn address(&self, lo: Term, hi: Term, extra: Sum) -> Result<Address, &'static str> {
        let s = self.sum(lo);
        let Some(base) = s.base else {
            if let Some((param, dword, rest)) =
                self.split_const_base(lo, hi, extra.dynamic.is_some())
            {
                if rest.dynamic.is_some() && extra.dynamic.is_some() {
                    return Err("an address with two computed terms");
                }
                return Ok(Address {
                    param,
                    dword,
                    offset: rest.offset.wrapping_add(extra.offset),
                    dynamic: rest.dynamic.or(extra.dynamic),
                });
            }
            if s.dynamic.is_some() && extra.dynamic.is_some() {
                return Err("an address with two computed terms");
            }
            return Err(self.why_no_base(s.dynamic));
        };
        if s.dynamic.is_some() && extra.dynamic.is_some() {
            return Err("an address with two computed terms");
        }
        if let Some(pc) = self.getpc(base) {
            // A table in the code: the base is the next instruction's offset.
            let high = pc.writes.get(1).map(|w| w.1);
            let match_pc = |t: Term| high.is_some_and(|h| t == Term::Value(h));
            let match_pc_v = |v: Value| high == Some(v);
            if !self.high_half(hi, &match_pc) && !self.carried(hi, &match_pc_v) {
                return Err("an address with an untracked high half");
            }
            return Ok(Address {
                param: CODE,
                dword: 0,
                offset: (pc.machine.offset as u32 + 4)
                    .wrapping_add(s.offset)
                    .wrapping_add(extra.offset),
                dynamic: s.dynamic.or(extra.dynamic),
            });
        }
        let (param, _) = self
            .param_value(base)
            .ok_or("an address not from a root parameter")?;
        let descriptor = match self.layout[param].root {
            RootType::Table | RootType::ResourceHeap | RootType::SamplerHeap => false,
            RootType::Descriptor => true,
            _ => return Err("an address from a root parameter that is no address"),
        };
        if descriptor {
            let has_high = self
                .p
                .inputs
                .iter()
                .any(|&(r, _)| self.param_of(r) == Some((param, 1)))
                || self.spilled.values().any(|&p| p == (param, 1));
            if !has_high {
                return Err("a root descriptor without its high half");
            }
        }
        // shortcut: with a computed term, a high half that adds anything to the
        // base's is taken as the carry of that term (offsets below 4 GB).
        let dynamic = s.dynamic.or(extra.dynamic);
        if !self.matches_high(hi, descriptor.then_some((param, 1)), dynamic.is_some()) {
            return Err("an address with an untracked high half");
        }
        Ok(Address {
            param,
            dword: 0,
            offset: s.offset.wrapping_add(extra.offset),
            dynamic,
        })
    }

    /// Resolves `t` as a 64-bit address in `RootType::Constants` `(param, dword)`
    /// paired with `(param, dword + 1)` in `hi`, plus remaining offset/dynamic terms.
    fn split_const_base(
        &self,
        t: Term,
        hi: Term,
        extra_dyn: bool,
    ) -> Option<(usize, u32, Sum)> {
        let Term::Value(v) = t else { return None };
        if let Some((param, dword)) = self.param_value(v)
            && self.layout[param].root == RootType::Constants
            && dword + 1 < self.layout[param].sgprs
            && self.matches_high(hi, Some((param, dword + 1)), extra_dyn)
        {
            return Some((
                param,
                dword,
                Sum {
                    base: None,
                    offset: 0,
                    dynamic: None,
                },
            ));
        }
        let [a, b] = self.add_terms(v)?;
        for (cand, rest) in [(a, b), (b, a)] {
            let other = self.sum(rest);
            if other.base.is_some() {
                continue;
            }
            if let Some((param, dword, mut inner)) =
                self.split_const_base(cand, hi, extra_dyn || other.dynamic.is_some())
            {
                if inner.dynamic.is_some() && other.dynamic.is_some() {
                    continue;
                }
                inner.offset = inner.offset.wrapping_add(other.offset);
                inner.dynamic = inner.dynamic.or(other.dynamic);
                return Some((param, dword, inner));
            }
        }
        None
    }

    fn matches_high(&self, hi: Term, high_param: Option<(usize, u32)>, has_dynamic: bool) -> bool {
        let is_high_term = |t: Term| self.is_param_high(t, high_param);
        let is_high_val = |v: Value| high_param.is_some() && self.is_param_high(Term::Value(v), high_param);
        self.high_half(hi, &is_high_term) || (has_dynamic && self.carried(hi, &is_high_val))
    }

    fn is_param_high(&self, t: Term, param: Option<(usize, u32)>) -> bool {
        match (param, t) {
            (None, Term::Const(0)) => true,
            (Some((p, d)), Term::Value(v)) => {
                self.param_value(v) == Some((p, d))
                    || matches!(self.strip_hi_mask(v), Some(x) if self.is_param_high(x, Some((p, d))))
            }
            _ => false,
        }
    }

    /// `hi` satisfies `want` (0 for a table, high half for descriptor/code), possibly plus a carry.
    fn high_half(&self, hi: Term, want: &dyn Fn(Term) -> bool) -> bool {
        want(hi)
            || matches!(hi, Term::Value(v) if self.carry_add(v).is_some_and(
                |[a, b]| want(a) && b == Term::Const(0) || want(b) && a == Term::Const(0)
            ))
    }

    /// `hi` adds something (with carry) to `want`: `s_addc_u32` or
    /// `v_add_co_ci_u32` with a value matching `want` as an operand.
    fn carried(&self, hi: Term, want: &dyn Fn(Value) -> bool) -> bool {
        let Term::Value(v) = hi else { return false };
        if want(v) {
            return true;
        }
        if let Some([a, b]) = self.carry_add(v) {
            return self.carried(a, want) || self.carried(b, want);
        }
        false
    }

    fn why_no_base(&self, dynamic: Option<Value>) -> &'static str {
        let Some(v) = dynamic else {
            return "a constant address";
        };
        if self
            .leaves(v)
            .any(|l| self.param_value(l).is_some_and(|(p, _)| !is_address_root(self.layout[p].root)))
        {
            return "an address from a root parameter that is no address";
        }
        if self.inputs[v.0 as usize].is_some() {
            return "an address in an SGPR that is no root parameter";
        }
        if self
            .leaves(v)
            .any(|l| self.def(l).is_some_and(|i| i.name() == "s_getpc_b64"))
        {
            return "an address relative to the program counter";
        }
        match self.def(v).map(|i| i.name()) {
            Some(n) if n.starts_with("s_load_") => "an address loaded from memory",
            None => "an address from a phi",
            _ => "an address not from a root parameter",
        }
    }

    /// The terms of the additions that compute `v`.
    fn leaves(&self, v: Value) -> Box<dyn Iterator<Item = Value> + '_> {
        match self.add_terms(v) {
            Some(terms) => Box::new(terms.into_iter().flat_map(move |t| match t {
                Term::Value(x) => self.leaves(x),
                Term::Const(_) => Box::new(std::iter::empty()),
            })),
            None => Box::new(std::iter::once(v)),
        }
    }

    fn smem_address(
        &self,
        inst: &Inst,
        base: &Operand,
        offset: i32,
        soffset: &Operand,
    ) -> Result<Address, &'static str> {
        let Operand::Scalar { code, .. } = *base else {
            return Err("an address not in SGPRs");
        };
        let word = |k| Some(self.origin(inst.read(Reg::S(code + k))?));
        let (Some(lo), Some(hi)) = (word(0), word(1)) else {
            return Err("an address not in SGPRs");
        };
        let mut extra = self.sum(
            self.term(inst, soffset)
                .ok_or("an unknown offset operand")?,
        );
        if extra.base.is_some() {
            return Err("an offset from a root parameter");
        }
        extra.offset = extra.offset.wrapping_add(offset as u32);
        self.address(lo, hi, extra)
    }

    fn descriptor(&self, inst: &Inst, op: &Operand) -> Result<Source, &'static str> {
        let Operand::Scalar { code, dwords } = *op else {
            return Err("a descriptor not in SGPRs");
        };
        let words = (0..dwords)
            .map(|k| inst.read(Reg::S(code + k)).map(|v| self.origin(v)))
            .collect::<Option<Vec<_>>>()
            .ok_or("a descriptor not in SGPRs")?;
        self.classify(&words, &mut Vec::new())
    }

    /// The source of a descriptor's words; `joins` are the phi blocks
    /// being resolved.
    fn classify(&self, words: &[Term], joins: &mut Vec<usize>) -> Result<Source, &'static str> {
        if let Some(source) = self.phi(words, joins)? {
            return Ok(source);
        }
        if let Some(consts) = words
            .iter()
            .map(|w| match *w {
                Term::Const(c) => Some(c),
                Term::Value(_) => None,
            })
            .collect::<Option<Vec<_>>>()
        {
            return Ok(Source::Constant(consts));
        }
        if let Some(source) = self.loaded(words) {
            return Ok(source);
        }
        if let [lo, hi, Term::Const(w2), Term::Const(w3)] = *words {
            let (mut hi, mut bits) = (hi, 0);
            while let Term::Value(v) = hi
                && let Some((x, more)) = self.set_bits(v)
            {
                (hi, bits) = (x, bits | more);
            }
            let address = self.address(
                lo,
                hi,
                Sum {
                    base: None,
                    offset: 0,
                    dynamic: None,
                },
            )?;
            return Ok(Source::Built {
                address,
                bits,
                words: [w2, w3],
            });
        }
        let loaded = |w: &Term| {
            matches!(*w, Term::Value(v)
                if self.def(v).is_some_and(|i| i.name().starts_with("s_load_")))
        };
        Err(if words.iter().any(loaded) {
            "a descriptor changed after loading"
        } else {
            "a descriptor neither loaded nor built"
        })
    }

    /// Words that phis of one block join, traced per predecessor (words
    /// defined before the join hold on every edge).
    fn phi(&self, words: &[Term], joins: &mut Vec<usize>) -> Result<Option<Source>, &'static str> {
        let phi_of = |w: &Term| match *w {
            Term::Value(v) => self.phis[v.0 as usize],
            Term::Const(_) => None,
        };
        let Some((block, _)) = words.iter().find_map(phi_of) else {
            return Ok(None);
        };
        if words.iter().filter_map(phi_of).any(|(b, _)| b != block) {
            return Err("a descriptor joined in several blocks");
        }
        if joins.contains(&block) {
            return Err("a descriptor that changes in a loop");
        }
        // Each level multiplies the sources by the predecessor count.
        if joins.len() >= MAX_JOINS {
            return Err("a descriptor joined too often");
        }
        joins.push(block);
        let phis = &self.p.blocks[block].phis;
        let args = (0..self.p.blocks[block].preds.len())
            .map(|j| {
                let edge: Vec<Term> = words
                    .iter()
                    .map(|w| match phi_of(w) {
                        Some((_, i)) => self.origin(phis[i].args[j]),
                        None => *w,
                    })
                    .collect();
                self.classify(&edge, joins)
            })
            .collect::<Result<_, _>>();
        joins.pop();
        Ok(Some(Source::Phi { block, args: args? }))
    }

    /// `v` as some other dword with constant bits set.
    fn set_bits(&self, v: Value) -> Option<(Term, u32)> {
        if let Some((_, [x, Term::Const(bits)] | [Term::Const(bits), x])) =
            self.sop2(v, &["s_or_b32"])
        {
            return Some((x, bits));
        }
        let inst = self.def(v)?;
        let Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            src,
            ..
        } = &inst.fields
        else {
            return None;
        };
        match (inst.name(), self.term(inst, src.first()?)?) {
            ("s_bitset1_b32", Term::Const(bit)) => {
                Some((self.origin(inst.read(Reg::S(*code))?), 1 << (bit & 31)))
            }
            _ => None,
        }
    }

    /// Consecutive dwords of one scalar load.
    fn loaded(&self, words: &[Term]) -> Option<Source> {
        let Term::Value(v0) = words[0] else {
            return None;
        };
        let load = self.defs[v0.0 as usize]?;
        let inst = self.inst(load);
        if !inst.name().starts_with("s_load_") {
            return None;
        }
        let first = inst.writes.iter().position(|w| w.1 == v0)?;
        let same = words.iter().enumerate().all(|(k, w)| {
            inst.writes
                .get(first + k)
                .is_some_and(|x| *w == Term::Value(x.1))
        });
        same.then_some(Source::Loaded {
            load,
            first: first as u8,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Moved {
    Const(u32),
    Value(Value),
}

/// Resolves every `v_readlane_b32` reading a spilled SGPR back to its scalar value or constant.
pub(crate) fn spilled_sgprs(p: &Program) -> std::collections::HashMap<Site, Moved> {
    let (defs, phis, _) = index_values(p);
    let mut out = std::collections::HashMap::new();
    for (b, block) in p.blocks.iter().enumerate() {
        for (i, inst) in block.insts.iter().enumerate() {
            if let Some(m) = restored_with(p, &defs, &phis, inst) {
                out.insert((b, i), m);
            }
        }
    }
    out
}

fn restored_with(
    p: &Program,
    defs: &[Option<Site>],
    phis: &[Option<(usize, usize)>],
    inst: &Inst,
) -> Option<Moved> {
    let Fields::Valu(v) = &inst.fields else {
        return None;
    };
    if inst.name() != "v_readlane_b32" {
        return None;
    }
    let [Operand::Vector { reg, .. }, lane] = v.src[..] else {
        return None;
    };
    spilled_with(
        p,
        defs,
        phis,
        inst.read(Reg::V(reg))?,
        lane_of(lane)?,
        &mut Vec::new(),
    )
    .ok()
    .flatten()
}

fn spilled_with(
    p: &Program,
    defs: &[Option<Site>],
    phis: &[Option<(usize, usize)>],
    v: Value,
    lane: u32,
    seen: &mut Vec<Value>,
) -> Result<Option<Moved>, ()> {
    if let Some((b, i)) = phis[v.0 as usize] {
        if seen.contains(&v) {
            return Ok(None);
        }
        seen.push(v);
        let mut found = None;
        for &a in &p.blocks[b].phis[i].args {
            if let Some(m) = spilled_with(p, defs, phis, a, lane, seen)? {
                if found.is_some_and(|f| f != m) {
                    return Err(());
                }
                found = Some(m);
            }
        }
        return Ok(found);
    }
    let (b, i) = defs[v.0 as usize].ok_or(())?;
    let inst = &p.blocks[b].insts[i];
    let Fields::Valu(w) = &inst.fields else {
        return Err(());
    };
    let (Some(Operand::Vector { reg, .. }), [value, at]) = (w.dst, &w.src[..]) else {
        return Err(());
    };
    if inst.name() != "v_writelane_b32" {
        return Err(());
    }
    if lane_of(*at).ok_or(())? != lane {
        return spilled_with(p, defs, phis, inst.read(Reg::V(reg)).ok_or(())?, lane, seen);
    }
    match *value {
        Operand::Scalar { code, .. } if code != NULL => inst.read(Reg::S(code)).map(Moved::Value),
        Operand::Literal(x) => Some(Moved::Const(x)),
        Operand::Const(c) => inline(c).map(Moved::Const),
        _ => None,
    }
    .map(Some)
    .ok_or(())
}

impl Tracer<'_> {
    /// For `v_readlane_b32` of a constant lane: the SGPR spilled to that
    /// lane, found by following the VGPR's `v_writelane_b32` writes.
    fn restored(&self, inst: &Inst) -> Option<Moved> {
        restored_with(self.p, &self.defs, &self.phis, inst)
    }
}

/// A lane select given as a constant.
fn lane_of(op: Operand) -> Option<u32> {
    match op {
        Operand::Const(c) => inline(c),
        Operand::Literal(x) => Some(x),
        _ => None,
    }
}

/// What dword `k` of a move's result copies.
fn copy_source(inst: &Inst, k: usize) -> Option<Moved> {
    let Fields::Sop { src, imm, .. } = &inst.fields else {
        return None;
    };
    if inst.name() == "s_movk_i32" {
        return Some(Moved::Const(i32::from((*imm)? as i16) as u32));
    }
    let src = src.first()?;
    match (inst.name(), *src) {
        ("s_mov_b32" | "s_mov_b64", Operand::Scalar { code, .. }) if code != NULL => {
            inst.read(Reg::S(code + k as u8)).map(Moved::Value)
        }
        ("s_mov_b32" | "s_mov_b64", Operand::Scalar { .. }) => Some(Moved::Const(0)),
        ("s_mov_b32", Operand::Literal(x)) => Some(Moved::Const(x)),
        ("s_mov_b32", Operand::Const(c)) => inline(c).map(Moved::Const),
        // shortcut: only the one-operand ways the compiler makes a constant
        // that has no inline form; fold more if descriptors need it.
        ("s_brev_b32", Operand::Const(c)) => inline(c).map(|x| Moved::Const(x.reverse_bits())),
        ("s_not_b32", Operand::Const(c)) => inline(c).map(|x| Moved::Const(!x)),
        // 64-bit inline integers are sign-extended.
        ("s_mov_b64", Operand::Const(c @ 128..=208)) => {
            let low = inline(c)?;
            Some(Moved::Const(if k == 0 {
                low
            } else {
                ((low as i32) >> 31) as u32
            }))
        }
        _ => None,
    }
}

/// The 32-bit value of an inline constant operand.
pub(crate) fn inline(code: u8) -> Option<u32> {
    let float = |f: f32| Some(f.to_bits());
    match code {
        128..=192 => Some(u32::from(code) - 128),
        193..=208 => Some((192 - i32::from(code)) as u32),
        240 => float(0.5),
        241 => float(-0.5),
        242 => float(1.0),
        243 => float(-1.0),
        244 => float(2.0),
        245 => float(-2.0),
        246 => float(4.0),
        247 => float(-4.0),
        248 => Some(0x3E22_F983),
        _ => None,
    }
}

/// How many user SGPRs the hardware loads for `stage` before the rest of the user data spills to memory: GFX10
/// compute has 16, the graphics stages 32.
fn user_sgpr_limit(stage: HwStage) -> u32 {
    if matches!(stage, HwStage::Cs) { 16 } else { 32 }
}

fn is_address_root(root: RootType) -> bool {
    matches!(
        root,
        RootType::Table
            | RootType::ResourceHeap
            | RootType::SamplerHeap
            | RootType::Descriptor
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::RootType::*;

    const S_ENDPGM: u32 = 0xBF81_0000;

    fn track_words(words: &[u32], layout: &[RootParam], stage: HwStage) -> Resources {
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        track(&crate::ir::build(&code, false, 256).unwrap(), layout, stage).unwrap()
    }

    fn param(root: RootType, user_data: u8, sgprs: u32) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data,
            slot: 0,
        }
    }

    fn at(param: usize, offset: u32) -> Address {
        Address {
            param,
            dword: 0,
            offset,
            dynamic: None,
        }
    }

    #[test]
    fn table_descriptors_are_loaded_from_their_range() {
        let r = track_words(
            &[
                0xBE82_0309, // s_mov_b32 s2, s9
                0xBE89_0380, // s_mov_b32 s9, 0
                0xBE83_0380, // s_mov_b32 s3, 0
                0xF408_0204,
                0xFA00_0000, // s_load_dwordx4 s[8:11], s[8:9], null
                0xF40C_0001,
                0xFA00_0000, // s_load_dwordx8 s[0:7], s[2:3], null
                0xF080_0F08,
                0x0040_0002, // image_sample v[0:3], v[2:3], s[0:7], s[8:11]
                S_ENDPGM,
            ],
            &[param(Table, 8, 1), param(Table, 9, 1)],
            HwStage::Ps,
        );
        assert_eq!(r.scalar, [((0, 3), at(0, 0)), ((0, 4), at(1, 0))]);
        let roles: Vec<_> = r.descriptors.iter().map(|d| (d.role, &d.source)).collect();
        assert_eq!(
            roles,
            [
                (
                    Role::Resource,
                    &Source::Loaded {
                        load: (0, 4),
                        first: 0
                    }
                ),
                (
                    Role::Sampler,
                    &Source::Loaded {
                        load: (0, 3),
                        first: 0
                    }
                ),
            ]
        );
    }

    #[test]
    fn root_descriptors_are_built_in_code() {
        let r = track_words(
            &[
                0x8801_FF0B,
                0x0010_0000, // s_or_b32 s1, s11, 0x100000
                0xBE80_030A, // s_mov_b32 s0, s10
                0xBE82_03C1, // s_mov_b32 s2, -1
                0xBE83_03FF,
                0x1101_4FAC, // s_mov_b32 s3, 0x11014fac
                0xF430_0000,
                0xFA00_0000, // s_buffer_load_dwordx16 s[0:15], s[0:3], null
                S_ENDPGM,
            ],
            &[param(Descriptor, 10, 2)],
            HwStage::Vs,
        );
        assert_eq!(
            r.descriptors[0].source,
            Source::Built {
                address: at(0, 0),
                bits: 0x10_0000,
                words: [u32::MAX, 0x1101_4FAC]
            }
        );
        assert_eq!(
            r.params.iter().map(|p| (p.1, p.2)).collect::<Vec<_>>(),
            [(0, 0), (0, 1)]
        );
    }

    #[test]
    fn heap_indices_from_root_constants_in_a_merged_stage() {
        // User data starts at s8: the constant (index 6) is s14, the heap
        // (index 10) s18.
        let r = track_words(
            &[
                0x8F08_850E, // s_lshl_b32 s8, s14, 5
                0xBE93_0380, // s_mov_b32 s19, 0
                0xF408_0209,
                0x1000_0000, // s_load_dwordx4 s[8:11], s[18:19], s8
                0xE000_2000,
                0x8002_0400, // buffer_load_format_x v4, v0, s[8:11], 0 idxen
                S_ENDPGM,
            ],
            &[param(Constants, 6, 1), param(ResourceHeap, 10, 1)],
            HwStage::Ms,
        );
        let index = r.params.iter().find(|p| p.1 == 0).unwrap().0;
        let shift = r.scalar[0].1.dynamic.unwrap();
        assert_eq!(r.scalar[0].1.param, 1);
        // The computed term is the shifted constant, not the constant itself.
        assert_ne!(shift, index);
        assert_eq!(
            r.descriptors[0].source,
            Source::Loaded {
                load: (0, 2),
                first: 0
            }
        );
    }

    #[test]
    fn spilled_sgprs_are_followed_through_vgpr_lanes() {
        let r = track_words(
            &[
                0xD761_0000,
                0x0001_0609, // v_writelane_b32 v0, s9, 3
                0xD760_0002,
                0x0001_0700, // v_readlane_b32 s2, v0, 3
                0xBE83_0380, // s_mov_b32 s3, 0
                0xF40C_0001,
                0xFA00_0020, // s_load_dwordx8 s[0:7], s[2:3], 0x20
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        assert_eq!(r.scalar, [((0, 3), at(0, 0x20))]);
    }

    #[test]
    fn descriptors_joined_by_control_flow_are_traced_per_edge() {
        let r = track_words(
            &[
                0xBE82_0309, // s_mov_b32 s2, s9
                0xBE83_0380, // s_mov_b32 s3, 0
                0xBF06_8001, // s_cmp_eq_u32 s1, 0
                0xBF85_0003, // s_cbranch_scc1 -> 0x1c
                0xF408_0001,
                0xFA00_0000, // s_load_dwordx4 s[0:3], s[2:3], null
                0xBF82_0002, // s_branch -> 0x24
                0xF408_0001,
                0xFA00_0010, // 0x1c: s_load_dwordx4 s[0:3], s[2:3], 0x10
                0xF420_0100,
                0xFA00_0000, // 0x24: s_buffer_load_dword s4, s[0:3], null
                S_ENDPGM,
            ],
            &[param(Table, 9, 1)],
            HwStage::Ps,
        );
        let Source::Phi { args, .. } = &r.descriptors[0].source else {
            panic!("{:?}", r.descriptors)
        };
        let offsets: Vec<_> = args
            .iter()
            .map(|a| match a {
                Source::Loaded { load, .. } => {
                    r.scalar.iter().find(|s| s.0 == *load).unwrap().1.offset
                }
                a => panic!("{a:?}"),
            })
            .collect();
        assert_eq!(offsets.len(), 2);
        assert!(
            offsets.contains(&0) && offsets.contains(&0x10),
            "{offsets:?}"
        );
    }

    #[test]
    fn untraceable_addresses_fail_with_a_reason() {
        let code: Vec<u8> = [0xF408_0001, 0xFA00_0000, S_ENDPGM] // s_load_dwordx4 s[0:3], s[2:3]
            .iter()
            .flat_map(|w: &u32| w.to_le_bytes())
            .collect();
        let p = crate::ir::build(&code, false, 256).unwrap();
        let err = track(&p, &[param(Table, 9, 1)], HwStage::Ps).unwrap_err();
        assert_eq!(
            err,
            ShaderError::Unsupported {
                offset: 0,
                what: "an address in an SGPR that is no root parameter"
            }
        );
    }

    fn smem_load(op: u32, dst: u8, base: u8, soff: u8, offset: u32) -> [u32; 2] {
        [
            (0x3D << 26) | (op << 18) | (u32::from(dst) << 6) | u32::from(base >> 1),
            (u32::from(soff) << 25) | (offset & 0x1F_FFFF),
        ]
    }

    fn s_and_b32_imm(dst: u8, src: u8, imm: u32) -> [u32; 2] {
        [0x8700_FF00 | (u32::from(dst) << 16) | u32::from(src), imm]
    }

    fn s_mov_b32_s(dst: u8, src: u8) -> u32 {
        0xBE80_0300 | (u32::from(dst) << 16) | u32::from(src)
    }

    #[test]
    fn spill_table_loads_resolve_constants_and_root_descriptors() {
        let mut words = Vec::new();
        // s_and_b32 s3, s1, 0xff; s_mov_b32 s2, s0
        words.extend_from_slice(&s_and_b32_imm(3, 1, 0xff));
        words.push(s_mov_b32_s(2, 0));
        // 0: s_load_dwordx16 s[8:23], s[2:3], 0x10 -> user_data 20..35 (Constants, param 2)
        words.extend_from_slice(&smem_load(4, 8, 2, 125, 0x10));
        // 1: s_load_dwordx2 s[0:1], s[2:3], 0x08 -> user_data 18..19 (Descriptor, param 1)
        words.extend_from_slice(&smem_load(1, 0, 2, 125, 0x08));
        // Scalar load from the spilled root descriptor at offset 0x20.
        words.extend_from_slice(&smem_load(2, 4, 0, 125, 0x20));
        // Build a buffer descriptor around the spilled root descriptor s[0:1] -> s[0:3].
        words.extend_from_slice(&[
            0x8801_FF01,
            0x0010_0000, // s_or_b32 s1, s1, 0x100000
            0xBE82_03C1, // s_mov_b32 s2, -1
            0xBE83_03FF,
            0x1101_4FAC, // s_mov_b32 s3, 0x11014fac
            0xF430_0000,
            0xFA00_0000, // s_buffer_load_dwordx16 s[0:15], s[0:3], null
            S_ENDPGM,
        ]);
        let layout = [
            param(Descriptor, 16, 2),
            param(Descriptor, 18, 2),
            param(Constants, 20, 16),
        ];
        let r = track_words(&words, &layout, HwStage::Cs);
        assert_eq!(r.spill_loads, [(0, 2), (0, 3)]);
        assert_eq!(r.scalar, [((0, 4), at(1, 0x20))]);
        assert_eq!(
            r.descriptors[0].source,
            Source::Built {
                address: at(1, 0),
                bits: 0x10_0000,
                words: [u32::MAX, 0x1101_4FAC],
            }
        );
        // 16 dwords from Constants (param 2) + 2 dwords from Descriptor (param 1).
        assert_eq!(r.params.len(), 18);
        assert!(
            (0..16).all(|d| r.params.iter().any(|&(_, p, dw)| p == 2 && dw == d)),
            "{:?}",
            r.params
        );
        assert!(r.params.iter().any(|&(_, p, dw)| p == 1 && dw == 0));
        assert!(r.params.iter().any(|&(_, p, dw)| p == 1 && dw == 1));
    }

    #[test]
    fn spill_table_works_in_merged_stages_and_for_descriptor_tables() {
        // Mesh stage: user SGPRs start at s8, so the spill pointer arrives in s[8:9].
        let mut words = Vec::new();
        words.extend_from_slice(&s_and_b32_imm(11, 9, 0xff));
        words.push(s_mov_b32_s(10, 8));
        // s_load_dword s4, s[10:11], 0x00 -> user_data 32 (Table, param 0): graphics stages load 32 user SGPRs
        words.extend_from_slice(&smem_load(0, 4, 10, 125, 0x00));
        words.push(0xBE85_0380); // s_mov_b32 s5, 0
        // s_load_dwordx4 s[12:15], s[4:5], 0x20 -> descriptor from Table
        words.extend_from_slice(&smem_load(2, 12, 4, 125, 0x20));
        words.extend_from_slice(&[
            0xE000_2000,
            0x8003_0400, // buffer_load_format_x v4, v0, s[12:15], 0 idxen
            S_ENDPGM,
        ]);
        let r = track_words(&words, &[param(Table, 32, 1)], HwStage::Ms);
        assert_eq!(r.spill_loads, [(0, 2)]);
        assert_eq!(r.scalar, [((0, 4), at(0, 0x20))]);
        assert_eq!(
            r.descriptors[0].source,
            Source::Loaded {
                load: (0, 4),
                first: 0,
            }
        );
    }

    #[test]
    fn spill_table_rejects_dynamic_unaligned_and_out_of_layout_loads() {
        let track_err = |words: &[u32], layout: &[RootParam]| {
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let p = crate::ir::build(&code, false, 256).unwrap();
            track(&p, layout, HwStage::Cs).unwrap_err()
        };
        let mut prefix = Vec::new();
        prefix.extend_from_slice(&s_and_b32_imm(3, 1, 0xff));
        prefix.push(s_mov_b32_s(2, 0));

        // Dynamic soffset (s4).
        let mut dyn_off = prefix.clone();
        dyn_off.extend_from_slice(&smem_load(0, 8, 2, 4, 0));
        dyn_off.push(S_ENDPGM);
        assert_eq!(
            track_err(&dyn_off, &[param(Constants, 16, 4)]),
            ShaderError::Unsupported {
                offset: 12,
                what: "a spill-table load at a dynamic offset",
            }
        );

        // Unaligned byte offset (0x02).
        let mut unaligned = prefix.clone();
        unaligned.extend_from_slice(&smem_load(0, 8, 2, 125, 2));
        unaligned.push(S_ENDPGM);
        assert_eq!(
            track_err(&unaligned, &[param(Constants, 16, 4)]),
            ShaderError::Unsupported {
                offset: 12,
                what: "an unaligned spill-table load",
            }
        );

        // Offset past the root layout (user_data 32 when layout only has 16..20).
        let mut outside = prefix.clone();
        outside.extend_from_slice(&smem_load(0, 8, 2, 125, 0x40));
        outside.push(S_ENDPGM);
        assert_eq!(
            track_err(&outside, &[param(Constants, 16, 4)]),
            ShaderError::Unsupported {
                offset: 12,
                what: "a spill-table load outside the root layout",
            }
        );

        // Without user_data >= 16, s[0:1] is not a spill pointer.
        let mut no_spill = prefix;
        no_spill.extend_from_slice(&smem_load(0, 8, 2, 125, 0));
        no_spill.push(S_ENDPGM);
        assert_eq!(
            track_err(&no_spill, &[param(Table, 9, 1)]),
            ShaderError::Unsupported {
                offset: 12,
                what: "an address in an SGPR that is no root parameter",
            }
        );
    }

    fn mubuf(op: u32, offset: u16, srsrc: u8, vdata: u8, vaddr: u8) -> [u32; 2] {
        [
            (0x38 << 26) | (op << 18) | (u32::from(offset) & 0xFFF),
            (0x80 << 24)
                | ((u32::from(srsrc >> 2) & 0x1F) << 16)
                | (u32::from(vdata) << 8)
                | u32::from(vaddr),
        ]
    }

    fn s_mov_b32_imm(dst: u8, imm: u32) -> [u32; 2] {
        [0xBE80_03FF | (u32::from(dst) << 16), imm]
    }

    fn s_or_b32_imm(dst: u8, src: u8, imm: u32) -> [u32; 2] {
        [0x8800_FF00 | (u32::from(dst) << 16) | u32::from(src), imm]
    }

    fn s_add_u32_imm(dst: u8, src: u8, imm: u32) -> [u32; 2] {
        [0x8000_FF00 | (u32::from(dst) << 16) | u32::from(src), imm]
    }

    fn s_add_u32_s(dst: u8, src0: u8, src1: u8) -> u32 {
        0x8000_0000 | (u32::from(dst) << 16) | (u32::from(src1) << 8) | u32::from(src0)
    }

    fn s_addc_u32_zero(dst: u8, src0: u8) -> u32 {
        0x8200_8000 | (u32::from(dst) << 16) | u32::from(src0)
    }

    #[test]
    fn spilled_root_uav_descriptor_tracks_stores_loads_and_scalar_reads() {
        let mut words = Vec::new();
        words.extend_from_slice(&s_and_b32_imm(3, 1, 0xff));
        words.push(s_mov_b32_s(2, 0));
        // s_load_dwordx2 s[0:1], s[2:3], 0x08 -> user_data 18..19 (Descriptor, param 1)
        words.extend_from_slice(&smem_load(1, 0, 2, 125, 0x08));
        words.extend_from_slice(&s_or_b32_imm(1, 1, 0x0010_0000));
        words.push(0xBE82_03C1); // s_mov_b32 s2, -1
        words.extend_from_slice(&s_mov_b32_imm(3, 0x1101_4FAC));
        words.extend_from_slice(&[0xF420_0100, 0xFA00_0000]); // s_buffer_load_dword s4, s[0:3], null
        words.extend_from_slice(&mubuf(0x1C, 16, 0, 0, 0)); // buffer_store_dword
        words.extend_from_slice(&mubuf(0x0C, 4, 0, 1, 0)); // buffer_load_dword
        words.push(S_ENDPGM);

        let layout = [
            param(Descriptor, 16, 2),
            param(Descriptor, 18, 2),
            param(Constants, 20, 16),
        ];
        let r = track_words(&words, &layout, HwStage::Cs);
        assert_eq!(r.spill_loads, [(0, 2)]);
        assert_eq!(r.descriptors.len(), 3);
        for d in &r.descriptors {
            assert_eq!(
                d.source,
                Source::Built {
                    address: at(1, 0),
                    bits: 0x0010_0000,
                    words: [u32::MAX, 0x1101_4FAC],
                }
            );
        }
        assert!(r.params.iter().any(|&(_, p, dw)| p == 1 && dw == 0));
        assert!(r.params.iter().any(|&(_, p, dw)| p == 1 && dw == 1));
    }

    #[test]
    fn constants_64bit_addresses_resolve_direct_spilled_and_arithmetic() {
        // Direct Constants (ud 0..4, param 0):
        // - dwords (2, 3) masked with s_and_b32 0xffff and built into a V#.
        // - dwords (0, 1) with scalar s_add_u32/s_addc_u32 +0x20 loaded via smem.
        // - dwords (0, 1) with vector v_add_co_u32/v_add_co_ci_u32 loaded via global_load_dword.
        let mut words = Vec::new();
        words.push(s_mov_b32_s(4, 2));
        words.extend_from_slice(&s_and_b32_imm(5, 3, 0xffff));
        words.extend_from_slice(&s_or_b32_imm(5, 5, 0x0010_0000));
        words.push(0xBE86_03C1); // s_mov_b32 s6, -1
        words.extend_from_slice(&s_mov_b32_imm(7, 0x1101_4FAC));
        words.extend_from_slice(&mubuf(0x0C, 0, 4, 0, 0));
        words.extend_from_slice(&s_add_u32_imm(8, 0, 0x20));
        words.push(s_addc_u32_zero(9, 1));
        words.extend_from_slice(&smem_load(0, 10, 8, 125, 0));
        // Vector 64-bit carry: v_mov_b32 v1, 0; v_add_co_u32 v0, vcc, s0, v4; v_add_co_ci_u32 v1, vcc, s1, v1, vcc
        words.push(0x7E02_0280);
        words.extend_from_slice(&[
            (0x35 << 26) | (0x30F << 16) | (106 << 8),
            (256 + 4) << 9, // s0 + v4
        ]);
        words.push((0x28 << 25) | (1 << 17) | (1 << 9) | 1);
        words.extend_from_slice(&[
            (0x37 << 26) | (0x0C << 18) | (2 << 14) | 16,
            (3 << 24) | (125 << 16), // global_load_dword v3, v[0:1], off offset:16
        ]);
        words.push(S_ENDPGM);

        let r = track_words(&words, &[param(Constants, 0, 4)], HwStage::Cs);
        assert_eq!(
            r.descriptors[0].source,
            Source::Built {
                address: Address {
                    param: 0,
                    dword: 2,
                    offset: 0,
                    dynamic: None,
                },
                bits: 0x0010_0000,
                words: [u32::MAX, 0x1101_4FAC],
            }
        );
        assert_eq!(
            r.scalar[0].1,
            Address {
                param: 0,
                dword: 0,
                offset: 0x20,
                dynamic: None,
            }
        );
        assert_eq!(r.scalar[1].1.param, 0);
        assert_eq!(r.scalar[1].1.dword, 0);
        assert_eq!(r.scalar[1].1.offset, 16);
        assert!(r.scalar[1].1.dynamic.is_some());

        // Spilled Constants (ud 20, param 2): read dwords (2, 3) at spill offset 0x18.
        let mut spilled = Vec::new();
        spilled.extend_from_slice(&s_and_b32_imm(3, 1, 0xff));
        spilled.push(s_mov_b32_s(2, 0));
        spilled.extend_from_slice(&smem_load(1, 4, 2, 125, 0x18));
        spilled.push(0xBE86_03C1); // s_mov_b32 s6, -1
        spilled.extend_from_slice(&s_mov_b32_imm(7, 0x1101_4FAC));
        spilled.extend_from_slice(&mubuf(0x1C, 0, 4, 0, 0));
        spilled.push(S_ENDPGM);
        let rs = track_words(
            &spilled,
            &[
                param(Descriptor, 16, 2),
                param(Descriptor, 18, 2),
                param(Constants, 20, 8),
            ],
            HwStage::Cs,
        );
        assert_eq!(
            rs.descriptors[0].source,
            Source::Built {
                address: Address {
                    param: 2,
                    dword: 2,
                    offset: 0,
                    dynamic: None,
                },
                bits: 0,
                words: [u32::MAX, 0x1101_4FAC],
            }
        );
    }

    #[test]
    fn constants_dword0_offsets_address_roots_and_invalid_non_address_fails() {
        // Table base in s0 (hi zeroed in s1) + Constants dword 0 in s2:
        // Constants dword 0 must act as dynamic offset, not a base address.
        let mut words = Vec::new();
        words.push(0xBE81_0380); // s_mov_b32 s1, 0
        words.push(s_add_u32_s(4, 0, 2));
        words.push(s_addc_u32_zero(5, 1));
        words.extend_from_slice(&smem_load(2, 8, 4, 125, 0));
        words.push(S_ENDPGM);

        let r = track_words(
            &words,
            &[param(Table, 0, 1), param(Constants, 2, 1)],
            HwStage::Ps,
        );
        assert_eq!(r.scalar.len(), 1);
        let a = r.scalar[0].1;
        assert_eq!((a.param, a.dword, a.offset), (0, 0, 0));
        assert!(a.dynamic.is_some());

        // Single-dword Constant + 0 (not a 64-bit consecutive pair) still fails
        // with "an address from a root parameter that is no address".
        let track_err = |words: &[u32], layout: &[RootParam]| {
            let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let p = crate::ir::build(&code, false, 256).unwrap();
            track(&p, layout, HwStage::Cs).unwrap_err()
        };
        let mut bad = vec![0xBE81_0380]; // s_mov_b32 s1, 0
        bad.extend_from_slice(&smem_load(0, 4, 0, 125, 0));
        bad.push(S_ENDPGM);
        assert_eq!(
            track_err(&bad, &[param(Constants, 0, 2)]),
            ShaderError::Unsupported {
                offset: 4,
                what: "an address from a root parameter that is no address",
            }
        );

        // Non-consecutive Constant dwords (s0, s2) also fail cleanly.
        let mut non_adj = vec![s_mov_b32_s(4, 0), s_mov_b32_s(5, 2)];
        non_adj.extend_from_slice(&smem_load(0, 8, 4, 125, 0));
        non_adj.push(S_ENDPGM);
        assert_eq!(
            track_err(&non_adj, &[param(Constants, 0, 4)]),
            ShaderError::Unsupported {
                offset: 8,
                what: "an address from a root parameter that is no address",
            }
        );
    }
}
