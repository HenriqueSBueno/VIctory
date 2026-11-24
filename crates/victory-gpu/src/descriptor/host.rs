//! Host descriptor heaps behind the title's shader-visible heaps (decision
//! 0007, step 3). Translated shaders index the heaps directly at the
//! title's descriptor positions, so each host heap keeps the title heap's
//! order: slot `i` is made from what the title wrote at slot `i`, when it
//! writes it (view creation and descriptor copies) and again when a draw
//! reads it through a table and finds the slot changed.

use std::sync::{Arc, Mutex};

use victory_shader::ResourceType;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::{VIEW_TAG, View, check_live, forget_epoch, view_at_epoch};
use crate::host::host;
use crate::root::{DescriptorRange, RangeKind};

/// Host synchronization work needed when a mirrored descriptor slot is bound
/// by a command list (refreshing host buffer copies, recording UAV writebacks,
/// uploading/transitioning textures).
#[derive(Clone)]
pub(crate) enum SlotSync {
    Buffer {
        shadow: Arc<crate::shadow::Shadow>,
        address: u64,
        len: u64,
        uav: bool,
    },
    Texture {
        resource: usize,
        uav: bool,
    },
}

/// Backend operations for [`MirrorCache`], abstracted for CPU unit testing.
pub(crate) trait MirrorBackend {
    fn lookup(&self, i: u32, generation: u32) -> Option<(View, u64)>;
    fn check_live(&self, pairs: &[(u32, u32)], out: &mut Vec<bool>) -> u64;
    fn forget_epoch(&self) -> u64;
    fn is_buffer(&self, view: &View) -> bool;
    fn make_view(&self, index: u32, view: &View, later: bool) -> Option<SlotSync>;
    fn make_null(&self, index: u32);
    #[cfg(feature = "diagnostics")]
    fn note_unknown(&self, _index: u32, _handle: u64) {}
}

/// Per-slot cache of the title descriptor record `(generation << 32) | index`
/// each static host slot was last made from, plus its validation epoch and
/// cached host resource sync metadata.
pub(crate) struct MirrorCache {
    made: Vec<u64>,
    epoch: Vec<u64>,
    sync_id: Vec<u32>,
    syncs: Vec<Option<SlotSync>>,
    free_syncs: Vec<u32>,
    stale_indices: Vec<u32>,
    stale_pairs: Vec<(u32, u32)>,
    stale_live: Vec<bool>,
}

impl MirrorCache {
    pub(crate) fn new(count: u32) -> Self {
        let n = count as usize;
        Self {
            made: vec![0; n],
            epoch: vec![0; n],
            sync_id: vec![0; n],
            syncs: Vec::new(),
            free_syncs: Vec::new(),
            stale_indices: Vec::new(),
            stale_pairs: Vec::new(),
            stale_live: Vec::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn is_made(&self, index: u32) -> bool {
        self.made.get(index as usize).copied().unwrap_or(0) != 0
    }

    #[cfg(any(feature = "diagnostics", test))]
    pub(crate) fn made_key(&self, index: u32) -> Option<u64> {
        self.made.get(index as usize).copied()
    }

    fn clear_sync(&mut self, idx: usize) {
        let id = std::mem::take(&mut self.sync_id[idx]);
        if id != 0 {
            self.syncs[id as usize - 1] = None;
            self.free_syncs.push(id);
        }
    }

    fn set_sync(&mut self, idx: usize, sync: Option<SlotSync>) {
        let Some(sync) = sync else {
            self.clear_sync(idx);
            return;
        };
        let existing = self.sync_id[idx];
        if existing != 0 {
            self.syncs[existing as usize - 1] = Some(sync);
        } else if let Some(id) = self.free_syncs.pop() {
            self.syncs[id as usize - 1] = Some(sync);
            self.sync_id[idx] = id;
        } else {
            self.syncs.push(Some(sync));
            self.sync_id[idx] = self.syncs.len() as u32;
        }
    }

    fn clear_slot(&mut self, idx: usize) {
        self.made[idx] = 0;
        self.epoch[idx] = 0;
        self.clear_sync(idx);
    }

    fn push_sync(&self, k: u32, idx: usize, out_syncs: &mut Option<&mut Vec<(u32, SlotSync)>>) {
        let Some(out) = out_syncs.as_deref_mut() else {
            return;
        };
        let id = self.sync_id[idx];
        if id != 0
            && let Some(sync) = &self.syncs[id as usize - 1]
        {
            out.push((k, sync.clone()));
        }
    }

    /// Reads `[tag, index, generation]` from `count` contiguous title slots
    /// starting at `first_handle` (`step` bytes apart) and re-mirrors any slot
    /// whose record differs from the cached host slot (including slots written
    /// by plain memory copies or invalidated by resource release).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn sync_range(
        &mut self,
        first: u32,
        first_handle: u64,
        step: u32,
        count: u32,
        for_draw: bool,
        backend: &impl MirrorBackend,
        mut out_syncs: Option<&mut Vec<(u32, SlotSync)>>,
    ) {
        if let Some(out) = out_syncs.as_deref_mut() {
            out.clear();
        }
        if count == 0 || first_handle == 0 {
            return;
        }
        let Some(end) = first.checked_add(count) else {
            return;
        };
        if (end as usize) > self.made.len() {
            return;
        }
        let cur_epoch = backend.forget_epoch();
        self.stale_indices.clear();
        self.stale_pairs.clear();
        for k in 0..count {
            let idx = (first + k) as usize;
            let handle = first_handle + u64::from(k) * u64::from(step);
            // SAFETY: `first + k < self.made.len()`, inside the title heap.
            let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
            let key = if tag == VIEW_TAG && generation != 0 {
                u64::from(generation) << 32 | u64::from(i)
            } else {
                0
            };
            if key == 0 {
                #[cfg(feature = "diagnostics")]
                if for_draw {
                    backend.note_unknown(first + k, handle);
                }
                if self.made[idx] != 0 {
                    self.clear_slot(idx);
                    backend.make_null(first + k);
                }
                continue;
            }
            if self.made[idx] == key {
                if self.epoch[idx] == cur_epoch {
                    self.push_sync(k, idx, &mut out_syncs);
                } else {
                    self.stale_indices.push(k);
                    self.stale_pairs.push((i, generation));
                }
                continue;
            }
            let Some((view, view_epoch)) = backend.lookup(i, generation) else {
                if self.made[idx] != 0 {
                    self.clear_slot(idx);
                    backend.make_null(first + k);
                }
                continue;
            };
            let later = !for_draw && backend.is_buffer(&view);
            let sync = backend.make_view(first + k, &view, later);
            if later {
                self.clear_slot(idx);
            } else {
                self.made[idx] = key;
                self.epoch[idx] = view_epoch;
                self.set_sync(idx, sync.clone());
                if let (Some(sync), Some(out)) = (sync, out_syncs.as_deref_mut()) {
                    out.push((k, sync));
                }
            }
        }
        if !self.stale_pairs.is_empty() {
            let live_epoch = backend.check_live(&self.stale_pairs, &mut self.stale_live);
            for n in 0..self.stale_indices.len() {
                let k = self.stale_indices[n];
                let idx = (first + k) as usize;
                if self.stale_live.get(n).copied().unwrap_or(false) {
                    self.epoch[idx] = live_epoch;
                    self.push_sync(k, idx, &mut out_syncs);
                } else {
                    self.clear_slot(idx);
                    backend.make_null(first + k);
                }
            }
        }
    }
}

/// Maximum slots scanned at bind time for an unbounded root descriptor table
/// range on a static host heap (`4096` slots = 128 KiB of 32-byte views).
pub(crate) const UNBOUNDED_STATIC_LIMIT: u32 = 4096;

/// Maximum slots copied per unbounded root descriptor table onto a `fresh`
/// dynamic lease (bounded so dynamic slots are not exhausted on transient
/// tables).
pub(crate) const UNBOUNDED_FRESH_LIMIT: u32 = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct UsedSlot {
    pub(crate) slot_offset: u32,
    pub(crate) kind: ResourceType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TableRange {
    pub(crate) start: u32,
    pub(crate) end: u32,
    pub(crate) kind: Option<ResourceType>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TablePlan {
    Static {
        ranges: Vec<TableRange>,
    },
    Fresh {
        n: u32,
        ranges: Vec<TableRange>,
    },
}

fn range_kind_to_res(kind: RangeKind) -> ResourceType {
    match kind {
        RangeKind::Srv => ResourceType::Srv,
        RangeKind::Uav => ResourceType::Uav,
        RangeKind::Cbv => ResourceType::Cbv,
        RangeKind::Sampler => ResourceType::Sampler,
    }
}

/// Computes the slot ranges to mirror for a bound descriptor table and whether
/// it must use dynamic `fresh` slots (CBV alignment or NVIDIA capped-heap
/// overflow). Finite ranges use their exact descriptor count; unbounded ranges
/// (`count == None`, or finite counts exceeding [`UNBOUNDED_STATIC_LIMIT`]) are
/// bounded by [`UNBOUNDED_STATIC_LIMIT`] / [`UNBOUNDED_FRESH_LIMIT`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_table(
    sig_ranges: Option<&[DescriptorRange]>,
    used_slots: &[UsedSlot],
    base_index: u32,
    heap_count: u32,
    host_count: u32,
    dynamic_cap: u32,
    samplers: bool,
    table: u64,
    step: u32,
    is_unaligned_cbv: impl Fn(u64) -> bool,
    has_record: impl Fn(u64) -> bool,
) -> TablePlan {
    let remaining = heap_count.saturating_sub(base_index);
    if remaining == 0 || step == 0 {
        return TablePlan::Static { ranges: Vec::new() };
    }
    let used_end = used_slots
        .iter()
        .map(|u| u.slot_offset.saturating_add(1))
        .max()
        .unwrap_or(0)
        .min(remaining);
    let fresh_unbounded_cap = UNBOUNDED_FRESH_LIMIT.min(dynamic_cap.max(1));

    let has_used_cbv = used_slots.iter().any(|u| u.kind == ResourceType::Cbv);
    let past_static_base = base_index >= host_count;
    let past_static_used = used_end > 0 && base_index.saturating_add(used_end) > host_count;
    let mut past_static_range = false;
    let mut has_unaligned_cbv = false;

    if !samplers {
        let slot_addr = |k: u32| table + u64::from(k) * u64::from(step);
        if let Some(sig) = sig_ranges.filter(|s| !s.is_empty()) {
            for r in sig {
                if r.offset >= remaining {
                    continue;
                }
                let finite = r.count.filter(|&c| c <= UNBOUNDED_STATIC_LIMIT);
                if let Some(c) = finite {
                    let end = r.offset.saturating_add(c).min(remaining);
                    if end > r.offset && base_index.saturating_add(end) > host_count {
                        past_static_range = true;
                    }
                    if r.kind == RangeKind::Cbv {
                        for k in r.offset..end {
                            if is_unaligned_cbv(slot_addr(k)) {
                                has_unaligned_cbv = true;
                                break;
                            }
                        }
                    }
                } else {
                    if !past_static_base {
                        let static_left = host_count.saturating_sub(base_index).max(r.offset);
                        let scan_end = r
                            .offset
                            .saturating_add(fresh_unbounded_cap)
                            .max(used_end)
                            .min(remaining);
                        if (static_left..scan_end).any(|k| has_record(slot_addr(k))) {
                            past_static_range = true;
                        }
                    }
                    if r.kind == RangeKind::Cbv {
                        let cbv_end = r
                            .offset
                            .saturating_add(fresh_unbounded_cap)
                            .max(used_end)
                            .min(remaining);
                        for k in r.offset..cbv_end {
                            if is_unaligned_cbv(slot_addr(k)) {
                                has_unaligned_cbv = true;
                                break;
                            }
                        }
                    }
                }
            }
        } else {
            for k in 0..used_end {
                if is_unaligned_cbv(slot_addr(k)) {
                    has_unaligned_cbv = true;
                    break;
                }
            }
        }
    }

    let fresh = !samplers
        && (has_used_cbv
            || past_static_base
            || past_static_used
            || past_static_range
            || has_unaligned_cbv);

    if fresh {
        let mut ranges = Vec::new();
        let n = if let Some(sig) = sig_ranges.filter(|s| !s.is_empty()) {
            let mut max_end = used_end;
            for r in sig {
                if r.offset >= remaining {
                    continue;
                }
                let end = match r.count {
                    Some(c) if c <= UNBOUNDED_STATIC_LIMIT => {
                        r.offset.saturating_add(c).min(remaining)
                    }
                    _ => {
                        let cap = r
                            .offset
                            .saturating_add(fresh_unbounded_cap)
                            .max(used_end)
                            .min(remaining);
                        let min_start = max_end.max(r.offset);
                        let last = (min_start..cap)
                            .rev()
                            .find(|&k| has_record(table + u64::from(k) * u64::from(step)));
                        match last {
                            Some(k) => k + 1,
                            None => min_start,
                        }
                    }
                };
                if end > r.offset {
                    ranges.push(TableRange {
                        start: r.offset,
                        end,
                        kind: Some(range_kind_to_res(r.kind)),
                    });
                    max_end = max_end.max(end);
                }
            }
            max_end.max(1).min(remaining)
        } else {
            used_end.max(1).min(remaining)
        };
        return TablePlan::Fresh { n, ranges };
    }

    let static_cap = host_count.saturating_sub(base_index).min(remaining);
    let mut ranges = Vec::new();
    if let Some(sig) = sig_ranges.filter(|s| !s.is_empty()) {
        for r in sig {
            if r.offset >= static_cap {
                continue;
            }
            let end = match r.count {
                Some(c) if c <= UNBOUNDED_STATIC_LIMIT => {
                    r.offset.saturating_add(c).min(static_cap)
                }
                _ => r
                    .offset
                    .saturating_add(UNBOUNDED_STATIC_LIMIT)
                    .max(used_end)
                    .min(static_cap),
            };
            if end > r.offset {
                ranges.push(TableRange {
                    start: r.offset,
                    end,
                    kind: Some(range_kind_to_res(r.kind)),
                });
            }
        }
        for u in used_slots {
            if u.slot_offset < static_cap
                && !ranges
                    .iter()
                    .any(|rg| rg.start <= u.slot_offset && u.slot_offset < rg.end)
            {
                ranges.push(TableRange {
                    start: u.slot_offset,
                    end: u.slot_offset + 1,
                    kind: Some(u.kind),
                });
            }
        }
    } else {
        let end = used_end.max(1).min(static_cap);
        if end > 0 {
            ranges.push(TableRange {
                start: 0,
                end,
                kind: None,
            });
        }
    }
    TablePlan::Static { ranges }
}

pub(crate) struct HostHeap {
    pub(crate) heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    /// Slots mirroring the title's; view heaps have `dynamic` more.
    count: u32,
    /// Dynamic slots past `count` (none in sampler heaps): [`DYNAMIC`]
    /// where the host takes them, else [`MIN_DYNAMIC`].
    dynamic: u32,
    sampler: bool,
    /// Per-slot mirror cache: last mirrored record, forget epoch, and sync info.
    cache: Mutex<MirrorCache>,
    /// The host resource each slot views, held while the slot does: shaders
    /// index heaps directly, so a slot the title has not rewritten since it
    /// released the resource may still be read, and a freed host resource
    /// read there is a GPU page fault (the console reads stale memory).
    kept: Mutex<Vec<Option<ID3D12Resource>>>,
    /// Where the search for free dynamic slots starts, and which are taken
    /// (by lists not yet reset).
    next: Mutex<(u32, Vec<bool>)>,
}

/// Host heaps' GPU ranges (start, bytes, bytes per slot, sampler), for
/// device-removed reports: a shader indexing a heap past its end faults.
static GPU_RANGES: Mutex<Vec<(u64, u64, u64, bool)>> = Mutex::new(Vec::new());

/// For device-removed reports: where host GPU address `va` lies against
/// the host descriptor heaps.
pub(crate) fn near_heaps(va: u64) -> String {
    let ranges = GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner());
    let mut text = String::new();
    for &(start, bytes, step, sampler) in ranges.iter() {
        let kind = if sampler { "sampler" } else { "view" };
        let place = if va < start {
            format!("{:#x} below", start - va)
        } else if va < start + bytes {
            format!("inside, slot {}", (va - start) / step)
        } else {
            format!("{:#x} past its end (slot {})", va - start - bytes, (va - start) / step)
        };
        text.push_str(&format!("; {kind} heap {start:#x}+{bytes:#x}: {place}"));
    }
    text
}

/// The host's limit for shader-visible view heaps on resource binding
/// tiers 1 and 2; tier 3 hardware may take more.
const PORTABLE_VIEWS: u32 = 1_000_000;

/// Slots past the title's in a view heap, for tables a draw needs made
/// afresh (constant buffers copied per draw). A list holds the slots it
/// took until it is reset (its submissions have run) or released.
/// 2^16 ran out seconds into the world (every world run on an RX 9070 XT,
/// each a device removal shortly after; #59).
const DYNAMIC: u32 = 1 << 18;

/// The dynamic slots of a heap whose host would not take [`DYNAMIC`] more
/// (a capped heap, NVIDIA): the title's slots keep their room first.
const MIN_DYNAMIC: u32 = 1 << 16;

impl HostHeap {
    pub(crate) fn new(sampler: bool, count: u32) -> Option<Self> {
        let host = host()?;
        let kind = if sampler {
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
        } else {
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV
        };
        let make = |count: u32| -> windows::core::Result<ID3D12DescriptorHeap> {
            // SAFETY: plain heap creation.
            unsafe {
                host.device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: kind,
                        NumDescriptors: count,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
            }
        };
        // The larger dynamic range only where the host takes it on top of
        // every title slot; else the smaller one, and the cap after that.
        let (heap, count, extra) = if sampler {
            match make(count) {
                Ok(heap) => (heap, count, 0),
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        } else if let Ok(heap) = make(count + DYNAMIC) {
            (heap, count, DYNAMIC)
        } else {
            match make(count + MIN_DYNAMIC) {
                Ok(heap) => (heap, count, MIN_DYNAMIC),
                Err(e) if count + MIN_DYNAMIC > PORTABLE_VIEWS => {
                    // shortcut: descriptors past the host's limit are left
                    // out; a title that reaches them needs remapping.
                    tracing::warn!(error = %e, count, "host view heap capped at {PORTABLE_VIEWS}");
                    (make(PORTABLE_VIEWS).ok()?, PORTABLE_VIEWS - MIN_DYNAMIC, MIN_DYNAMIC)
                }
                Err(e) => {
                    tracing::warn!(error = %e, count, sampler, "no host descriptor heap");
                    return None;
                }
            }
        };
        // SAFETY: a live heap.
        let (cpu, step) = unsafe {
            (
                heap.GetCPUDescriptorHandleForHeapStart().ptr,
                host.device.GetDescriptorHandleIncrementSize(kind),
            )
        };
        tracing::debug!(count, sampler, "host descriptor heap");
        // SAFETY: a live shader-visible heap.
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart().ptr };
        GPU_RANGES.lock().unwrap_or_else(|e| e.into_inner()).push((
            gpu,
            u64::from(count + extra) * u64::from(step),
            u64::from(step),
            sampler,
        ));
        let h = HostHeap {
            heap,
            cpu,
            step,
            count,
            dynamic: extra,
            sampler,
            cache: Mutex::new(MirrorCache::new(count)),
            kept: Mutex::new(vec![None; (count + extra) as usize]),
            next: Mutex::new((0, vec![false; extra as usize])),
        };
        // A slot never written is undefined: shaders reading one (a slot
        // the title wrote in a way we do not know, or a table entry no
        // used-resource entry names) read a wild address. Every slot starts
        // null instead.
        let started = std::time::Instant::now();
        if let Err(e) = h.fill_null(kind, count + extra) {
            tracing::warn!(error = %e, "host descriptor heap not made null");
        }
        tracing::debug!(ms = started.elapsed().as_millis(), "host descriptor heap made null");
        Some(h)
    }

    /// Makes the first `total` slots null: one null slot in a CPU-side heap,
    /// doubled by copies to a block, the block copied over the heap.
    fn fill_null(&self, kind: D3D12_DESCRIPTOR_HEAP_TYPE, total: u32) -> windows::core::Result<()> {
        const BLOCK: u32 = 1 << 16;
        let Some(host) = host() else { return Ok(()) };
        // SAFETY: plain heap creation and copies inside live heaps.
        unsafe {
            let block: ID3D12DescriptorHeap =
                host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: BLOCK,
                    ..Default::default()
                })?;
            let start = block.GetCPUDescriptorHandleForHeapStart();
            let at = |i: u32| D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: start.ptr + (i * self.step) as usize,
            };
            if self.sampler {
                null_sampler(start);
            } else {
                null_view(start);
            }
            let mut filled = 1;
            while filled < BLOCK {
                let n = filled.min(BLOCK - filled);
                host.device.CopyDescriptorsSimple(n, at(filled), start, kind);
                filled += n;
            }
            let mut done = 0;
            while done < total {
                let n = BLOCK.min(total - done);
                host.device.CopyDescriptorsSimple(n, self.cpu(done), start, kind);
                done += n;
            }
        }
        Ok(())
    }

    /// A plain point sampler at slot `index`, for sampler slots without a
    /// view.
    fn null_sampler(&self, index: u32) {
        null_sampler(self.cpu(index));
    }
}

/// A plain point sampler at `at`.
fn null_sampler(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_POINT,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        ..Default::default()
    };
    // SAFETY: a slot of a sampler heap of ours.
    unsafe { host.device.CreateSampler(&desc, at) };
}

impl HostHeap {
    /// `n` consecutive free dynamic slots (view heaps only), taken until
    /// [`HostHeap::release`]: the first's index.
    pub(crate) fn dynamic(&self, n: u32) -> Option<u32> {
        let cap = self.dynamic;
        if self.sampler || n == 0 || n > cap {
            if !self.sampler && n > cap {
                static TOO_BIG: std::sync::Once = std::sync::Once::new();
                TOO_BIG.call_once(|| {
                    tracing::warn!(
                        n,
                        cap,
                        "a table needs more dynamic descriptor slots than the heap has"
                    );
                });
            }
            return None;
        }
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let (next, taken) = &mut *guard;
        let n = n as usize;
        // Search from `next` on, then from the start.
        let starts = (*next as usize..self.dynamic as usize).chain(0..*next as usize);
        let mut run = 0;
        let mut found = None;
        let mut last = usize::MAX;
        for at in starts {
            // A run cannot wrap past the end.
            if at != last.wrapping_add(1) {
                run = 0;
            }
            last = at;
            run = if taken[at] { 0 } else { run + 1 };
            if run == n {
                found = Some(at + 1 - n);
                break;
            }
        }
        let Some(first) = found else {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(n, "no free dynamic descriptor slots: a table is not made"));
            return None;
        };
        taken[first..first + n].fill(true);
        *next = ((first + n) % self.dynamic as usize) as u32;
        Some(self.count + first as u32)
    }

    /// The title slots this heap has host slots for; fewer than the title's
    /// when the host caps its size.
    pub(crate) fn count(&self) -> u32 {
        self.count
    }

    /// Number of dynamic slots in this heap (0 for sampler heaps).
    pub(crate) fn dynamic_capacity(&self) -> u32 {
        self.dynamic
    }

    /// Gives back `n` dynamic slots from `first` (an index [`dynamic`]
    /// returned).
    ///
    /// [`dynamic`]: HostHeap::dynamic
    pub(crate) fn release(&self, first: u32, n: u32) {
        let Some(at) = first.checked_sub(self.count) else {
            return;
        };
        let mut guard = self.next.lock().unwrap_or_else(|e| e.into_inner());
        let end = (at + n).min(self.dynamic) as usize;
        if let Some(slots) = guard.1.get_mut(at as usize..end) {
            slots.fill(false);
        }
    }

    /// Makes slot `index` (a dynamic one) from `view`; a null view if the
    /// host cannot. Returns whether a host view was made.
    pub(crate) fn make_at(&self, index: u32, view: &View) -> bool {
        let kept = make(view, self.cpu(index));
        let made = kept.is_some();
        if !made {
            null_for(view, self.cpu(index));
        }
        self.keep(index, kept.flatten());
        made
    }

    /// Makes slot `index` read nothing: a null view, or a plain sampler.
    pub(crate) fn make_null(&self, index: u32) {
        if self.sampler {
            self.null_sampler(index);
        } else {
            null_view(self.cpu(index));
        }
        self.keep(index, None);
    }

    /// Slot `index` now views `resource` (or nothing of the host's).
    fn keep(&self, index: u32, resource: Option<ID3D12Resource>) {
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = kept.get_mut(index as usize) {
            *slot = resource;
        }
    }

    /// Makes slot `index` (a dynamic one) a constant buffer view of host
    /// memory (the list's own, which it holds).
    pub(crate) fn constant_at(&self, index: u32, gpu: u64, size: u32) {
        self.keep(index, None);
        let Some(host) = host() else { return };
        let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
            BufferLocation: gpu,
            SizeInBytes: size.next_multiple_of(256),
        };
        // SAFETY: a slot of our heap; the memory is the caller's.
        unsafe {
            host.device
                .CreateConstantBufferView(Some(&desc), self.cpu(index))
        };
    }

    /// Makes slot `index` (a dynamic one) a raw buffer view of `len` bytes
    /// of `buffer` from byte `offset` (a multiple of 16): a root SRV with a
    /// size, whose reads past it return zeros instead of reaching whatever
    /// the host placed after the buffer.
    pub(crate) fn raw_at(&self, index: u32, buffer: &ID3D12Resource, offset: u64, len: u64) {
        let Some(host) = host() else { return };
        let elements = (len / 4).clamp(1, u64::from(u32::MAX)) as u32;
        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_SRV {
                    FirstElement: offset / 4,
                    NumElements: elements,
                    StructureByteStride: 0,
                    Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                },
            },
        };
        // SAFETY: a slot of our heap and a live buffer covering the range.
        unsafe {
            host.device
                .CreateShaderResourceView(buffer, Some(&desc), self.cpu(index))
        };
        self.keep(index, Some(buffer.clone()));
    }

    /// Makes slot `index` a raw buffer view that reads zeros.
    pub(crate) fn null_raw_at(&self, index: u32) {
        null_buffer_view(self.cpu(index));
        self.keep(index, None);
    }

    /// Whether this is a sampler heap.
    pub(crate) fn is_sampler(&self) -> bool {
        self.sampler
    }

    /// The shader-visible handle of slot `index`.
    pub(crate) fn gpu_handle(&self, index: u32) -> D3D12_GPU_DESCRIPTOR_HANDLE {
        // SAFETY: a live shader-visible heap.
        let start = unsafe { self.heap.GetGPUDescriptorHandleForHeapStart().ptr };
        D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: start + u64::from(index) * u64::from(self.step),
        }
    }

    fn cpu(&self, index: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: self.cpu + index as usize * self.step as usize,
        }
    }

}

struct HostBackend<'a> {
    heap: &'a HostHeap,
    #[cfg(feature = "diagnostics")]
    diag: Option<(u8, Option<u32>)>,
}

impl MirrorBackend for HostBackend<'_> {
    fn lookup(&self, i: u32, generation: u32) -> Option<(View, u64)> {
        view_at_epoch(i, generation)
    }

    fn check_live(&self, pairs: &[(u32, u32)], out: &mut Vec<bool>) -> u64 {
        check_live(pairs, out)
    }

    fn forget_epoch(&self) -> u64 {
        forget_epoch()
    }

    fn is_buffer(&self, view: &View) -> bool {
        view.is_buffer()
    }

    fn make_view(&self, index: u32, view: &View, later: bool) -> Option<SlotSync> {
        let kept = if later {
            None
        } else {
            make(view, self.heap.cpu(index))
        };
        let made = kept.is_some();
        if !made && !self.heap.sampler {
            null_for(view, self.heap.cpu(index));
        }
        self.heap.keep(index, kept.flatten());
        made.then(|| slot_sync(view)).flatten()
    }

    fn make_null(&self, index: u32) {
        self.heap.make_null(index);
    }

    #[cfg(feature = "diagnostics")]
    fn note_unknown(&self, index: u32, handle: u64) {
        if !self.heap.sampler
            && let Some((root_param, user_data)) = self.diag
        {
            super::note_unknown_handle(index, handle, root_param, user_data);
        }
    }
}

/// Extracts the host buffer or texture synchronization needed when a draw or
/// dispatch reads `view`.
pub(crate) fn slot_sync(view: &View) -> Option<SlotSync> {
    let placed = placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    if let Some((address, len)) = buffer_range(view)
        && len > 0
        && let Some(shadow) = crate::shadow::find(address, len)
    {
        let uav = matches!(view, View::Unordered(_, _, _) | View::Placed(true, _, _));
        return Some(SlotSync::Buffer {
            shadow,
            address,
            len,
            uav,
        });
    }
    match view {
        View::Shader(resource, _) => {
            let _ = crate::resource::host_resource(*resource as crate::object::This)?;
            Some(SlotSync::Texture {
                resource: *resource,
                uav: false,
            })
        }
        View::Unordered(resource, _, _) => {
            let _ = crate::resource::host_resource(*resource as crate::object::This)?;
            Some(SlotSync::Texture {
                resource: *resource,
                uav: true,
            })
        }
        _ => None,
    }
}

impl HostHeap {
    /// Makes host slot `index` from the record the title has at `handle`
    /// if it changed since. Buffer views are made only for draws (`for_draw`),
    /// which also copy their data: a host copy of a buffer is made on first
    /// use.
    pub(crate) fn mirror(&self, index: u32, handle: u64, for_draw: bool) {
        if index >= self.count {
            // Diagnostic (NVIDIA heap cap): the highest title slot past the
            // host heap seen so far.
            static HIGHEST: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if HIGHEST.fetch_max(index, std::sync::atomic::Ordering::Relaxed) < index {
                tracing::warn!(
                    index,
                    count = self.count,
                    for_draw,
                    "title slot past the host heap"
                );
            }
            return;
        }
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        cache.sync_range(
            index,
            handle,
            0,
            1,
            for_draw,
            &HostBackend {
                heap: self,
                #[cfg(feature = "diagnostics")]
                diag: None,
            },
            None,
        );
    }

    /// Mirrors `count` contiguous title descriptor slots starting at
    /// `(first, first_handle)` for a draw or dispatch, appending any host
    /// resource sync work into `out_syncs` as `(offset_in_range, sync)`.
    pub(crate) fn mirror_range(
        &self,
        first: u32,
        first_handle: u64,
        step: u32,
        count: u32,
        out_syncs: &mut Vec<(u32, SlotSync)>,
        #[cfg(feature = "diagnostics")] root: (u8, Option<u32>),
    ) {
        let avail = self.count.saturating_sub(first);
        let count = count.min(avail);
        if count == 0 {
            out_syncs.clear();
            return;
        }
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        cache.sync_range(
            first,
            first_handle,
            step,
            count,
            true,
            &HostBackend {
                heap: self,
                #[cfg(feature = "diagnostics")]
                diag: Some(root),
            },
            Some(out_syncs),
        );
    }

    /// The view host slot `index` was made from, in a few words, with the
    /// title and host addresses of a buffer view (diagnostics). "null" for
    /// a slot never made or left null.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn describe_slot(&self, index: u32) -> Option<String> {
        let key = self
            .cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .made_key(index)?;
        if key == 0 {
            return Some("null".into());
        }
        let view = view_at_epoch(key as u32, (key >> 32) as u32)?.0;
        let mut text: String = format!("{view:?}").chars().take(120).collect();
        if let Some((address, len)) = buffer_range(&view) {
            let host = crate::shadow::find(address, len.max(1)).map(|s| s.gpu(address));
            text.push_str(&format!(" title {address:#x}+{len:#x} host {host:#x?}"));
        }
        Some(text)
    }
}

/// Logs `what` the first time it happens.
macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A view description from its words, with the format as the host has it.
fn srv_desc(words: &[u32; 10]) -> D3D12_SHADER_RESOURCE_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_SHADER_RESOURCE_VIEW_DESC>()
            .read_unaligned()
    }
}

fn uav_desc(words: &[u32; 10]) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut words = *words;
    words[0] = crate::resource::pc_format(words[0]);
    // SAFETY: 40 bytes in the host's layout.
    unsafe {
        std::ptr::from_ref(&words)
            .cast::<D3D12_UNORDERED_ACCESS_VIEW_DESC>()
            .read_unaligned()
    }
}

/// A buffer view of title memory at `address` on its host copy: the first
/// element moves by the copy's offset.
/// Bytes per element of a buffer view.
fn element_size(desc: &D3D12_SHADER_RESOURCE_VIEW_DESC) -> u64 {
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_SRV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    }
}

/// The title memory a constant, shader, or unordered access buffer view reads:
/// its address and length.
pub(crate) fn buffer_range(view: &View) -> Option<(u64, u64)> {
    let (base, words): (u64, &[u32; 10]) = match view {
        &View::Constant(address, size) => return Some((address, u64::from(size))),
        View::Shader(r, Some(desc)) => (
            crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64,
            desc,
        ),
        View::Placed(false, resource, desc) if resource[0] == 1 => (
            u64::from(desc[10]) | u64::from(desc[11]) << 32,
            desc[..10].try_into().ok()?,
        ),
        View::Unordered(r, _, Some(desc)) => {
            let base = crate::resource::buffer_bytes(*r as crate::object::This)?.0 as u64;
            return unordered_range(base, desc);
        }
        View::Placed(true, resource, desc) if resource[0] == 1 => {
            let base = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            return unordered_range(base, desc[..10].try_into().ok()?);
        }
        _ => return None,
    };
    let desc = srv_desc(words);
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// [`buffer_range`] of an unordered access view of memory at `base`.
fn unordered_range(base: u64, words: &[u32; 10]) -> Option<(u64, u64)> {
    let desc = uav_desc(words);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32)?.0
    };
    Some((
        base + buffer.FirstElement * element,
        u64::from(buffer.NumElements) * element,
    ))
}

/// The host resource a slot made by [`make`] views, if any.
type Kept = Option<ID3D12Resource>;

fn buffer_srv(
    address: u64,
    len: u64,
    mut desc: D3D12_SHADER_RESOURCE_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    let element = element_size(&desc);
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let offset = shadow.offset(address);
    if element == 0 || offset % element != 0 {
        once!(
            offset,
            element,
            "buffer view not on an element of its host copy"
        );
        return None;
    }
    buffer.FirstElement += offset / element;
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateShaderResourceView(&shadow.buffer, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// An unordered access view of title buffer memory at `address` (`len`
/// bytes) on its host copy, as [`buffer_srv`]; false if the copy cannot be
/// written by shaders (upload memory).
/// shortcut: what shaders write stays in the host copy; title memory (and
/// so the CPU, and copies run on the CPU) does not see it.
fn buffer_uav(
    address: u64,
    len: u64,
    mut desc: D3D12_UNORDERED_ACCESS_VIEW_DESC,
    at: D3D12_CPU_DESCRIPTOR_HANDLE,
) -> Option<Kept> {
    let shadow = crate::shadow::find(address, len)?;
    // SAFETY: a live buffer.
    let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
    if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
        once!("unordered access to a buffer whose host copy shaders cannot write");
        return None;
    }
    // SAFETY: a buffer view's union member.
    let buffer = unsafe { &mut desc.Anonymous.Buffer };
    let element = if buffer.StructureByteStride > 0 {
        u64::from(buffer.StructureByteStride)
    } else if buffer.Flags.0 & D3D12_BUFFER_UAV_FLAG_RAW.0 != 0 {
        4
    } else {
        crate::resource::element(desc.Format.0 as u32).map_or(0, |(bytes, _)| bytes)
    };
    let offset = shadow.offset(address);
    if element == 0 || offset % element != 0 {
        once!(offset, element, "unordered buffer view not on an element of its host copy");
        return None;
    }
    buffer.FirstElement += offset / element;
    buffer.CounterOffsetInBytes = 0;
    let host = host()?;
    // SAFETY: a live buffer and a slot of our heap.
    unsafe {
        host.device
            .CreateUnorderedAccessView(&shadow.buffer, None::<&ID3D12Resource>, Some(&desc), at)
    };
    Some(Some(shadow.buffer.clone()))
}

/// Whether the host can make a shader (or unordered access) view of
/// `texture` in `format` (none: the texture's own). The console allows any
/// view; the host refuses (and removes the device for) views without the
/// resource's permission, of typed depth, and in another format than a
/// typed resource's. A view of a typed texture in another format of its
/// family takes the texture's format.
/// shortcut: such a view (sRGB of UNORM, say) reads without its
/// conversion; cast with castable formats if colors come out wrong.
fn fits(texture: &ID3D12Resource, mut format: Option<&mut DXGI_FORMAT>, unordered: bool) -> bool {
    use crate::resource::{DEPTH_FORMATS, family};
    // SAFETY: a live resource.
    let d = unsafe { texture.GetDesc() };
    let allowed = if unordered {
        d.Flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 != 0
    } else {
        d.Flags.0 & D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE.0 == 0
    };
    let own = d.Format.0 as u32;
    if !allowed || DEPTH_FORMATS.iter().any(|&(typed, _)| typed == own) {
        once!(
            unordered,
            format = own,
            flags = d.Flags.0,
            "view the host does not allow"
        );
        return false;
    }
    // Every family has more than one format, its typeless one first.
    let typeless = family(own) == own && family(own + 1) == own;
    if let Some(format) = format.as_deref_mut()
        && unordered
        && let Some(&(_, linear)) = crate::resource::SRGB
            .iter()
            .find(|&&(s, _)| s == format.0 as u32)
    {
        format.0 = linear as i32;
    }
    let view = format.as_ref().map_or(0, |f| f.0 as u32);
    if view == 0 {
        return !typeless;
    }
    // A packed 16-bit target made R8G8B8A8_UNORM on this GPU
    // (`resource::target_format`): views name the format it has.
    if view != own && crate::resource::target_format(view) == own {
        if let Some(format) = format {
            *format = d.Format;
        }
        return true;
    }
    if family(view) != family(own) {
        once!(view, own, "view format of another family");
        return false;
    }
    if let Some(format) = format
        && !typeless
    {
        *format = d.Format;
    }
    true
}

/// Makes the host descriptor for `view` at `at`: the host resource it
/// views, or `None` if it cannot.
fn make(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Kept> {
    let host = host()?;
    let device = &host.device;
    match view {
        &View::Constant(address, size) => {
            let shadow = crate::shadow::find(address, u64::from(size))?;
            let gpu = shadow.gpu(address);
            if gpu % 256 != 0 {
                once!(
                    address = format_args!("{address:#x}"),
                    "constant buffer not 256-byte aligned on the host"
                );
                return None;
            }
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC {
                BufferLocation: gpu,
                SizeInBytes: size.next_multiple_of(256),
            };
            // SAFETY: a slot of our heap; the range lies in the host copy.
            unsafe { device.CreateConstantBufferView(Some(&desc), at) };
            Some(Some(shadow.buffer.clone()))
        }
        View::Shader(resource, desc) => {
            let resource = *resource as crate::object::This;
            if let Some(texture) = crate::resource::host_resource(resource) {
                let mut desc = desc.as_ref().map(srv_desc);
                if !fits(texture, desc.as_mut().map(|d| &mut d.Format), false) {
                    return None;
                }
                // SAFETY: a live texture and a slot of our heap.
                unsafe {
                    device.CreateShaderResourceView(
                        texture,
                        desc.as_ref().map(std::ptr::from_ref),
                        at,
                    )
                };
                Some(Some(texture.clone()))
            } else if let (Some((data, len)), Some(desc)) =
                (crate::resource::buffer_bytes(resource), desc)
            {
                buffer_srv(data as u64, len, srv_desc(desc), at)
            } else {
                None
            }
        }
        View::Unordered(resource, _counter, desc) => {
            let resource = *resource as crate::object::This;
            let Some(texture) = crate::resource::host_resource(resource) else {
                return match (crate::resource::buffer_bytes(resource), desc) {
                    (Some((data, len)), Some(desc)) => {
                        buffer_uav(data as u64, len, uav_desc(desc), at)
                    }
                    _ => None,
                };
            };
            let mut desc = desc.as_ref().map(uav_desc);
            if !fits(texture, desc.as_mut().map(|d| &mut d.Format), true) {
                return None;
            }
            // SAFETY: a live texture and a slot of our heap.
            unsafe {
                device.CreateUnorderedAccessView(
                    texture,
                    None::<&ID3D12Resource>,
                    desc.as_ref().map(std::ptr::from_ref),
                    at,
                )
            };
            Some(Some(texture.clone()))
        }
        View::Sampler(words) => {
            // SAFETY: 13 words in the host's layout.
            let desc = unsafe {
                std::ptr::from_ref(words)
                    .cast::<D3D12_SAMPLER_DESC>()
                    .read_unaligned()
            };
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::ConsoleSampler(words) => {
            let desc = crate::sampler::console(words);
            // SAFETY: a slot of our sampler heap.
            unsafe { device.CreateSampler(&desc, at) };
            Some(None)
        }
        View::Placed(unordered, resource, desc) => {
            if let Some(view) = placed_texture(view) {
                return make(&view, at);
            }
            const BUFFER: u32 = 1;
            if resource[0] != BUFFER {
                // shortcut: unordered access to a texture the title placed
                // itself is not mirrored; give it a resource as shader
                // views of such textures get when a dispatch writes one.
                once!(dimension = resource[0], "placed unordered texture view not mirrored");
                return None;
            }
            // The resource description's width, after the alignment.
            let len = u64::from(resource[4]) | u64::from(resource[5]) << 32;
            let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
            let words: &[u32; 10] = desc[..10].try_into().unwrap_or(&[0; 10]);
            if *unordered {
                buffer_uav(address, len, uav_desc(words), at)
            } else {
                buffer_srv(address, len, srv_desc(words), at)
            }
        }
    }
}

/// A shader view of a texture the title placed in memory itself, as a view
/// of the resource made for that memory (on first use).
pub(crate) fn placed_texture(view: &View) -> Option<View> {
    const BUFFER: u32 = 1;
    let View::Placed(false, resource, desc) = view else {
        return None;
    };
    if resource[0] == BUFFER {
        return None;
    }
    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
    let texture = crate::resource::texture_at(address, resource)?;
    Some(View::Shader(texture as usize, desc[..10].try_into().ok()))
}

/// A null view of the kind `view` is, for a slot whose view the host
/// cannot make: unordered access (to a buffer or a 2D texture) or a shader
/// view. Writes to it are dropped, reads give zeros.
fn null_for(view: &View, at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let is_buffer = |r: &usize| crate::resource::buffer_bytes(*r as crate::object::This).is_some();
    // Unordered views: whether a buffer, and the title's view dimension.
    let (buffer, dimension) = match view {
        View::Unordered(r, _, desc) => (
            is_buffer(r) || desc.is_some_and(|d| d[1] == 1),
            desc.map(|d| d[1]),
        ),
        View::Placed(true, resource, desc) => (resource[0] == 1, Some(desc[1])),
        // Shader views of buffers read nothing through a null buffer view:
        // a null 2D view where the shader expects a buffer is a mismatch
        // the hardware does not survive in general (GPU-based validation
        // reports it).
        View::Shader(r, desc) if is_buffer(r) || desc.is_some_and(|d| d[1] == 1) => {
            return null_buffer_view(at);
        }
        View::Placed(false, resource, _) if resource[0] == 1 => return null_buffer_view(at),
        // Textures: a null view of the title's dimension (1D, arrays, 3D,
        // cubes), not always 2D.
        View::Shader(_, Some(desc)) => return null_like(desc, at),
        View::Placed(false, _, desc) => {
            return null_like(desc[..10].try_into().unwrap_or(&[0; 10]), at);
        }
        View::Constant(..) => {
            let desc = D3D12_CONSTANT_BUFFER_VIEW_DESC::default();
            // SAFETY: a slot of our heap; a zero location makes a null view.
            return unsafe { host.device.CreateConstantBufferView(Some(&desc), at) };
        }
        _ => return null_view(at),
    };
    let desc = if buffer {
        D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        }
    } else {
        null_uav_desc(dimension)
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device.CreateUnorderedAccessView(
            None::<&ID3D12Resource>,
            None::<&ID3D12Resource>,
            Some(&desc),
            at,
        )
    };
}

/// A null texture UAV of the title's view dimension (1D, 1D or 2D arrays,
/// 3D), 2D if it names none: a 2D null view where the shader writes a 3D
/// texture is a mismatch GPU-based validation reports (the world's wind
/// field, CS_CopyWindFieldBeforeUpdateWithSmooth, 2026-11-24).
fn null_uav_desc(dimension: Option<u32>) -> D3D12_UNORDERED_ACCESS_VIEW_DESC {
    let mut desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
        ..Default::default()
    };
    match dimension.map(|d| D3D12_UAV_DIMENSION(d as i32)) {
        Some(D3D12_UAV_DIMENSION_TEXTURE1D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE1DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1DARRAY;
            desc.Anonymous.Texture1DArray = D3D12_TEX1D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE2DARRAY) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            desc.Anonymous.Texture2DArray = D3D12_TEX2D_ARRAY_UAV {
                ArraySize: 1,
                ..Default::default()
            };
        }
        Some(D3D12_UAV_DIMENSION_TEXTURE3D) => {
            desc.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE3D;
            desc.Anonymous.Texture3D = D3D12_TEX3D_UAV {
                WSize: u32::MAX,
                ..Default::default()
            };
        }
        _ => {}
    }
    desc
}

/// A buffer view that reads zeros (raw, as translated shaders mostly read
/// buffers).
fn null_buffer_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_TYPELESS,
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                NumElements: 1,
                Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A texture view that reads zeros with the dimension (and mip and array
/// counts) of the title's description `words`; 2D if it names none.
fn null_like(words: &[u32; 10], at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    const TEXTURES: std::ops::RangeInclusive<u32> = 2..=10;
    if !TEXTURES.contains(&words[1]) {
        return null_view(at);
    }
    let Some(host) = host() else { return };
    // One mip, one slice or cube, plane 0: fields a null view still checks.
    let mut w = [0u32; 10];
    w[1] = words[1];
    match words[1] {
        6 => {}
        7 => w[5] = 1,
        3 | 5 | 10 => (w[5], w[7]) = (1, 1),
        _ => w[5] = 1,
    }
    let mut desc = srv_desc(&w);
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

/// A view that reads zeros, for slots whose view the host cannot make.
/// shortcut: slots with no view at all (kind unknown) get this null 2D
/// view; null buffer views there did not stop the world's low-address
/// page faults (2026-11-24, run w10), so they are not the cause.
fn null_view(at: D3D12_CPU_DESCRIPTOR_HANDLE) {
    let Some(host) = host() else { return };
    let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MipLevels: 1,
                ..Default::default()
            },
        },
    };
    // SAFETY: a slot of our heap; a null resource is allowed with a desc.
    unsafe {
        host.device
            .CreateShaderResourceView(None::<&ID3D12Resource>, Some(&desc), at)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A null texture UAV keeps the title's dimension (3D for the world's
    /// wind field), and is 2D when the title names none.
    #[test]
    fn null_uavs_keep_the_titles_dimension() {
        let d = null_uav_desc(Some(8));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE3D);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture3D.WSize }, u32::MAX);
        let d = null_uav_desc(Some(5));
        assert_eq!(d.ViewDimension, D3D12_UAV_DIMENSION_TEXTURE2DARRAY);
        // SAFETY: the member the dimension names.
        assert_eq!(unsafe { d.Anonymous.Texture2DArray.ArraySize }, 1);
        let dimension = |d| null_uav_desc(d).ViewDimension;
        assert_eq!(dimension(Some(2)), D3D12_UAV_DIMENSION_TEXTURE1D);
        assert_eq!(dimension(Some(4)), D3D12_UAV_DIMENSION_TEXTURE2D);
        assert_eq!(dimension(None), D3D12_UAV_DIMENSION_TEXTURE2D);
        // Multisampled UAVs need a cap; they stay 2D.
        assert_eq!(dimension(Some(6)), D3D12_UAV_DIMENSION_TEXTURE2D);
    }

    /// A null 3D UAV made on the real device leaves it alive.
    #[test]
    fn null_3d_uav_is_accepted_by_the_device() {
        let Some(host) = host() else {
            return; // no host GPU
        };
        let Some(heap) = HostHeap::new(false, 4) else {
            return;
        };
        let view = View::Unordered(0, 0, Some([28, 8, 0, 0, 0, 0, 0, 0, 0, 0]));
        null_for(&view, heap.cpu(0));
        // SAFETY: a plain device query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }

    /// Dynamic slots stay taken until given back, and runs do not wrap.
    #[test]
    fn dynamic_slots_are_held_until_released() {
        let Some(heap) = HostHeap::new(false, 4) else {
            return; // no host GPU
        };
        let a = heap.dynamic(heap.dynamic - 100).unwrap();
        let b = heap.dynamic(60).unwrap();
        assert_eq!((a, b), (4, 4 + heap.dynamic - 100));
        // 40 left at the end: 50 do not fit until something is given back.
        assert_eq!(heap.dynamic(50), None);
        heap.release(a, 50);
        assert_eq!(heap.dynamic(50), Some(4));
        assert_eq!(heap.dynamic(40), Some(4 + heap.dynamic - 40));
    }

    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Debug, PartialEq, Eq)]
    enum MockOp {
        MakeView(u32, bool),
        MakeNull(u32),
        CheckLive(usize),
    }

    struct MockBackend {
        epoch: AtomicU64,
        entries: Mutex<std::collections::HashMap<(u32, u32), View>>,
        ops: Mutex<Vec<MockOp>>,
        #[cfg(feature = "diagnostics")]
        unknown: Mutex<Vec<(u32, u64)>>,
    }

    impl MockBackend {
        fn new(initial: &[(u32, u32, View)]) -> Self {
            let mut entries = std::collections::HashMap::new();
            for &(i, generation, ref v) in initial {
                entries.insert((i, generation), v.clone());
            }
            Self {
                epoch: AtomicU64::new(1),
                entries: Mutex::new(entries),
                ops: Mutex::new(Vec::new()),
                #[cfg(feature = "diagnostics")]
                unknown: Mutex::new(Vec::new()),
            }
        }

        fn take_ops(&self) -> Vec<MockOp> {
            std::mem::take(&mut *self.ops.lock().unwrap())
        }
    }

    impl MirrorBackend for MockBackend {
        fn lookup(&self, i: u32, generation: u32) -> Option<(View, u64)> {
            let view = self.entries.lock().unwrap().get(&(i, generation)).cloned()?;
            Some((view, self.epoch.load(Ordering::Acquire)))
        }

        fn check_live(&self, pairs: &[(u32, u32)], out: &mut Vec<bool>) -> u64 {
            out.clear();
            let entries = self.entries.lock().unwrap();
            for &(i, generation) in pairs {
                out.push(entries.contains_key(&(i, generation)));
            }
            self.ops.lock().unwrap().push(MockOp::CheckLive(pairs.len()));
            self.epoch.load(Ordering::Acquire)
        }

        fn forget_epoch(&self) -> u64 {
            self.epoch.load(Ordering::Acquire)
        }

        fn is_buffer(&self, view: &View) -> bool {
            matches!(view, View::Constant(..) | View::Placed(..))
        }

        fn make_view(&self, index: u32, view: &View, later: bool) -> Option<SlotSync> {
            self.ops.lock().unwrap().push(MockOp::MakeView(index, later));
            if later {
                return None;
            }
            match view {
                View::Shader(r, _) => Some(SlotSync::Texture {
                    resource: *r,
                    uav: false,
                }),
                View::Unordered(r, _, _) => Some(SlotSync::Texture {
                    resource: *r,
                    uav: true,
                }),
                _ => None,
            }
        }

        fn make_null(&self, index: u32) {
            self.ops.lock().unwrap().push(MockOp::MakeNull(index));
        }

        #[cfg(feature = "diagnostics")]
        fn note_unknown(&self, index: u32, handle: u64) {
            self.unknown.lock().unwrap().push((index, handle));
        }
    }

    fn write_rec(slot: &mut [u32; 8], i: u32, generation: u32) {
        slot[..3].copy_from_slice(&[VIEW_TAG, i, generation]);
    }

    fn sync_tags(syncs: &[(u32, SlotSync)]) -> Vec<(u32, usize, bool)> {
        syncs
            .iter()
            .filter_map(|(k, s)| match s {
                SlotSync::Texture { resource, uav } => Some((*k, *resource, *uav)),
                SlotSync::Buffer { .. } => None,
            })
            .collect()
    }

    #[test]
    fn mirror_cache_sync_range_tracks_memcpy_overwrite_and_zero_clear() {
        let srv = View::Shader(0x1000, None);
        let uav = View::Unordered(0x2000, 0, None);
        let backend = MockBackend::new(&[(1, 1, srv), (2, 1, uav)]);
        let mut cache = MirrorCache::new(16);
        let mut heap = vec![[0u32; 8]; 8];
        // Populate skipped slot 0 and dynamically indexed slot 3 by raw memory writes.
        write_rec(&mut heap[0], 1, 1);
        write_rec(&mut heap[3], 2, 1);
        let mut syncs = Vec::new();
        let table_base = heap.as_ptr() as u64;

        cache.sync_range(4, table_base, 32, 6, true, &backend, Some(&mut syncs));
        assert_eq!(
            backend.take_ops(),
            vec![MockOp::MakeView(4, false), MockOp::MakeView(7, false)]
        );
        assert!(cache.is_made(4));
        assert!(!cache.is_made(5));
        assert!(cache.is_made(7));
        assert_eq!(cache.made_key(7), Some((1u64 << 32) | 2));
        assert_eq!(sync_tags(&syncs), vec![(0, 0x1000, false), (3, 0x2000, true)]);

        // Re-scanning an unchanged range performs zero host descriptor writes.
        cache.sync_range(4, table_base, 32, 6, true, &backend, Some(&mut syncs));
        assert!(backend.take_ops().is_empty());
        assert_eq!(sync_tags(&syncs), vec![(0, 0x1000, false), (3, 0x2000, true)]);

        // Plain memcpy overwrites slot 3 with another record and zeroes slot 0.
        heap[0][..3].copy_from_slice(&[0, 0, 0]);
        let src = heap[3];
        let mut replacement = [0u32; 8];
        write_rec(&mut replacement, 1, 1);
        heap[3].copy_from_slice(&replacement);
        assert_ne!(src, heap[3]);

        cache.sync_range(4, table_base, 32, 6, true, &backend, Some(&mut syncs));
        assert_eq!(
            backend.take_ops(),
            vec![MockOp::MakeNull(4), MockOp::MakeView(7, false)]
        );
        assert!(!cache.is_made(4));
        assert_eq!(cache.made_key(7), Some((1u64 << 32) | 1));
        assert_eq!(sync_tags(&syncs), vec![(3, 0x1000, false)]);
    }

    #[test]
    fn mirror_cache_invalidates_released_views_after_forget_epoch() {
        let srv_a = View::Shader(0x1000, None);
        let srv_b = View::Shader(0x2000, None);
        let backend = MockBackend::new(&[(1, 1, srv_a), (2, 1, srv_b)]);
        let mut cache = MirrorCache::new(8);
        let mut heap = [[0u32; 8]; 4];
        write_rec(&mut heap[0], 1, 1);
        write_rec(&mut heap[2], 2, 1);
        let mut syncs = Vec::new();
        let table_base = heap.as_ptr() as u64;

        cache.sync_range(0, table_base, 32, 4, true, &backend, Some(&mut syncs));
        assert_eq!(
            backend.take_ops(),
            vec![MockOp::MakeView(0, false), MockOp::MakeView(2, false)]
        );

        // Simulate resource destruction releasing entry (2, 1) and bumping forget epoch.
        backend.entries.lock().unwrap().remove(&(2, 1));
        backend.epoch.store(2, Ordering::Release);

        cache.sync_range(0, table_base, 32, 4, true, &backend, Some(&mut syncs));
        assert_eq!(
            backend.take_ops(),
            vec![MockOp::CheckLive(2), MockOp::MakeNull(2)]
        );
        assert!(cache.is_made(0));
        assert!(!cache.is_made(2));
        assert_eq!(sync_tags(&syncs), vec![(0, 0x1000, false)]);

        // Subsequent draws at the same epoch skip re-validation.
        cache.sync_range(0, table_base, 32, 4, true, &backend, Some(&mut syncs));
        assert!(backend.take_ops().is_empty());
        assert_eq!(sync_tags(&syncs), vec![(0, 0x1000, false)]);
    }

    #[test]
    fn mirror_cache_defers_buffers_off_draw_then_makes_on_draw() {
        let cbuf = View::Constant(0x4000, 256);
        let backend = MockBackend::new(&[(5, 1, cbuf)]);
        let mut cache = MirrorCache::new(4);
        let mut heap = [[0u32; 8]; 1];
        write_rec(&mut heap[0], 5, 1);
        let base = heap.as_ptr() as u64;

        cache.sync_range(0, base, 0, 1, false, &backend, None);
        assert_eq!(backend.take_ops(), vec![MockOp::MakeView(0, true)]);
        assert!(!cache.is_made(0));

        cache.sync_range(0, base, 0, 1, true, &backend, None);
        assert_eq!(backend.take_ops(), vec![MockOp::MakeView(0, false)]);
        assert!(cache.is_made(0));
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn mirror_cache_sync_range_notes_unknown_slots_only_for_draws() {
        let srv = View::Shader(0x1000, None);
        let backend = MockBackend::new(&[(1, 1, srv)]);
        let mut cache = MirrorCache::new(8);
        let mut heap = [[0u32; 8]; 3];
        write_rec(&mut heap[0], 1, 1);
        heap[1] = [0xabcd_1234, 2, 0x10, 0x9000_0fac, 0, 0, 0, 0];
        let base = heap.as_ptr() as u64;

        cache.sync_range(4, base, 32, 3, false, &backend, None);
        assert!(backend.unknown.lock().unwrap().is_empty());

        cache.sync_range(4, base, 32, 3, true, &backend, None);
        let noted = std::mem::take(&mut *backend.unknown.lock().unwrap());
        assert_eq!(noted, vec![(5, base + 32), (6, base + 64)]);
    }

    #[test]
    fn plan_table_covers_finite_ranges_and_skipped_slots() {
        let sig = [
            DescriptorRange {
                kind: RangeKind::Srv,
                offset: 0,
                count: Some(4),
            },
            DescriptorRange {
                kind: RangeKind::Uav,
                offset: 6,
                count: Some(2),
            },
        ];
        // Shader statically declared only s0+2; s0+0..1, s0+3, u0..1 are dynamic/skipped.
        let used = [UsedSlot {
            slot_offset: 2,
            kind: ResourceType::Srv,
        }];
        let plan = plan_table(
            Some(&sig),
            &used,
            10,
            1000,
            1000,
            1024,
            false,
            0x1000,
            32,
            |_| false,
            |_| false,
        );
        assert_eq!(
            plan,
            TablePlan::Static {
                ranges: vec![
                    TableRange {
                        start: 0,
                        end: 4,
                        kind: Some(ResourceType::Srv),
                    },
                    TableRange {
                        start: 6,
                        end: 8,
                        kind: Some(ResourceType::Uav),
                    },
                ],
            }
        );
    }

    #[test]
    fn plan_table_bounds_unbounded_ranges_on_static_and_fresh() {
        let sig = [DescriptorRange {
            kind: RangeKind::Srv,
            offset: 0,
            count: None,
        }];
        let used_srv = [UsedSlot {
            slot_offset: 1,
            kind: ResourceType::Srv,
        }];
        let static_plan = plan_table(
            Some(&sig),
            &used_srv,
            100,
            10_000,
            10_000,
            1 << 16,
            false,
            0x10_0000,
            32,
            |_| false,
            |_| false,
        );
        assert_eq!(
            static_plan,
            TablePlan::Static {
                ranges: vec![TableRange {
                    start: 0,
                    end: UNBOUNDED_STATIC_LIMIT,
                    kind: Some(ResourceType::Srv),
                }],
            }
        );

        // Using a CBV forces Fresh; unbounded SRV range trims to highest written record + 1.
        let used_cbv = [UsedSlot {
            slot_offset: 0,
            kind: ResourceType::Cbv,
        }];
        let base = 0x20_0000u64;
        let fresh_plan = plan_table(
            Some(&sig),
            &used_cbv,
            0,
            10_000,
            10_000,
            1 << 16,
            false,
            base,
            32,
            |_| false,
            |at| matches!((at - base) / 32, 0 | 5 | 11),
        );
        assert_eq!(
            fresh_plan,
            TablePlan::Fresh {
                n: 12,
                ranges: vec![TableRange {
                    start: 0,
                    end: 12,
                    kind: Some(ResourceType::Srv),
                }],
            }
        );

        // Populated records past UNBOUNDED_FRESH_LIMIT are bounded at UNBOUNDED_FRESH_LIMIT.
        let capped_fresh = plan_table(
            Some(&sig),
            &used_cbv,
            0,
            10_000,
            10_000,
            1 << 16,
            false,
            base,
            32,
            |_| false,
            |_| true,
        );
        assert_eq!(
            capped_fresh,
            TablePlan::Fresh {
                n: UNBOUNDED_FRESH_LIMIT,
                ranges: vec![TableRange {
                    start: 0,
                    end: UNBOUNDED_FRESH_LIMIT,
                    kind: Some(ResourceType::Srv),
                }],
            }
        );
    }

    #[test]
    fn plan_table_handles_capped_heaps_samplers_and_unaligned_cbvs() {
        let unbounded_srv = [DescriptorRange {
            kind: RangeKind::Srv,
            offset: 0,
            count: None,
        }];
        let used = [UsedSlot {
            slot_offset: 0,
            kind: ResourceType::Srv,
        }];
        let base = 0x30_0000u64;

        // Capped heap stays static when unbounded table has no populated slots >= host_count.
        let clamped_static = plan_table(
            Some(&unbounded_srv),
            &used,
            10,
            200,
            100,
            1024,
            false,
            base,
            32,
            |_| false,
            |at| (at - base) / 32 == 5,
        );
        assert_eq!(
            clamped_static,
            TablePlan::Static {
                ranges: vec![TableRange {
                    start: 0,
                    end: 90,
                    kind: Some(ResourceType::Srv),
                }],
            }
        );

        // Populating a slot past host_count promotes to Fresh on NVIDIA capped heaps.
        let promoted_unbounded = plan_table(
            Some(&unbounded_srv),
            &used,
            10,
            200,
            100,
            1024,
            false,
            base,
            32,
            |_| false,
            |at| (at - base) / 32 == 95,
        );
        assert_eq!(
            promoted_unbounded,
            TablePlan::Fresh {
                n: 96,
                ranges: vec![TableRange {
                    start: 0,
                    end: 96,
                    kind: Some(ResourceType::Srv),
                }],
            }
        );

        // An unbounded secondary range whose offset starts at/past host_count also promotes
        // when populated.
        let offset_unbounded = [
            DescriptorRange {
                kind: RangeKind::Srv,
                offset: 0,
                count: Some(2),
            },
            DescriptorRange {
                kind: RangeKind::Uav,
                offset: 10,
                count: None,
            },
        ];
        let promoted_offset = plan_table(
            Some(&offset_unbounded),
            &used,
            90,
            200,
            100,
            1024,
            false,
            base,
            32,
            |_| false,
            |at| (at - base) / 32 == 12,
        );
        assert_eq!(
            promoted_offset,
            TablePlan::Fresh {
                n: 13,
                ranges: vec![
                    TableRange {
                        start: 0,
                        end: 2,
                        kind: Some(ResourceType::Srv),
                    },
                    TableRange {
                        start: 10,
                        end: 13,
                        kind: Some(ResourceType::Uav),
                    },
                ],
            }
        );

        // Finite table crossing host_count also promotes to Fresh.
        let finite_srv = [DescriptorRange {
            kind: RangeKind::Srv,
            offset: 0,
            count: Some(10),
        }];
        let promoted_finite = plan_table(
            Some(&finite_srv),
            &used,
            95,
            200,
            100,
            1024,
            false,
            base,
            32,
            |_| false,
            |_| false,
        );
        assert_eq!(
            promoted_finite,
            TablePlan::Fresh {
                n: 10,
                ranges: vec![TableRange {
                    start: 0,
                    end: 10,
                    kind: Some(ResourceType::Srv),
                }],
            }
        );

        // Unaligned CBV inside a root-signature CBV range promotes to Fresh even if not in used.
        let cbv_range = [DescriptorRange {
            kind: RangeKind::Cbv,
            offset: 0,
            count: Some(4),
        }];
        let cbv_fresh = plan_table(
            Some(&cbv_range),
            &[],
            0,
            1000,
            1000,
            1024,
            false,
            base,
            32,
            |at| (at - base) / 32 == 2,
            |_| false,
        );
        assert_eq!(
            cbv_fresh,
            TablePlan::Fresh {
                n: 4,
                ranges: vec![TableRange {
                    start: 0,
                    end: 4,
                    kind: Some(ResourceType::Cbv),
                }],
            }
        );

        // Sampler heaps always stay static.
        let sampler_range = [DescriptorRange {
            kind: RangeKind::Sampler,
            offset: 0,
            count: Some(8),
        }];
        let sampler_plan = plan_table(
            Some(&sampler_range),
            &[],
            0,
            2048,
            2048,
            0,
            true,
            base,
            16,
            |_| true,
            |_| true,
        );
        assert_eq!(
            sampler_plan,
            TablePlan::Static {
                ranges: vec![TableRange {
                    start: 0,
                    end: 8,
                    kind: Some(ResourceType::Sampler),
                }],
            }
        );
    }

    #[test]
    fn parsed_root_signature_ranges_drive_plan_table() {
        use crate::object::{Object, S_OK, This, slot};
        use windows::Win32::Graphics::Direct3D12::{
            D3D12_DESCRIPTOR_RANGE, D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
            D3D12_DESCRIPTOR_RANGE_TYPE_SRV, D3D12_DESCRIPTOR_RANGE_TYPE_UAV, D3D12_ROOT_CONSTANTS,
            D3D12_ROOT_DESCRIPTOR_TABLE, D3D12_ROOT_PARAMETER, D3D12_ROOT_PARAMETER_0,
            D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS, D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            D3D12_SHADER_VISIBILITY_ALL,
        };

        let ranges = [
            D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: 3,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
                OffsetInDescriptorsFromTableStart: 1,
            },
            D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                NumDescriptors: u32::MAX,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
                OffsetInDescriptorsFromTableStart: D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
            },
        ];
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: 4,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: ranges.len() as u32,
                        pDescriptorRanges: ranges.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];
        let desc = crate::root::Desc {
            parameter_count: params.len() as u32,
            parameters: params.as_ptr().cast(),
            sampler_count: 0,
            samplers: std::ptr::null(),
            flags: 0,
        };
        let (mut blob, mut error) = (std::ptr::null_mut(), std::ptr::null_mut());
        assert_eq!(crate::root::serialize(&desc, 1, &mut blob, &mut error), S_OK);
        let blob = blob.cast_const().cast::<Object>();
        let data = slot::<extern "system" fn(This) -> *const u8>(blob, 0x18)(blob);
        let size = slot::<extern "system" fn(This) -> usize>(blob, 0x20)(blob);
        // SAFETY: `blob` is a live ID3DBlob of `size` bytes returned by `serialize`.
        let bytes = unsafe { std::slice::from_raw_parts(data, size) };
        let parsed = crate::root::parse_tables(bytes).expect("root signature parses");
        slot::<extern "system" fn(This) -> u32>(blob, 0x10)(blob);

        assert_eq!(parsed.len(), 2);
        assert!(parsed[0].is_none());
        let rt = parsed[1].as_deref().unwrap();
        assert_eq!(
            rt,
            &[
                DescriptorRange {
                    kind: RangeKind::Srv,
                    offset: 1,
                    count: Some(3),
                },
                DescriptorRange {
                    kind: RangeKind::Uav,
                    offset: 4,
                    count: None,
                },
            ]
        );
        let plan = plan_table(
            Some(rt),
            &[],
            0,
            10_000,
            10_000,
            1024,
            false,
            0x40_0000,
            32,
            |_| false,
            |_| false,
        );
        assert_eq!(
            plan,
            TablePlan::Static {
                ranges: vec![
                    TableRange {
                        start: 1,
                        end: 4,
                        kind: Some(ResourceType::Srv),
                    },
                    TableRange {
                        start: 4,
                        end: 4 + UNBOUNDED_STATIC_LIMIT,
                        kind: Some(ResourceType::Uav),
                    },
                ],
            }
        );
    }
}
